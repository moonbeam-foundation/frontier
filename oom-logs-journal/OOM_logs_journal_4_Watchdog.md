# Watchdog for 2.4: force-closing slow logs subscribers

The goal: detect and terminate `eth_subscribe("logs")` subscribers whose broadcast receiver has fallen behind the journal's current sequence by more than a threshold, **before** they hit tokio's `RecvError::Lagged` (which only fires at `capacity` lag, i.e. up to 128 entries × 4 MiB = 512 MiB pinned in the ring).

The cleanest design: embed last-delivered-seq tracking directly into each subscriber and have the journal periodically sweep for stragglers. Two sub-designs to pick from — I'll recommend one and implement it.

---

## Design choice

**Option A — cooperative (subscriber self-reports)**: each subscription publishes its last-delivered `seq` into a shared registry via an `AtomicU64`. The journal's head also lives in an `AtomicU64`. A periodic sweeper compares and closes stragglers by firing a `oneshot` cancellation.

**Option B — surveillance (journal tracks every receiver)**: wrap the `broadcast::Receiver` in a type that reports `len()` (tokio exposes this) per tick.

Option A is better because:
- It captures actual *delivery* lag (post-filter, post-serialize) not just ring-buffer lag.
- Works even if the subscriber is stuck in user-code rather than in `recv()`.
- Lets us expose per-subscriber metrics cheaply.

I'll implement Option A, with Option B's `broadcast::Receiver::len()` as a secondary cheap check inside the sweeper.

---

## 1. New subscriber handle type

Add to `client/rpc/src/logs_journal.rs`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;

/// Per-subscription bookkeeping used by the lag watchdog.
///
/// Each `eth_subscribe("logs")` task owns one of these. It updates
/// `last_delivered_seq` every time it successfully forwards an entry to the
/// websocket, and listens on `cancel_rx` for a forced shutdown from the
/// watchdog.
pub struct LogsSubscriberHandle {
    id: u64,
    last_delivered_seq: AtomicU64,
    cancel_tx: Mutex<Option<oneshot::Sender<CancelReason>>>,
    // Captured at subscribe time for diagnostics.
    subscribed_at: std::time::Instant,
}

#[derive(Clone, Copy, Debug)]
pub enum CancelReason {
    LaggedEntries(u64),
    LaggedDuration(std::time::Duration),
    JournalShutdown,
}

impl LogsSubscriberHandle {
    pub fn last_delivered_seq(&self) -> u64 {
        self.last_delivered_seq.load(Ordering::Relaxed)
    }

    pub fn note_delivered(&self, seq: u64) {
        // Monotonic: only advance.
        let mut cur = self.last_delivered_seq.load(Ordering::Relaxed);
        while seq > cur {
            match self.last_delivered_seq.compare_exchange_weak(
                cur, seq, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Called by the watchdog to force-close this subscription.
    /// Returns `false` if cancellation was already fired.
    fn cancel(&self, reason: CancelReason) -> bool {
        if let Some(tx) = self.cancel_tx.lock().take() {
            let _ = tx.send(reason);
            true
        } else {
            false
        }
    }
}
```

---

## 2. Registry and head tracking on `LogsJournal`

Extend `LogsJournal` (or its state) with a subscriber registry and a published-head counter. The head is updated every time the worker appends an entry:

```rust
pub struct LogsJournal {
    // ... existing fields ...
    head_seq: AtomicU64,
    subscribers: Mutex<Slab<Arc<LogsSubscriberHandle>>>,
    next_subscriber_id: AtomicU64,
    watchdog_config: WatchdogConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct WatchdogConfig {
    /// Max allowed `head_seq - last_delivered_seq` before force-close.
    pub max_lag_entries: u64,
    /// Max wall-clock time a subscriber may stay behind by ≥1 entry.
    pub max_lag_duration: std::time::Duration,
    /// How often the sweeper runs.
    pub sweep_interval: std::time::Duration,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            // Roughly half the broadcast ring — kick out well before `Lagged`.
            max_lag_entries: 32,
            max_lag_duration: std::time::Duration::from_secs(15),
            sweep_interval: std::time::Duration::from_secs(2),
        }
    }
}
```

When the worker pushes an entry:

```rust
// Inside the worker, after state.push(...):
self.head_seq.store(entry.seq, Ordering::Relaxed);
```

---

## 3. Subscribe API that returns both the stream and the handle

Replace the current direct `broadcast::Sender::subscribe()` call with a wrapper method on `LogsJournal`:

```rust
impl LogsJournal {
    pub fn subscribe_logs(
        self: &Arc<Self>,
    ) -> (Arc<LogsSubscriberHandle>, broadcast::Receiver<Arc<LogsJournalEntry>>, oneshot::Receiver<CancelReason>) {
        let (cancel_tx, cancel_rx) = oneshot::channel();

        let handle = Arc::new(LogsSubscriberHandle {
            id: self.next_subscriber_id.fetch_add(1, Ordering::Relaxed),
            last_delivered_seq: AtomicU64::new(self.head_seq.load(Ordering::Relaxed)),
            cancel_tx: Mutex::new(Some(cancel_tx)),
            subscribed_at: std::time::Instant::now(),
        });

        let receiver = self.broadcast_tx.subscribe();
        self.subscribers.lock().insert(handle.clone());

        (handle, receiver, cancel_rx)
    }

    /// Called from the handle's Drop via a helper — see section 5.
    fn unregister_subscriber(&self, id: u64) {
        let mut subs = self.subscribers.lock();
        subs.retain(|_, h| h.id != id);
    }
}
```

(If you want to avoid the `retain` scan, store the slab key inside the handle. Minor optimization — skipped for clarity.)

---

## 4. The sweeper task

Spawn this once at journal construction time, alongside the worker:

```rust
fn spawn_watchdog(
    spawn_handle: &SpawnTaskHandle,
    journal: Arc<LogsJournal>,
) {
    let cfg = journal.watchdog_config;
    let fut = async move {
        let mut ticker = tokio::time::interval(cfg.sweep_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        // Per-subscriber first-seen-behind timestamps, keyed by subscriber id.
        let mut behind_since: HashMap<u64, std::time::Instant> = HashMap::new();

        loop {
            ticker.tick().await;

            let head = journal.head_seq.load(Ordering::Relaxed);
            let now = std::time::Instant::now();

            // Snapshot under lock, act outside lock.
            let snapshot: Vec<_> = {
                let subs = journal.subscribers.lock();
                subs.iter().map(|(_, h)| h.clone()).collect()
            };

            let mut live_ids = HashSet::with_capacity(snapshot.len());

            for handle in snapshot {
                live_ids.insert(handle.id);
                let last = handle.last_delivered_seq();
                let lag = head.saturating_sub(last);

                if lag == 0 {
                    behind_since.remove(&handle.id);
                    continue;
                }

                if lag > cfg.max_lag_entries {
                    if handle.cancel(CancelReason::LaggedEntries(lag)) {
                        log::warn!(
                            target: "frontier-logs-journal",
                            "Force-closing logs subscription {} after {} entries behind head (head={}, last={})",
                            handle.id, lag, head, last,
                        );
                    }
                    behind_since.remove(&handle.id);
                    continue;
                }

                // Duration-based check: been behind for too long, even if lag is small.
                let first = *behind_since.entry(handle.id).or_insert(now);
                if now.duration_since(first) > cfg.max_lag_duration {
                    if handle.cancel(CancelReason::LaggedDuration(now.duration_since(first))) {
                        log::warn!(
                            target: "frontier-logs-journal",
                            "Force-closing logs subscription {} after being ≥1 entry behind for {:?} (lag={})",
                            handle.id, now.duration_since(first), lag,
                        );
                    }
                    behind_since.remove(&handle.id);
                }
            }

            // Garbage collect timestamps for dropped subscribers.
            behind_since.retain(|id, _| live_ids.contains(id));
        }
    };

    spawn_handle.spawn("frontier-logs-journal-watchdog", None, fut);
}
```

Two independent triggers:
- **Lag in entries**: catches bursty slow subscribers fast.
- **Lag in wall-clock**: catches subscribers that are always just-a-bit-behind and never catch up — the silent memory pinner that lag-by-entries alone would miss on low-traffic chains.

`MissedTickBehavior::Delay` avoids a thundering sweep after a node pause.

---

## 5. Subscription loop integration (`eth_pubsub.rs`)

Patch the `Kind::Logs` branch to:
1. Use the new `subscribe_logs` API.
2. Call `handle.note_delivered(entry.seq)` after each successful push to the client.
3. `select!` on the broadcast stream + the cancel oneshot + subscription close.
4. Drop the handle (and thereby unregister) on exit.

```rust
Kind::Logs => {
    let filtered_params = FilteredParams::new(Some(filter.clone()));

    let (handle, mut rx, mut cancel_rx) = pubsub.logs_journal.subscribe_logs();

    // RAII unregister via a small guard — because Arc<LogsSubscriberHandle>
    // alone doesn't know about the slab key. Simplest: closure on Drop.
    struct UnregisterOnDrop {
        journal: Arc<LogsJournal>,
        id: u64,
    }
    impl Drop for UnregisterOnDrop {
        fn drop(&mut self) { self.journal.unregister_subscriber(self.id); }
    }
    let _unregister = UnregisterOnDrop {
        journal: pubsub.logs_journal.clone(),
        id: handle.id,
    };

    let sink = pending.accept().await?;
    let mut buffered: BoundedVecDeque<PubSubResult> = BoundedVecDeque::new(16);

    loop {
        tokio::select! {
            biased;

            // 1. Watchdog cancellation wins over delivering more entries.
            reason = &mut cancel_rx => {
                log::debug!(
                    target: "frontier-rpc",
                    "Closing logs subscription (reason: {:?})", reason.ok(),
                );
                break;
            }

            // 2. Sink closed by the client.
            _ = sink.closed() => break,

            // 3. Next journal entry.
            recv = rx.recv() => match recv {
                Ok(entry) => {
                    if !entry.complete {
                        // Matches existing fail-closed policy.
                        break;
                    }
                    for log in entry.logs.iter() {
                        if log_matches_filter(&logs_params, log, false) {
                            let msg = PubSubResult::Log(Box::new(log.clone()));
                            if sink.send(&msg).await.is_err() {
                                return; // client dropped
                            }
                        }
                    }
                    // CRITICAL: only update after successful delivery.
                    handle.note_delivered(entry.seq);
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    log::warn!(
                        target: "frontier-rpc",
                        "Logs subscription lagged by {} entries; closing", n,
                    );
                    break;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    }
    // `_unregister` drops here, removing us from the journal's subscriber list.
}
```

Important subtlety: `note_delivered` is called **after** the `sink.send().await` completes. That means the AtomicU64 reflects the last seq the client actually acked at the TCP level (backpressure-aware). A subscriber that's stuck in `sink.send().await` because the websocket TX buffer is full will *not* advance its counter — which is exactly what the watchdog needs to see.

---

## 6. Wiring + defaults

In `template/node/src/rpc/eth.rs`, the journal construction gets one extra call:

```rust
let logs_journal = LogsJournal::new(/* existing args */, WatchdogConfig::default());
spawn_watchdog(&spawn_handle, logs_journal.clone());
```

And expose tuning knobs as CLI flags if you want operators to dial them (I'd default to the values in `WatchdogConfig::default()` — the math:

- `max_lag_entries = 32`, `max_bytes_per_entry = 4 MiB` → worst-case 128 MiB pinned per slow subscriber before kick.
- Multiply by `receiver_count` in your real deployment to size your memory budget.

---

## 7. Metrics

Three new gauges pay for themselves the first time you debug this:

```rust
frontier_logs_journal_subscribers        // current count
frontier_logs_journal_max_subscriber_lag // max(head - last_delivered) across subs
frontier_logs_journal_watchdog_kicks     // counter, by reason label
```

The sweeper already has all of this in hand — emit during the sweep.

---

## 8. What this does and doesn't close

Closes:
- Broadcast ring slots pinned beyond `max_lag_entries` — **hard cap on journal-side memory pinning per slow subscriber**.
- Silently-always-slightly-behind subscribers — **no more un-evictable Arcs held indefinitely**.
- Clients that vanish without RST — caught by either the lag check or the sweeper's periodic scan (and also by tokio's `sink.closed()`, but the sweeper is a backstop).

Doesn't close:
- A subscriber that's fast enough to keep up but has a deep per-sub `BoundedVecDeque::new(16)` of cloned logs — that's 2.6, separate fix (needs `Arc<Log>` in `PubSubResult`).
- Memory pinned by the `tracing_unbounded` ingress backlog — that's 2.3, which we'll do next.

---

Ready for the bounded-ingress `SinkRegistry` variant? That one's smaller but has a subtle policy question: on ingress overflow, do you (a) drop the notification silently, (b) push a gap marker into the journal and mark the current entry incomplete, or (c) block the producer (bad — back-pressures block import)? I'll recommend (b) since it matches the existing fail-closed design, but tell me if you'd rather (a) for simplicity.