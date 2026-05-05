# Bounded ingress with fail-closed gap marker

The plan: swap the unbounded ingress channel for a bounded one, and on `TrySendError::Full`, **record the drop and push a gap marker into the journal** so every downstream consumer fails closed exactly the same way they already do for reorg-too-deep / oversized-entry cases. This reuses the existing `complete: false` machinery end-to-end — no new failure mode for subscribers or filters to learn.

---

## 1. Policy recap and why (b) fits

Existing fail-closed paths in `logs_journal.rs`:
- `build_journal_payload` returns `(complete=false, vec![])` when a delta exceeds `max_blocks_per_entry` / `max_logs_per_entry` / `max_bytes_per_entry`.
- The worker pushes a gap marker on stream reconnect (deduplicated).
- Subscribers break out on `!entry.complete`.
- `snapshot_since` returns `IncompleteEntry` when a cursor lands on one, and filter.rs removes that filter from the pool.

So: any overflow on ingress should funnel into exactly that same path. A dropped notification = a block we can't describe = the journal is now incomplete from that seq onward for anyone who cares. Clients then re-bootstrap via `eth_getLogs`, which is the intended recovery mechanism.

---

## 2. Bounded-ingress `SinkRegistry`

Extend the registry from the earlier patch with a `try_broadcast` and a capacity parameter per-registration:

```rust
use sc_utils::mpsc::{tracing_unbounded, TracingUnboundedSender, TracingUnboundedReceiver};
// Swap to a bounded channel. Pick ONE:
//   - `tokio::sync::mpsc` (bounded, async send, async recv).
//   - `futures::channel::mpsc` (bounded, sync try_send, async recv; matches
//      fc-mapping-sync's existing style better).
// I'll use futures::channel::mpsc because the producer side is a synchronous
// block-import notification handler that can't `.await`.
use futures::channel::mpsc as fmpsc;

pub struct SinkRegistry<T> {
    inner: Mutex<Slab<RegisteredSink<T>>>,
}

struct RegisteredSink<T> {
    sender: fmpsc::Sender<T>,
    /// Counts notifications dropped due to `TrySendError::Full` since last observed.
    dropped_since_last_send: AtomicU64,
    name: &'static str,
}

pub struct BroadcastOutcome {
    pub delivered: usize,
    pub dropped_full: usize,
    pub dropped_closed: usize,
}

impl<T: Clone + Send + 'static> SinkRegistry<T> {
    pub fn register(
        self: &Arc<Self>,
        name: &'static str,
        capacity: usize,
    ) -> (SinkGuard<T>, fmpsc::Receiver<T>) {
        let (sender, receiver) = fmpsc::channel(capacity);
        let key = self.inner.lock().insert(RegisteredSink {
            sender,
            dropped_since_last_send: AtomicU64::new(0),
            name,
        });
        (SinkGuard { registry: Arc::downgrade(self), key: Some(key) }, receiver)
    }

    /// Producer-side broadcast. Never blocks, never allocates on zero subs.
    pub fn try_broadcast(&self, mut make_msg: impl FnMut() -> T) -> BroadcastOutcome {
        let mut inner = self.inner.lock();
        let mut delivered = 0;
        let mut dropped_full = 0;
        let mut dropped_closed = 0;

        inner.retain(|_, sink| {
            match sink.sender.try_send(make_msg()) {
                Ok(()) => { delivered += 1; true }
                Err(e) if e.is_full() => {
                    sink.dropped_since_last_send.fetch_add(1, Ordering::Relaxed);
                    dropped_full += 1;
                    true // keep the sink; it's just temporarily full
                }
                Err(_) => {
                    dropped_closed += 1;
                    false // receiver dropped; prune
                }
            }
        });

        BroadcastOutcome { delivered, dropped_full, dropped_closed }
    }

    /// Consumer-side helper: atomically take the overflow counter for a sink.
    /// Returns (and resets) the number of notifications dropped since the
    /// last call.
    pub fn take_drop_count(&self, key: usize) -> u64 {
        self.inner
            .lock()
            .get(key)
            .map(|s| s.dropped_since_last_send.swap(0, Ordering::Relaxed))
            .unwrap_or(0)
    }
}
```

Two design notes:

- I kept `try_broadcast` calling `make_msg()` before `try_send` — `try_send` consumes the value. If the message clone is expensive (it's an `EthereumBlockNotification<B>` with two `Vec<Hash>` in reorg info, small-ish), you can optimize by checking `sender.capacity()` first, but honestly: not worth the complexity, broadcast is per-block.
- The guard exposes its `key` so the worker can call `take_drop_count`. Add `pub fn key(&self) -> Option<usize>` to `SinkGuard`.

---

## 3. Worker integration in `logs_journal.rs`

The worker has two responsibilities now:
1. On every received notification, check if there were drops since the last one, and if so, **push a gap marker before processing the current notification**.
2. On disconnect/reconnect, keep existing gap-marker behavior.

```rust
let (mut sink_guard, mut notifications) = sink_registry.register(
    "logs_journal_notification_stream",
    INGRESS_CAPACITY, // e.g., 1024
);

let worker_fut = async move {
    loop {
        while let Some(notification) = notifications.next().await {
            // --- Check for ingress drops since last iteration ---
            if let Some(key) = sink_guard.key() {
                let dropped = sink_registry.take_drop_count(key);
                if dropped > 0 {
                    log::warn!(
                        target: "frontier-logs-journal",
                        "Ingress overflow: {} block notification(s) dropped before seq {}; \
                         inserting gap marker and re-anchoring",
                        dropped,
                        state.current_head_seq(),
                    );
                    // Push a gap marker. This is idempotent/deduped by existing
                    // logic that avoids stacking consecutive incomplete entries.
                    state.push_gap_marker(GapReason::IngressOverflow { dropped });
                    // Metric:
                    LOGS_JOURNAL_INGRESS_DROPS.inc_by(dropped);
                }
            }

            // --- Normal path ---
            let (complete, logs) = build_journal_payload(&notification, &storage_override);
            state.push(complete, logs);
            broadcast_tx.send(/* arc of new entry */).ok();
        }

        // Stream ended — existing reconnect logic with gap marker + backoff.
        state.push_gap_marker(GapReason::StreamReconnect);
        tokio::time::sleep(LOGS_JOURNAL_RECONNECT_BACKOFF).await;
        drop(sink_guard);
        let (g, n) = sink_registry.register("logs_journal_notification_stream", INGRESS_CAPACITY);
        sink_guard = g;
        notifications = n;
    }
};
```

Tagging the `GapReason` is optional but pays off the first time you're debugging an incident. An enum with `IngressOverflow { dropped }`, `StreamReconnect`, `OversizedEntry { blocks, logs, bytes }` lets you count each in Prometheus separately.

---

## 4. Producer (mapping-sync) side

Your previous patch replaced the Vec iteration with `sink_registry.broadcast(|| n.clone())`. Rename to `try_broadcast` and optionally log/meter the outcome:

```rust
let outcome = sink_registry.try_broadcast(|| notification.clone());
if outcome.dropped_full > 0 {
    // Debug log only — the journal worker will log a WARN with the exact count
    // once it resumes. Double-logging here would be noisy.
    log::debug!(
        target: "fc-mapping-sync",
        "try_broadcast: delivered={} dropped_full={} dropped_closed={}",
        outcome.delivered, outcome.dropped_full, outcome.dropped_closed,
    );
}
```

The producer **never blocks** on full ingress — which was the whole point: block import must not be back-pressured by an RPC-layer journal.

---

## 5. Capacity sizing

How deep should `INGRESS_CAPACITY` be?

The journal worker's per-notification cost is:
- `storage_override.current_block(hash)` → one DB read (hot path) or cache hit,
- `storage_override.current_transaction_statuses(hash)` → one DB read,
- `build_journal_payload` clone/filter/bound work,
- `state.push` + `broadcast_tx.send`.

On a healthy node this is sub-millisecond. Under pressure (reorg storm, DB under compaction), it can be 10s of ms. A capacity of `1024` gives roughly 1024 × worst-case = 10–20 seconds of slack before triggering fail-closed. That's usually enough to ride out a DB stall without marking the journal incomplete.

Tune by observing the new `dropped_full` metric in your environment. If you never see drops: capacity is fine or oversized. If you see frequent drops: either increase capacity modestly, *or* lower journal config costs (smaller `max_bytes_per_entry` → less cloning per entry → worker runs faster).

**Do not** set capacity to, say, 100k "to be safe" — that reintroduces the unbounded behavior this patch is fixing, just with a larger limit.

---

## 6. Interaction with the watchdog

Relevant subtlety: a gap marker pushed due to ingress overflow advances `head_seq` by one entry. That means **every current logs subscriber will see one-entry lag** after a gap event. The watchdog's `max_lag_entries = 32` is comfortably above 1, so a single overflow doesn't cause a subscriber kick. But a flurry of overflows (DB severely stuck) could produce many gap markers in quick succession and trigger kicks.

That's *correct* behavior: if the journal is producing gaps faster than subscribers consume them, those subscribers are also behind in real terms and should fail closed. But flag it to operators via:

```rust
// In the watchdog's sweep:
if let CancelReason::LaggedEntries(_) = reason {
    // Annotate: was the lag filled with gap markers?
    // Can be exposed as a derived metric ratio.
}
```

Not strictly necessary — the existing `frontier_logs_journal_watchdog_kicks{reason="lagged_entries"}` metric combined with `frontier_logs_journal_ingress_drops` lets you correlate externally.

---

## 7. `snapshot_since` and filters

No change required. The existing path already handles incomplete entries:
- `snapshot_since(cursor)` returns `IncompleteEntry` when any entry in the requested range has `complete = false`.
- `filter.rs` removes the filter from the pool on that error.

That means ingress overflow causes `eth_getFilterChanges` clients to get an RPC error on their next poll and have to re-install the filter. Standard behavior and matches the documented fail-closed contract.

---

## 8. Summary of the three patches stacked

| Problem | Patch | What it bounds |
|---|---|---|
| 2.1/2.2 sink leaks | `SinkRegistry` + RAII `SinkGuard` | Registry size = live subscribers (no growth) |
| 2.4 broadcast ring pinning | `LogsSubscriberHandle` + watchdog | Per-subscriber lag ≤ 32 entries or 15 s |
| 2.3 unbounded ingress | Bounded `SinkRegistry` + gap marker on drop | Ingress backlog ≤ `INGRESS_CAPACITY` |

Still not covered:
- **2.5**: just drop the `max_total_bytes` default from 512 MiB to 64 MiB in `LogsJournalConfig::default()` — one-line policy change.
- **2.6**: needs API-level change to `PubSubResult::Log(Arc<Log>)`. Worth doing but more invasive; I'd ship the three patches above and measure first.

With all three in place, **every memory source in the logs_journal subsystem has a hard bound proportional to either (a) live subscriber count or (b) compile-time constants**. No time-linear or churn-linear growth remains. If you still see OOM after these, it's almost certainly 2.6 (per-subscriber owned log clones) or something outside `logs_journal` entirely (mapping-sync cache, EVM state cache, etc.), and the diagnostic work becomes cleanly narrowed.

---

Want me to also write the 2.6 patch (convert `PubSubResult::Log` to `Arc<Log>`), or pivot to a combined diagnostic/metrics cheat-sheet for validating all three patches in production?