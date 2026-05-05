# Re-audit against current `master`

Good news: the code has evolved substantially since #1862/#1881 and several things I flagged are now hardened. But **the core sink leak is still there** — and now in a more insidious form — plus there are at least two other plausible OOM contributors. Here's the line-by-line re-audit.

---

## 1. What's now hardened (credit where due)

From `logs_journal.rs`:

- **Per-entry caps** (`max_blocks_per_entry = 128`, `max_logs_per_entry = 10_000`, `max_bytes_per_entry = 4 MiB`) — my old #2.4 concern about reorg-induced giant entries is addressed; `build_journal_payload` returns `(false, Vec::new())` and stores an incomplete marker rather than a huge payload.
- **Three-way eviction** on the `VecDeque`: `max_entries` OR `max_total_logs` OR `max_total_bytes`.
- **Broadcast ring clamped**: `broadcast_capacity(max_entries).clamp(1, 4_096)`.
- **Gap-marker dedup**: worker won't stack consecutive incomplete markers.
- **Filter pool self-heals**: in `filter.rs`, when `snapshot_since` returns `CursorTooOld` or `IncompleteEntry`, the filter is removed from the pool — the "replay from seq 0" bug on fresh filters is also fixed (`create_filter` sets `last_log_journal_seq = Some(self.logs_journal.cursor())`).
- **Pub/sub fails closed on lag**: `RecvError::Lagged(n)` closes the subscription (`"Closing logs subscription; lagged behind logs journal by {n} entries"`), and `!entry.complete` closes it too.

So the obvious "infinite VecDeque growth" and "giant reorg blob" paths are closed.

---

## 2. The surviving culprits

### 2.1 ★ The worker reconnect loop still leaks sinks into `pubsub_notification_sinks`

This is the same bug I flagged last time, structurally unchanged. Look carefully:

```rust
fn register_notification_stream(
    pubsub_notification_sinks: &Arc<Mutex<Vec<TracingUnboundedSender<...>>>>,
) -> TracingUnboundedReceiver<EthereumBlockNotification<B>> {
    let (inner_sink, notifications) =
        sc_utils::mpsc::tracing_unbounded("logs_journal_notification_stream", 100_000);
    pubsub_notification_sinks.lock().push(inner_sink);   // ← append-only
    notifications
}
```

And in the worker:

```rust
let worker_fut = async move {
    let mut notifications = initial_notifications;
    loop {
        while let Some(notification) = notifications.next().await { ... }
        // gap marker ...
        tokio::time::sleep(LOGS_JOURNAL_RECONNECT_BACKOFF).await;
        notifications = register_notification_stream(&pubsub_notification_sinks); // ← leak
    }
};
```

Every time `notifications.next()` returns `None`, the worker:

1. Drops the receiver (old sender in the vec becomes a "send to dead receiver"),
2. Sleeps 50 ms,
3. **Appends a fresh sender** to `pubsub_notification_sinks`.

The old sender is never removed. The 50 ms backoff throttles it but does not bound it. If the upstream stream terminates regularly — which it can under these conditions:

- Mapping-sync pipeline restarts / re-spawns the fan-out,
- Any transient panic or drop in the block-import notifier chain,
- The stream simply ending because the fan-out was rebuilt for any reason,

…then `pubsub_notification_sinks` grows linearly with time. Each leaked entry is a `TracingUnboundedSender` with its own channel state. More importantly, **the producer side (fc-mapping-sync) fans out to every sink in that Vec on every block**, so each leaked sink adds per-block CPU cost *and* briefly allocates a message that gets dropped. The vec itself is unbounded.

This alone is a slow, monotonically increasing leak that #1881's byte cap cannot touch, because it lives **outside** the journal state.

**Validation test**: patch the worker to log `pubsub_notification_sinks.lock().len()` after each `register_notification_stream` call. If you see it climb over hours/days on your nodes, this is confirmed.

**Fix**: the sender needs to be removed from the Vec when the worker drops its receiver. Since `TracingUnboundedSender` doesn't implement `PartialEq`, the usual pattern is to store `Weak` handles and prune the Vec lazily, or to keep an index and remove by index on reconnect.

### 2.2 ★★ `eth_subscribe("newHeads")` also leaks sinks (and shares the same Vec for the pubsub path)

This is separate from the journal but uses a shared pattern and is almost certainly contributing. In `eth_pubsub.rs`:

```rust
Kind::NewHeads => {
    let (inner_sink, block_notification_stream) =
        sc_utils::mpsc::tracing_unbounded("pubsub_notification_stream", 100_000);
    pubsub.pubsub_notification_sinks.lock().push(inner_sink);
    let flat_stream = block_notification_stream.flat_map(move |notification| {
        pubsub.new_heads_from_notification(notification)
    });
    PendingSubscription::from(pending)
        .pipe_from_stream(flat_stream, BoundedVecDeque::new(16))
        .await
}
```

Every `eth_subscribe("newHeads")` RPC call **pushes a sender into the shared `pubsub_notification_sinks`**. When the client disconnects and the subscription task exits, the `block_notification_stream` (receiver) is dropped, but **the sender in the shared Vec is never removed**. It stays there for the lifetime of the process.

For nodes serving dapps/indexers that reconnect often (every block explorer page load, every wallet that opens a websocket and re-opens after a timeout), this Vec grows at roughly the rate of new websocket connections. Each leaked sender is still receiving every block notification from mapping-sync fan-out — meaning every block iteration the producer pays O(n) cost and briefly allocates messages for n dead subscribers.

This leak predates the logs_journal feature but the feature **compounds** it: the same `pubsub_notification_sinks` Vec is now shared between the journal worker (which itself leaks — see 2.1) and every newHeads subscription.

**Validation test**: graph `pubsub_notification_sinks.lock().len()` over the lifetime of a node, and correlate with websocket connection churn metrics.

### 2.3 Unbounded ingress channel (`tracing_unbounded`, 100_000 is only a warn)

`register_notification_stream` creates `tracing_unbounded(..., 100_000)`. The 100_000 is a **warning threshold**, not a hard cap. If the worker falls behind because `storage_override.current_block` / `current_transaction_statuses` is slow on a particular node (DB pressure, large reorgs, contended storage cache), the channel grows without bound. Each queued `EthereumBlockNotification<B>` pins `reorg_info: Option<Arc<ReorgInfo<B>>>` which contains `retracted: Vec<B::Hash>` + `enacted: Vec<B::Hash>` — small per message but unbounded in count.

Worse: combined with 2.1 and 2.2, every **leaked** sink has its own per-channel state. Under the right conditions, each leaked sink can accumulate its own backlog if the producer drops messages asynchronously.

### 2.4 Broadcast ring pinning by a single slow `eth_subscribe("logs")` consumer

The broadcast channel is tokio's. Its ring buffer slots are only reclaimed when **every** receiver has advanced past them. With defaults (`max_total_bytes = 512 MiB`, `max_bytes_per_entry = 4 MiB`), `max_entries = 128`, so the broadcast capacity is `min(128, 4096) = 128` slots. Each slot holds an `Arc<LogsJournalEntry>` with up to 4 MiB of logs.

Until a lagging subscriber falls `capacity` behind and triggers `RecvError::Lagged` (which closes the subscription — good), the ring buffer can pin up to ~**512 MiB** of log data *in addition to* the 512 MiB in the VecDeque. Note that `Arc` deduplicates most of this with the state, so the actual memory doesn't double — but evicted entries still held by slow broadcast subscribers **are retained past eviction**, effectively un-evictable until the subscriber advances.

If you have many concurrent logs subscribers and one is consistently slow but not slow *enough* to trigger `Lagged`, journal memory growth can exceed 512 MiB and the state's byte budget becomes a lower bound, not an upper one.

### 2.5 Default 512 MiB baseline is just… high

`DEFAULT_LOGS_JOURNAL_MAX_TOTAL_BYTES = 512 * 1024 * 1024`. That's a permanent 512 MiB ceiling for the journal VecDeque alone, before you add:

- `Arc`-shared entries pinned by broadcast receivers,
- per-subscription `BoundedVecDeque::new(16)` rings (each holding up to 16 `PubSubResult::Log` clones — and note that logs subscribers `.map(|log| PubSubResult::Log(Box::new(log.clone())))` so these are *owned clones*, not `Arc`-shared with the journal),
- the `pubsub_notification_sinks` Vec + per-sink channel state,
- the `tracing_unbounded` ingress backlog.

On chains with heavy log traffic (Moonbeam-class) running many websocket clients, **steady-state** RSS attributable to this feature alone can easily land in the 1–2 GiB range without any leak at all. Whether that's "OOM" or "fine" depends entirely on your node's memory budget.

### 2.6 Per-subscription owned clones in the logs stream

One more subtle one. In `eth_pubsub.rs`, the Kind::Logs branch:

```rust
let results: Vec<_> = entry
    .logs
    .iter()
    .filter(|log| log_matches_filter(&logs_params, log, false))
    .map(|log| PubSubResult::Log(Box::new(log.clone())))   // full clone per subscriber
    .collect();
```

Each log is cloned for each subscriber. With 4 MiB entries and N subscribers, a single entry can produce up to `N * 4 MiB` of transient allocations that get buffered into `BoundedVecDeque::new(16)` per subscription. The ring caps *count* but not byte size, so a fast producer + 16-deep ring of 4 MiB allocations per subscriber ≈ **64 MiB of owned clones per subscriber** buffered at worst case, on top of everything else.

---

## 3. Why #1881's bound "didn't help" on your nodes

Because #1881 only bounds the journal state VecDeque. The OOM comes from outside that VecDeque:

| Source | Bounded by #1881? |
|---|---|
| `LogsJournalState.entries` (the VecDeque) | **Yes** (byte-capped to 512 MiB) |
| `pubsub_notification_sinks` leak from journal worker reconnects (2.1) | No |
| `pubsub_notification_sinks` leak from newHeads subscriptions (2.2) | No |
| `tracing_unbounded` ingress channel (2.3) | No |
| Broadcast ring pinning by slow subscribers (2.4) | No |
| Per-subscription owned log clones (2.6) | Only by count (16), not bytes |

Bounding one out of six memory sources is exactly the shape of a fix that ships, closes the obvious leak, and leaves a slower leak plus several amplification paths intact. That matches your observation.

---

## 4. What I'd do on your nodes

**Immediate diagnostic (no code change)**:
1. Graph `process_resident_memory_bytes` against time and against websocket churn metrics (new connections per minute). A linear-with-connections trend → 2.2. A linear-with-time-even-under-no-load trend → 2.1.
2. Flip logs-journal default budget down (`LogsJournalConfig::from_max_total_bytes(64 * 1024 * 1024)` via whatever wiring your template uses) and see if RSS stabilizes proportionally. If RSS growth is unchanged → the OOM is **not** in the VecDeque and is one of 2.1–2.4.
3. Temporarily stop accepting `eth_subscribe("newHeads")` clients (or restart the websocket listener daily) and check if the leak rate drops. If yes → 2.2 is a significant contributor.

**Minimal patches worth cherry-picking**:
1. Make `register_notification_stream` return an index/handle, and remove the sender from the Vec on reconnect.
2. Mirror the same fix on the `Kind::NewHeads` path in `eth_pubsub.rs` — store a guard that removes the sink from `pubsub_notification_sinks` when the subscription task exits.
3. Replace `tracing_unbounded` with a bounded channel with `try_send` and a "mark journal incomplete on backpressure" policy (semantically consistent with the existing fail-closed design).
4. Optionally: drop the default `max_total_bytes` from 512 MiB to something saner like 64–128 MiB unless you specifically need deep replay.

**Stopgap in production**:
If you need breathing room while upstream iterates, Moonbeam's fork reverted #1862 (the commit I saw earlier). That's a viable immediate mitigation while 2.1 and 2.2 get fixed properly.

---

Want me to sketch the actual patch for the sink-removal fix (2.1 + 2.2 share a pattern)? The cleanest approach is swapping the `Vec<Sender>` for a `slab::Slab<Sender>` or `HashMap<u64, Sender>` so each caller can hold a `RemoveOnDrop` guard keyed by their slot.