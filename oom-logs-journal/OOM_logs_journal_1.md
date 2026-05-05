# Deep dive on Frontier's `logs_journal` and the residual OOM

Let me walk through the architecture introduced in #1862, what #1881 actually fixed (and didn't), and then the concrete suspects I'd chase first in your nodes. I only have the detailed content for #1862 from the scrape — #1881's title ("bound logs journal memory and derive retention from total bytes") tells us it replaced the entry-count cap with a byte-budget cap, but I'll reason about what that change does and doesn't close.

---

## 1. Architecture recap

Key pieces added in `client/rpc/src/logs_journal.rs`:

- A shared `Arc<LogsJournal>` is constructed once in `template/node/src/rpc/eth.rs` and injected into **both** `EthFilter` and `EthPubSub`.
- Internal state:
  ```
  struct LogsJournalState {
      entries: VecDeque<Arc<LogsJournalEntry>>,
      next_seq: u64,
      max_entries: usize,     // #1862
      // #1881 likely added a max_bytes / current_bytes pair here
  }
  struct LogsJournalEntry {
      seq: u64,
      complete: bool,
      logs: Vec<Log>,         // all logs for the delta, with removed flags
  }
  ```
- A background worker pulls `EthereumBlockNotification`s via `sc_utils::mpsc::tracing_unbounded("logs_journal_notification_stream", 100_000)`, materializes the logs by calling `storage_override.current_block(hash)` and `storage_override.current_transaction_statuses(hash)`, builds an entry (with reorg deltas and `removed` flags), pushes to the `VecDeque`, evicts from the front, and **broadcasts** the `Arc<LogsJournalEntry>` to subscribers via a `tokio::sync::broadcast` channel.
- `EthPubSub` subscribes to the broadcast for `eth_subscribe("logs")`.
- `EthFilter` stores `last_log_journal_seq` in each `FilterPoolItem` and on poll calls `snapshot_since(cursor)` to return the missed entries.

What #1881 changed: retention is now keyed on **total payload bytes**, not on number of entries. Good in theory, but it only controls one of several places memory can grow.

---

## 2. Why the bound in #1881 is not sufficient

There are at least **five independent memory sources** on the logs_journal path. #1881 only bounded one of them (the `VecDeque` itself). Here are the culprits in rough priority order.

### 2.1 Sink leak in the worker reconnection loop (★ highest suspicion)

Look at this structure (paraphrased from the CodeRabbit review and the diff summary):

```rust
let worker_fut = async move {
    let mut had_stream = false;
    loop {
        let (inner_sink, mut notifications) =
            sc_utils::mpsc::tracing_unbounded("logs_journal_notification_stream", 100_000);
        pubsub_notification_sinks.lock().push(inner_sink);   // <-- never removed

        while let Some(n) = notifications.next().await { /* build + push + broadcast */ }

        if had_stream { state.push(false, Vec::new()); }     // gap marker
        // loops and pushes a NEW sink again
    }
};
```

`pubsub_notification_sinks` is a `Vec<TracingUnboundedSender<...>>` that the block-import notifier fans out to. Every time the inner stream terminates, the loop:

1. Drops the receiver,
2. **Allocates a new sender and appends it** to `pubsub_notification_sinks`.

If the stream ever flaps (which it does under load, on pruning, on service restarts, or simply because `Stream::next()` returned `None`), the sink list grows without bound. Each entry corresponds to a *dead* channel that the notifier still holds and still tries to send to. Two costs:

- The `Vec<Sender>` itself grows.
- More importantly, the **notification producer** fans out to every sink in that list. Even if send fails, some implementations keep buffering until the receiver is explicitly dropped on the other end. Combined with `tracing_unbounded`, you can end up with each producer call allocating on every stale sink.

CodeRabbit's own comment flagged exactly this: *"recreates sinks indefinitely (creating new `pubsub_notification_sinks` entries)"*. The author added a gap-marker fix but did **not** address the sink leak itself.

This is the single most plausible explanation for a monotonically-increasing RSS that #1881's byte cap cannot touch, because the leak is outside the `VecDeque`.

**What to look for in a heap dump / metrics:** count of entries in `pubsub_notification_sinks`, RSS growth correlated with block-import notifier stream flaps, and the `"logs_journal_notification_stream"` counter in tracing_unbounded metrics going up without recovery.

### 2.2 Unbounded ingress channel (`tracing_unbounded` with soft warn)

`sc_utils::mpsc::tracing_unbounded(_, 100_000)` is **not** bounded — the 100_000 is only the warning threshold. If the worker ever falls behind (because `storage_override.current_block` / `current_transaction_statuses` becomes slow, e.g., during heavy block import, during DB contention, or on a node whose storage-cache layer is under pressure), the channel grows indefinitely, each `EthereumBlockNotification` pinning block hashes.

Combined with 2.1, every stale sink also has its own unbounded channel.

### 2.3 Broadcast channel + slow/lagging subscribers

The journal broadcasts `Arc<LogsJournalEntry>` to all `eth_subscribe("logs")` consumers through a `tokio::sync::broadcast`. Properties:

- The broadcast has a fixed ring buffer (capacity set at `channel()` time). The **slowest** subscriber pins entries in the buffer — items are only reclaimable once every receiver has advanced past them.
- If you run many websocket clients and even one is slow (backpressured TCP socket, blocked event loop in a third-party consumer, an indexer that blocks on DB writes), the ring buffer stays full of `Arc<LogsJournalEntry>`s.
- `Arc` means refs are shared, so entries evicted from the `VecDeque` stay alive as long as any subscriber's broadcast slot still holds them.

This is why #1881's byte cap on the VecDeque does not bound overall memory: evicted entries remain live via broadcast buffers and via in-flight `snapshot_since` results.

The design is advertised as "fail closed under backpressure" for lagging subscribers — but broadcast's `RecvError::Lagged` only fires when a subscriber falls more than `capacity` behind. Between 0 and `capacity` lag, memory grows with no signal.

### 2.4 Reorg payload amplification — a single entry can be huge

`build_journal_payload` on a canonical transition builds one entry that concatenates:

- `removed=true` log clones for every block reorged out,
- `removed=false` log clones for every new canonical block.

On chains with high log density (Moonbeam is a prime example — popular ERC-20s, bridges, oracles emit dozens of logs per tx, hundreds per block), a reorg of even 3–5 blocks can produce an entry with thousands of `Log` structs. Each `Log` is non-trivial (address + topics vec + data bytes).

Implications:

- A single entry can exceed the entire byte budget introduced in #1881. Depending on how the bound is enforced (I'd bet it's "evict oldest until total ≤ max_bytes"), the journal may evict everything else to make room but still hold one pathological entry.
- If there's a bug where the bound is checked before adding the new entry or the new entry itself is never rejected, this is unbounded in the worst case.
- Even without a bug, transient spikes during reorg storms (known to happen on parachains during relay-chain hiccups) can push RSS very high, and since allocators rarely release memory back to the OS, the high-water mark sticks.

### 2.5 Filter cursor anchoring / filter pool growth

Two sub-issues:

- `FilterPoolItem.last_log_journal_seq: Option<u64>` is per installed filter. If your node has many `eth_newFilter` consumers that never call `eth_uninstallFilter` (common with misbehaving dapps/indexers), the filter pool itself grows. Frontier does time-based cleanup (`filter_pool` prune), but if polling is frequent enough, the `last_poll` keeps getting refreshed and entries never age out.
- In the initial 1862 code, log filters were seeded with `last_log_journal_seq = Some(0)`, which causes every new filter to replay the **entire retained journal** on its first `eth_getFilterChanges`. CodeRabbit flagged this and it was supposedly changed to the current cursor — worth verifying that the fix actually landed on the code path you're running, because mis-seeding causes large transient allocations proportional to journal size × number of fresh filters.

### 2.6 Indirect: StorageOverride cache amplification

`build_journal_payload` calls `current_block(hash)` and `current_transaction_statuses(hash)` on every canonical transition. If your `StorageOverride` backend is `SchemaV3Override` with an LRU cache (the default setup), you are now warming that cache on every block from the RPC side, not just on demand. This can inflate the cache above what the node sized it for, especially on re-orgs where both the canonical and non-canonical hashes are queried.

This is not a leak per se, but it does increase steady-state RSS after enabling the feature, and can interact with 2.2 (slowing the worker and backing up the unbounded channel).

---

## 3. How I'd diagnose on your nodes

In order of speed-to-signal:

1. **Metric**: expose/graph the length of `pubsub_notification_sinks` (or patch the worker to log it every N blocks). If it grows monotonically → it's 2.1.
2. **Metric**: the tracing_unbounded gauge for `"logs_journal_notification_stream"`. Growth past 100k and not coming down → 2.2. Growth combined with 2.1 → confirms both.
3. **Heap profile** (e.g., `heaptrack`, `jemalloc` profiling, or `bytehound`) and look for:
   - Large retained allocations rooted in `TracingUnboundedSender` / `broadcast::Sender` — points to 2.1 or 2.3.
   - Many `LogsJournalEntry` / `Vec<Log>` allocations rooted in `broadcast::Receiver` slots — 2.3.
   - One or two gigantic `Vec<Log>` allocations — 2.4.
4. **Reproduce synthetically**: spin up a test node with 100 concurrent `eth_subscribe("logs")` clients where one client deliberately stops reading (don't close the socket; just stop draining). Drive log-heavy traffic. If RSS climbs and never recovers after that slow client disconnects → 2.3 with a likely additional bug where broadcast slots aren't reclaimed on disconnect.
5. **Correlate with reorgs**: log reorg depth and RSS delta. Spikes aligned with reorgs → 2.4.
6. **Disable the feature**: if you can ship a build that constructs `EthFilter`/`EthPubSub` with a no-op journal (or reverts to the pre-1862 path), and the OOM disappears, that's your smoking gun. Moonbeam foundation actually did exactly this on 2026-04-23 — the scrape shows a `Revert "feat: reorged-out logs..."` commit on their fork. That's a strong signal the feature is not yet production-safe.

---

## 4. Mitigations you can apply now

Short-term patches while upstream iterates:

- **Fix the sink leak**: in the worker `loop`, before re-creating the sink, remove the old one from `pubsub_notification_sinks` (retain by `Sender::same_channel` or store an index). This is a ~10-line fix and almost certainly the highest-leverage one.
- **Bound the ingress channel**: replace `tracing_unbounded` with a bounded channel (e.g., capacity 1024) and drop/lag on overflow — matches the "fail closed" design goal stated in the PR description.
- **Cap per-entry size**: in `build_journal_payload`, if the entry exceeds, say, 25% of the byte budget, mark it `complete=false` and emit a short incomplete marker so subscribers re-bootstrap instead of receiving a giant blob. Aligns with fail-closed semantics.
- **Bound broadcast capacity tightly** and ensure slow subscribers are actually dropped: `broadcast::Sender::receiver_count` + a watchdog that closes subscriptions whose last-delivered-seq hasn't advanced within X seconds.
- **Audit filter lifecycle**: confirm the "seed with current cursor, not 0" patch is on your build, and verify the filter-pool pruner runs even when filters are being polled.

If you need a stopgap in production today, the pragmatic move is to revert to the pre-1862 log path (the Moonbeam revert commit is in the scrape and you can cherry-pick it), and re-enable once upstream closes the sink-leak and broadcast-retention issues.

---

Want me to go deeper on any one of these — for example, sketch the exact patch for the sink leak, or walk through the broadcast-channel memory math for your actual subscriber count and log rate?
