Yes. Based on the `worker.rs` and `sink_registry.rs` you pasted/scraped, the old append-only sink leak is mostly addressed, but there are still several realistic OOM paths. The two biggest suspects are:

1. `best_at_import` in `MappingSyncWorker` growing without a hard cap.
2. `SinkRegistry` still using `tracing_unbounded`, so `len()` can look fine while per-sink pending queues are huge.

If the memory is attributed to `frontier-mapping-sync-worker`, that makes both especially plausible because both are allocated from inside that task’s poll/sync/broadcast path.

## 1. Biggest red flag: `best_at_import` is unbounded

This field lives directly inside `MappingSyncWorker`:

```rust
best_at_import: HashMap<Block::Hash, BestBlockInfo<Block>>,
```

And every import notification that is `is_new_best` inserts:

```rust
if notification.is_new_best {
    let reorg_info = notification.tree_route.as_ref().map(|tree_route| {
        Arc::new(ReorgInfo::from_tree_route(tree_route, notification.hash))
    });

    self.best_at_import.insert(
        notification.hash,
        BestBlockInfo {
            block_number: *notification.header.number(),
            reorg_info,
        },
    );
}
```

The comment says entries are pruned when blocks become finalized, but in this file there is no local hard cap. If `sync_blocks` stalls, finality stalls, the node is catching up, state pruning makes sync fail, or import notifications arrive much faster than mapping-sync can process them, this map can grow.

This is especially suspicious because the worker drains all currently available import notifications before calling `sync_blocks`:

```rust
loop {
    match Stream::poll_next(Pin::new(&mut self.import_notifications), cx) {
        Poll::Pending => break,
        Poll::Ready(Some(notification)) => {
            fire = true;
            if notification.is_new_best {
                self.best_at_import.insert(...);
            }
        }
        Poll::Ready(None) => return Poll::Ready(None),
    }
}
```

So after a stall, it can suddenly drain a large backlog and insert a large number of `BestBlockInfo` entries in one poll cycle. That matches “memory jumps suddenly inside `frontier-mapping-sync-worker`”.

### Why this can get large

Each entry contains:

```rust
pub struct BestBlockInfo<Block: BlockT> {
    pub block_number: Block::Number,
    pub reorg_info: Option<Arc<ReorgInfo<Block>>>,
}
```

Even if most `ReorgInfo` objects are small, the `HashMap` overhead plus key/value storage adds up. If `tree_route` is present for many blocks, each entry also allocates a `ReorgInfo`, likely containing `Vec`s of enacted/retracted hashes.

If there is a big reorg, finality lag, or sync stall, this can become substantial.

### First diagnostic

Add a metric/log for:

```rust
self.best_at_import.len()
```

and ideally also:

```rust
client.info().best_number
client.info().finalized_number
best - finalized gap
sync_from
have_next
sync_oracle.is_major_syncing()
```

Temporarily add something like this around the `sync_blocks` call:

```rust
let chain_info = self.client.info();
let best_at_import_len_before = self.best_at_import.len();
let sink_stats = self.pubsub_notification_sinks.stats();

log::warn!(
    target: "mapping-sync",
    "mapping-sync diag before sync_blocks: sync_from={:?} have_next={} major_syncing={} \
     best_number={:?} finalized_number={:?} best_at_import_len={} \
     sinks={} sink_capacity={} pending_total={} pending_max={}",
    self.sync_from,
    self.have_next,
    self.sync_oracle.is_major_syncing(),
    chain_info.best_number,
    chain_info.finalized_number,
    best_at_import_len_before,
    sink_stats.sinks,
    sink_stats.capacity,
    sink_stats.pending_total,
    sink_stats.pending_max,
);
```

If `best_at_import_len` climbs monotonically or jumps before the OOM, that is likely your culprit.

## 2. `SinkRegistry` fixed the logical leak, but not all queue memory

Your new `SinkRegistry` is much better than the old append-only `Vec`. It removes guards on drop and prunes closed/lagging sinks during broadcast.

But it still uses:

```rust
tracing_unbounded(name, warn_threshold)
```

That means each sink still has an unbounded channel internally.

You try to enforce a cap here:

```rust
if sink.len() >= max_pending {
    let _ = sink.close();
    to_remove.push(id);
    continue;
}

if sink.unbounded_send(make_msg()).is_err() {
    to_remove.push(id);
}
```

This is better than nothing, but memory can still grow up to roughly:

```text
number_of_sinks × max_pending × size_of_EthereumBlockNotification
```

plus channel node overhead.

If `max_pending` is large, this can absolutely OOM. The `register(..., 100_000)` argument is only the tracing warning threshold, not a hard cap. The real question is: what value is passed to `broadcast(max_pending, ...)` from `emit_block_notification`?

If `max_pending` is also anywhere near `100_000`, that is dangerous.

### Important: `len()` alone is not enough

`SinkRegistry::len()` only tells you how many sinks exist. You also need to know how much is queued inside those sinks.

Add this to `sink_registry.rs`:

```rust
#[derive(Debug, Clone, Default)]
pub struct SinkRegistryStats {
    pub sinks: usize,
    pub capacity: usize,
    pub pending_total: usize,
    pub pending_max: usize,
    pub closed: usize,
}

impl<T> SinkRegistry<T> {
    pub fn stats(&self) -> SinkRegistryStats {
        let inner = self.inner.lock();

        let mut stats = SinkRegistryStats {
            sinks: inner.sinks.len(),
            capacity: inner.sinks.capacity(),
            pending_total: 0,
            pending_max: 0,
            closed: 0,
        };

        for sink in inner.sinks.values() {
            let pending = sink.len();
            stats.pending_total = stats.pending_total.saturating_add(pending);
            stats.pending_max = stats.pending_max.max(pending);

            if sink.is_closed() {
                stats.closed = stats.closed.saturating_add(1);
            }
        }

        stats
    }
}
```

Then log/export:

```text
frontier_pubsub_sink_registry_len
frontier_pubsub_sink_registry_capacity
frontier_pubsub_sink_pending_total
frontier_pubsub_sink_pending_max
frontier_pubsub_sink_closed
```

If `pending_total` or `pending_max` spikes before OOM, the culprit is still the notification channel fanout.

## 3. `HashMap` capacity in `SinkRegistry` can retain memory after churn

Another subtle issue: `HashMap::remove()` and `HashMap::clear()` do not necessarily release allocated bucket memory.

Your registry does:

```rust
inner.sinks.remove(&id);
```

and:

```rust
inner.sinks.clear();
```

but never shrinks.

So if a public RPC node experiences a subscription storm — even if all sinks are later removed — the registry’s internal `HashMap` may retain its high-water capacity. `len()` can return a small number while the map still owns memory for a much larger historical number of sinks.

Add `capacity` to the stats above. If you see:

```text
len = 20
capacity = 500_000
```

then you found retained memory.

Patch:

```rust
fn maybe_shrink<T>(inner: &mut Inner<T>) {
    let len = inner.sinks.len();
    let cap = inner.sinks.capacity();

    if cap > 4096 && cap > len.saturating_mul(4).max(1) {
        inner.sinks.shrink_to_fit();
    }
}
```

Call it after removals:

```rust
for id in to_remove {
    inner.sinks.remove(&id);
}

maybe_shrink(&mut inner);
```

And in `clear_on_major_sync`:

```rust
pub fn clear_on_major_sync(&self) {
    let mut inner = self.inner.lock();
    inner.generation = inner.generation.saturating_add(1);
    inner.sinks.clear();
    inner.sinks.shrink_to_fit();
}
```

## 4. `broadcast` can be made less allocation-spiky

Current code allocates a `Vec` for removals:

```rust
let mut to_remove = Vec::new();
```

If the registry temporarily has many zombie sinks, `to_remove` itself can get large. Not the main OOM, but avoidable.

You can rewrite `broadcast` using `retain`:

```rust
pub fn broadcast(&self, max_pending: usize, mut make_msg: impl FnMut() -> T)
where
    T: Clone,
{
    let mut inner = self.inner.lock();

    if inner.sinks.is_empty() {
        return;
    }

    let mut dropped_closed = 0usize;
    let mut dropped_lagging = 0usize;
    let mut dropped_send_error = 0usize;

    inner.sinks.retain(|_, sink| {
        if sink.is_closed() {
            dropped_closed += 1;
            return false;
        }

        let pending = sink.len();

        if pending >= max_pending {
            log::warn!(
                target: "mapping-sync",
                "Dropping lagging pubsub subscriber: pending={}, max_pending={}",
                pending,
                max_pending,
            );

            let _ = sink.close();
            dropped_lagging += 1;
            return false;
        }

        match sink.unbounded_send(make_msg()) {
            Ok(()) => true,
            Err(_) => {
                dropped_send_error += 1;
                false
            }
        }
    });

    if dropped_closed > 0 || dropped_lagging > 0 || dropped_send_error > 0 {
        log::warn!(
            target: "mapping-sync",
            "pubsub sink pruning: closed={} lagging={} send_error={} remaining={} capacity={}",
            dropped_closed,
            dropped_lagging,
            dropped_send_error,
            inner.sinks.len(),
            inner.sinks.capacity(),
        );
    }

    maybe_shrink(&mut inner);
}
```

This removes the temporary `to_remove` allocation and gives you useful logs.

## 5. `clear_on_major_sync` may not free queued messages immediately

This line removes senders:

```rust
inner.sinks.clear();
```

But if a receiver still exists and has queued messages, the queue may stay alive until the receiver is dropped or drained.

This matters for slow WebSocket clients. If a subscription task is stuck trying to write to a dead/slow network socket, it may not promptly poll its receiver and observe closure. The sender side is gone, but queued messages may remain pinned by the receiver side.

So `clear_on_major_sync` helps stop future sends, but it is not a guaranteed immediate memory release for already queued messages.

This is another reason `max_pending` must be small, and why a bounded channel is safer than `tracing_unbounded`.

## 6. The old leak may be fixed, but slow live subscribers can still OOM you

The old bug was mostly:

```text
dead subscriptions leave dead senders in Vec forever
```

Your `SinkRegistry` addresses that.

But a different problem remains:

```text
live but slow subscriptions accumulate queued notifications
```

This affects especially:

- `eth_subscribe("newHeads")`
- logs journal notification stream
- any client that keeps TCP open but stops reading
- public RPC nodes under subscription abuse

Even with perfect RAII removal, a malicious or slow client can force queue growth until `max_pending`.

If `max_pending` is too large, the OOM persists.

Immediate mitigation: set `max_pending` low. For example:

```rust
const PUBSUB_MAX_PENDING_NOTIFICATIONS: usize = 64;
```

or even lower for `newHeads`.

For `logs_journal`, maybe:

```rust
const LOGS_JOURNAL_MAX_PENDING_NOTIFICATIONS: usize = 128;
```

Do not use `100_000` as a real memory cap.

## 7. The best fix: make `SinkRegistry` truly bounded

Right now it is logically bounded by checking `sink.len()`, but physically it is still an unbounded channel.

A stronger design is to replace:

```rust
TracingUnboundedSender<T>
```

with a bounded channel and use `try_send`.

Policy:

```text
if send succeeds:
    ok
if receiver closed:
    prune
if channel full:
    close/prune sink and/or mark logs journal incomplete
```

For pubsub/newHeads, dropping/closing a lagging subscriber is fine.

For logs journal, full ingress should insert a gap marker so filters/subscribers fail closed and rebootstrap.

That would eliminate the class where `mapping-sync-worker` allocates more and more queue nodes.

## 8. `best_at_import` should be capped or pruned earlier

Even if `sync_blocks` prunes finalized entries, I would not rely on that as the only bound.

Add a local prune before/after draining import notifications:

```rust
fn prune_best_at_import<Block: BlockT>(
    best_at_import: &mut HashMap<Block::Hash, BestBlockInfo<Block>>,
    finalized_number: Block::Number,
) {
    best_at_import.retain(|_, info| info.block_number > finalized_number);
}
```

Call it periodically:

```rust
let chain_info = self.client.info();
prune_best_at_import(&mut self.best_at_import, chain_info.finalized_number);
```

Then add a hard emergency cap:

```rust
const MAX_BEST_AT_IMPORT_ENTRIES: usize = 65_536;
```

If exceeded, drop oldest entries by block number:

```rust
fn cap_best_at_import<Block: BlockT>(
    best_at_import: &mut HashMap<Block::Hash, BestBlockInfo<Block>>,
) {
    const MAX: usize = 65_536;

    if best_at_import.len() <= MAX {
        return;
    }

    let excess = best_at_import.len() - MAX;

    let mut hashes_by_number: Vec<_> = best_at_import
        .iter()
        .map(|(hash, info)| (*hash, info.block_number))
        .collect();

    hashes_by_number.sort_by_key(|(_, number)| *number);

    for (hash, _) in hashes_by_number.into_iter().take(excess) {
        best_at_import.remove(&hash);
    }

    log::warn!(
        target: "mapping-sync",
        "Capped best_at_import by dropping {} oldest entries; remaining={}",
        excess,
        best_at_import.len(),
    );
}
```

You may need minor trait bounds depending on exact `Block::Hash` and `Block::Number` traits, but the idea is straightforward.

Also consider not storing full `ReorgInfo` unless there are actually retracted blocks. For normal linear block extension, you probably do not need to allocate a `ReorgInfo` object at all.

Pseudo-policy:

```rust
if notification.is_new_best {
    let reorg_info = notification.tree_route.as_ref().and_then(|tree_route| {
        if tree_route.retracted().is_empty() {
            None
        } else {
            Some(Arc::new(ReorgInfo::from_tree_route(tree_route, notification.hash)))
        }
    });

    self.best_at_import.insert(...);
}
```

Adapt to the actual `tree_route` API.

## 9. Watch for mapping sync stuck behind pruned state

This part is also suspicious:

```rust
state_pruning_blocks: Option<Block::Number>,
```

and the worker comment:

```rust
/// If set, blocks below the live state window are skipped during catch-up so the sync tip
/// does not get stuck behind pruned state.
```

If `state_pruning_blocks` is wrong or `None` on a non-archive/pruned node, mapping-sync can get stuck trying to sync old blocks whose state is unavailable.

Then this happens:

```rust
Err(e) => {
    self.have_next = false;
    debug!(target: "mapping-sync", "Syncing failed with error {e:?}, retrying.");
    Poll::Ready(Some(()))
}
```

That log is only `debug`. In production you may not notice that mapping sync is failing repeatedly while import notifications keep accumulating and `best_at_import` grows.

Temporarily promote this to `warn`:

```rust
log::warn!(
    target: "mapping-sync",
    "Syncing failed with error {e:?}, retrying. sync_from={:?} best_at_import_len={}",
    self.sync_from,
    self.best_at_import.len(),
);
```

If you see repeated failures, fix `state_pruning_blocks` / sync cursor first.

## 10. Practical diagnosis matrix

Run with the new stats and classify:

| Observation | Likely cause |
|---|---|
| `best_at_import_len` grows continuously | Unbounded `best_at_import`, finality lag, sync failure, or import backlog |
| `sink_registry_len` is stable but `pending_total` grows | Slow live subscriber or logs journal falling behind |
| `sink_registry_len` low but `capacity` huge | HashMap high-water retention after subscription storm |
| `pending_max` near `max_pending` | One very slow sink; probably logs journal or a stuck WS subscription |
| OOM disappears when public WS is disabled | Pubsub queue/subscriber path |
| OOM still happens with no WS clients | `best_at_import`, `sync_blocks`, DB/status/log materialization |
| OOM correlates with `major_syncing=true` | `best_at_import` and/or sync failure during catch-up |
| OOM correlates with reorg | Large `ReorgInfo` pinned in map/queues |
| OOM correlates with log-heavy block | Mapping-sync materializes receipts/logs before logs journal caps apply |

## My strongest current bet

Given your `SinkRegistry` already handles guard-drop and closed-sink pruning, I would stop looking only at `sink_registry.len()`.

My top two bets are:

1. `best_at_import` is growing because mapping sync is stalled, finality is lagging, or the node is catching up/importing faster than `sync_blocks` can prune.
2. Pending queues inside `tracing_unbounded` are growing even though the number of sinks is not leaking.

Add `best_at_import_len`, `sink pending_total`, `sink pending_max`, and `sink HashMap capacity` metrics. Those four numbers should tell you very quickly which side is responsible.

The immediate defensive patches I would ship are:

1. Add a hard cap and finalized-number prune to `best_at_import`.
2. Lower `max_pending` drastically.
3. Add `SinkRegistry::stats()`.
4. Shrink the registry `HashMap` after large prune/clear events.
5. Replace `tracing_unbounded` with a bounded channel plus `try_send`.

If you paste `client/mapping-sync/src/lib.rs`, especially the `sync_blocks` / `emit_block_notification` path and the actual `broadcast(max_pending, ...)` value, I can point to the exact line most likely allocating the OOM volume.