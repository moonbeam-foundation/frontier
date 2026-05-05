Yes — strong hint: if the profiler says `frontier-mapping-sync-worker`, that does **not** rule out the `logs_journal` / pubsub sink issue. In Moonbeam’s wiring, the mapping-sync worker is the **producer** of the notifications consumed by `LogsJournal` and `EthPubSub`.

From the scraped `rpc.rs`, the same `pubsub_notification_sinks` object is passed to all three places:

```rust
let logs_journal = Arc::new(LogsJournal::new(
    subscription_task_executor.clone(),
    overrides.clone(),
    pubsub_notification_sinks.clone(),
));
```

```rust
EthPubSub::new(
    pool,
    Arc::clone(&client),
    sync.clone(),
    subscription_task_executor,
    overrides,
    pubsub_notification_sinks.clone(),
    logs_journal,
)
```

And then in the task you’re seeing:

```rust
MappingSyncWorker::new(
    params.client.import_notification_stream(),
    Duration::new(6, 0),
    params.client.clone(),
    params.substrate_backend.clone(),
    params.overrides.clone(),
    b.clone(),
    3,
    0,
    params.state_pruning.and_then(|mode| { ... }),
    SyncStrategy::Parachain,
    sync.clone(),
    pubsub_notification_sinks.clone(),
)
.for_each(|()| futures::future::ready(()))
```

So the path is basically:

```text
Substrate import notification
  -> frontier-mapping-sync-worker
      -> write Frontier KV / SQL mapping
      -> build EthereumBlockNotification
      -> send notification to pubsub_notification_sinks
           -> LogsJournal receiver
           -> eth_subscribe("newHeads") receivers
           -> possibly other pubsub consumers
```

That means allocations caused by sending to leaked, slow, or unbounded receivers can be attributed to `frontier-mapping-sync-worker`, because that task is where `unbounded_send(notification.clone())` happens.

## Main suspect: `pubsub_notification_sinks`

The immediate thing I would inspect is the fanout inside `fc-mapping-sync`, not only `fc-rpc`.

The likely shape is something like:

```rust
for sink in pubsub_notification_sinks.lock().iter() {
    let _ = sink.unbounded_send(notification.clone());
}
```

If so, this is dangerous for two reasons.

### 1. Dead sinks may never be pruned

If every `eth_subscribe("newHeads")` connection pushes a sender into `pubsub_notification_sinks`, but disconnect does not remove it, then the vec grows forever.

Same for `LogsJournal` if it reconnects its internal receiver and registers a new sink without removing the old one.

Then mapping-sync pays \(O(n)\) work every block where \(n\) is total historical subscriptions/reconnects, not current live subscriptions.

That can look like:

```text
hour 0: 100 sinks
hour 6: 20_000 sinks
then one block/reorg/catch-up burst:
  mapping-sync suddenly allocates/sends/clones like crazy
```

So memory appears to explode “inside” `frontier-mapping-sync-worker`.

### 2. The channels are probably unbounded

If the sink is a `TracingUnboundedSender`, the “100_000” number used in Frontier code is usually a warning threshold, not a hard cap.

So if the `LogsJournal` receiver falls behind — for example because it is doing expensive `StorageOverride` reads, filtering logs, or dealing with reorg info — the mapping-sync worker continues pushing notifications into an unbounded queue.

Again: allocation site is the mapping-sync task.

## The fastest confirmation

Add this temporary logging in Moonbeam’s `spawn_essential_tasks`, right before spawning `frontier-mapping-sync-worker`:

```rust
{
    let sinks = pubsub_notification_sinks.clone();

    params.task_manager.spawn_handle().spawn(
        "frontier-pubsub-sinks-len-logger",
        Some("frontier"),
        async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Adjust this if EthereumBlockNotificationSinks is not a Mutex<Vec<_>>
                let len = sinks.lock().len();

                log::warn!(
                    target: "frontier",
                    "frontier pubsub_notification_sinks len = {}",
                    len,
                );
            }
        },
    );
}
```

If that number grows monotonically with websocket churn, you found the leak.

Expected healthy value should be roughly:

```text
1 logs_journal sink
+ currently active eth_subscribe("newHeads") subscribers
+ maybe a small number of transient subscribers
```

It should **not** be thousands or millions after a public RPC node has been running for a while.

## First hotfix: prune failed sinks in mapping-sync

Even before doing the cleaner RAII `SinkRegistry` patch, you can harden the producer. In `fc-mapping-sync`, find the place where it broadcasts `EthereumBlockNotification` to `pubsub_notification_sinks`.

Replace this style:

```rust
for sink in self.pubsub_notification_sinks.lock().iter() {
    let _ = sink.unbounded_send(notification.clone());
}
```

with this:

```rust
let mut sinks = self.pubsub_notification_sinks.lock();

let before = sinks.len();
let mut failed = 0usize;

sinks.retain(|sink| {
    match sink.unbounded_send(notification.clone()) {
        Ok(()) => true,
        Err(_) => {
            failed += 1;
            false
        }
    }
});

let after = sinks.len();

if failed > 0 || before > 1_000 {
    log::warn!(
        target: "fc-mapping-sync",
        "pubsub notification fanout: before={} after={} pruned_failed={} block_hash={:?}",
        before,
        after,
        failed,
        notification.hash,
    );
}
```

This does not solve slow live receivers, but it does fix the dead-sink accumulation class.

Important: patch both paths if Moonbeam can use either backend:

```rust
fc_db::Backend::KeyValue => fc_mapping_sync::kv::MappingSyncWorker
fc_db::Backend::Sql      => fc_mapping_sync::sql::SyncWorker
```

Your scraped file shows both branches pass `pubsub_notification_sinks.clone()`.

## Second hotfix: bound the producer-side notification channels

If the sink count is stable but memory still jumps in `frontier-mapping-sync-worker`, then the likely problem is a **live but slow receiver**, especially the `LogsJournal` receiver.

The current pipeline is probably:

```text
mapping-sync-worker --unbounded_send--> logs_journal_notification_stream
```

That needs to become bounded.

Policy should be:

```text
If notification channel is full:
  do not block mapping-sync
  drop the notification
  mark logs journal incomplete / insert gap marker
  force logs subscribers and filters to re-bootstrap
```

This matters because blocking mapping-sync is bad: it can delay Frontier DB indexing and create a bigger sync backlog.

So the bounded policy should be fail-closed, not backpressure block import.

Conceptually:

```rust
match sink.try_send(notification.clone()) {
    Ok(()) => {}
    Err(TrySendError::Full(_)) => {
        // Do not block mapping-sync.
        // Increment a dropped-notification counter for that sink.
        // LogsJournal should observe the drop counter and push complete=false gap marker.
    }
    Err(TrySendError::Closed(_)) => {
        // prune sink
    }
}
```

If you only apply the watchdog on `eth_subscribe("logs")`, this mapping-sync OOM can still happen, because the watchdog protects the **broadcast/output side** of `LogsJournal`, not the **ingress side** from mapping-sync into `LogsJournal`.

Given the task attribution you’re seeing, the ingress/fanout side is now the priority.

## Why this can be sudden

This is the pattern I would expect on a public RPC node:

```text
1. newHeads/logs subscribers churn for hours.
2. pubsub_notification_sinks grows silently.
3. Mapping-sync keeps running, but fanout cost increases.
4. A reorg, catch-up, or log-heavy block arrives.
5. Mapping-sync emits many notifications quickly.
6. Every notification is cloned/sent to every stale sink.
7. Memory explodes inside frontier-mapping-sync-worker.
```

So the root leak can be slow, but the OOM event can look sudden.

## Also check log-heavy blocks

If `pubsub_notification_sinks.len()` is stable, then look at actual block payloads.

`logs_journal` caps do **not** protect mapping-sync itself. Mapping-sync still has to read/write Ethereum blocks, receipts, transaction statuses, and logs to the Frontier backend. A pathological block with many logs or huge log data can allocate heavily before the journal ever applies its own limits.

Quick TypeScript scanner to identify suspicious recent blocks:

```ts
import { createPublicClient, http } from "viem";

const RPC = process.env.RPC ?? "http://127.0.0.1:9944";

const client = createPublicClient({
  transport: http(RPC),
});

function hexBytes(hex: string): number {
  if (!hex || hex === "0x") return 0;
  return (hex.length - 2) / 2;
}

async function main() {
  const latest = await client.getBlockNumber();
  const from = latest > 500n ? latest - 500n : 0n;

  const rows: Array<{
    block: bigint;
    txs: number;
    logs: number;
    logBytes: number;
  }> = [];

  for (let n = from; n <= latest; n++) {
    const block = await client.getBlock({
      blockNumber: n,
      includeTransactions: false,
    });

    let logs = 0;
    let logBytes = 0;

    for (const txHash of block.transactions) {
      const receipt = await client.getTransactionReceipt({ hash: txHash });
      for (const log of receipt.logs) {
        logs++;
        logBytes += 20; // address
        logBytes += log.topics.length * 32;
        logBytes += hexBytes(log.data);
      }
    }

    rows.push({
      block: n,
      txs: block.transactions.length,
      logs,
      logBytes,
    });

    if (n % 50n === 0n) {
      console.error(`scanned block ${n}/${latest}`);
    }
  }

  rows.sort((a, b) => b.logBytes - a.logBytes);

  console.log("Top blocks by log bytes:");
  for (const r of rows.slice(0, 20)) {
    console.log(
      `block=${r.block} txs=${r.txs} logs=${r.logs} logBytes=${r.logBytes} ` +
      `logMiB=${(r.logBytes / 1024 / 1024).toFixed(2)}`
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
```

If your OOM aligns with a block or reorg containing unusually huge logs, then the culprit may be mapping-sync’s receipt/status materialization rather than the pubsub sink leak.

## Diagnosis matrix

Use this to narrow it quickly:

| Observation | Likely cause |
|---|---|
| `pubsub_notification_sinks.len()` grows forever | Dead sink leak from `newHeads`, `LogsJournal`, or reconnects |
| Sink length stable, but memory grows with no subscribers | LogsJournal ingress channel falling behind or mapping-sync DB/status allocation |
| Memory spike aligns with WS reconnect storm | newHeads/logs subscription sink churn |
| Memory spike aligns with reorg/catch-up | mapping-sync emits a burst into many sinks or unbounded queues |
| Memory spike aligns with one block with huge receipts/logs | mapping-sync materialization / Frontier DB write path |
| Disabling public WS makes the issue disappear | pubsub sink path, not DB mapping |
| Disabling `LogsJournal` makes the issue disappear even without clients | LogsJournal ingress backlog or journal construction path |

## Immediate mitigation order

I would do this in order:

1. **Instrument `pubsub_notification_sinks.len()`** in Moonbeam’s `rpc.rs`.
2. **Patch mapping-sync fanout to `retain` only live sinks**.
3. **Add warning logs when fanout length is high**, e.g. `> 1_000`.
4. **Bound the notification channels** from mapping-sync to RPC/logs-journal.
5. **Temporarily disable or gate `LogsJournal`** on one staging node to prove whether the OOM disappears.
6. If sink count is clean, **scan recent blocks for log volume** and correlate OOM time with block/reorg events.

My strongest current bet: the memory is still related to `pubsub_notification_sinks`, but the allocation is now showing up under `frontier-mapping-sync-worker` because that worker is the producer doing the unbounded sends.