# Concrete patch: `SinkRegistry<T>` with RAII removal

Below is a minimal, self-contained patch that fixes both 2.1 (logs_journal worker reconnect leak) and 2.2 (newHeads subscription leak) by replacing the shared `Vec<TracingUnboundedSender<_>>` with a slab-backed registry that returns a `SinkGuard` on registration. The guard removes the sink from the registry on drop.

This spans three places in the workspace: a new shared type, the journal worker, the newHeads branch, and the fc-mapping-sync producer. I'll show each piece and then list the wiring touch-ups.

---

## 1. The new type

Put this somewhere shared — `client/rpc-core/src/sink_registry.rs` or under `fc-mapping-sync` if you want to minimize the dep graph change. Add `slab = "0.4"` to the crate's `Cargo.toml`.

```rust
use parking_lot::Mutex;
use sc_utils::mpsc::{tracing_unbounded, TracingUnboundedReceiver, TracingUnboundedSender};
use slab::Slab;
use std::sync::{Arc, Weak};

/// A shared registry of async sinks with RAII-based removal.
///
/// Each caller of [`SinkRegistry::register`] receives a [`SinkGuard`]; when the
/// guard is dropped the corresponding sink is removed from the registry. The
/// producer side iterates with [`SinkRegistry::broadcast`], which also lazily
/// prunes any sink whose receiver has been dropped.
pub struct SinkRegistry<T> {
    inner: Mutex<Slab<TracingUnboundedSender<T>>>,
}

impl<T: Clone + Send + 'static> SinkRegistry<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(Slab::new()) })
    }

    /// Register a new sink. The returned [`SinkGuard`] MUST be kept alive for
    /// as long as the receiver is in use; dropping it removes the sink.
    pub fn register(
        self: &Arc<Self>,
        name: &'static str,
        warn_threshold: usize,
    ) -> (SinkGuard<T>, TracingUnboundedReceiver<T>) {
        let (sender, receiver) = tracing_unbounded(name, warn_threshold);
        let key = self.inner.lock().insert(sender);
        (
            SinkGuard { registry: Arc::downgrade(self), key: Some(key) },
            receiver,
        )
    }

    /// Send `msg` to every live sink, pruning any whose receiver has been
    /// dropped. The closure form avoids cloning when there are zero sinks.
    pub fn broadcast(&self, mut make_msg: impl FnMut() -> T) {
        let mut inner = self.inner.lock();
        if inner.is_empty() {
            return;
        }
        inner.retain(|_, sender| {
            if sender.is_closed() {
                return false;
            }
            // `unbounded_send` only fails if the receiver is gone; treat as closed.
            sender.unbounded_send(make_msg()).is_ok()
        });
    }

    /// Diagnostic — export this as a Prometheus gauge.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }
}

pub struct SinkGuard<T> {
    registry: Weak<SinkRegistry<T>>,
    key: Option<usize>,
}

impl<T> Drop for SinkGuard<T> {
    fn drop(&mut self) {
        if let (Some(registry), Some(key)) = (self.registry.upgrade(), self.key.take()) {
            let _ = registry.inner.lock().try_remove(key);
        }
    }
}
```

Two things to note:

- `broadcast` prunes on every send when it encounters a closed sink, so even if a guard is forgotten (shouldn't happen with the patches below), closed senders don't accumulate forever.
- `SinkGuard` holds a `Weak` so teardown order between registry and guards never matters.

---

## 2. Patch `client/rpc/src/logs_journal.rs`

Replace the helper and the worker loop. The key change: the guard lives **outside** the inner `while let` so it's re-created on each reconnect, and the old guard is dropped exactly when the old receiver is dropped.

```rust
// Signature change — take the registry directly.
pub fn spawn_logs_journal_worker<B, C, BE>(
    spawn_handle: &SpawnTaskHandle,
    sink_registry: Arc<SinkRegistry<EthereumBlockNotification<B>>>,
    // ... other args unchanged ...
) where /* ... */ {
    // Initial registration.
    let (mut sink_guard, mut notifications) =
        sink_registry.register("logs_journal_notification_stream", 100_000);

    let worker_fut = async move {
        loop {
            while let Some(notification) = notifications.next().await {
                // ... existing build_journal_payload + push + broadcast logic ...
            }

            // Stream ended. Insert gap marker (existing dedup logic).
            // ...

            tokio::time::sleep(LOGS_JOURNAL_RECONNECT_BACKOFF).await;

            // Re-register. Dropping the old guard removes the old sender
            // from the registry BEFORE we insert the new one.
            drop(sink_guard);
            let (new_guard, new_notifications) =
                sink_registry.register("logs_journal_notification_stream", 100_000);
            sink_guard = new_guard;
            notifications = new_notifications;
        }
    };

    spawn_handle.spawn("frontier-logs-journal", None, worker_fut);
}
```

Delete `register_notification_stream` entirely — its push-only semantics were the bug.

---

## 3. Patch `client/rpc/src/eth_pubsub.rs` (newHeads branch)

The subscription future must **own** the guard so the sink is removed when the client disconnects.

```rust
Kind::NewHeads => {
    let (sink_guard, block_notification_stream) = pubsub
        .sink_registry
        .register("pubsub_notification_stream", 100_000);

    let pubsub_cl = pubsub.clone();
    let flat_stream = block_notification_stream.flat_map(move |notification| {
        pubsub_cl.new_heads_from_notification(notification)
    });

    // Move the guard into the async task so it's dropped when the task ends,
    // regardless of whether the client disconnected cleanly or the stream errored.
    let subscription = PendingSubscription::from(pending)
        .pipe_from_stream(flat_stream, BoundedVecDeque::new(16));

    async move {
        let _sink_guard = sink_guard; // held until task exit
        subscription.await
    }
    .await
}
```

Apply the identical pattern to any other `Kind::*` branch that pushes into `pubsub_notification_sinks`. In current master only `Kind::NewHeads` does, but double-check your local fork if you've added custom subscription kinds.

---

## 4. Patch the producer side (`client/mapping-sync/`)

Wherever the pipeline currently does:

```rust
// old
for sink in sinks.lock().iter() {
    let _ = sink.unbounded_send(notification.clone());
}
```

replace with:

```rust
// new
sink_registry.broadcast(|| notification.clone());
```

And in the pipeline constructor, swap the `Arc<Mutex<Vec<TracingUnboundedSender<EthereumBlockNotification<B>>>>>` parameter for `Arc<SinkRegistry<EthereumBlockNotification<B>>>`.

---

## 5. Wiring in `template/node/src/rpc/eth.rs`

Replace the sink Vec construction with a single registry that's passed to both the mapping-sync pipeline and the RPC builders:

```rust
// before
let pubsub_notification_sinks: Arc<Mutex<Vec<TracingUnboundedSender<_>>>> =
    Arc::new(Mutex::new(Vec::new()));

// after
let sink_registry: Arc<SinkRegistry<EthereumBlockNotification<Block>>> =
    SinkRegistry::new();
```

Pass `sink_registry.clone()` into:
- the mapping-sync pipeline spawn,
- `spawn_logs_journal_worker`,
- `EthPubSub::new(..)`.

---

## 6. Observability — expose the sink count

Add a Prometheus gauge so this never silently regresses again:

```rust
let sink_count_gauge = register(
    Gauge::new("frontier_pubsub_sink_registry_len",
               "Current number of registered pubsub/logs-journal sinks")?,
    &registry,
)?;

// In a lightweight metrics task:
spawn_handle.spawn("frontier-sink-metrics", None, {
    let sink_registry = sink_registry.clone();
    async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(10));
        loop {
            ticker.tick().await;
            sink_count_gauge.set(sink_registry.len() as i64);
        }
    }
});
```

On a healthy node this should settle around `1 + active_newHeads_subscribers`. Any upward drift is a regression.

---

## 7. What this patch does NOT fix

Being explicit so you don't ship this expecting it to solve the OOM alone:

- **2.3 (unbounded ingress channel)**: `tracing_unbounded` still has no hard cap. If you want this closed too, swap `tracing_unbounded` for a bounded channel inside `SinkRegistry::register` (take a `capacity` param) and in `broadcast` use `try_send`; on error, mark the entry closed the same way as `is_closed()`.
- **2.4 (broadcast ring pinning by slow logs subscribers)**: separate concern in `logs_journal.rs`; needs a per-subscriber watchdog that force-closes subscribers whose last-delivered seq hasn't advanced for N seconds.
- **2.5 (512 MiB default budget)**: policy change in `LogsJournalConfig` defaults.
- **2.6 (per-subscription owned log clones)**: needs `Arc<Log>` instead of `Log` in `PubSubResult::Log`, which is a public API change and probably a bigger patch than you want.

Realistically, on a node that OOMs at the multi-GB scale, **2.6 + 2.4 are more likely the dominant contributors than the sink leak**. The sink leak is slow (linear in websocket churn, small per entry). I'd ship this patch because it's unambiguously correct, but I'd also run the diagnostics from section 3 of the previous message to confirm where the actual gigabytes are going before declaring victory.

Want me to sketch the bounded-ingress variant of `SinkRegistry`, or the watchdog for 2.4 next?