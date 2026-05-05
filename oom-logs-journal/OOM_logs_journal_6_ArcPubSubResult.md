# 2.6 patch: eliminate per-subscriber log cloning via `Arc<Log>`

The problem, restated: every `eth_subscribe("logs")` subscriber currently allocates an **owned clone** of every matching log, boxes it, and pushes it into a per-subscription 16-deep ring. With N subscribers and M matching logs per entry, the system allocates `N × M` full `Log` clones per journal entry. A `Log` with a couple of topics and a few KB of data, multiplied across a Moonbeam-class log volume and a few hundred subscribers, is a lot of transient allocation — and if any subscriber's ring is ever full, some of that memory is retained.

Fix: share logs via `Arc`. The journal already holds `Arc<LogsJournalEntry>` — we extend that sharing one level deeper, down to the individual `Log`, so the subscription task sends a reference, not a copy.

This patch is more invasive than the previous three because it touches a public-ish type: `PubSubResult`. I'll show the change, all the call sites, and the serialization/compat considerations.

---

## 1. The core type change

In `fc-rpc-core` (or wherever `PubSubResult` is defined — typically `client/rpc-core/src/types/pubsub.rs`):

```rust
// Before
#[derive(Debug, Serialize, Eq, PartialEq, Clone)]
#[serde(untagged)]
pub enum PubSubResult {
    Header(Box<RichHeader>),
    Log(Box<Log>),
    TransactionHash(H256),
    SyncState(PubSubSyncStatus),
}

// After
#[derive(Debug, Clone)]
pub enum PubSubResult {
    Header(Arc<RichHeader>),
    Log(Arc<Log>),
    TransactionHash(H256),
    SyncState(PubSubSyncStatus),
}
```

Two changes worth calling out:

**Boxed → Arc.** `Box` implies unique ownership and forces a clone when we want to hand the same payload to multiple places. `Arc` is exactly what we want here: produce once, hand out cheaply to every subscriber.

**Manual `Serialize`.** `#[derive(Serialize)]` won't cleanly work across an `Arc` with `#[serde(untagged)]` (it will, via `&T` deref, but the `Eq/PartialEq` derives break on `Arc<Log>` unless `Log: Eq`). Simplest and most explicit:

```rust
impl Serialize for PubSubResult {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Header(h) => h.as_ref().serialize(s),
            Self::Log(l) => l.as_ref().serialize(s),
            Self::TransactionHash(h) => h.serialize(s),
            Self::SyncState(ss) => ss.serialize(s),
        }
    }
}
```

The wire format is **byte-identical** to the old `Box<Log>` version — `Arc<T>` serializes the same as `T` and as `Box<T>` (all via `Deref`). Every websocket client continues to work unchanged.

Drop the `Eq, PartialEq` derives or implement them by-value if something in the codebase depends on them (the existing tests mostly construct fresh values rather than compare them — quick `grep PubSubResult ==` will tell you).

---

## 2. Share logs inside `LogsJournalEntry`

In `logs_journal.rs`:

```rust
// Before
pub struct LogsJournalEntry {
    pub seq: u64,
    pub complete: bool,
    pub logs: Vec<Log>,
}

// After
pub struct LogsJournalEntry {
    pub seq: u64,
    pub complete: bool,
    pub logs: Vec<Arc<Log>>,
}
```

`build_journal_payload` becomes:

```rust
fn build_journal_payload(...) -> (bool, Vec<Arc<Log>>) {
    let mut out = Vec::new();
    // For each retracted/enacted block:
    for log in block_logs_iter {
        let mut l = log; // `Log` as before
        l.removed = retracted;
        out.push(Arc::new(l));
        // existing budget checks on count/bytes unchanged
        if over_budget { return (false, Vec::new()); }
    }
    (true, out)
}
```

One allocation per log, at construction time, amortized across all subscribers and all filter snapshot calls.

Byte-accounting for the state's `max_total_bytes` should count `Arc` strong-counted bytes once per entry, not once per subscriber — which is already what the state does (it sums over `entries`, not subscribers), so no change needed.

---

## 3. `snapshot_since` for filters

The filter path in `filter.rs` collects logs into the RPC response. It currently does something like:

```rust
// Before
let mut out = Vec::new();
for entry in snapshot {
    for log in &entry.logs {
        if log_matches_filter(&params, log, false) {
            out.push(log.clone());
        }
    }
}
```

With `Arc<Log>`, the natural rewrite is:

```rust
// After: clone the Arc (cheap), deref for filtering.
let mut out: Vec<Log> = Vec::new();
for entry in snapshot {
    for log in &entry.logs {
        if log_matches_filter(&params, log.as_ref(), false) {
            out.push((**log).clone()); // must deep-clone here: RPC returns owned Log
        }
    }
}
```

The filter RPC response is a `Vec<Log>` (owned) by convention — there's no way around cloning at the serialization boundary. But critically: **we only deep-clone for logs that actually match the filter**, and each subscriber/filter pays for its own matches only. That's already the case in the current code; the key improvement from this patch lands in the pub/sub path, not the filter path. The filter path is arguably unchanged in cost.

(If you want to push further: make the RPC return type `Vec<Cow<'a, Log>>` or serialize directly from `&Log`. Not worth it for filter polling, which is low-frequency compared to pub/sub.)

---

## 4. The pub/sub hot path — this is where the win is

In `eth_pubsub.rs` `Kind::Logs`:

```rust
// Before
for log in entry.logs.iter() {
    if log_matches_filter(&logs_params, log, false) {
        let msg = PubSubResult::Log(Box::new(log.clone()));
        if sink.send(&msg).await.is_err() { return; }
    }
}

// After
for log in entry.logs.iter() {
    if log_matches_filter(&logs_params, log.as_ref(), false) {
        // Arc::clone is atomic refcount bump — no allocation, no memcpy.
        let msg = PubSubResult::Log(Arc::clone(log));
        if sink.send(&msg).await.is_err() { return; }
    }
}
```

For N subscribers all interested in the same log, this changes the cost profile from:

- **Before**: N allocations, N memcpys of the full `Log` payload, N `Box` headers.
- **After**: N atomic refcount bumps. Zero allocations, zero memcpys.

Per-subscription `BoundedVecDeque::new(16)` now holds 16 `Arc<Log>`s worth of refcount pointers (+ `PubSubResult` enum tag overhead) rather than 16 fully-owned `Log`s. That's roughly **16 × N × sizeof(Log)** of transient heap pressure gone per subscriber ring, and it was *the* pressure that spiked during log-heavy blocks.

---

## 5. `new_heads` gets the same treatment for free

While we're touching the type:

```rust
// In new_heads_from_notification, before:
PubSubResult::Header(Box::new(rich_header))

// After:
PubSubResult::Header(Arc::new(rich_header))
```

With many `newHeads` subscribers (every wallet on the network), this is a meaningful reduction too. If you want to go further, cache the last-constructed `Arc<RichHeader>` keyed by block hash so even the single `Arc::new` allocation is shared — but honestly, header construction is cheap enough to not matter.

---

## 6. Breaking-change considerations

`PubSubResult` is `pub`. External crates that construct `PubSubResult::Log(Box::new(log))` will fail to compile. That's a semver-major change to `fc-rpc-core`.

Two acceptable approaches:

**Option A — breaking bump**: just ship it, bump the major version of `fc-rpc-core`, update the changelog to note "`PubSubResult::{Header,Log}` now wrap `Arc` instead of `Box`; wire format unchanged". Anyone constructing these manually changes one line. Given this is a client-library crate primarily consumed by node templates, the blast radius is small.

**Option B — keep the old signature, change internals**:

```rust
pub enum PubSubResult {
    Header(Box<RichHeader>),
    Log(Box<Log>),
    // ...
}

// Add a constructor that internalizes the Arc→Box conversion:
impl PubSubResult {
    pub fn log_shared(log: Arc<Log>) -> Self {
        // This still allocates + copies on construction.
        Self::Log(Box::new((*log).clone()))
    }
}
```

But **option B gets you nothing** — you're back to per-subscriber clones at the construction site. The whole point of this patch is sharing, so option A is the only version that actually fixes 2.6.

---

## 7. Interaction with `jsonrpsee`'s `SubscriptionSink::send`

`SubscriptionSink::send` signature is roughly `async fn send<T: Serialize>(&self, msg: &T) -> Result<...>`. It serializes by reference, so passing `&PubSubResult::Log(Arc<Log>)` serializes exactly like `&PubSubResult::Log(Box<Log>)` did — single trip through `Serialize`, single byte output. No per-send clone on the jsonrpsee side.

One very minor detail: jsonrpsee internally buffers the serialized bytes before writing to the websocket. That buffer is per-subscription and `String`-based. So the *serialized* bytes are still per-subscriber (there's no way around that with per-connection TLS and framing). The win is on the *structured* side, before serialization: one `Log` in memory, N serializations of it.

If you want to go even further and share serialized bytes across subscribers with identical filters, you'd need to pre-serialize once per entry and broadcast `Arc<str>` — but that's a much bigger architectural change and loses per-subscriber filtering flexibility. Not recommended unless profiling says the serializer is the bottleneck (it almost never is compared to allocation).

---

## 8. Verification

Before/after microbench with `criterion`, in `client/rpc/benches/`:

```rust
fn bench_broadcast_logs(c: &mut Criterion) {
    let entry = make_entry_with_n_logs(500); // realistic Moonbeam-ish block
    let subscribers = 100;

    c.bench_function("pubsub_broadcast_old", |b| {
        b.iter(|| {
            for _ in 0..subscribers {
                for log in &entry.logs {
                    let _msg = PubSubResult::Log(Box::new(log.clone()));
                    black_box(_msg);
                }
            }
        });
    });

    c.bench_function("pubsub_broadcast_new", |b| {
        b.iter(|| {
            for _ in 0..subscribers {
                for log in &entry.logs {
                    let _msg = PubSubResult::Log(Arc::clone(log));
                    black_box(_msg);
                }
            }
        });
    });
}
```

On my reasoning (not measured): expect roughly 20–100× speedup on the hot loop, proportional to `sizeof(Log)`, and **zero** heap allocations in the `new` variant.

In production: watch `process_resident_memory_bytes` on a log-heavy block. Pre-patch, you'll see a brief RSS spike tracking `N × logs_per_block × avg_log_size`. Post-patch, RSS should be flat across that same block (allocation locality still matters, but the *quantity* of bytes touched collapses).

---

## 9. Final stacked-patch summary

All four patches together:

| Memory source | Patch | Bound after patch |
|---|---|---|
| Worker reconnect sink leak (2.1) | `SinkRegistry` + RAII guard | O(live subscribers) |
| `newHeads` sub sink leak (2.2) | Same `SinkRegistry` | O(live subscribers) |
| Unbounded ingress channel (2.3) | Bounded registry + gap marker | `INGRESS_CAPACITY` × `sizeof(notif)` |
| Broadcast ring pinning by slow subs (2.4) | `LogsSubscriberHandle` + watchdog | `max_lag_entries` × `max_bytes_per_entry` per sub |
| Per-sub owned log clones (2.6) | `PubSubResult::Log(Arc<Log>)` | Amortized: 1 × `sizeof(entry)` across all subs |

Plus the one-line policy fix for 2.5 (drop default `max_total_bytes` from 512 MiB to something more realistic like 64 MiB).

At this point every identified growth path in `logs_journal` is either (a) a hard compile-time bound, (b) proportional only to the number of actually-connected subscribers, or (c) failing closed via the existing gap-marker mechanism.

If your nodes still OOM after all of this lands, the culprit is genuinely outside `logs_journal` — most likely candidates in order:

- `SchemaV3Override` storage cache (resize via CLI flags),
- Substrate's state cache / trie cache (same),
- EVM execution cache (`--eth-statuses-cache` if your fork exposes it),
- something in your runtime itself (contract storage growth).

Want a consolidated **validation playbook** — which metrics to watch, which logs to grep for, which synthetic load tests to run — to confirm each of these patches is doing what it's supposed to in your staging environment before rolling to prod?