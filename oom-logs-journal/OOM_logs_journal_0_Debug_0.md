# Monitoring logs_journal memory growth on Frontier master

A reproducible harness that (a) generates realistic load against a Frontier node, (b) records memory + sink-registry + broadcast metrics over time, and (c) produces a comparable baseline report. Run it twice — once on unpatched master, once on your patched branch — and diff the reports.

The harness is deliberately **black-box friendly**: it uses standard `/metrics`, `/proc/<pid>/status`, and RPC endpoints. The only thing that requires a node-side change is one optional instrumentation patch (section 3) which exposes sink-registry depth — without it, you can still measure RSS + broadcast lag + subscriber churn, which covers 80% of the signal.

---

## 1. What we're measuring and why

Five time series, collected every 5 seconds:

| Series | Source | Detects |
|---|---|---|
| `process_resident_memory_bytes` | Substrate's built-in `/metrics` | Overall RSS growth (2.x summed) |
| `process_virtual_memory_bytes` | Same | VSZ growth — catches mmap leaks that don't touch RSS |
| `frontier_pubsub_sink_registry_len` | Custom gauge (section 3) | Sink leak (2.1, 2.2) |
| `frontier_logs_journal_entries_total_bytes` | Custom gauge | Journal state growth (bounded since #1881) |
| `frontier_logs_journal_broadcast_lag_max` | Custom gauge | Broadcast ring pinning (2.4) |

Plus three workload variables we deliberately drive:

| Driver | What it stresses |
|---|---|
| **WebSocket subscription churn**: connect/disconnect `eth_subscribe("newHeads")` and `("logs")` clients on a ~30s cycle | 2.1, 2.2 (sink leak per reconnect) |
| **Slow-subscriber**: one client that deliberately stops reading but keeps the TCP socket open | 2.3, 2.4 (broadcast/ingress pinning) |
| **Steady-state subscribers**: a stable pool of N well-behaved subscribers for baseline | Normal load separation |

If, across a 2–4 hour run, we can show these series grow linearly on master and are flat on your patched build, that's the proof.

---

## 2. Assumptions

- Frontier dev node or a testnet node you control, reachable at `http://127.0.0.1:9944` (HTTP RPC) and `ws://127.0.0.1:9944` (WS RPC). Adjust in `config.ts`.
- Node started with `--prometheus-external --prometheus-port 9615` (default port).
- A way to *produce logs* at a steady rate. Two options: (a) submit transactions to a log-heavy contract, or (b) run a Frontier dev chain with `--sealing=interval=2000` and an existing deployed ERC-20/event-emitter. The harness includes an optional tx-spammer using a prefunded dev account.
- Node.js 20+ and `pnpm`. We'll use `viem` for RPC, `ws` for raw websocket behavior, and `prom-client` / plain HTTP scraping for metrics.

---

## 3. One optional node-side patch (adds `frontier_pubsub_sink_registry_len`)

Without this gauge, you can still detect sink leaks indirectly via RSS growth under churn load, but you can't attribute the growth precisely. If you can patch the node you're running, add this to `template/node/src/rpc/eth.rs` (or wherever the `SinkRegistry` is constructed — use the `Vec<Sender>` equivalent on unpatched master):

```rust
use substrate_prometheus_endpoint::{register, Gauge, U64};

let sink_registry_len = register(
    Gauge::<U64>::new(
        "frontier_pubsub_sink_registry_len",
        "Number of registered pubsub/logs-journal sinks",
    )?,
    &prometheus_registry,
)?;

// On unpatched master, where the sinks are a Vec<Sender>:
spawn_handle.spawn("frontier-sink-metrics", None, {
    let sinks = pubsub_notification_sinks.clone();
    async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            ticker.tick().await;
            sink_registry_len.set(sinks.lock().len() as u64);
        }
    }
});
```

Drop-in equivalent for the patched `SinkRegistry` type (from the earlier patch series): replace `sinks.lock().len()` with `sink_registry.len()`.

Add the other two gauges similarly if you want them — `logs_journal_entries_total_bytes` by reading `state.lock().current_bytes()` (adds a cheap accessor), and `broadcast_lag_max` by iterating the subscriber registry if you have the watchdog patch.

---

## 4. The harness — file-by-file

Project layout:

```
frontier-oom-harness/
├── package.json
├── tsconfig.json
├── config.ts
├── src/
│   ├── main.ts              # orchestrator
│   ├── metrics-scraper.ts   # /metrics + /proc RSS poller
│   ├── churn-driver.ts      # connect/disconnect WS subs on a cycle
│   ├── slow-subscriber.ts   # one client that stops reading
│   ├── steady-subscribers.ts
│   ├── tx-spammer.ts        # optional: emits log-heavy txs
│   └── report.ts            # post-run analysis + plot
└── runs/                    # written per-run; one subdir per run
    └── <run-id>/
        ├── config.json
        ├── samples.ndjson   # one metrics snapshot per line
        └── report.md
```

### 4.1 `package.json`

```json
{
  "name": "frontier-oom-harness",
  "private": true,
  "type": "module",
  "scripts": {
    "run:master": "tsx src/main.ts --label=master",
    "run:patched": "tsx src/main.ts --label=patched",
    "report": "tsx src/report.ts"
  },
  "dependencies": {
    "viem": "^1^.21.0",
    "ws": "^8.18.0",
    "undici": "^6.19.0",
    "asciichart": "^1.5.25"
  },
  "devDependencies": {
    "tsx": "^4.19.0",
    "typescript": "^5.5.0",
    "@types/node": "^20.14.0",
    "@types/ws": "^2^.5.12"
  }
}
```

### 4.2 `config.ts`

```typescript
export const CONFIG = {
  node: {
    httpRpc: "http://127.0.0.1:9944",
    wsRpc:   "ws://127.0.0.1:9944",
    metrics: "http://127.0.0.1:9615/metrics",
    // Optional: path to node's /proc/<pid>/status for RSS cross-check.
    // If the harness runs on the same host, fill in the PID or set to null
    // and the harness will try to resolve it from the RPC endpoint PID.
    pid: null as number | null,
  },

  run: {
    // Total duration. For a real baseline use ≥ 2 hours. 15min = smoke test.
    durationMs: 2 * 60 * 60 * 1000,
    sampleIntervalMs: 5_000,
  },

  load: {
    // Steady well-behaved subs.
    steadyNewHeads: 20,
    steadyLogs: 20,

    // Churn driver: connect & disconnect this many subs per cycle.
    churnNewHeadsPerCycle: 10,
    churnLogsPerCycle: 10,
    churnCycleMs: 30_000,

    // Slow subscriber.
    slowSubscriberEnabled: true,

    // Optional tx spammer (logs-generating contract).
    txSpammer: {
      enabled: false,
      contract: "0x0000000000000000000000000000000000000000",
      // Pre-funded Substrate dev account (Alice's EVM-mapped key).
      privateKey: "0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133",
      txPerMinute: 60,
    },
  },

  // Filter used for all logs subscriptions. Empty = all logs.
  logsFilter: { topics: [] as string[] },
};
```

### 4.3 `src/metrics-scraper.ts`

```typescript
import { request } from "undici";
import { readFile } from "node:fs/promises";
import { CONFIG } from "../config.js";

export type MetricsSample = {
  ts: number;
  rss_bytes: number | null;
  vsz_bytes: number | null;
  proc_rss_bytes: number | null;
  sink_registry_len: number | null;
  journal_entries_bytes: number | null;
  broadcast_lag_max: number | null;
  // Raw gauges we don't analyze but keep for forensics.
  substrate_block_height: number | null;
  websocket_sessions: number | null;
};

const GAUGES = {
  rss_bytes: "process_resident_memory_bytes",
  vsz_bytes: "process_virtual_memory_bytes",
  sink_registry_len: "frontier_pubsub_sink_registry_len",
  journal_entries_bytes: "frontier_logs_journal_entries_total_bytes",
  broadcast_lag_max: "frontier_logs_journal_broadcast_lag_max",
  substrate_block_height: "substrate_block_height",
  websocket_sessions: "substrate_sub_libp2p_open_streams_count", // best-effort
} as const;

export async function scrapeMetrics(): Promise<MetricsSample> {
  const ts = Date.now();
  let body = "";
  try {
    const { body: b } = await request(CONFIG.node.metrics);
    body = await b.text();
  } catch {
    // Node unreachable — return sentinels so the gap is visible in samples.
    return emptySample(ts);
  }

  const parsed: Record<string, number | null> = {};
  for (const [key, metricName] of Object.entries(GAUGES)) {
    parsed[key] = parsePromGauge(body, metricName);
  }

  // Optional: cross-check RSS via /proc/<pid>/status.
  let proc_rss_bytes: number | null = null;
  if (CONFIG.node.pid) {
    proc_rss_bytes = await readProcRss(CONFIG.node.pid).catch(() => null);
  }

  return { ts, proc_rss_bytes, ...(parsed as any) };
}

function parsePromGauge(body: string, name: string): number | null {
  // Matches: `name value` or `name{...} value`. Ignores comment lines.
  const re = new RegExp(`^${name}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)\\s*$`, "m");
  const m = body.match(re);
  return m ? Number(m[1]) : null;
}

async function readProcRss(pid: number): Promise<number> {
  const status = await readFile(`/proc/${pid}/status`, "utf8");
  const m = status.match(/VmRSS:\s+(\d+)\s+kB/);
  if (!m) throw new Error("VmRSS not found");
  return Number(m[1]) * 1024;
}

function emptySample(ts: number): MetricsSample {
  return {
    ts,
    rss_bytes: null, vsz_bytes: null, proc_rss_bytes: null,
    sink_registry_len: null, journal_entries_bytes: null, broadcast_lag_max: null,
    substrate_block_height: null, websocket_sessions: null,
  };
}
```

### 4.4 `src/churn-driver.ts`

This is the critical driver for exposing the sink leaks. We deliberately connect, subscribe, briefly idle, then **close the websocket** — no `eth_unsubscribe` call — which is what real misbehaving clients do and is precisely the path that stresses the sink-teardown logic.

```typescript
import WebSocket from "ws";
import { CONFIG } from "../config.js";

export class ChurnDriver {
  private running = false;

  async start(): Promise<void> {
    this.running = true;
    while (this.running) {
      await this.runCycle().catch((e) => {
        console.error(`[churn] cycle error: ${e.message}`);
      });
      await sleep(CONFIG.load.churnCycleMs);
    }
  }

  stop() { this.running = false; }

  private async runCycle(): Promise<void> {
    const subs: Promise<void>[] = [];
    for (let i = 0; i < CONFIG.load.churnNewHeadsPerCycle; i++) {
      subs.push(openAndClose("newHeads"));
    }
    for (let i = 0; i < CONFIG.load.churnLogsPerCycle; i++) {
      subs.push(openAndClose("logs", CONFIG.logsFilter));
    }
    await Promise.allSettled(subs);
  }
}

async function openAndClose(kind: string, filter?: unknown): Promise<void> {
  const ws = new WebSocket(CONFIG.node.wsRpc);
  await waitOpen(ws);

  const params = filter !== undefined ? [kind, filter] : [kind];
  ws.send(JSON.stringify({
    jsonrpc: "2.0", id: 1, method: "eth_subscribe", params,
  }));

  // Receive a few messages, then abruptly terminate — no eth_unsubscribe.
  await new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, 5_000);
    ws.once("close", () => { clearTimeout(timer); resolve(); });
    ws.once("error", () => { clearTimeout(timer); resolve(); });
  });

  // Abrupt close — simulates a client process dying or a network drop.
  ws.terminate();
}

function waitOpen(ws: WebSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    ws.once("open", resolve);
    ws.once("error", reject);
  });
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
```

### 4.5 `src/slow-subscriber.ts`

```typescript
import WebSocket from "ws";
import { CONFIG } from "../config.js";

// A single long-lived WS connection that subscribes to logs, receives
// ~10 messages to prime the broadcast channel, then stops calling recv.
// The TCP socket stays open; the server's broadcast slot for this client
// cannot be reclaimed until tokio's `Lagged` fires or our watchdog kicks.
export class SlowSubscriber {
  private ws?: WebSocket;

  async start(): Promise<void> {
    this.ws = new WebSocket(CONFIG.node.wsRpc);
    await new Promise<void>((res, rej) => {
      this.ws!.once("open", res);
      this.ws!.once("error", rej);
    });

    this.ws.send(JSON.stringify({
      jsonrpc: "2.0", id: 1, method: "eth_subscribe",
      params: ["logs", CONFIG.logsFilter],
    }));

    let received = 0;
    this.ws.on("message", () => {
      received++;
      if (received >= 10) {
        // Pause the stream by removing all listeners. The socket's receive
        // buffer will fill, TCP window will close, and the server's send()
        // will backpressure. Broadcast ring on server pins entries.
        this.ws!.removeAllListeners("message");
      }
    });
  }

  stop() { this.ws?.terminate(); }
}
```

Note: pausing the `message` handler doesn't actually stop `ws` from draining the socket — `ws` reads eagerly. To genuinely stall the server, we need to stop reading at the TCP level. That requires going below `ws` to the underlying socket:

```typescript
// After opening and subscribing:
const sock = (this.ws as any)._socket as import("net").Socket;
sock.pause(); // stops reading from TCP; server send() will eventually block
```

Added as a `.pause()` call after the first 10 messages. That's the real stall.

### 4.6 `src/steady-subscribers.ts`

Simple: N well-behaved, keep-alive subscribers. They do nothing interesting except provide a baseline so churn effects stand out.

```typescript
import WebSocket from "ws";
import { CONFIG } from "../config.js";

export class SteadySubscribers {
  private sockets: WebSocket[] = [];

  async start(): Promise<void> {
    const promises: Promise<WebSocket>[] = [];
    for (let i = 0; i < CONFIG.load.steadyNewHeads; i++) {
      promises.push(this.open("newHeads"));
    }
    for (let i = 0; i < CONFIG.load.steadyLogs; i++) {
      promises.push(this.open("logs", CONFIG.logsFilter));
    }
    this.sockets = await Promise.all(promises);
  }

  stop() { for (const s of this.sockets) s.terminate(); }

  private open(kind: string, filter?: unknown): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(CONFIG.node.wsRpc);
      ws.once("open", () => {
        const params = filter !== undefined ? [kind, filter] : [kind];
        ws.send(JSON.stringify({
          jsonrpc: "2.0", id: 1, method: "eth_subscribe", params,
        }));
        // Drain messages to avoid being mistaken for slow.
        ws.on("message", () => {});
        resolve(ws);
      });
      ws.once("error", reject);
    });
  }
}
```

### 4.7 `src/tx-spammer.ts` (optional)

```typescript
import { createWalletClient, http, parseAbi } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { CONFIG } from "../config.js";

// Assumes a deployed contract with a public `emitLogs(uint256 count)` method.
// Deploy this minimal contract once on your test chain if needed:
//
//   event Ping(uint256 indexed i, bytes32 data);
//   function emitLogs(uint256 n) external {
//     for (uint i=0; i<n; i++) emit Ping(i, keccak256(abi.encode(i, block.number)));
//   }

const ABI = parseAbi(["function emitLogs(uint256 n)"]);

export class TxSpammer {
  private running = false;

  async start(): Promise<void> {
    if (!CONFIG.load.txSpammer.enabled) return;
    this.running = true;

    const account = privateKeyToAccount(CONFIG.load.txSpammer.privateKey as `0x${string}`);
    const client = createWalletClient({ account, transport: http(CONFIG.node.httpRpc) });

    const intervalMs = 60_000 / CONFIG.load.txSpammer.txPerMinute;
    while (this.running) {
      try {
        await client.writeContract({
          address: CONFIG.load.txSpammer.contract as `0x${string}`,
          abi: ABI,
          functionName: "emitLogs",
          args: [50n],
          chain: null,
        });
      } catch (e: any) {
        console.error(`[spammer] ${e.message}`);
      }
      await sleep(intervalMs);
    }
  }

  stop() { this.running = false; }
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
```

### 4.8 `src/main.ts`

```typescript
import { mkdir, writeFile, appendFile } from "node:fs/promises";
import { join } from "node:path";
import { CONFIG } from "../config.js";
import { scrapeMetrics } from "./metrics-scraper.js";
import { ChurnDriver } from "./churn-driver.js";
import { SlowSubscriber } from "./slow-subscriber.js";
import { SteadySubscribers } from "./steady-subscribers.js";
import { TxSpammer } from "./tx-spammer.js";

async function main() {
  const label = process.argv.find(a => a.startsWith("--label="))?.split("=")[1] ?? "run";
  const runId = `${label}-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const dir = join("runs", runId);
  await mkdir(dir, { recursive: true });
  const samplesPath = join(dir, "samples.ndjson");
  await writeFile(join(dir, "config.json"), JSON.stringify({ label, ...CONFIG }, null, 2));

  console.log(`[run ${runId}] starting; duration=${CONFIG.run.durationMs / 60_000}min`);

  const steady = new SteadySubscribers();
  const churn = new ChurnDriver();
  const slow = new SlowSubscriber();
  const spammer = new TxSpammer();

  await steady.start();
  if (CONFIG.load.slowSubscriberEnabled) await slow.start();
  void churn.start();
  void spammer.start();

  const deadline = Date.now() + CONFIG.run.durationMs;
  let samples = 0;
  while (Date.now() < deadline) {
    const sample = await scrapeMetrics();
    await appendFile(samplesPath, JSON.stringify(sample) + "\n");
    samples++;
    if (samples % 12 === 0) {
      // Print a compact heartbeat every ~1 minute.
      console.log(
        `[${new Date().toISOString()}] rss=${fmtMb(sample.rss_bytes)} ` +
        `sinks=${sample.sink_registry_len ?? "?"} ` +
        `journal_bytes=${fmtMb(sample.journal_entries_bytes)} ` +
        `bc_lag=${sample.broadcast_lag_max ?? "?"}`
      );
    }
    await sleep(CONFIG.run.sampleIntervalMs);
  }

  churn.stop(); slow.stop(); spammer.stop(); steady.stop();
  console.log(`[run ${runId}] done; wrote ${samples} samples to ${samplesPath}`);
  console.log(`[run ${runId}] generate report with: pnpm report ${runId}`);
}

const fmtMb = (b: number | null) => b == null ? "?" : `${(b / 1024 / 1024).toFixed(1)}MiB`;
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

main().catch(e => { console.error(e); process.exit(1); });
```

### 4.9 `src/report.ts`

```typescript
import { readFile, writeFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import asciichart from "asciichart";

async function main() {
  const runId = process.argv[2];
  if (!runId) {
    const runs = (await readdir("runs")).sort();
    console.error("usage: pnpm report <run-id>");
    console.error("available runs:"); runs.forEach(r => console.error("  " + r));
    process.exit(1);
  }

  const dir = join("runs", runId);
  const raw = await readFile(join(dir, "samples.ndjson"), "utf8");
  const samples = raw.trim().split("\n").map(l => JSON.parse(l));
  if (samples.length < 2) { console.error("not enough samples"); process.exit(1); }

  const t0 = samples[0].ts;
  const dur = (samples.at(-1)!.ts - t0) / 1000;

  const summary = {
    runId,
    durationSeconds: dur,
    sampleCount: samples.length,
    rss: summarize(samples, "rss_bytes"),
    vsz: summarize(samples, "vsz_bytes"),
    sinkRegistry: summarize(samples, "sink_registry_len"),
    journalBytes: summarize(samples, "journal_entries_bytes"),
    broadcastLag: summarize(samples, "broadcast_lag_max"),
  };

  const rssChart = asciichart.plot(
    samples.map(s => (s.rss_bytes ?? 0) / 1024 / 1024),
    { height: 12 },
  );
  const sinkChart = asciichart.plot(
    samples.map(s => s.sink_registry_len ?? 0),
    { height: 8 },
  );

  const md = [
    `# Run report: ${runId}`, "",
    `Duration: ${dur.toFixed(0)}s, samples: ${samples.length}`, "",
    "## Summary", "```json", JSON.stringify(summary, null, 2), "```", "",
    "## RSS (MiB) over time", "```", rssChart, "```", "",
    "## Sink registry length over time", "```", sinkChart, "```",
  ].join("\n");

  await writeFile(join(dir, "report.md"), md);
  console.log(md);
  console.log(`\nReport saved to ${join(dir, "report.md")}`);
}

function summarize(samples: any[], key: string) {
  const values = samples.map(s => s[key]).filter((v): v is number => v != null);
  if (values.length < 2) return { samples: values.length, note: "insufficient data" };
  const first = values[0];
  const last = values.at(-1)!;
  const max = Math.max(...values);
  const min = Math.min(...values);
  // Linear regression slope, per second.
  const firstTs = samples.find(s => s[key] != null)!.ts;
  const points = samples
    .filter(s => s[key] != null)
    .map(s => ({ x: (s.ts - firstTs) / 1000, y: s[key] as number }));
  const slopePerSec = linearSlope(points);
  const slopePerHour = slopePerSec * 3600;
  return { first, last, min, max, delta: last - first, slopePerHour, samples: values.length };
}

function linearSlope(pts: { x: number; y: number }[]): number {
  const n = pts.length;
  const sx = pts.reduce((a, p) => a + p.x, 0);
  const sy = pts.reduce((a, p) => a + p.y, 0);
  const sxx = pts.reduce((a, p) => a + p.x * p.x, 0);
  const sxy = pts.reduce((a, p) => a + p.x * p.y, 0);
  const denom = n * sxx - sx * sx;
  return denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
}

main().catch(e => { console.error(e); process.exit(1); });
```

---

## 5. Running it: baseline vs. patched

### Step 1: baseline on unpatched master

```bash
# Terminal 1 — node
./frontier-node \
  --dev --tmp \
  --rpc-external --ws-external \
  --rpc-cors=all \
  --prometheus-external --prometheus-port 9615 \
  --sealing=interval=2000

# Terminal 2 — harness
pnpm install
pnpm run:master
# ... wait 2 hours ...
pnpm report master-2026-04-23T...
```

### Step 2: patched build

Rebuild with your patches applied. Same node command. Same harness config.

```bash
pnpm run:patched
pnpm report patched-2026-04-23T...
```

### Step 3: diff

The key numbers to compare between the two `summary.json` blocks:

| Metric | Master (expected) | Patched (expected) |
|---|---|---|
| `rss.slopePerHour` | Positive, ideally > 50 MiB/hr under churn | Near zero (within noise) |
| `sinkRegistry.slopePerHour` | Positive and ~linear with `churnCycleMs` | Zero — size tracks live subs only |
| `sinkRegistry.max` | Unbounded growth over run | Bounded by `steady + transient churn` |
| `broadcastLag.max` | Can grow toward `max_entries` | Bounded by watchdog's `max_lag_entries` |
| `journalBytes.max` | ≤ 512 MiB (already bounded by #1881) | ≤ 512 MiB (unchanged) or lower if you patched 2.5 |

If the patched run shows flat RSS and flat sink_registry under the same load, the fix lands.

---

## 6. Things to watch for that can confound the comparison

- **Warmup**: Substrate caches warm over the first ~10 minutes. Discard the first 600 s when computing slopes if you want a cleaner signal. Easy addition to `report.ts`: `samples.filter(s => s.ts - t0 > 600_000)`.
- **Block time drift**: `--sealing=interval=2000` is deterministic, but manual sealing isn't. Use interval sealing for apples-to-apples.
- **Allocator behavior**: jemalloc doesn't always release memory back to the OS, so RSS can look worse than actual allocator heap. If you suspect this, run a second pass with `MALLOC_CONF=background_thread:true,narenas:1,dirty_decay_ms:1000`.
- **OS-level cache pressure**: RSS can grow because of page cache pressure from DB reads, which is fine. Cross-check with `VmRSS` from `/proc` (already in the scraper) and with `journalBytes` growth — if RSS grows but journal bytes are flat and sink registry is flat, the growth is somewhere else (not our concern here).
- **Churn driver's own memory**: the harness itself allocates. Run it on a different host if possible, or at least note its own RSS separately.

---

## 7. Minimum viable smoke test

If two hours is too long for a first pass, start here to validate the harness works at all:

```typescript
// In config.ts, temporarily:
run: { durationMs: 15 * 60 * 1000, sampleIntervalMs: 2_000 },
load: {
  steadyNewHeads: 5, steadyLogs: 5,
  churnNewHeadsPerCycle: 20, churnLogsPerCycle: 20,
  churnCycleMs: 5_000,  // aggressive churn
  slowSubscriberEnabled: true,
  txSpammer: { enabled: false, /* ... */ },
},
```

15-minute run with aggressive churn. On unpatched master you should already see `sink_registry_len` climbing visibly in the ASCII chart.

---

## 8. Packaging for your team

Check the whole `frontier-oom-harness/` directory into a repo (internal or a fork of frontier's `ci/` tree). Three commit-level artifacts worth preserving:

- `runs/master-<ts>/report.md` — the smoking gun.
- `runs/patched-<ts>/report.md` — the proof.
- A short `FINDINGS.md` that pastes the two `summary.json` blocks side by side.

That triad is what you want to attach to the upstream issue or PR when you file the fix.

---

I will flesh out the `emitLogs` Solidity contract + deployment script so the tx-spammer path works out of the box.