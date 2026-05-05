# Frontier OOM harness

Load generator and metrics sampler for investigating **logs journal / pubsub** memory behavior on a Frontier (or Frontier-based) node. It drives realistic WebSocket churn, a deliberately slow `eth_subscribe("logs")` client, steady subscribers, and optionally log-heavy transactions, while recording Prometheus samples to disk and producing a short Markdown report.

The harness is **black-box** except that the template node exposes three extra gauges when built from this repo (see [Prometheus metrics](#prometheus-metrics)).

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Node.js** | 20+ recommended |
| **pnpm** | `pnpm install` in this directory |
| **Running node** | HTTP RPC `9944`, WebSocket `9944`, Prometheus `9615` (defaults in `config.ts`) |
| **Foundry** (`forge`, `cast`, `jq`) | Only if you use `contracts/deploy.sh` or `make deploy` |

If your shell picks up an unrelated npm package named `forge`, `deploy.sh` prepends `~/.foundry/bin` to `PATH` when that directory exists.

## Node flags (example)

Start the node with external RPC, WebSocket, CORS, Prometheus, and steady sealing so runs are comparable:

```bash
./target/release/frontier-template-node \
  --dev --tmp \
  --rpc-external \
  --rpc-cors=all \
  --prometheus-external --prometheus-port 9615
```

Adjust the binary path if you build elsewhere.

## Install

```bash
cd frontier-oom-harness
pnpm install
```

## Configuration

Edit `config.ts` for RPC URLs, load shape, and sampling interval. A few behaviors are controlled by environment variables:

| Variable | Effect |
|----------|--------|
| `DURATION_MS` | Run length in milliseconds (default: 2 hours). `make smoke` sets 15 minutes. |
| `TX_SPAMMER_ENABLED` | `true` / `1` forces the tx spammer on; `false` / `0` forces it off. If unset, the spammer is **on** when `runs/.log-emitter-address` exists and contains a non-zero address (see [Log volume](#optional-log-volume)). |
| `NODE_PID` | PID of the `frontier-template-node` process. When set, the harness fills **RSS** (and **VSZ** on macOS / Linux) from the OS if Prometheus does not export `process_resident_memory_bytes` / `process_virtual_memory_bytes` (common on **macOS**). You can also set `PID_FROM_CONFIG` in `config.ts`. |

### RSS / VSZ on macOS

Substrate’s Prometheus export on Darwin often **does not** include `process_resident_memory_bytes` or `process_virtual_memory_bytes`, even though `/metrics` is healthy (you still get chain, RPC, and Frontier gauges). The harness then leaves those fields empty **unless** you pass the node PID, for example:

```bash
NODE_PID=$(pgrep -f frontier-template-node | head -1) pnpm run run:master
```

RSS and VSZ are then sampled with `ps` (macOS) or `/proc/<pid>/status` (Linux) and copied into the same `rss_bytes` / `vsz_bytes` fields used by the report.

## Prometheus metrics

The harness scrapes `http://127.0.0.1:9615/metrics` every `sampleIntervalMs` (default 5s). When the runtime exports them, standard process gauges include `process_resident_memory_bytes` and `process_virtual_memory_bytes`. This repository’s template node also registers:

| Metric | Meaning |
|--------|---------|
| `frontier_pubsub_sink_registry_len` | Length of the Ethereum block pubsub sink pool |
| `frontier_pubsub_sink_registry_capacity` | HashMap backing capacity for the sink registry (can exceed `len` after churn) |
| `frontier_pubsub_sink_closed` | Closed senders still present until the next broadcast prune |
| `frontier_mapping_sync_best_at_import_entries` | KV mapping-sync worker `best_at_import` map size (zero on SQL-only backend) |
| `frontier_logs_journal_entries_total_bytes` | Bytes retained in the logs journal deque |
| `frontier_logs_journal_broadcast_lag_max` | Max observed `Lagged(n)` skip count for logs subscribers |

If you run a node without these Frontier gauges, those fields are `null`. RSS/VSZ from Prometheus may also be `null` on macOS; use `NODE_PID` as above so reports still include memory curves.

## Optional log volume

To stress log throughput (helps exercise the journal and pubsub path):

1. With the dev node running: `make deploy` or `cd contracts && ./deploy.sh`
2. That writes the deployed address to `runs/.log-emitter-address` and enables the spammer on the next harness run (unless `TX_SPAMMER_ENABLED=false`).

The spammer uses the standard Substrate dev EVM private key (Alice’s mapped key). Only use on `--dev` chains.

Gas is estimated per call and clamped to Frontier’s per-transaction cap (`2^24` gas, EIP-7825). The previous hard-coded `20_000_000` exceeded that cap and was rejected as `exceeds transaction gas limit cap`. Each submission waits for inclusion before the next, which avoids pool `already known` noise from overlapping broadcasts.

## Running the harness

From `frontier-oom-harness/`:

```bash
# Label is only used in the output directory name (e.g. baseline vs patched comparison)
pnpm run run:master
# or
pnpm run run:patched
```

Artifacts:

- `runs/<label>-<iso-timestamp>/config.json` — resolved config snapshot  
- `runs/<label>-<iso-timestamp>/samples.ndjson` — one JSON object per line per scrape  

Heartbeat lines print roughly once per minute to stderr.

## Reports

After a run:

```bash
pnpm run report -- <run-id>
```

Example: `pnpm run report -- master-2026-04-23T12-00-00-000Z` (use the actual folder name under `runs/`).

The command writes `report.md` plus **SVG line charts** (`chart-*.svg`) next to the samples, prints the Markdown to stdout, and embeds the charts so they render in GitHub and VS Code. The JSON summary still includes RSS, VSZ, sink registry, journal bytes, and broadcast lag.

`make report-latest` runs the report for the lexicographically last run directory (handy if you only care about the newest run).

## Makefile shortcuts

| Target | What it does |
|--------|----------------|
| `make deploy` | Build Solidity and deploy `LogEmitter`; write `runs/.log-emitter-address` |
| `make smoke` | Deploy, then a **15 minute** labeled `master` run (`DURATION_MS` preset) |
| `make baseline` | Deploy, then default-duration `run:master` |
| `make patched` | Deploy, then `run:patched` |
| `make report-latest` | Report the latest run directory |

All targets assume you invoke `make` from `frontier-oom-harness/` with the node already running for deploy and harness steps.

## Comparing builds

Typical workflow:

1. Build and run the **baseline** node; run the harness with `pnpm run run:master` (or `make baseline`).
2. Switch to your **patched** branch, rebuild, restart the node with the same flags.
3. Run `pnpm run run:patched` (or `make patched`).
4. Generate `report.md` for each run and compare summaries (RSS slope, sink count growth, journal bytes, broadcast lag).

Warm the node for ~10 minutes before trusting slopes if you need a clean signal; the journal in `oom-logs-journal/` discusses confounders.

## Run artifacts and Git

`runs/.gitignore` ignores everything under `runs/` except itself so large NDJSON files stay local. Commit reports separately if you need them in Git.

## License

Same as the parent Frontier repository unless stated otherwise.
