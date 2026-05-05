import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

function readDeployedAddress(): string {
  const path = join(__dirname, "runs", ".log-emitter-address");
  if (existsSync(path)) {
    return readFileSync(path, "utf8").trim();
  }
  return "0x0000000000000000000000000000000000000000";
}

const ZERO = "0x0000000000000000000000000000000000000000";
const deployed = readDeployedAddress();
/** Process id for OS-level RSS/VSZ when Prometheus omits `process_*_memory_bytes` (typical on macOS). */
const PID_FROM_CONFIG: number | null = null;

const nodePidRaw = process.env.NODE_PID;
const nodePidFromEnv =
  nodePidRaw != null &&
  nodePidRaw !== "" &&
  Number.isFinite(Number(nodePidRaw)) &&
  Number(nodePidRaw) > 0
    ? Number(nodePidRaw)
    : null;
const nodePid = nodePidFromEnv ?? PID_FROM_CONFIG;

const txSpammerEnv = process.env.TX_SPAMMER_ENABLED;
const txSpammerEnabled =
  txSpammerEnv === "0" || txSpammerEnv === "false"
    ? false
    : txSpammerEnv === "1" || txSpammerEnv === "true"
      ? true
      : deployed !== ZERO;

export const CONFIG = {
  // node: {
    // httpRpc: "http://127.0.0.1:9944",
    // wsRpc: "ws://127.0.0.1:9944",
    // metrics: "http://127.0.0.1:9615/metrics",
    // pid: nodePid,
  // },
  node: {
    httpRpc: "http://127.0.0.1:8800",
    wsRpc: "ws://127.0.0.1:8800",
    metrics: "http://127.0.0.1:57537/metrics",
    pid: nodePid,
  },

  run: {
    durationMs: Number(process.env.DURATION_MS ?? 2 * 60 * 60 * 1000),
    sampleIntervalMs: 5_000,
  },

  load: {
    steadyNewHeads: 20,
    steadyLogs: 20,

    churnNewHeadsPerCycle: 10,
    churnLogsPerCycle: 10,
    churnCycleMs: 30_000,

    slowSubscriberEnabled: true,

    txSpammer: {
      enabled: txSpammerEnabled,
      contract: deployed as `0x${string}`,
      privateKey:
        "0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133",
      txPerMinute: 60,
      logsPerTx: 50,
      accounts: [] as `0x${string}`[],
    },
  },

  logsFilter: { topics: [] as string[] },
};
