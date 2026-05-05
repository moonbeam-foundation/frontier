import { execFile } from "child_process";
import { readFile } from "node:fs/promises";
import { request } from "undici";
import { CONFIG } from "../config.js";

export type MetricsSample = {
  ts: number;
  rss_bytes: number | null;
  vsz_bytes: number | null;
  proc_rss_bytes: number | null;
  sink_registry_len: number | null;
  sink_registry_capacity: number | null;
  sink_closed: number | null;
  mapping_sync_best_at_import_entries: number | null;
  journal_entries_bytes: number | null;
  broadcast_lag_max: number | null;
  substrate_block_height: number | null;
  websocket_sessions: number | null;
};

const GAUGES = {
  rss_bytes: "process_resident_memory_bytes",
  vsz_bytes: "process_virtual_memory_bytes",
  sink_registry_len: "frontier_pubsub_sink_registry_len",
  sink_registry_capacity: "frontier_pubsub_sink_registry_capacity",
  sink_closed: "frontier_pubsub_sink_closed",
  mapping_sync_best_at_import_entries:
    "frontier_mapping_sync_best_at_import_entries",
  journal_entries_bytes: "frontier_logs_journal_entries_total_bytes",
  broadcast_lag_max: "frontier_logs_journal_broadcast_lag_max",
  substrate_block_height: "substrate_block_height",
  websocket_sessions: "substrate_sub_libp2p_open_streams_count",
} as const;

export async function scrapeMetrics(): Promise<MetricsSample> {
  const ts = Date.now();
  let body = "";
  try {
    const { body: b } = await request(CONFIG.node.metrics);
    body = await b.text();
  } catch {
    return emptySample(ts);
  }

  const parsed: Record<string, number | null> = {};
  for (const [key, metricName] of Object.entries(GAUGES)) {
    parsed[key] = parsePromGauge(body, metricName);
  }

  let proc_rss_bytes: number | null = null;
  let proc_vsz_bytes: number | null = null;
  if (CONFIG.node.pid != null) {
    if (process.platform === "darwin") {
      proc_rss_bytes = await readDarwinRssBytes(CONFIG.node.pid);
      proc_vsz_bytes = await readDarwinVszBytes(CONFIG.node.pid);
    } else {
      const st = await readProcStatusLinux(CONFIG.node.pid).catch(() => null);
      if (st) {
        proc_rss_bytes = st.rssBytes;
        proc_vsz_bytes = st.vszBytes;
      }
    }
  }

  let rss_bytes = parsed.rss_bytes ?? null;
  let vsz_bytes = parsed.vsz_bytes ?? null;
  if (rss_bytes == null && proc_rss_bytes != null) {
    rss_bytes = proc_rss_bytes;
  }
  if (vsz_bytes == null && proc_vsz_bytes != null) {
    vsz_bytes = proc_vsz_bytes;
  }

  return {
    ts,
    proc_rss_bytes,
    rss_bytes,
    vsz_bytes,
    sink_registry_len: parsed.sink_registry_len ?? null,
    sink_registry_capacity: parsed.sink_registry_capacity ?? null,
    sink_closed: parsed.sink_closed ?? null,
    mapping_sync_best_at_import_entries:
      parsed.mapping_sync_best_at_import_entries ?? null,
    journal_entries_bytes: parsed.journal_entries_bytes ?? null,
    broadcast_lag_max: parsed.broadcast_lag_max ?? null,
    substrate_block_height: parsed.substrate_block_height ?? null,
    websocket_sessions: parsed.websocket_sessions ?? null,
  };
}

function parsePromGauge(body: string, name: string): number | null {
  const re = new RegExp(
    `^${escapeRegex(name)}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)\\s*$`,
    "m",
  );
  const m = body.match(re);
  return m ? Number(m[1]) : null;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function readProcStatusLinux(pid: number): Promise<{
  rssBytes: number;
  vszBytes: number;
}> {
  const status = await readFile(`/proc/${pid}/status`, "utf8");
  const rss = status.match(/VmRSS:\s+(\d+)\s+kB/);
  const vsz = status.match(/VmSize:\s+(\d+)\s+kB/);
  if (!rss) throw new Error("VmRSS not found");
  const rssBytes = Number(rss[1]) * 1024;
  const vszBytes = vsz ? Number(vsz[1]) * 1024 : rssBytes;
  return { rssBytes, vszBytes };
}

/** macOS `ps` RSS column is resident memory in KiB. */
function execFileUtf8(
  file: string,
  args: readonly string[],
): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(file, [...args], { encoding: "utf8" }, (err, stdout) => {
      if (err) reject(err);
      else resolve(typeof stdout === "string" ? stdout : String(stdout));
    });
  });
}

async function readDarwinRssBytes(pid: number): Promise<number | null> {
  try {
    const stdout = await execFileUtf8("ps", ["-p", String(pid), "-o", "rss="]);
    const kib = Number(stdout.trim().split(/\s+/)[0]);
    if (!Number.isFinite(kib) || kib < 0) return null;
    return Math.round(kib * 1024);
  } catch {
    return null;
  }
}

/**
 * macOS `ps -o vsz=` reports **virtual size in bytes** (not KiB × 1024); scaling
 * KiB produced multi‑TB bogus values in reports.
 */
async function readDarwinVszBytes(pid: number): Promise<number | null> {
  try {
    const stdout = await execFileUtf8("ps", ["-p", String(pid), "-o", "vsz="]);
    const b = Number(stdout.trim().split(/\s+/)[0]);
    if (!Number.isFinite(b) || b < 0) return null;
    return Math.round(b);
  } catch {
    return null;
  }
}

function emptySample(ts: number): MetricsSample {
  return {
    ts,
    rss_bytes: null,
    vsz_bytes: null,
    proc_rss_bytes: null,
    sink_registry_len: null,
    sink_registry_capacity: null,
    sink_closed: null,
    mapping_sync_best_at_import_entries: null,
    journal_entries_bytes: null,
    broadcast_lag_max: null,
    substrate_block_height: null,
    websocket_sessions: null,
  };
}
