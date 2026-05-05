import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { CONFIG } from "../config.js";
import { ChurnDriver } from "./churn-driver.js";
import { scrapeMetrics } from "./metrics-scraper.js";
import { SlowSubscriber } from "./slow-subscriber.js";
import { SteadySubscribers } from "./steady-subscribers.js";
import { TxSpammer } from "./tx-spammer.js";

async function main(): Promise<void> {
  const argv = process.argv.filter((a) => a !== "--");
  const label =
    argv.find((a) => a.startsWith("--label="))?.split("=")[1] ?? "run";
  const runId = `${label}-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const dir = join("runs", runId);
  await mkdir(dir, { recursive: true });
  const samplesPath = join(dir, "samples.ndjson");
  await writeFile(
    join(dir, "config.json"),
    JSON.stringify({ label, ...CONFIG }, null, 2),
  );

  console.log(
    `[run ${runId}] starting; duration=${CONFIG.run.durationMs / 60_000}min`,
  );

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
      console.log(
        `[${new Date().toISOString()}] rss=${fmtMb(sample.rss_bytes)} ` +
          `sinks=${sample.sink_registry_len ?? "?"} ` +
          `sink_cap=${sample.sink_registry_capacity ?? "?"} ` +
          `closed=${sample.sink_closed ?? "?"} ` +
          `best_at_import=${sample.mapping_sync_best_at_import_entries ?? "?"} ` +
          `journal_bytes=${fmtMb(sample.journal_entries_bytes)} ` +
          `bc_lag=${sample.broadcast_lag_max ?? "?"}`,
      );
    }
    await sleep(CONFIG.run.sampleIntervalMs);
  }

  churn.stop();
  slow.stop();
  spammer.stop();
  steady.stop();
  console.log(`[run ${runId}] done; wrote ${samples} samples to ${samplesPath}`);
  console.log(`[run ${runId}] generate report with: pnpm run report -- ${runId}`);
}

const fmtMb = (b: number | null) =>
  b == null ? "?" : `${(b / 1024 / 1024).toFixed(1)}MiB`;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
