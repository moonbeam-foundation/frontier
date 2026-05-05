import { readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { lineChartSvg, type TimePoint } from "./chart-svg.js";

type Sample = Record<string, unknown>;

function seriesFromSamples(
  samples: Sample[],
  key: string,
  transform: (v: number) => number,
): TimePoint[] {
  const out: TimePoint[] = [];
  for (const s of samples) {
    const v = s[key];
    if (typeof v === "number" && Number.isFinite(v)) {
      out.push({ t: s.ts as number, y: transform(v) });
    }
  }
  return out;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2).filter((a) => a !== "--");
  const runId = args[0];
  if (!runId) {
    const entries = await readdir("runs", { withFileTypes: true });
    const runs = entries
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();
    console.error("usage: pnpm run report -- <run-id>");
    console.error("available runs:");
    for (const r of runs) console.error("  " + r);
    process.exit(1);
  }

  const dir = join("runs", runId);
  const raw = await readFile(join(dir, "samples.ndjson"), "utf8");
  const samples = raw
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Sample);
  if (samples.length < 2) {
    console.error("not enough samples");
    process.exit(1);
  }

  const t0 = samples[0]!.ts as number;
  const dur = ((samples.at(-1)!.ts as number) - t0) / 1000;

  const summary = {
    runId,
    durationSeconds: dur,
    sampleCount: samples.length,
    rss: summarize(samples, "rss_bytes"),
    vsz: summarize(samples, "vsz_bytes"),
    sinkRegistry: summarize(samples, "sink_registry_len"),
    sinkRegistryCapacity: summarize(samples, "sink_registry_capacity"),
    sinkClosed: summarize(samples, "sink_closed"),
    mappingSyncBestAtImport: summarize(
      samples,
      "mapping_sync_best_at_import_entries",
    ),
    journalBytes: summarize(samples, "journal_entries_bytes"),
    broadcastLag: summarize(samples, "broadcast_lag_max"),
  };

  const rssPts = seriesFromSamples(samples, "rss_bytes", (v) => v / 1024 / 1024);
  const sinkPts = seriesFromSamples(samples, "sink_registry_len", (v) => v);
  const sinkCapPts = seriesFromSamples(
    samples,
    "sink_registry_capacity",
    (v) => v,
  );
  const sinkClosedPts = seriesFromSamples(samples, "sink_closed", (v) => v);
  const bestAtImportPts = seriesFromSamples(
    samples,
    "mapping_sync_best_at_import_entries",
    (v) => v,
  );
  const journalPts = seriesFromSamples(samples, "journal_entries_bytes", (v) => v / 1024 / 1024);
  const lagPts = seriesFromSamples(samples, "broadcast_lag_max", (v) => v);
  const vszPts = seriesFromSamples(samples, "vsz_bytes", (v) => v / 1024 / 1024);

  const vszBytesVals = samples
    .map((s) => s.vsz_bytes)
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const maxVszBytes = vszBytesVals.length ? Math.max(...vszBytesVals) : 0;
  /** Skip VSZ chart when samples look bogus (e.g. old Darwin `ps` scaling bug). */
  const includeVszChart = maxVszBytes > 0 && maxVszBytes < 512 * 1024 ** 3;

  const chartFiles: { name: string; svg: string }[] = [
    {
      name: "chart-rss-mib.svg",
      svg: lineChartSvg({
        title: "Resident set size (RSS)",
        subtitle: "Mebibytes — from Prometheus or NODE_PID / ps",
        points: rssPts,
        yAxisLabel: "MiB",
        formatY: (v) => v.toFixed(1),
        stroke: "#1d4ed8",
      }),
    },
    {
      name: "chart-sinks.svg",
      svg: lineChartSvg({
        title: "Ethereum block pubsub sink pool",
        subtitle: "frontier_pubsub_sink_registry_len",
        points: sinkPts,
        yAxisLabel: "sinks",
        formatY: (v) => (Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1)),
        stroke: "#b45309",
      }),
    },
    {
      name: "chart-journal-mib.svg",
      svg: lineChartSvg({
        title: "Logs journal retained size",
        subtitle: "frontier_logs_journal_entries_total_bytes → MiB",
        points: journalPts,
        yAxisLabel: "MiB",
        formatY: (v) => v.toFixed(1),
        stroke: "#047857",
      }),
    },
  ];

  if (sinkCapPts.length >= 2) {
    chartFiles.push({
      name: "chart-sink-registry-capacity.svg",
      svg: lineChartSvg({
        title: "Pubsub sink registry HashMap capacity",
        subtitle: "frontier_pubsub_sink_registry_capacity (can exceed len after churn)",
        points: sinkCapPts,
        yAxisLabel: "capacity",
        formatY: (v) => (Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1)),
        stroke: "#7c3aed",
      }),
    });
  }

  if (bestAtImportPts.length >= 2) {
    chartFiles.push({
      name: "chart-mapping-best-at-import.svg",
      svg: lineChartSvg({
        title: "Mapping sync KV best_at_import map size",
        subtitle: "frontier_mapping_sync_best_at_import_entries",
        points: bestAtImportPts,
        yAxisLabel: "entries",
        formatY: (v) => (Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1)),
        stroke: "#0e7490",
      }),
    });
  }

  if (includeVszChart && vszPts.length >= 2) {
    chartFiles.push({
      name: "chart-vsz-gib.svg",
      svg: lineChartSvg({
        title: "Virtual memory size (VSZ)",
        subtitle: "Mebibytes — from Prometheus or NODE_PID / ps / proc",
        points: vszPts,
        yAxisLabel: "MiB",
        formatY: (v) =>
          v >= 1024 ? `${(v / 1024).toFixed(1)} GiB` : `${v.toFixed(0)} MiB`,
        stroke: "#6d28d9",
      }),
    });
  }

  if (lagPts.length >= 2 && lagPts.some((p) => p.y > 0)) {
    chartFiles.push({
      name: "chart-broadcast-lag.svg",
      svg: lineChartSvg({
        title: "Logs journal broadcast lag (max skipped)",
        subtitle: "frontier_logs_journal_broadcast_lag_max",
        points: lagPts,
        yAxisLabel: "skipped msgs",
        formatY: (v) => String(Math.round(v)),
        stroke: "#be123c",
      }),
    });
  }

  if (sinkClosedPts.length >= 2 && sinkClosedPts.some((p) => p.y > 0)) {
    chartFiles.push({
      name: "chart-sink-closed.svg",
      svg: lineChartSvg({
        title: "Closed sink senders (awaiting prune)",
        subtitle: "frontier_pubsub_sink_closed",
        points: sinkClosedPts,
        yAxisLabel: "closed",
        formatY: (v) => String(Math.round(v)),
        stroke: "#64748b",
      }),
    });
  }

  for (const { name, svg } of chartFiles) {
    await writeFile(join(dir, name), svg, "utf8");
  }

  const mdParts: string[] = [
    `# Run report: ${runId}`,
    "",
    `Duration: **${dur.toFixed(0)} s** · samples: **${samples.length}** (spacing ≈ ${((dur / Math.max(samples.length - 1, 1)) * 1000).toFixed(0)} ms)`,
    "",
    "## Summary",
    "",
    "```json",
    JSON.stringify(summary, null, 2),
    "```",
    "",
    "## Charts (SVG)",
    "",
    "Vector charts are saved next to this file and embedded below so they render in GitHub, VS Code, and most PDF/HTML exports.",
    "",
    "### RSS (MiB)",
    "",
    "![RSS — MiB](chart-rss-mib.svg)",
    "",
    "### Pubsub sink registry (count)",
    "",
    "![Sink registry](chart-sinks.svg)",
    "",
    "### Logs journal retained (MiB)",
    "",
    "![Journal — MiB](chart-journal-mib.svg)",
    "",
  ];

  if (sinkCapPts.length >= 2) {
    mdParts.push(
      "### Pubsub sink registry capacity",
      "",
      "![Sink registry capacity](chart-sink-registry-capacity.svg)",
      "",
    );
  }

  if (bestAtImportPts.length >= 2) {
    mdParts.push(
      "### Mapping sync best_at_import entries",
      "",
      "![Mapping sync best_at_import](chart-mapping-best-at-import.svg)",
      "",
    );
  }

  if (includeVszChart && vszPts.length >= 2) {
    mdParts.push("### Virtual memory (MiB)", "", "![VSZ — MiB](chart-vsz-gib.svg)", "");
  } else {
    mdParts.push(
      "### Virtual memory (VSZ)",
      "",
      "_Chart omitted:_ VSZ samples looked unphysical (often a `ps` unit mismatch on older runs). Re-run the harness after updating to the latest `metrics-scraper` Darwin VSZ fix, or rely on RSS above.",
      "",
    );
  }

  if (lagPts.some((p) => p.y > 0)) {
    mdParts.push("### Broadcast lag (max)", "", "![Broadcast lag](chart-broadcast-lag.svg)", "");
  }

  if (sinkClosedPts.some((p) => p.y > 0)) {
    mdParts.push("### Closed sinks", "", "![Sink closed](chart-sink-closed.svg)", "");
  }

  mdParts.push(
    "## Reading the summary",
    "",
    "- **`slopePerHour`** is a least-squares line through all points for that series. It is useful for **monotonic** trends; if a series **fills then plateaus** (for example the bounded journal), the slope can look extreme even when the curve is flat at the end.",
    "- **RSS** reflects process memory pressure; combine with **sink**, **sink capacity**, **journal**, and **mapping sync** charts when attributing growth.",
    "- **`sink_registry_capacity`** can stay high after subscription churn even when **`sink_registry_len`** is low (HashMap bucket retention).",
    "",
  );

  const md = mdParts.join("\n");

  await writeFile(join(dir, "report.md"), md, "utf8");
  console.log(md);
  console.log(`\nReport saved to ${join(dir, "report.md")}`);
  for (const { name } of chartFiles) {
    console.log(`Chart: ${join(dir, name)}`);
  }
}

function summarize(samples: Sample[], key: string) {
  const values = samples
    .map((s) => s[key])
    .filter((v): v is number => typeof v === "number" && v != null);
  if (values.length < 2) return { samples: values.length, note: "insufficient data" };
  const first = values[0]!;
  const last = values.at(-1)!;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const firstTs = (samples.find((s) => s[key] != null)!.ts as number);
  const points = samples
    .filter((s) => s[key] != null)
    .map((s) => ({
      x: ((s.ts as number) - firstTs) / 1000,
      y: s[key] as number,
    }));
  const slopePerSec = linearSlope(points);
  const slopePerHour = slopePerSec * 3600;
  return {
    first,
    last,
    min,
    max,
    delta: last - first,
    slopePerHour,
    samples: values.length,
  };
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

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
