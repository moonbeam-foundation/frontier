/** Minimal line-chart SVGs for run reports (no native deps). */

export type TimePoint = { t: number; y: number };

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function niceStep(range: number, targetTicks: number): number {
  if (range <= 0 || !Number.isFinite(range)) return 1;
  const raw = range / targetTicks;
  const p10 = 10 ** Math.floor(Math.log10(raw));
  const err = raw / p10;
  const nice = err < 1.5 ? 1 : err < 3.5 ? 2 : err < 7.5 ? 5 : 10;
  return nice * p10;
}

/**
 * Single-series time chart: `t` = unix ms, `y` = display value (already scaled).
 */
export function lineChartSvg(opts: {
  title: string;
  subtitle?: string;
  points: TimePoint[];
  yAxisLabel: string;
  formatY: (v: number) => string;
  stroke?: string;
  width?: number;
  height?: number;
}): string {
  const {
    title,
    subtitle,
    points,
    yAxisLabel,
    formatY,
    stroke = "#1d4ed8",
    width = 920,
    height = 320,
  } = opts;

  if (points.length === 0) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect fill="#f8fafc" width="100%" height="100%"/><text x="50%" y="50%" text-anchor="middle" fill="#64748b" font-family="system-ui,sans-serif" font-size="14">No data</text></svg>`;
  }

  const pad = { l: 72, r: 28, t: subtitle ? 52 : 40, b: 44 };
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;

  const t0 = points[0]!.t;
  const t1 = points.at(-1)!.t;
  const span = Math.max(t1 - t0, 1);
  const ys = points.map((p) => p.y);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const padY = (yMax - yMin) * 0.06;
  yMin -= padY;
  yMax += padY;
  const yRange = Math.max(yMax - yMin, 1e-9);

  const xOf = (t: number) => pad.l + ((t - t0) / span) * iw;
  const yOf = (y: number) => pad.t + ih - ((y - yMin) / yRange) * ih;

  const poly = points.map((p) => `${xOf(p.t).toFixed(2)},${yOf(p.y).toFixed(2)}`).join(" ");

  const yTicks = 5;
  const step = niceStep(yRange, yTicks);
  const y0 = Math.floor(yMin / step) * step;
  const lines: string[] = [];
  const labels: string[] = [];
  let tickCount = 0;
  for (let y = y0; y <= yMax + step * 0.001 && tickCount < 48; y += step) {
    tickCount++;
    const yy = yOf(y);
    lines.push(
      `<line x1="${pad.l}" y1="${yy.toFixed(2)}" x2="${(pad.l + iw).toFixed(2)}" y2="${yy.toFixed(2)}" stroke="#e2e8f0" stroke-width="1"/>`,
    );
    labels.push(
      `<text x="${pad.l - 8}" y="${(yy + 4).toFixed(2)}" text-anchor="end" fill="#475569" font-family="system-ui,sans-serif" font-size="11">${escapeXml(formatY(y))}</text>`,
    );
  }

  const xTicks = Math.min(8, Math.max(3, Math.floor(points.length / 25)));
  const xLabelEls: string[] = [];
  for (let i = 0; i <= xTicks; i++) {
    const t = t0 + (span * i) / xTicks;
    const x = xOf(t);
    xLabelEls.push(
      `<text x="${x.toFixed(2)}" y="${(height - 12).toFixed(2)}" text-anchor="middle" fill="#475569" font-family="system-ui,sans-serif" font-size="11">${escapeXml(`${((t - t0) / 1000).toFixed(0)}s`)}</text>`,
    );
  }

  const sub = subtitle
    ? `<text x="${pad.l}" y="36" fill="#64748b" font-family="system-ui,sans-serif" font-size="12">${escapeXml(subtitle)}</text>`
    : "";

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect fill="#ffffff" width="100%" height="100%"/>`,
    `<text x="${pad.l}" y="26" font-weight="600" fill="#0f172a" font-family="system-ui,sans-serif" font-size="15">${escapeXml(title)}</text>`,
    sub,
    `<text transform="rotate(-90 ${12} ${pad.t + ih / 2})" x="12" y="${(pad.t + ih / 2).toFixed(0)}" text-anchor="middle" fill="#64748b" font-family="system-ui,sans-serif" font-size="12">${escapeXml(yAxisLabel)}</text>`,
    ...lines,
    ...labels,
    ...xLabelEls,
    `<rect x="${pad.l}" y="${pad.t}" width="${iw}" height="${ih}" fill="none" stroke="#cbd5e1" stroke-width="1"/>`,
    `<polyline fill="none" stroke="${stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${poly}" />`,
    `</svg>`,
  ].join("");
}
