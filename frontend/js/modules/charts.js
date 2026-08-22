// Hand-rolled SVG charts, no library, matching the house style established in
// security-anti-patterns and momentum-factor. Every chart is a plain function
// that sets container.innerHTML.

import { normalQuantile } from "./distributions.js";

const LINE_COLOR = { light: "#7a4f22", dark: "#d0a46a" };
const SECOND_COLOR = { light: "#2a78d6", dark: "#3987e5" };
const BAND_COLOR = { light: "rgba(122,79,34,0.18)", dark: "rgba(208,164,106,0.22)" };
const BASELINE_COLOR = { light: "#a89f92", dark: "#6b6459" };
const THRESHOLD_COLOR = { light: "#b3261e", dark: "#ff8a80" };
const SERIES_COLORS = {
  light: ["#7a4f22", "#2a78d6", "#1baf7a"],
  dark: ["#d0a46a", "#3987e5", "#199e70"],
};

function ensureChartStyles() {
  if (document.getElementById("factor-chart-styles")) return;
  const style = document.createElement("style");
  style.id = "factor-chart-styles";
  style.textContent = `
    .viz-root { --viz-surface: var(--card-background); color-scheme: light dark; }
    .viz-grid { stroke: var(--border-color); stroke-width: 1; }
    .viz-baseline { stroke: var(--muted-text); stroke-width: 1; stroke-dasharray: 4 3; opacity: 0.6; }
    .viz-threshold { stroke-width: 1; stroke-dasharray: 5 3; }
    .viz-tick { font-size: 10px; fill: var(--muted-text); font-family: inherit; }
    .viz-legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; font-size: 0.82rem; color: var(--muted-text); }
    .viz-legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .viz-legend-swatch { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
    .viz-heat-cell { stroke: var(--card-background); stroke-width: 1; }
    .viz-heat-label { font-size: 10px; font-family: inherit; }
  `;
  document.head.appendChild(style);
}
ensureChartStyles();

function niceLinearTicks(minValue, maxValue, targetCount = 5) {
  const range = maxValue - minValue || 1;
  const roughStep = range / targetCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const normalized = roughStep / magnitude;
  const niceNormalized = normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10;
  const step = niceNormalized * magnitude;
  const start = Math.floor(minValue / step) * step;
  const end = Math.ceil(maxValue / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Math.round(v * 10000) / 10000);
  return ticks;
}

function color(pair, isDark) {
  return isDark ? pair.dark : pair.light;
}

function selectIndexTicks(n, maxTicks = 8) {
  if (n <= maxTicks) return Array.from({ length: n }, (_, i) => i);
  const step = (n - 1) / (maxTicks - 1);
  const indices = new Set();
  for (let i = 0; i < maxTicks; i++) indices.add(Math.round(i * step));
  return [...indices].sort((a, b) => a - b);
}

// --- 1. Grouped bar chart: |t-stat| per term x per SE type ---
export function tstatBarChart(container, { terms, seTypes, valuesByType, isDark, threshold = 1.96 }) {
  const width = 720;
  const height = 340;
  const padding = { top: 16, right: 20, bottom: 46, left: 44 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const allValues = seTypes.flatMap((se) => valuesByType[se]);
  const ticks = niceLinearTicks(0, Math.max(threshold, ...allValues));
  const maxTick = ticks[ticks.length - 1];
  const yFor = (v) => padding.top + plotHeight - (v / maxTick) * plotHeight;

  const groupWidth = plotWidth / terms.length;
  const barWidth = (groupWidth * 0.72) / seTypes.length;
  const colors = SERIES_COLORS[isDark ? "dark" : "light"];

  const gridlines = ticks.map((tick) => {
    const y = yFor(tick);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />
      <text x="${padding.left - 6}" y="${y + 3}" text-anchor="end" class="viz-tick">${tick.toFixed(1)}</text>`;
  }).join("");

  const bars = terms.map((term, termIndex) => {
    const groupStart = padding.left + termIndex * groupWidth + groupWidth * 0.14;
    const termBars = seTypes.map((se, seIndex) => {
      const value = valuesByType[se][termIndex];
      const x = groupStart + seIndex * barWidth;
      const y = yFor(value);
      const barColor = colors[seIndex % colors.length];
      return `<rect x="${x}" y="${y}" width="${barWidth * 0.9}" height="${padding.top + plotHeight - y}" fill="${barColor}" />`;
    }).join("");
    const label = `<text x="${groupStart + (barWidth * seTypes.length) / 2}" y="${height - padding.bottom + 16}" text-anchor="middle" class="viz-tick">${term}</text>`;
    return termBars + label;
  }).join("");

  const thresholdY = yFor(threshold);
  const thresholdLine = `<line x1="${padding.left}" y1="${thresholdY}" x2="${width - padding.right}" y2="${thresholdY}" class="viz-threshold" style="stroke:${color(THRESHOLD_COLOR, isDark)}" />`;

  const legend = seTypes.map((se, i) => `<span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${colors[i % colors.length]}"></span>${se}</span>`).join("")
    + `<span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${color(THRESHOLD_COLOR, isDark)}"></span>|t| = ${threshold} (5% significance)</span>`;

  container.innerHTML = `<div class="viz-root">
    <div class="viz-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="t-statistic magnitude by term and standard error assumption">
      ${gridlines}${bars}${thresholdLine}
    </svg>
  </div>`;
}

// --- 2. Residuals vs fitted scatter ---
export function residualsVsFittedChart(container, { fitted, residuals, isDark }) {
  const width = 640, height = 320;
  const padding = { top: 16, right: 20, bottom: 34, left: 52 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const xTicks = niceLinearTicks(Math.min(...fitted), Math.max(...fitted), 5);
  const yTicks = niceLinearTicks(Math.min(...residuals), Math.max(...residuals), 5);
  const xFor = (v) => padding.left + ((v - xTicks[0]) / (xTicks[xTicks.length - 1] - xTicks[0] || 1)) * plotWidth;
  const yFor = (v) => padding.top + plotHeight - ((v - yTicks[0]) / (yTicks[yTicks.length - 1] - yTicks[0] || 1)) * plotHeight;

  const gridlines = yTicks.map((tick) => {
    const y = yFor(tick);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />
      <text x="${padding.left - 6}" y="${y + 3}" text-anchor="end" class="viz-tick">${tick.toFixed(3)}</text>`;
  }).join("");
  const xLabels = xTicks.map((tick) => `<text x="${xFor(tick)}" y="${height - 4}" text-anchor="middle" class="viz-tick">${tick.toFixed(3)}</text>`).join("");

  const zeroY = yFor(0);
  const baseline = `<line x1="${padding.left}" y1="${zeroY}" x2="${width - padding.right}" y2="${zeroY}" class="viz-baseline" />`;
  const dotColor = color(LINE_COLOR, isDark);
  const dots = fitted.map((f, i) => `<circle cx="${xFor(f)}" cy="${yFor(residuals[i])}" r="2.6" fill="${dotColor}" opacity="0.55" />`).join("");

  container.innerHTML = `<div class="viz-root"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Residuals versus fitted values">
    ${gridlines}${baseline}${dots}${xLabels}
  </svg></div>`;
}

// --- 3. Normal Q-Q plot ---
export function qqPlotChart(container, { residuals, isDark }) {
  const width = 640, height = 320;
  const padding = { top: 16, right: 20, bottom: 34, left: 52 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const n = residuals.length;
  const mean = residuals.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(residuals.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const sorted = [...residuals].sort((a, b) => a - b);
  const points = sorted.map((value, i) => {
    const p = (i + 0.5) / n;
    return { theoretical: normalQuantile(p), sample: (value - mean) / std };
  });

  const allValues = points.flatMap((p) => [p.theoretical, p.sample]);
  const ticks = niceLinearTicks(Math.min(...allValues), Math.max(...allValues), 5);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const xFor = (v) => padding.left + ((v - lo) / (hi - lo || 1)) * plotWidth;
  const yFor = (v) => padding.top + plotHeight - ((v - lo) / (hi - lo || 1)) * plotHeight;

  const gridlines = ticks.map((tick) => {
    const y = yFor(tick);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />
      <text x="${padding.left - 6}" y="${y + 3}" text-anchor="end" class="viz-tick">${tick.toFixed(1)}</text>`;
  }).join("");
  const xLabels = ticks.map((tick) => `<text x="${xFor(tick)}" y="${height - 4}" text-anchor="middle" class="viz-tick">${tick.toFixed(1)}</text>`).join("");

  const diagonal = `<line x1="${xFor(lo)}" y1="${yFor(lo)}" x2="${xFor(hi)}" y2="${yFor(hi)}" class="viz-baseline" />`;
  const dotColor = color(LINE_COLOR, isDark);
  const dots = points.map((p) => `<circle cx="${xFor(p.theoretical)}" cy="${yFor(p.sample)}" r="2.6" fill="${dotColor}" opacity="0.55" />`).join("");

  container.innerHTML = `<div class="viz-root"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Normal Q-Q plot of residuals">
    ${gridlines}${diagonal}${dots}${xLabels}
  </svg></div>`;
}

// --- 4. Residual ACF bar chart ---
export function acfBarChart(container, { autocorrelations, significanceBand, isDark }) {
  const width = 640, height = 260;
  const padding = { top: 16, right: 20, bottom: 34, left: 44 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const maxAbs = Math.max(significanceBand, ...autocorrelations.map((v) => Math.abs(v)));
  const ticks = niceLinearTicks(-maxAbs, maxAbs, 4);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const yFor = (v) => padding.top + plotHeight - ((v - lo) / (hi - lo || 1)) * plotHeight;
  const zeroY = yFor(0);

  const barWidth = (plotWidth / autocorrelations.length) * 0.6;
  const barColor = color(LINE_COLOR, isDark);
  const bars = autocorrelations.map((value, i) => {
    const x = padding.left + (i + 0.2) * (plotWidth / autocorrelations.length);
    const y = yFor(Math.max(value, 0));
    const barHeight = Math.abs(yFor(value) - zeroY);
    const barY = value >= 0 ? yFor(value) : zeroY;
    return `<rect x="${x}" y="${barY}" width="${barWidth}" height="${barHeight}" fill="${barColor}" />`;
  }).join("");

  const bandColor = color(THRESHOLD_COLOR, isDark);
  const upperBand = `<line x1="${padding.left}" y1="${yFor(significanceBand)}" x2="${width - padding.right}" y2="${yFor(significanceBand)}" class="viz-threshold" style="stroke:${bandColor}" />`;
  const lowerBand = `<line x1="${padding.left}" y1="${yFor(-significanceBand)}" x2="${width - padding.right}" y2="${yFor(-significanceBand)}" class="viz-threshold" style="stroke:${bandColor}" />`;

  const gridlines = ticks.map((tick) => {
    const y = yFor(tick);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />
      <text x="${padding.left - 6}" y="${y + 3}" text-anchor="end" class="viz-tick">${tick.toFixed(2)}</text>`;
  }).join("");

  const xTickIndices = selectIndexTicks(autocorrelations.length, 12);
  const xLabels = xTickIndices.map((i) => `<text x="${padding.left + (i + 0.5) * (plotWidth / autocorrelations.length)}" y="${height - 4}" text-anchor="middle" class="viz-tick">${i + 1}</text>`).join("");

  container.innerHTML = `<div class="viz-root"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Residual autocorrelation by lag">
    ${gridlines}${bars}${upperBand}${lowerBand}${xLabels}
  </svg></div>`;
}

// --- 5. Residual histogram + normal fit ---
export function residualHistogramChart(container, { residuals, isDark }) {
  const width = 640, height = 300;
  const padding = { top: 16, right: 20, bottom: 34, left: 44 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const n = residuals.length;
  const mean = residuals.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(residuals.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const min = Math.min(...residuals), max = Math.max(...residuals);
  const binCount = 30;
  const binWidth = (max - min) / binCount || 1;
  const bins = new Array(binCount).fill(0);
  residuals.forEach((v) => {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((v - min) / binWidth)));
    bins[idx] += 1;
  });
  const density = bins.map((count) => count / (n * binWidth));

  const xTicks = niceLinearTicks(min, max, 5);
  const xFor = (v) => padding.left + ((v - xTicks[0]) / (xTicks[xTicks.length - 1] - xTicks[0] || 1)) * plotWidth;
  const maxDensity = Math.max(...density, normalPdfValue(mean, mean, std));
  const yTicks = niceLinearTicks(0, maxDensity, 4);
  const yFor = (v) => padding.top + plotHeight - (v / yTicks[yTicks.length - 1]) * plotHeight;

  const barColor = color(LINE_COLOR, isDark);
  const bars = density.map((d, i) => {
    const x0 = xFor(min + i * binWidth);
    const x1 = xFor(min + (i + 1) * binWidth);
    return `<rect x="${x0}" y="${yFor(d)}" width="${Math.max(0, x1 - x0 - 1)}" height="${padding.top + plotHeight - yFor(d)}" fill="${barColor}" opacity="0.55" />`;
  }).join("");

  const curvePoints = [];
  const steps = 100;
  for (let i = 0; i <= steps; i++) {
    const x = min + (i / steps) * (max - min);
    curvePoints.push(`${i === 0 ? "M" : "L"}${xFor(x)},${yFor(normalPdfValue(x, mean, std))}`);
  }
  const curveColor = color(THRESHOLD_COLOR, isDark);
  const curve = `<path d="${curvePoints.join(" ")}" fill="none" stroke="${curveColor}" stroke-width="2" />`;

  const gridlines = yTicks.map((tick) => {
    const y = yFor(tick);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />`;
  }).join("");
  const xLabels = xTicks.map((tick) => `<text x="${xFor(tick)}" y="${height - 4}" text-anchor="middle" class="viz-tick">${tick.toFixed(3)}</text>`).join("");

  container.innerHTML = `<div class="viz-root">
    <div class="viz-legend"><span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${curveColor}"></span>Normal fit</span></div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Residual distribution versus a fitted normal curve">
      ${gridlines}${bars}${curve}${xLabels}
    </svg></div>`;
}

function normalPdfValue(x, mean, std) {
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}

// --- 6. Correlation heatmap ---
export function correlationHeatmap(container, { names, matrix, isDark }) {
  const cell = 56;
  const labelSpace = 90;
  const width = labelSpace + cell * names.length + 10;
  const height = labelSpace + cell * names.length + 10;

  function heatColor(value) {
    const t = (value + 1) / 2;
    const negative = isDark ? [255, 138, 128] : [179, 38, 30];
    const positive = isDark ? [63, 135, 229] : [42, 120, 214];
    const mix = value >= 0 ? positive : negative;
    const strength = Math.abs(value);
    const base = isDark ? [28, 26, 23] : [255, 253, 249];
    const rgb = mix.map((channel, i) => Math.round(base[i] + (channel - base[i]) * strength));
    return `rgb(${rgb.join(",")})`;
  }

  const cells = [];
  for (let row = 0; row < names.length; row++) {
    for (let col = 0; col < names.length; col++) {
      const value = matrix[row][col];
      const x = labelSpace + col * cell;
      const y = labelSpace + row * cell;
      const textColor = Math.abs(value) > 0.5 ? "#fff" : (isDark ? "#f4efe7" : "#2b2722");
      cells.push(`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${heatColor(value)}" class="viz-heat-cell" />`);
      cells.push(`<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle" class="viz-heat-label" fill="${textColor}">${value.toFixed(2)}</text>`);
    }
  }

  const colLabels = names.map((name, i) => `<text x="${labelSpace + i * cell + cell / 2}" y="${labelSpace - 8}" text-anchor="middle" class="viz-tick">${name}</text>`).join("");
  const rowLabels = names.map((name, i) => `<text x="${labelSpace - 8}" y="${labelSpace + i * cell + cell / 2 + 4}" text-anchor="end" class="viz-tick">${name}</text>`).join("");

  container.innerHTML = `<div class="viz-root"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Factor correlation heatmap">
    ${cells.join("")}${colLabels}${rowLabels}
  </svg></div>`;
}

// --- 7. Rolling coefficient line + band ---
export function rollingBandChart(container, { estimate, lower, upper, fullSample, label, isDark }) {
  const width = 720, height = 320;
  const padding = { top: 16, right: 20, bottom: 34, left: 52 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const allValues = [...estimate, ...lower, ...upper, ...(fullSample != null ? [fullSample] : [])];
  const ticks = niceLinearTicks(Math.min(...allValues), Math.max(...allValues), 5);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const n = estimate.length;
  const xFor = (i) => padding.left + (n <= 1 ? 0 : (i / (n - 1)) * plotWidth);
  const yFor = (v) => padding.top + plotHeight - ((v - lo) / (hi - lo || 1)) * plotHeight;

  const gridlines = ticks.map((tick) => {
    const y = yFor(tick);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />
      <text x="${padding.left - 6}" y="${y + 3}" text-anchor="end" class="viz-tick">${tick.toFixed(2)}</text>`;
  }).join("");

  const bandPath = [
    ...upper.map((v, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(v)}`),
    ...lower.map((v, i) => `L${xFor(n - 1 - i)},${yFor(lower[n - 1 - i])}`).slice(0, n),
    "Z",
  ].join(" ");

  const lineColor = color(LINE_COLOR, isDark);
  const bandColor = color(BAND_COLOR, isDark);
  const line = estimate.map((v, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(v)}`).join(" ");

  const zeroY = yFor(0);
  const zeroLine = lo < 0 && hi > 0 ? `<line x1="${padding.left}" y1="${zeroY}" x2="${width - padding.right}" y2="${zeroY}" class="viz-baseline" />` : "";
  const fullSampleLine = fullSample != null
    ? `<line x1="${padding.left}" y1="${yFor(fullSample)}" x2="${width - padding.right}" y2="${yFor(fullSample)}" class="viz-threshold" style="stroke:${color(THRESHOLD_COLOR, isDark)}" />`
    : "";

  const legend = `<span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${lineColor}"></span>Rolling ${label}</span>`
    + (fullSample != null ? `<span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${color(THRESHOLD_COLOR, isDark)}"></span>Full-sample estimate</span>` : "");

  container.innerHTML = `<div class="viz-root">
    <div class="viz-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Rolling ${label} with 95% band">
      ${gridlines}${zeroLine}
      <path d="${bandPath}" fill="${bandColor}" stroke="none" />
      <path d="${line}" fill="none" stroke="${lineColor}" stroke-width="2" />
      ${fullSampleLine}
    </svg></div>`;
}

// --- 8. Cumulative fit, dual line, log scale ---
export function cumulativeFitChart(container, { actualGrowth, explainedGrowth, isDark, modelLabel }) {
  const width = 720, height = 340;
  const padding = { top: 16, right: 20, bottom: 34, left: 52 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const allValues = [...actualGrowth, ...explainedGrowth].filter((v) => v > 0);
  const logMin = Math.log10(Math.min(...allValues));
  const logMax = Math.log10(Math.max(...allValues));
  const n = actualGrowth.length;
  const xFor = (i) => padding.left + (n <= 1 ? 0 : (i / (n - 1)) * plotWidth);
  const yFor = (v) => padding.top + plotHeight - ((Math.log10(v) - logMin) / (logMax - logMin || 1)) * plotHeight;

  const tickExponents = [];
  for (let e = Math.floor(logMin); e <= Math.ceil(logMax); e++) tickExponents.push(e);
  const gridlines = tickExponents.map((e) => {
    const value = Math.pow(10, e);
    const y = yFor(value);
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="viz-grid" />
      <text x="${padding.left - 6}" y="${y + 3}" text-anchor="end" class="viz-tick">${value < 1 ? value.toFixed(2) : value.toFixed(0)}x</text>`;
  }).join("");

  const actualColor = color(LINE_COLOR, isDark);
  const explainedColor = color(SECOND_COLOR, isDark);
  const actualPath = actualGrowth.map((v, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(v)}`).join(" ");
  const explainedPath = explainedGrowth.map((v, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(v)}`).join(" ");

  const legend = `<span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${actualColor}"></span>Actual excess return</span>
    <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:${explainedColor}"></span>Explained by ${modelLabel}</span>`;

  container.innerHTML = `<div class="viz-root">
    <div class="viz-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Actual versus factor-explained cumulative excess return, log scale">
      ${gridlines}
      <path d="${actualPath}" fill="none" stroke="${actualColor}" stroke-width="2" />
      <path d="${explainedPath}" fill="none" stroke="${explainedColor}" stroke-width="2" stroke-dasharray="6 4" />
    </svg></div>`;
}
