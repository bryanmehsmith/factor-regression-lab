import { wireDisclosureControls } from "./utils/disclosure.js";
import { bindSliderOutput } from "./utils/sliderOutput.js";
import { fit, coefficientTable, compareStandardErrors, compareNested, SE_TYPES, ALPHA_LABEL } from "./modules/regression.js";
import { runAll, varianceInflationFactors, factorCorrelations, alphaVerdict, SIGNIFICANCE } from "./modules/diagnostics.js";
import { rollingEstimates, stabilitySummary } from "./modules/rolling.js";
import {
  tstatBarChart, residualsVsFittedChart, qqPlotChart, acfBarChart,
  residualHistogramChart, correlationHeatmap, rollingBandChart, cumulativeFitChart,
} from "./modules/charts.js";

const API_BASE = "/demos/factor-regression/api";
const FACTOR_MODELS = {
  CAPM: ["Mkt-RF"],
  FF3: ["Mkt-RF", "SMB", "HML"],
  FF5: ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
  "FF5+Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
};
const MODEL_ORDER = Object.keys(FACTOR_MODELS);

wireDisclosureControls(document.body);

const el = (id) => document.getElementById(id);
const industrySelect = el("industry-select");
const industryField = el("industry-field");
const tickerField = el("ticker-field");
const tickerSelect = el("ticker-select");
const customTicker = el("custom-ticker");
const liveRefresh = el("live-refresh");
const liveStatus = el("live-status");
const modelSelect = el("model-select");
const seSelect = el("se-select");
const hacSlider = el("hac-slider");
const hacOutput = el("hac-output");
const windowSlider = el("window-slider");
const windowOutput = el("window-output");
const startYearSlider = el("start-year-slider");
const endYearSlider = el("end-year-slider");
const periodOutput = el("period-output");
const emptyWarning = el("empty-warning");
const resultsEl = el("results");

let bundledFactors = null;
let bundledIndustries = null;
let catalog = null;
let liveFactors = null;
let liveIndustries = null;
let tickerCache = new Map();

function activeFactors() { return liveFactors ?? bundledFactors; }
function activeIndustries() { return liveIndustries ?? bundledIndustries; }

function isDarkMode() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function seriesAsMap(dates, values) {
  const map = new Map();
  dates.forEach((d, i) => { if (values[i] != null) map.set(d, values[i]); });
  return map;
}

// Aligns an asset's returns with the factor model's regressors + RF, builds
// { dates, excess, regressorNames, regressors }, mirrors data.build_regression_frame.
function buildFrame(assetDates, assetValues, factorsData, model) {
  const regressorNames = FACTOR_MODELS[model];
  const assetMap = seriesAsMap(assetDates, assetValues);
  const factorMaps = {};
  for (const name of [...regressorNames, "RF"]) {
    factorMaps[name] = seriesAsMap(factorsData.dates, factorsData.series[name]);
  }

  const dates = [];
  const excess = [];
  const regressors = Object.fromEntries(regressorNames.map((name) => [name, []]));

  for (const date of factorsData.dates) {
    if (!assetMap.has(date)) continue;
    const rf = factorMaps.RF.get(date);
    if (rf == null) continue;
    const factorValues = regressorNames.map((name) => factorMaps[name].get(date));
    if (factorValues.some((v) => v == null)) continue;

    dates.push(date);
    excess.push(assetMap.get(date) - rf);
    regressorNames.forEach((name, i) => regressors[name].push(factorValues[i]));
  }

  return { dates, excess, regressorNames, regressors };
}

function sliceByYearRange(frame, startYear, endYear) {
  const indices = [];
  frame.dates.forEach((date, i) => {
    const year = Number(date.slice(0, 4));
    if (year >= startYear && year <= endYear) indices.push(i);
  });
  return {
    dates: indices.map((i) => frame.dates[i]),
    excess: indices.map((i) => frame.excess[i]),
    regressorNames: frame.regressorNames,
    regressors: Object.fromEntries(frame.regressorNames.map((name) => [name, indices.map((i) => frame.regressors[name][i])])),
  };
}

function formatPercent(value, decimals = 2) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(decimals)}%`;
}
function formatNumber(value, decimals = 3) {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(decimals);
}

function populateIndustrySelect() {
  const names = Object.keys(activeIndustries().series);
  industrySelect.innerHTML = names.map((name) => `<option value="${name}">${name}</option>`).join("");
}

function populateTickerSelect() {
  tickerSelect.innerHTML = catalog.map((row) => `<option value="${row.ticker}">${row.ticker} (${row.name})</option>`).join("");
}

function currentSource() {
  return document.querySelector('input[name="source"]:checked').value;
}

async function fetchTicker(symbol) {
  if (tickerCache.has(symbol)) return tickerCache.get(symbol);
  const response = await fetch(`${API_BASE}/ticker?symbol=${encodeURIComponent(symbol)}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
  tickerCache.set(symbol, body);
  return body;
}

async function getAssetSeries() {
  if (currentSource() === "industry") {
    const name = industrySelect.value;
    const industries = activeIndustries();
    return { name, dates: industries.dates, values: industries.series[name] };
  }
  const symbol = (customTicker.value.trim() || tickerSelect.value).toUpperCase();
  liveStatus.textContent = `Fetching ${symbol} from Yahoo Finance...`;
  try {
    const body = await fetchTicker(symbol);
    liveStatus.textContent = "";
    return { name: symbol, dates: body.dates, values: body.values };
  } catch (err) {
    liveStatus.textContent = `Could not load ${symbol} from Yahoo Finance (${err.message}). Yahoo Finance rate-limits heavily; the industry portfolios need no live data.`;
    throw err;
  }
}

function renderCoefficientTable(regression) {
  const rows = coefficientTable(regression);
  const tbody = document.querySelector("#coefficient-table tbody");
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.term}</td>
      <td>${formatNumber(row.estimate, 4)}</td>
      <td>${formatNumber(row.stdError, 4)}</td>
      <td>${formatNumber(row.tStat, 2)}</td>
      <td>${formatNumber(row.pValue, 4)}</td>
      <td>[${formatNumber(row.ciLower, 4)}, ${formatNumber(row.ciUpper, 4)}]</td>
    </tr>`).join("");
}

function renderSeComparisonTable(comparison, terms) {
  const table = el("se-comparison-table");
  const head = `<thead><tr><th>term</th>${SE_TYPES.map((se) => `<th colspan="3">${se}</th>`).join("")}</tr></thead>`;
  const rows = terms.map((term) => {
    const cells = SE_TYPES.map((se) => {
      const row = coefficientTable(comparison[se]).find((r) => r.term === term);
      return `<td>${formatNumber(row.estimate, 4)}</td><td>${formatNumber(row.tStat, 2)}</td><td>${formatNumber(row.pValue, 4)}</td>`;
    }).join("");
    return `<tr><td>${term}</td>${cells}</tr>`;
  }).join("");
  table.innerHTML = `${head}<tbody>${rows}</tbody>`;
}

function renderDiagnosticsTable(tests) {
  const tbody = document.querySelector("#diagnostics-table tbody");
  tbody.innerHTML = tests.map((test) => `
    <tr class="${test.concerning ? "row-review" : ""}">
      <td>${test.name}</td>
      <td>${formatNumber(test.statistic)}</td>
      <td>${test.pValue == null ? "-" : formatNumber(test.pValue, 4)}</td>
      <td>${test.nullHypothesis}</td>
      <td>${test.concerning ? "review" : "ok"}</td>
    </tr>`).join("");

  el("diagnostics-details").innerHTML = tests.map((test) => `
    <details class="tech">
      <summary>${test.name} (${test.concerning ? "review" : "ok"})</summary>
      <p><strong>Null hypothesis:</strong> ${test.nullHypothesis}</p>
      <p>${test.reading}</p>
    </details>`).join("");
}

function renderVifTable(vif) {
  const tbody = document.querySelector("#vif-table tbody");
  tbody.innerHTML = vif.map((row) => `
    <tr class="${row.flag === "review" ? "row-review" : ""}">
      <td>${row.regressor}</td><td>${formatNumber(row.vif, 2)}</td>
    </tr>`).join("");
}

function renderStabilityTable(summary) {
  const tbody = document.querySelector("#stability-table tbody");
  tbody.innerHTML = summary.map((row) => `
    <tr><td>${row.term}</td><td>${formatNumber(row.min)}</td><td>${formatNumber(row.max)}</td><td>${formatNumber(row.range)}</td></tr>`).join("");
}

async function recompute() {
  if (!bundledFactors || !bundledIndustries || !catalog) return;

  const model = modelSelect.value;
  const seType = seSelect.value;
  const hacLags = Number(hacSlider.value);
  const rollingWindowMonths = Number(windowSlider.value);

  let asset;
  try {
    asset = await getAssetSeries();
  } catch {
    resultsEl.style.display = "none";
    emptyWarning.style.display = "block";
    emptyWarning.textContent = "Could not load this asset's return series.";
    return;
  }

  const rawFrame = buildFrame(asset.dates, asset.values, activeFactors(), model);
  if (rawFrame.dates.length === 0) {
    resultsEl.style.display = "none";
    emptyWarning.style.display = "block";
    emptyWarning.textContent = "No overlapping months between this asset and the factor data.";
    return;
  }

  const years = [...new Set(rawFrame.dates.map((d) => Number(d.slice(0, 4))))].sort((a, b) => a - b);
  const minYear = years[0], maxYear = years[years.length - 1];
  if (Number(startYearSlider.max) !== maxYear - minYear) {
    startYearSlider.min = "0"; startYearSlider.max = String(maxYear - minYear);
    endYearSlider.min = "0"; endYearSlider.max = String(maxYear - minYear);
    startYearSlider.value = "0"; endYearSlider.value = String(maxYear - minYear);
  }
  const startYear = minYear + Number(startYearSlider.value);
  const endYear = minYear + Number(endYearSlider.value);
  periodOutput.textContent = `${startYear} to ${endYear}`;

  const frame = sliceByYearRange(rawFrame, Math.min(startYear, endYear), Math.max(startYear, endYear));
  const minimumMonths = FACTOR_MODELS[model].length + 24;
  if (frame.dates.length < minimumMonths) {
    resultsEl.style.display = "none";
    emptyWarning.style.display = "block";
    emptyWarning.textContent = `Only ${frame.dates.length} months in this sample; at least ${minimumMonths} are needed to estimate ${model} with any confidence. Widen the sample period.`;
    return;
  }

  emptyWarning.style.display = "none";
  resultsEl.style.display = "block";

  const regression = fit(frame, { model, seType, hacLags });
  const tests = runAll(fit(frame, { model, seType: "Classical (OLS)" }));

  el("asset-title").textContent = `${asset.name} on ${model}`;
  el("asset-caption").textContent = `${frame.dates[0]} to ${frame.dates[frame.dates.length - 1]}, ${regression.n} monthly observations. The dependent variable is the return in excess of the risk-free rate.`;

  const verdict = alphaVerdict(regression, tests);
  const verdictEl = el("verdict-callout");
  verdictEl.textContent = verdict;
  verdictEl.className = regression.alphaPvalue < SIGNIFICANCE ? "callout callout-safe" : "callout callout-info";

  el("metric-alpha").textContent = formatPercent(regression.alphaAnnualized);
  el("metric-tstat-label").textContent = formatNumber(regression.alphaTstat, 2);
  el("metric-tstat-sub").textContent = `Alpha t-stat (${seType.split(" ")[0]})`;
  el("metric-adjr2").textContent = formatNumber(regression.adjRSquared);
  el("metric-months").textContent = String(regression.n);

  renderCoefficientTable(regression);

  const comparison = compareStandardErrors(frame, hacLags);
  const terms = [ALPHA_LABEL, ...frame.regressorNames];
  const isDark = isDarkMode();
  tstatBarChart(el("tstat-chart"), {
    terms,
    seTypes: SE_TYPES,
    valuesByType: Object.fromEntries(SE_TYPES.map((se) => [se, coefficientTable(comparison[se]).map((r) => Math.abs(r.tStat))])),
    isDark,
  });
  renderSeComparisonTable(comparison, terms);

  const position = MODEL_ORDER.indexOf(model);
  const nestedCaption = el("nested-caption");
  const nestedBody = el("nested-body");
  if (position === 0) {
    nestedCaption.textContent = `${model} is the simplest model available, so there is nothing simpler to test it against.`;
    nestedBody.style.display = "none";
  } else {
    const smaller = MODEL_ORDER[position - 1];
    const nested = compareNested(frame, FACTOR_MODELS[smaller], { smallModel: smaller, largeModel: model, seType, hacLags });
    nestedCaption.textContent = `Adding ${nested.added.join(" and ")} to ${smaller} raises R-squared mechanically, because extra regressors always do. The joint test asks whether the improvement is larger than chance would produce.`;
    nestedBody.style.display = "block";
    el("metric-nested-p").textContent = formatNumber(nested.pValue, 4);
    el("metric-nested-r2").textContent = formatNumber(nested.largeAdjRSquared);
    el("metric-nested-alpha").textContent = formatPercent(nested.largeAlphaAnnualized);
    el("nested-verdict").textContent = nested.pValue < SIGNIFICANCE
      ? `${nested.added.join(" and ")} carry information ${smaller} misses, so the alpha from ${smaller} was partly just unmodelled factor exposure.`
      : `${nested.added.join(" and ")} add nothing here. The extra fit is within what chance would produce, so ${smaller} is the better description of this asset.`;
  }

  renderDiagnosticsTable(tests);
  residualsVsFittedChart(el("residual-scatter-chart"), { fitted: regression.fitted, residuals: regression.residuals, isDark });
  qqPlotChart(el("qq-chart"), { residuals: regression.residuals, isDark });
  const maxLags = Math.min(24, Math.max(1, Math.floor(regression.residuals.length / 4)));
  const autocorrelations = Array.from({ length: maxLags }, (_, k) => autocorrelationAt(regression.residuals, k + 1));
  acfBarChart(el("acf-chart"), { autocorrelations, significanceBand: 1.96 / Math.sqrt(regression.residuals.length), isDark });
  residualHistogramChart(el("histogram-chart"), { residuals: regression.residuals, isDark });

  const vif = varianceInflationFactors(frame);
  renderVifTable(vif);
  const correlations = factorCorrelations(frame);
  correlationHeatmap(el("correlation-chart"), { names: correlations.names, matrix: correlations.matrix, isDark });

  if (rollingWindowMonths > frame.dates.length) {
    el("rolling-chart").innerHTML = `<p class="callout callout-unsafe">The rolling window (${rollingWindowMonths} months) is longer than the sample (${frame.dates.length} months).</p>`;
    document.querySelector("#stability-table tbody").innerHTML = "";
  } else {
    const estimates = rollingEstimates(frame, rollingWindowMonths);
    const summary = stabilitySummary(estimates);
    renderStabilityTable(summary);

    const rollingTermSelect = el("rolling-term-select");
    const previousTerm = rollingTermSelect.value;
    rollingTermSelect.innerHTML = terms.map((term) => `<option value="${term}">${term}</option>`).join("");
    if (terms.includes(previousTerm)) rollingTermSelect.value = previousTerm;

    const selectedTerm = rollingTermSelect.value;
    const series = estimates[selectedTerm];
    const fullSample = selectedTerm === ALPHA_LABEL ? regression.alphaAnnualized : regression.beta[regression.termNames.indexOf(selectedTerm)];
    rollingBandChart(el("rolling-chart"), { estimate: series.estimate, lower: series.lower, upper: series.upper, fullSample, label: selectedTerm, isDark });
  }

  let growth = 1;
  const actualGrowth = frame.excess.map((r) => (growth *= 1 + r));
  growth = 1;
  const explainedGrowth = regression.fitted.map((r) => (growth *= 1 + r));
  cumulativeFitChart(el("cumulative-chart"), { actualGrowth, explainedGrowth, isDark, modelLabel: model });
}

function autocorrelationAt(residuals, lag) {
  const n = residuals.length;
  const mean = residuals.reduce((s, v) => s + v, 0) / n;
  const centered = residuals.map((v) => v - mean);
  let numerator = 0;
  for (let t = 0; t < n - lag; t++) numerator += centered[t] * centered[t + lag];
  const denominator = centered.reduce((s, v) => s + v * v, 0);
  return numerator / denominator;
}

function safeRecompute() {
  recompute().catch((err) => console.error(err));
}

document.querySelectorAll('input[name="source"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    industryField.style.display = currentSource() === "industry" ? "flex" : "none";
    tickerField.style.display = currentSource() === "ticker" ? "flex" : "none";
    safeRecompute();
  });
});
industrySelect.addEventListener("change", safeRecompute);
tickerSelect.addEventListener("change", () => { customTicker.value = ""; safeRecompute(); });
customTicker.addEventListener("change", safeRecompute);
modelSelect.addEventListener("change", safeRecompute);
seSelect.addEventListener("change", safeRecompute);
startYearSlider.addEventListener("input", safeRecompute);
endYearSlider.addEventListener("input", safeRecompute);
el("rolling-term-select").addEventListener("change", safeRecompute);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", safeRecompute);

bindSliderOutput(hacSlider, hacOutput, { onChange: safeRecompute });
bindSliderOutput(windowSlider, windowOutput, { onChange: safeRecompute });

liveRefresh.addEventListener("change", async () => {
  if (!liveRefresh.checked) {
    liveFactors = null; liveIndustries = null;
    liveStatus.textContent = "";
    safeRecompute();
    return;
  }
  liveRefresh.disabled = true;
  liveStatus.textContent = "Refreshing factors and industry portfolios from the Ken French Data Library...";
  try {
    const [factorsRes, industriesRes] = await Promise.all([
      fetch(`${API_BASE}/factors?refresh=true`),
      fetch(`${API_BASE}/industries?refresh=true`),
    ]);
    const [factorsBody, industriesBody] = await Promise.all([factorsRes.json(), industriesRes.json()]);
    if (!factorsRes.ok) throw new Error(factorsBody.error ?? "factors refresh failed");
    if (!industriesRes.ok) throw new Error(industriesBody.error ?? "industries refresh failed");
    liveFactors = factorsBody;
    liveIndustries = industriesBody;
    liveStatus.textContent = "Live data loaded.";
    safeRecompute();
  } catch (err) {
    liveStatus.textContent = `Live refresh failed (${err.message}); showing the bundled snapshot instead.`;
    liveRefresh.checked = false;
    liveFactors = null; liveIndustries = null;
  } finally {
    liveRefresh.disabled = false;
  }
});

async function init() {
  const [factorsRes, industriesRes, catalogRes] = await Promise.all([
    fetch("js/data/factors.json"),
    fetch("js/data/industries.json"),
    fetch("js/data/catalog.json"),
  ]);
  bundledFactors = await factorsRes.json();
  bundledIndustries = await industriesRes.json();
  catalog = await catalogRes.json();

  populateIndustrySelect();
  populateTickerSelect();
  safeRecompute();
}

init().catch((err) => {
  console.error(err);
  liveStatus.textContent = "Could not load the bundled factor/industry data.";
});
