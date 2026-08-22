// JS port of src/factor_lab/rolling.py: refit OLS over a sliding window, so a
// full-sample beta reads as a summary rather than a constant. No HAC in the
// rolling estimator here either, matching statsmodels' RollingOLS (classical
// SE only) -- refitting each window is cheap enough client-side at this scale
// that no incremental/rolling-update algebra is worth the complexity.

import { transpose, multiply, multiplyVector, subtractVectors, inverse, designMatrix } from "./linalg.js";
import { ALPHA_LABEL, PERIODS_PER_YEAR } from "./regression.js";

const Z_95 = 1.959963984540054;

function windowFit(y, X) {
  const Xt = transpose(X);
  const XtXInverse = inverse(multiply(Xt, X));
  const Xty = Xt.map((row) => row.reduce((sum, v, i) => sum + v * y[i], 0));
  const beta = multiplyVector(XtXInverse, Xty);
  const fitted = X.map((row) => row.reduce((sum, v, i) => sum + v * beta[i], 0));
  const residuals = subtractVectors(y, fitted);
  const k = X[0].length;
  const n = y.length;
  const sigma2 = residuals.reduce((s, u) => s + u * u, 0) / (n - k);
  const standardErrors = XtXInverse.map((row, i) => Math.sqrt(row[i] * sigma2));
  return { beta, standardErrors };
}

// frame: { dates, excess, regressorNames, regressors }
export function rollingEstimates(frame, window) {
  const n = frame.excess.length;
  if (window < 12) throw new Error("window must be at least 12 months");
  if (window > n) throw new Error(`window of ${window} exceeds the ${n} available months`);

  const termNames = [ALPHA_LABEL, ...frame.regressorNames];
  const columns = frame.regressorNames.map((name) => frame.regressors[name]);
  const fullX = designMatrix(columns);

  const estimates = Object.fromEntries(termNames.map((term) => [term, { dates: [], estimate: [], lower: [], upper: [] }]));

  for (let end = window; end <= n; end++) {
    const start = end - window;
    const y = frame.excess.slice(start, end);
    const X = fullX.slice(start, end);
    const { beta, standardErrors } = windowFit(y, X);

    termNames.forEach((term, i) => {
      const scale = term === ALPHA_LABEL ? PERIODS_PER_YEAR : 1;
      const series = estimates[term];
      series.dates.push(frame.dates[end - 1]);
      series.estimate.push(beta[i] * scale);
      series.lower.push((beta[i] - Z_95 * standardErrors[i]) * scale);
      series.upper.push((beta[i] + Z_95 * standardErrors[i]) * scale);
    });
  }

  return estimates;
}

export function stabilitySummary(estimates) {
  return Object.entries(estimates).map(([term, series]) => {
    const values = series.estimate;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { term, min, max, range: max - min, changedSign: values.some((v) => v > 0) && values.some((v) => v < 0) };
  });
}
