// JS port of src/factor_lab/diagnostics.py: residual diagnostics and VIF.

import { transpose, multiply, multiplyVector, subtractVectors, inverse, designMatrix } from "./linalg.js";
import { chiSquareCdf } from "./distributions.js";

export const SIGNIFICANCE = 0.05;
export const VIF_CONCERN = 5.0;

function ols(y, X) {
  const Xt = transpose(X);
  const XtXInverse = inverse(multiply(Xt, X));
  const Xty = Xt.map((row) => row.reduce((sum, v, i) => sum + v * y[i], 0));
  const beta = multiplyVector(XtXInverse, Xty);
  const fitted = X.map((row) => row.reduce((sum, v, i) => sum + v * beta[i], 0));
  const residuals = subtractVectors(y, fitted);
  return { beta, fitted, residuals };
}

function rSquaredOf(y, X) {
  const { residuals } = ols(y, X);
  const n = y.length;
  const mean = y.reduce((s, v) => s + v, 0) / n;
  const totalSumSquares = y.reduce((s, v) => s + (v - mean) ** 2, 0);
  const residualSumSquares = residuals.reduce((s, v) => s + v * v, 0);
  return 1 - residualSumSquares / totalSumSquares;
}

function breuschPagan(regression) {
  const residualsSquared = regression.residuals.map((u) => u * u);
  const rSquared = rSquaredOf(residualsSquared, regression.X);
  const n = regressionN(regression);
  const df = regression.regressorNames.length;
  const statistic = n * rSquared;
  const pValue = 1 - chiSquareCdf(statistic, df);
  const concerning = pValue < SIGNIFICANCE;

  const reading = concerning
    ? "Residual variance changes with the factor values (heteroskedasticity). Classical standard errors are biased here, so prefer White or Newey-West. Volatility clustering makes this the normal finding for asset returns."
    : "No evidence of heteroskedasticity; classical standard errors are not obviously wrong on this count.";

  return {
    name: "Breusch-Pagan (heteroskedasticity)",
    statistic, pValue, concerning,
    nullHypothesis: "Residual variance is constant (homoskedastic)",
    reading,
  };
}

function durbinWatson(regression) {
  const u = regression.residuals;
  let diffSumSquares = 0;
  for (let i = 1; i < u.length; i++) diffSumSquares += (u[i] - u[i - 1]) ** 2;
  const sumSquares = u.reduce((s, v) => s + v * v, 0);
  const statistic = diffSumSquares / sumSquares;
  const concerning = statistic < 1.5 || statistic > 2.5;

  let reading;
  if (statistic < 1.5) {
    reading = `${statistic.toFixed(2)} is well below 2, indicating positive first-order autocorrelation. Classical standard errors understate uncertainty when this happens, which overstates the t-statistic on alpha. Newey-West is the appropriate correction.`;
  } else if (statistic > 2.5) {
    reading = `${statistic.toFixed(2)} is above 2, indicating negative first-order autocorrelation (month-to-month reversal in what the factors fail to explain).`;
  } else {
    reading = `${statistic.toFixed(2)} is close to 2, so there is little first-order autocorrelation. Note this statistic only sees lag 1; check Ljung-Box for longer lags.`;
  }

  return {
    name: "Durbin-Watson (lag-1 autocorrelation)",
    statistic, pValue: null, concerning,
    nullHypothesis: "No first-order autocorrelation (statistic near 2)",
    reading,
  };
}

// Sample autocorrelation at `lag`, biased estimator (divide by n, not n-lag),
// matching statsmodels.tsa.stattools.acf's default.
function autocorrelation(residuals, lag) {
  const n = residuals.length;
  const mean = residuals.reduce((s, v) => s + v, 0) / n;
  const centered = residuals.map((v) => v - mean);
  let numerator = 0;
  for (let t = 0; t < n - lag; t++) numerator += centered[t] * centered[t + lag];
  const denominator = centered.reduce((s, v) => s + v * v, 0);
  return numerator / denominator;
}

function ljungBox(regression, lags) {
  const residuals = regression.residuals;
  const n = residuals.length;
  const effectiveLags = lags ?? Math.min(12, Math.max(1, Math.floor(n / 5)));

  let statistic = 0;
  for (let k = 1; k <= effectiveLags; k++) {
    const r = autocorrelation(residuals, k);
    statistic += (r * r) / (n - k);
  }
  statistic *= n * (n + 2);

  const pValue = 1 - chiSquareCdf(statistic, effectiveLags);
  const concerning = pValue < SIGNIFICANCE;
  const reading = concerning
    ? `Residual autocorrelation is present somewhere in the first ${effectiveLags} lags. This is the specific condition Newey-West standard errors exist to handle, so quote the HAC column and set the lag length to at least this horizon.`
    : `Residuals look serially uncorrelated over ${effectiveLags} lags, so the HAC correction should barely move the standard errors.`;

  return {
    name: `Ljung-Box (autocorrelation, ${effectiveLags} lags)`,
    statistic, pValue, concerning,
    nullHypothesis: `No autocorrelation in residuals up to lag ${effectiveLags}`,
    reading,
  };
}

function jarqueBera(regression) {
  const residuals = regression.residuals;
  const n = residuals.length;
  const mean = residuals.reduce((s, v) => s + v, 0) / n;
  const centered = residuals.map((v) => v - mean);
  const m2 = centered.reduce((s, v) => s + v * v, 0) / n;
  const m3 = centered.reduce((s, v) => s + v ** 3, 0) / n;
  const m4 = centered.reduce((s, v) => s + v ** 4, 0) / n;
  const skew = m3 / Math.pow(m2, 1.5);
  const kurtosis = m4 / (m2 * m2);
  const statistic = (n / 6) * (skew * skew + ((kurtosis - 3) ** 2) / 4);
  const pValue = 1 - chiSquareCdf(statistic, 2);
  const concerning = pValue < SIGNIFICANCE;

  const reading = concerning
    ? `Residuals are not normal (skew ${skew.toFixed(2)}, kurtosis ${kurtosis.toFixed(2)}). At n = ${n} this is the least worrying item on the list: OLS estimates stay unbiased and the t-statistics are asymptotically valid either way. It does mean exact small-sample inference and normal-theory prediction intervals are off, and fat tails are typical of monthly asset returns.`
    : `No evidence against normal residuals (skew ${skew.toFixed(2)}, kurtosis ${kurtosis.toFixed(2)}); normal-theory intervals are reasonable here.`;

  return {
    name: "Jarque-Bera (normality)",
    statistic, pValue, concerning,
    nullHypothesis: "Residuals are normally distributed",
    reading,
  };
}

function regressionN(regression) {
  return regression.residuals.length;
}

export function runAll(regression) {
  return [breuschPagan(regression), durbinWatson(regression), ljungBox(regression), jarqueBera(regression)];
}

// `frame`: same shape as regression.js's fit() input ({ excess, regressorNames, regressors }).
export function varianceInflationFactors(frame) {
  const names = frame.regressorNames;
  return names.map((target) => {
    const y = frame.regressors[target];
    const others = names.filter((name) => name !== target);
    // A single-regressor model (e.g. CAPM) has no "other" regressors to
    // build a design matrix from; regressing on the intercept alone gives
    // R^2 = 0 and VIF = 1, same as statsmodels does in this edge case.
    const X = others.length > 0 ? designMatrix(others.map((name) => frame.regressors[name])) : y.map(() => [1]);
    const rSquared = rSquaredOf(y, X);
    const vif = 1 / (1 - rSquared);
    return { regressor: target, vif, flag: vif > VIF_CONCERN ? "review" : "ok" };
  });
}

export function factorCorrelations(frame) {
  const names = frame.regressorNames;
  const n = frame.regressors[names[0]].length;
  const means = names.map((name) => frame.regressors[name].reduce((s, v) => s + v, 0) / n);
  const stds = names.map((name, i) => Math.sqrt(
    frame.regressors[name].reduce((s, v) => s + (v - means[i]) ** 2, 0) / n,
  ));

  const matrix = names.map((_, i) => names.map((__, j) => {
    let covariance = 0;
    for (let t = 0; t < n; t++) covariance += (frame.regressors[names[i]][t] - means[i]) * (frame.regressors[names[j]][t] - means[j]);
    covariance /= n;
    return covariance / (stds[i] * stds[j]);
  }));

  return { names, matrix };
}

export function alphaVerdict(regression, tests) {
  const significant = regression.alphaPvalue < SIGNIFICANCE;
  const annual = regression.alphaAnnualized;
  const autocorrelated = tests.some((test) => test.concerning && test.name.toLowerCase().includes("autocorrelation"));

  let verdict = significant
    ? `Alpha is ${(annual * 100).toFixed(2)}% a year and is statistically distinguishable from zero under ${regression.seType} standard errors (t = ${regression.alphaTstat.toFixed(2)}, p = ${regression.alphaPvalue.toFixed(3)}).`
    : `Alpha is ${(annual * 100).toFixed(2)}% a year but is not statistically distinguishable from zero under ${regression.seType} standard errors (t = ${regression.alphaTstat.toFixed(2)}, p = ${regression.alphaPvalue.toFixed(3)}); the ${regression.model} factors account for this asset's excess return.`;

  if (autocorrelated && regression.seType !== "Newey-West (HAC)") {
    verdict += " The residuals are autocorrelated, so this classical t-statistic is optimistic; read the Newey-West column instead.";
  }
  return verdict;
}
