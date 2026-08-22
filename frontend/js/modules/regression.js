// JS port of src/factor_lab/regression.py. Point estimates are identical
// across se_type; only the covariance (and so SE/t/p/CI) changes.

import { transpose, multiply, multiplyVector, subtractVectors, inverse, designMatrix } from "./linalg.js";
import { studentTCdf, studentTQuantile, normalCdf, normalQuantile, fCdf } from "./distributions.js";

export const PERIODS_PER_YEAR = 12;
export const ALPHA_LABEL = "alpha";
export const SE_TYPES = ["Classical (OLS)", "White (HC1)", "Newey-West (HAC)"];
export const DEFAULT_SE_TYPE = "Newey-West (HAC)";

export function defaultHacLags(nObs) {
  if (nObs <= 0) return 0;
  return Math.floor(4 * Math.pow(nObs / 100, 2 / 9));
}

function ordinaryLeastSquares(y, X) {
  const Xt = transpose(X);
  const XtX = multiply(Xt, X);
  const XtXInverse = inverse(XtX);
  const Xty = Xt.map((row) => row.reduce((sum, v, i) => sum + v * y[i], 0));
  const beta = multiplyVector(XtXInverse, Xty);
  const fitted = X.map((row) => row.reduce((sum, v, i) => sum + v * beta[i], 0));
  const residuals = subtractVectors(y, fitted);
  return { beta, fitted, residuals, XtXInverse, Xt };
}

function classicalCovariance(XtXInverse, residuals, k) {
  const n = residuals.length;
  const sigma2 = residuals.reduce((sum, u) => sum + u * u, 0) / (n - k);
  return XtXInverse.map((row) => row.map((v) => v * sigma2));
}

function sandwich(XtXInverse, meat) {
  return multiply(multiply(XtXInverse, meat), XtXInverse);
}

function hc1Covariance(X, XtXInverse, residuals, k) {
  const n = residuals.length;
  const meat = meatFromWeightedOuterProducts(X, residuals);
  const scale = n / (n - k);
  return sandwich(XtXInverse, meat).map((row) => row.map((v) => v * scale));
}

// Sum_i u_i^2 * x_i * x_i' -- the "meat" shared by HC0/HC1 and as the lag-0
// term of the HAC meat below.
function meatFromWeightedOuterProducts(X, residuals) {
  const k = X[0].length;
  const n = X.length;
  const meat = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < n; i++) {
    const w = residuals[i] * residuals[i];
    if (w === 0) continue;
    const row = X[i];
    for (let a = 0; a < k; a++) {
      const wa = w * row[a];
      for (let b = 0; b < k; b++) meat[a][b] += wa * row[b];
    }
  }
  return meat;
}

// Newey-West HAC, Bartlett kernel. Matches statsmodels'
// sm.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L}).
function hacCovariance(X, XtXInverse, residuals, lags) {
  const n = X.length;
  const k = X[0].length;
  const meat = meatFromWeightedOuterProducts(X, residuals);

  for (let lag = 1; lag <= lags; lag++) {
    const weight = 1 - lag / (lags + 1);
    const cross = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = lag; i < n; i++) {
      const w = residuals[i] * residuals[i - lag];
      if (w === 0) continue;
      const rowT = X[i];
      const rowS = X[i - lag];
      for (let a = 0; a < k; a++) {
        const wa = w * rowT[a];
        for (let b = 0; b < k; b++) cross[a][b] += wa * rowS[b];
      }
    }
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < k; b++) meat[a][b] += weight * (cross[a][b] + cross[b][a]);
    }
  }

  return sandwich(XtXInverse, meat);
}

function covarianceFor(seType, X, XtXInverse, residuals, k, hacLags) {
  if (seType === "Classical (OLS)") return classicalCovariance(XtXInverse, residuals, k);
  if (seType === "White (HC1)") return hc1Covariance(X, XtXInverse, residuals, k);
  if (seType === "Newey-West (HAC)") return hacCovariance(X, XtXInverse, residuals, hacLags);
  throw new Error(`unknown se_type ${seType}`);
}

// frame: { dates, excess: number[], regressors: { [name]: number[] } }, regressorNames in order.
export function fit(frame, { model = "custom", seType = DEFAULT_SE_TYPE, hacLags = null } = {}) {
  if (!SE_TYPES.includes(seType)) throw new Error(`unknown se_type ${seType}`);

  const regressorNames = frame.regressorNames;
  const columns = regressorNames.map((name) => frame.regressors[name]);
  const X = designMatrix(columns);
  const y = frame.excess;
  const n = y.length;
  const k = regressorNames.length + 1;
  const termNames = [ALPHA_LABEL, ...regressorNames];

  const { beta, fitted, residuals, XtXInverse } = ordinaryLeastSquares(y, X);

  const lags = seType === "Newey-West (HAC)" ? (hacLags ?? defaultHacLags(n)) : null;
  const covariance = covarianceFor(seType, X, XtXInverse, residuals, k, lags);
  const standardErrors = covariance.map((row, i) => Math.sqrt(Math.max(row[i], 0)));

  const dfResid = n - k;
  // statsmodels only uses the Student-t reference distribution for the
  // classical (nonrobust) covariance; HC1 and HAC default to use_t=False, a
  // plain normal/z reference for p-values and confidence intervals. This
  // only affects the reference distribution, not the point estimates or SEs
  // themselves, which are identical either way.
  const useT = seType === "Classical (OLS)";
  const tStats = beta.map((b, i) => b / standardErrors[i]);
  const pValues = tStats.map((t) => (useT ? 2 * (1 - studentTCdf(Math.abs(t), dfResid)) : 2 * (1 - normalCdf(Math.abs(t)))));
  const critical = useT ? studentTQuantile(0.975, dfResid) : normalQuantile(0.975);
  const ciLower = beta.map((b, i) => b - critical * standardErrors[i]);
  const ciUpper = beta.map((b, i) => b + critical * standardErrors[i]);

  const meanY = y.reduce((s, v) => s + v, 0) / n;
  const totalSumSquares = y.reduce((s, v) => s + (v - meanY) ** 2, 0);
  const residualSumSquares = residuals.reduce((s, v) => s + v * v, 0);
  const rSquared = 1 - residualSumSquares / totalSumSquares;
  const adjRSquared = 1 - (1 - rSquared) * (n - 1) / dfResid;

  return {
    model, seType, hacLags: lags, termNames, regressorNames,
    n, dfResid, beta, fitted, residuals, X, XtXInverse, covariance,
    standardErrors, tStats, pValues, ciLower, ciUpper,
    rSquared, adjRSquared,
    alphaMonthly: beta[0],
    alphaAnnualized: beta[0] * PERIODS_PER_YEAR,
    alphaTstat: tStats[0],
    alphaPvalue: pValues[0],
  };
}

export function coefficientTable(regression) {
  return regression.termNames.map((term, i) => ({
    term,
    estimate: regression.beta[i],
    stdError: regression.standardErrors[i],
    tStat: regression.tStats[i],
    pValue: regression.pValues[i],
    ciLower: regression.ciLower[i],
    ciUpper: regression.ciUpper[i],
  }));
}

export function compareStandardErrors(frame, hacLags) {
  return Object.fromEntries(
    SE_TYPES.map((seType) => [seType, fit(frame, { seType, hacLags })]),
  );
}

// Nested F-test: are the regressors `large` adds over `small` jointly zero?
// When seType is robust, this is the robust Wald form statsmodels' f_test
// uses (robust covariance, F(q, df_resid) reference distribution), not a
// textbook F computed from RSS.
export function compareNested(frame, smallRegressorNames, { smallModel = "restricted", largeModel = "full", seType = DEFAULT_SE_TYPE, hacLags = null } = {}) {
  const largeRegressorNames = frame.regressorNames;
  const unknown = smallRegressorNames.filter((name) => !largeRegressorNames.includes(name));
  if (unknown.length > 0) throw new Error(`${unknown} are not regressors of the larger model, so the models are not nested`);

  const added = largeRegressorNames.filter((name) => !smallRegressorNames.includes(name));
  if (added.length === 0) throw new Error("the two models have identical regressors, so there is nothing to test");

  const large = fit(frame, { model: largeModel, seType, hacLags });
  const smallFrame = {
    excess: frame.excess,
    regressorNames: smallRegressorNames,
    regressors: Object.fromEntries(smallRegressorNames.map((name) => [name, frame.regressors[name]])),
  };
  const small = fit(smallFrame, { model: smallModel, seType, hacLags });

  const q = added.length;
  const dfDenom = large.dfResid;

  // Restriction rows selecting the added coefficients out of beta.
  const restriction = added.map((name) => {
    const row = new Array(large.termNames.length).fill(0);
    row[large.termNames.indexOf(name)] = 1;
    return row;
  });
  const restrictedBeta = restriction.map((row) => row.reduce((sum, r, i) => sum + r * large.beta[i], 0));
  // R * Cov * R'
  const RCov = multiply(restriction, large.covariance);
  const RCovRt = multiply(RCov, transpose(restriction));
  const RCovRtInverse = inverse(RCovRt);
  const wald = restrictedBeta.reduce(
    (sum, ri, i) => sum + ri * RCovRtInverse[i].reduce((s, v, j) => s + v * restrictedBeta[j], 0),
    0,
  );
  const fStat = wald / q;
  const pValue = 1 - fCdf(fStat, q, dfDenom);

  return {
    smallModel, largeModel, added,
    smallAdjRSquared: small.adjRSquared, largeAdjRSquared: large.adjRSquared,
    incrementalAdjRSquared: large.adjRSquared - small.adjRSquared,
    fStat, pValue, dfNum: q, dfDenom,
    smallAlphaAnnualized: small.alphaAnnualized, largeAlphaAnnualized: large.alphaAnnualized,
    smallAlphaTstat: small.alphaTstat, largeAlphaTstat: large.alphaTstat,
    nObs: large.n,
  };
}
