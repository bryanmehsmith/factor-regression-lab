import { readFileSync } from "fs";
import { fit, compareNested } from "../frontend/js/modules/regression.js";
import { runAll, varianceInflationFactors } from "../frontend/js/modules/diagnostics.js";
import { rollingEstimates, stabilitySummary } from "../frontend/js/modules/rolling.js";

const cases = JSON.parse(readFileSync("./regression_fixtures.json", "utf8"));

let failures = 0;
let worst = 0;

function close(actual, expected, tol, label) {
  if (actual == null || expected == null) {
    if (actual !== expected) { console.log(`FAIL ${label}: null mismatch (${actual} vs ${expected})`); failures++; }
    return;
  }
  const err = Math.abs(actual - expected);
  const rel = err / Math.max(Math.abs(expected), 1e-8);
  worst = Math.max(worst, Math.min(err, rel));
  if (err > tol && rel > tol) {
    console.log(`FAIL ${label}: got ${actual}, expected ${expected}, abs_err=${err}, rel_err=${rel}`);
    failures++;
  }
}

for (const testCase of cases) {
  const frame = testCase.frame;

  for (const expectedFit of testCase.fits) {
    const actual = fit(frame, { seType: expectedFit.seType, hacLags: expectedFit.hacLags });
    const label = `${testCase.label}/${expectedFit.seType}/lags=${expectedFit.hacLags}`;

    close(actual.n, expectedFit.nObs, 0, `${label} nObs`);
    close(actual.rSquared, expectedFit.rSquared, 1e-8, `${label} rSquared`);
    close(actual.adjRSquared, expectedFit.adjRSquared, 1e-8, `${label} adjRSquared`);
    close(actual.alphaAnnualized, expectedFit.alphaAnnualized, 1e-6, `${label} alphaAnnualized`);
    close(actual.alphaTstat, expectedFit.alphaTstat, 1e-4, `${label} alphaTstat`);
    close(actual.alphaPvalue, expectedFit.alphaPvalue, 1e-6, `${label} alphaPvalue`);

    for (const expectedCoef of expectedFit.coefficients) {
      const i = actual.termNames.indexOf(expectedCoef.term);
      const t = `${label}/${expectedCoef.term}`;
      close(actual.beta[i], expectedCoef.estimate, 1e-8, `${t} estimate`);
      close(actual.standardErrors[i], expectedCoef.stdError, 1e-6, `${t} stdError`);
      close(actual.tStats[i], expectedCoef.tStat, 1e-4, `${t} tStat`);
      close(actual.pValues[i], expectedCoef.pValue, 1e-6, `${t} pValue`);
      close(actual.ciLower[i], expectedCoef.ciLower, 1e-5, `${t} ciLower`);
      close(actual.ciUpper[i], expectedCoef.ciUpper, 1e-5, `${t} ciUpper`);
    }
  }

  if (testCase.nested) {
    const nestedFrame = testCase.nestedFrame;
    const smallNames = Object.keys(nestedFrame.regressors).filter((n) => testCase.nested.added.includes(n) === false);
    const actual = compareNested(nestedFrame, smallNames, { seType: "Classical (OLS)" });
    const label = `${testCase.label}/nested`;
    close(actual.fStat, testCase.nested.fStat, 1e-4, `${label} fStat`);
    close(actual.pValue, testCase.nested.pValue, 1e-6, `${label} pValue`);
    close(actual.dfNum, testCase.nested.dfNum, 0, `${label} dfNum`);
    close(actual.dfDenom, testCase.nested.dfDenom, 0, `${label} dfDenom`);
    close(actual.largeAdjRSquared, testCase.nested.largeAdjRSquared, 1e-6, `${label} largeAdjRSquared`);

    const actualHac = compareNested(nestedFrame, smallNames, { seType: "Newey-West (HAC)", hacLags: 6 });
    const labelHac = `${testCase.label}/nestedHac`;
    close(actualHac.fStat, testCase.nestedHac.fStat, 1e-3, `${labelHac} fStat`);
    close(actualHac.pValue, testCase.nestedHac.pValue, 1e-4, `${labelHac} pValue`);
  }

  if (testCase.diagnostics) {
    const classical = fit(frame, { seType: "Classical (OLS)" });
    const tests = runAll(classical);
    for (const expectedTest of testCase.diagnostics.tests) {
      const actualTest = tests.find((t) => t.name === expectedTest.name);
      const label = `${testCase.label}/diagnostics/${expectedTest.name}`;
      if (!actualTest) { console.log(`FAIL ${label}: not found`); failures++; continue; }
      close(actualTest.statistic, expectedTest.statistic, 1e-5, `${label} statistic`);
      close(actualTest.pValue, expectedTest.pValue, 1e-5, `${label} pValue`);
      if (actualTest.concerning !== expectedTest.concerning) {
        console.log(`FAIL ${label}: concerning got ${actualTest.concerning}, expected ${expectedTest.concerning}`);
        failures++;
      }
    }

    const vif = varianceInflationFactors(frame);
    for (const expectedVif of testCase.diagnostics.vif) {
      const actualVif = vif.find((v) => v.regressor === expectedVif.regressor);
      close(actualVif.vif, expectedVif.vif, 1e-5, `${testCase.label}/vif/${expectedVif.regressor}`);
    }
  }

  if (testCase.rolling) {
    const estimates = rollingEstimates(frame, testCase.rolling.window);
    for (const [term, expectedSeries] of Object.entries(testCase.rolling.estimates)) {
      const actualSeries = estimates[term];
      const label = `${testCase.label}/rolling/${term}`;
      close(actualSeries.estimate.length, expectedSeries.estimate.length, 0, `${label} length`);
      for (let i = 0; i < expectedSeries.estimate.length; i++) {
        close(actualSeries.estimate[i], expectedSeries.estimate[i], 1e-5, `${label}[${i}] estimate`);
        close(actualSeries.lower[i], expectedSeries.lower[i], 1e-4, `${label}[${i}] lower`);
        close(actualSeries.upper[i], expectedSeries.upper[i], 1e-4, `${label}[${i}] upper`);
      }
    }

    const summary = stabilitySummary(estimates);
    for (const expectedRow of testCase.rolling.summary) {
      const actualRow = summary.find((s) => s.term === expectedRow.term);
      const label = `${testCase.label}/rolling-summary/${expectedRow.term}`;
      close(actualRow.min, expectedRow.min, 1e-5, `${label} min`);
      close(actualRow.max, expectedRow.max, 1e-5, `${label} max`);
      close(actualRow.range, expectedRow.range, 1e-5, `${label} range`);
    }
  }
}

console.log(`\nworst abs/rel error: ${worst}`);
console.log(`cases: ${cases.length}, failures: ${failures}`);
process.exit(failures > 0 ? 1 : 0);
