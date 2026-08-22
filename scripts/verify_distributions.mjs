import { readFileSync } from "fs";
import {
  studentTCdf, fCdf, chiSquareCdf, normalCdf, studentTQuantile, normalQuantile,
} from "../frontend/js/modules/distributions.js";

const ref = JSON.parse(readFileSync("./dist_reference.json", "utf8"));
let worst = 0;
let failures = 0;

function check(label, actual, expected, tol) {
  const err = Math.abs(actual - expected);
  const rel = err / Math.max(Math.abs(expected), 1e-10);
  const bad = err > tol && rel > tol;
  if (bad) {
    failures++;
    console.log(`FAIL ${label}: got ${actual}, expected ${expected}, err=${err}`);
  }
  worst = Math.max(worst, Math.min(err, rel));
}

for (const [t, df, expected] of ref.t_cdf) check(`t_cdf(${t},${df})`, studentTCdf(t, df), expected, 1e-9);
for (const [f, d1, d2, expected] of ref.f_cdf) check(`f_cdf(${f},${d1},${d2})`, fCdf(f, d1, d2), expected, 1e-9);
for (const [x, k, expected] of ref.chi2_cdf) check(`chi2_cdf(${x},${k})`, chiSquareCdf(x, k), expected, 1e-9);
for (const [z, expected] of ref.normal_cdf) check(`normal_cdf(${z})`, normalCdf(z), expected, 1e-9);
for (const [p, df, expected] of ref.t_quantile) check(`t_quantile(${p},${df})`, studentTQuantile(p, df), expected, 1e-6);
for (const [p, expected] of ref.normal_quantile) check(`normal_quantile(${p})`, normalQuantile(p), expected, 1e-6);

console.log(`\nworst abs/rel error: ${worst}`);
console.log(`failures: ${failures}`);
process.exit(failures > 0 ? 1 : 0);
