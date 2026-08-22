// Hand-rolled special functions backing every p-value and confidence interval
// in this app. There is no bundler/CDN dependency in this project, so these
// are implemented from the standard numerical-recipes algorithms rather than
// pulled from a stats library. Every function here is checked against
// Python (scipy.stats/statsmodels) in the parity harness before being trusted,
// see scripts/verify_stats_parity.mjs.

const MAX_ITERATIONS = 200;
const EPSILON = 3e-16;
const FLOOR = 1e-300;

// Lanczos approximation of ln(Gamma(x)).
const LANCZOS_G = 7;
const LANCZOS_COEFFICIENTS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function logGamma(x) {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const z = x - 1;
  let a = LANCZOS_COEFFICIENTS[0];
  const t = z + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_G + 2; i++) a += LANCZOS_COEFFICIENTS[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

function logBeta(a, b) {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

// Continued fraction for the regularized incomplete beta (Numerical Recipes betacf).
function betaContinuedFraction(x, a, b) {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FLOOR) d = FLOOR;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= MAX_ITERATIONS; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FLOOR) d = FLOOR;
    c = 1 + aa / c;
    if (Math.abs(c) < FLOOR) c = FLOOR;
    d = 1 / d;
    h *= d * c;

    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FLOOR) d = FLOOR;
    c = 1 + aa / c;
    if (Math.abs(c) < FLOOR) c = FLOOR;
    d = 1 / d;
    const del = d * c;
    h *= del;

    if (Math.abs(del - 1) < EPSILON) break;
  }
  return h;
}

// Regularized incomplete beta I_x(a, b) = P(X <= x) for a Beta(a, b) variable.
export function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - logBeta(a, b));
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betaContinuedFraction(x, a, b)) / a;
  }
  return 1 - (front * betaContinuedFraction(1 - x, b, a)) / b;
}

// Series expansion for the regularized lower incomplete gamma, x < a + 1.
function gammaSeries(a, x) {
  if (x <= 0) return 0;
  let sum = 1 / a;
  let del = sum;
  let ap = a;
  for (let n = 1; n <= MAX_ITERATIONS; n++) {
    ap += 1;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * EPSILON) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

// Continued fraction for the regularized UPPER incomplete gamma, x >= a + 1.
function gammaContinuedFraction(a, x) {
  let b = x + 1 - a;
  let c = 1 / FLOOR;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= MAX_ITERATIONS; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FLOOR) d = FLOOR;
    c = b + an / c;
    if (Math.abs(c) < FLOOR) c = FLOOR;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPSILON) break;
  }
  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

// Regularized lower incomplete gamma P(a, x) = P(X <= x) for a Gamma(a, 1) variable.
export function regularizedLowerIncompleteGamma(a, x) {
  if (x < 0) throw new Error("x must be non-negative");
  if (x === 0) return 0;
  if (x < a + 1) return gammaSeries(a, x);
  return 1 - gammaContinuedFraction(a, x);
}

export function chiSquareCdf(x, degreesOfFreedom) {
  if (x <= 0) return 0;
  return regularizedLowerIncompleteGamma(degreesOfFreedom / 2, x / 2);
}

// Two-sided Student-t CDF: P(T <= t) for t distributed with `degreesOfFreedom`.
export function studentTCdf(t, degreesOfFreedom) {
  const x = degreesOfFreedom / (degreesOfFreedom + t * t);
  const tail = 0.5 * regularizedIncompleteBeta(x, degreesOfFreedom / 2, 0.5);
  return t >= 0 ? 1 - tail : tail;
}

export function fCdf(f, df1, df2) {
  if (f <= 0) return 0;
  const x = (df1 * f) / (df1 * f + df2);
  return regularizedIncompleteBeta(x, df1 / 2, df2 / 2);
}

export function normalCdf(z) {
  const p = regularizedLowerIncompleteGamma(0.5, (z * z) / 2);
  return z >= 0 ? 0.5 + 0.5 * p : 0.5 - 0.5 * p;
}

function studentTPdf(t, degreesOfFreedom) {
  const halfDf = degreesOfFreedom / 2;
  const logCoefficient = logGamma(halfDf + 0.5) - logGamma(halfDf) - 0.5 * Math.log(degreesOfFreedom * Math.PI);
  return Math.exp(logCoefficient - (halfDf + 0.5) * Math.log(1 + (t * t) / degreesOfFreedom));
}

function normalPdf(z) {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

// Newton-Raphson with a bisection fallback: robust for any smooth, monotonic
// CDF, which both the normal and Student-t CDFs are. Safer to verify than a
// dedicated rational-approximation quantile algorithm, and only ever called
// a few hundred times per chart render.
function invertMonotonicCdf(targetP, cdf, pdf, lowerBound, upperBound) {
  let lo = lowerBound;
  let hi = upperBound;
  let x = 0;
  for (let iteration = 0; iteration < 100; iteration++) {
    const fx = cdf(x) - targetP;
    if (Math.abs(fx) < 1e-12) return x;
    if (fx > 0) hi = x; else lo = x;

    const derivative = pdf(x);
    let next = derivative > 1e-300 ? x - fx / derivative : (lo + hi) / 2;
    if (!(next > lo && next < hi)) next = (lo + hi) / 2;
    if (Math.abs(next - x) < 1e-13) return next;
    x = next;
  }
  return x;
}

export function normalQuantile(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  return invertMonotonicCdf(p, normalCdf, normalPdf, -40, 40);
}

export function studentTQuantile(p, degreesOfFreedom) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const cdf = (t) => studentTCdf(t, degreesOfFreedom);
  const pdf = (t) => studentTPdf(t, degreesOfFreedom);
  return invertMonotonicCdf(p, cdf, pdf, -1e4, 1e4);
}
