"""Generates scipy reference values for verify_distributions.mjs.

Checks the hand-rolled t/F/chi2/normal CDF and quantile functions in
frontend/js/modules/distributions.js against scipy.stats, independent of the
regression-level fixtures (which exercise the same functions indirectly).

    uv run scripts/export_dist_reference.py
    node scripts/verify_distributions.mjs
"""

import json
from pathlib import Path

from scipy import stats

OUTPUT_PATH = Path(__file__).resolve().parent / "dist_reference.json"


def main() -> None:
    cases = {"t_cdf": [], "f_cdf": [], "chi2_cdf": [], "normal_cdf": [], "t_quantile": [], "normal_quantile": []}

    for t in [-5, -2.5, -1.96, -1, -0.3, 0, 0.3, 1, 1.96, 2.5, 5, 10]:
        for df in [3, 5, 10, 30, 60, 120, 500]:
            cases["t_cdf"].append([t, df, stats.t.cdf(t, df)])

    for f in [0.01, 0.5, 1, 2, 3.84, 5, 10, 50]:
        for d1 in [1, 2, 3, 5]:
            for d2 in [10, 30, 100, 500]:
                cases["f_cdf"].append([f, d1, d2, stats.f.cdf(f, d1, d2)])

    for x in [0.01, 0.5, 1, 2, 3.84, 5.99, 10, 20, 50]:
        for k in [1, 2, 3, 4, 6, 12]:
            cases["chi2_cdf"].append([x, k, stats.chi2.cdf(x, k)])

    for z in [-4, -2.5, -1.96, -1, -0.3, 0, 0.3, 1, 1.96, 2.5, 4]:
        cases["normal_cdf"].append([z, stats.norm.cdf(z)])

    for p in [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975, 0.99, 0.999]:
        for df in [3, 5, 10, 30, 60, 120, 500]:
            cases["t_quantile"].append([p, df, stats.t.ppf(p, df)])
        cases["normal_quantile"].append([p, stats.norm.ppf(p)])

    OUTPUT_PATH.write_text(json.dumps(cases))
    print(f"Wrote {OUTPUT_PATH} ({sum(len(v) for v in cases.values())} cases)")


if __name__ == "__main__":
    main()
