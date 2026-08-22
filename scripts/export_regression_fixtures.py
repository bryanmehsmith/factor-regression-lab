"""Generates JS-vs-Python parity fixtures for the frontend's regression/diagnostics/rolling port.

Builds a battery of synthetic regression frames (reusing the exact generators
in tests/conftest.py) covering a range of sample sizes, autocorrelation,
heteroskedasticity, and regressor counts, fits each with the real
regression.py/diagnostics.py/rolling.py, and dumps both the input frame and
every statistic the frontend reports. verify_stats_parity.mjs reads this file,
recomputes everything in JS, and diffs.

Run whenever the JS port changes or the Python reference changes:

    uv run scripts/export_regression_fixtures.py
    node scripts/verify_stats_parity.mjs
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import N_MONTHS, TRUE_BETAS, ar1_residuals, make_frame  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factor_lab import diagnostics, regression, rolling  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "regression_fixtures.json"


def frame_to_json(frame):
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "excess": frame["excess"].tolist(),
        "regressorNames": [c for c in frame.columns if c != "excess"],
        "regressors": {c: frame[c].tolist() for c in frame.columns if c != "excess"},
    }


def coefficient_table_json(fitted):
    table = regression.coefficient_table(fitted)
    return [
        {
            "term": term,
            "estimate": float(row["estimate"]),
            "stdError": float(row["std_error"]),
            "tStat": float(row["t_stat"]),
            "pValue": float(row["p_value"]),
            "ciLower": float(row["ci_lower_95%"]),
            "ciUpper": float(row["ci_upper_95%"]),
        }
        for term, row in table.iterrows()
    ]


def fit_case(frame, se_type, hac_lags=None):
    fitted = regression.fit(frame, se_type=se_type, hac_lags=hac_lags)
    return {
        "seType": se_type,
        "hacLags": fitted.hac_lags,
        "nObs": fitted.n_obs,
        "rSquared": float(fitted.r_squared),
        "adjRSquared": float(fitted.adj_r_squared),
        "alphaAnnualized": float(fitted.alpha_annualized),
        "alphaTstat": float(fitted.alpha_tstat),
        "alphaPvalue": float(fitted.alpha_pvalue),
        "coefficients": coefficient_table_json(fitted),
    }


def diagnostics_json(frame):
    fitted = regression.fit(frame, se_type="Classical (OLS)")
    tests = diagnostics.run_all(fitted)
    vif = diagnostics.variance_inflation_factors(frame)
    return {
        "tests": [
            {
                "name": t.name,
                "statistic": float(t.statistic),
                "pValue": None if t.p_value is None else float(t.p_value),
                "concerning": bool(t.concerning),
            }
            for t in tests
        ],
        "vif": [
            {"regressor": row["regressor"], "vif": float(row["vif"])}
            for _, row in vif.iterrows()
        ],
    }


def nested_json(frame, small_regressors, se_type="Classical (OLS)", hac_lags=None):
    nested = regression.compare_nested(
        frame, small_regressors=small_regressors, se_type=se_type, hac_lags=hac_lags
    )
    return {
        "added": nested.added,
        "fStat": float(nested.f_stat),
        "pValue": float(nested.p_value),
        "dfNum": nested.df_num,
        "dfDenom": nested.df_denom,
        "largeAdjRSquared": float(nested.large_adj_r_squared),
        "smallAdjRSquared": float(nested.small_adj_r_squared),
    }


def rolling_json(frame, window=60):
    estimates = rolling.rolling_estimates(frame, window=window)
    summary = rolling.stability_summary(estimates)
    return {
        "window": window,
        "estimates": {
            term: {
                "estimate": df["estimate"].tolist(),
                "lower": df["lower"].tolist(),
                "upper": df["upper"].tolist(),
            }
            for term, df in estimates.items()
        },
        "summary": [
            {"term": term, "min": float(row["min"]), "max": float(row["max"]), "range": float(row["range"])}
            for term, row in summary.iterrows()
        ],
    }


def main():
    cases = []

    seeds_and_labels = [
        (0, "clean_600"),
        (1, "clean_alt_seed"),
        (20260726, "clean_matches_pytest_rng"),
    ]
    for seed, label in seeds_and_labels:
        rng = np.random.default_rng(seed)
        frame = make_frame(rng)
        case = {"label": label, "frame": frame_to_json(frame)}
        case["fits"] = [
            fit_case(frame, "Classical (OLS)"),
            fit_case(frame, "White (HC1)"),
            fit_case(frame, "Newey-West (HAC)", hac_lags=6),
            fit_case(frame, "Newey-West (HAC)", hac_lags=0),
            fit_case(frame, "Newey-West (HAC)", hac_lags=12),
        ]
        case["diagnostics"] = diagnostics_json(frame)
        case["rolling"] = rolling_json(frame, window=60)

        # A nested test needs a larger model than the frame's own TRUE_BETAS-only
        # columns, so build a separate FF5-ish frame for that part of the case.
        nested_frame = make_frame(rng, betas={**TRUE_BETAS, "RMW": 0.45, "CMA": -0.15})
        case["nestedFrame"] = frame_to_json(nested_frame)
        case["nested"] = nested_json(nested_frame, list(TRUE_BETAS))
        case["nestedHac"] = nested_json(nested_frame, list(TRUE_BETAS), se_type="Newey-West (HAC)", hac_lags=6)
        cases.append(case)

    # Autocorrelated residuals: the HAC-vs-classical centerpiece case.
    for rho, label in [(0.3, "ar1_rho03"), (0.7, "ar1_rho07"), (0.9, "ar1_rho09")]:
        rng = np.random.default_rng(42)
        frame = make_frame(rng, residuals=ar1_residuals(rng, rho=rho))
        case = {"label": label, "frame": frame_to_json(frame)}
        case["fits"] = [
            fit_case(frame, "Classical (OLS)"),
            fit_case(frame, "White (HC1)"),
            fit_case(frame, "Newey-West (HAC)", hac_lags=6),
        ]
        case["diagnostics"] = diagnostics_json(frame)
        cases.append(case)

    # Heteroskedastic residuals: the Breusch-Pagan / HC1 centerpiece case.
    rng = np.random.default_rng(7)
    frame = make_frame(rng, residuals=np.zeros(N_MONTHS))
    scale = 0.02 + 0.6 * (frame["Mkt-RF"] - frame["Mkt-RF"].min())
    frame["excess"] = frame["excess"] + rng.normal(0, 1, len(frame)) * scale
    case = {"label": "heteroskedastic", "frame": frame_to_json(frame)}
    case["fits"] = [fit_case(frame, "Classical (OLS)"), fit_case(frame, "White (HC1)")]
    case["diagnostics"] = diagnostics_json(frame)
    cases.append(case)

    # Small samples and varying regressor counts, to stress degrees-of-freedom edges.
    for n_months, betas_subset, label in [
        (36, ["Mkt-RF"], "small_n_capm"),
        (48, ["Mkt-RF", "SMB", "HML"], "small_n_ff3"),
        (120, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"], "medium_n_ff5"),
    ]:
        rng = np.random.default_rng(99)
        betas = {k: v for k, v in TRUE_BETAS.items() if k in betas_subset}
        betas = {**{b: 0.3 for b in betas_subset if b not in betas}, **betas}
        frame = make_frame(rng, betas=betas, n_months=n_months)
        case = {"label": label, "frame": frame_to_json(frame)}
        case["fits"] = [
            fit_case(frame, "Classical (OLS)"),
            fit_case(frame, "White (HC1)"),
            fit_case(frame, "Newey-West (HAC)", hac_lags=regression.default_hac_lags(n_months)),
        ]
        case["diagnostics"] = diagnostics_json(frame)
        cases.append(case)

    OUTPUT_PATH.write_text(json.dumps(cases))
    print(f"Wrote {OUTPUT_PATH} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
