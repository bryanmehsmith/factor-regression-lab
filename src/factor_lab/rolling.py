"""Rolling-window estimation, to show that a single full-sample beta is a summary, not a constant.

A full-sample regression reports one number per factor and implies the exposure
held for sixty years. Re-estimating over a moving window shows how much of that
is an artifact of averaging.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from factor_lab.regression import ALPHA_LABEL, PERIODS_PER_YEAR

DEFAULT_WINDOW = 60
# Two-sided normal critical value; the bands are indicative rather than exact.
Z_95 = 1.959963984540054


def rolling_estimates(
    frame: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> dict[str, pd.DataFrame]:
    """Rolling coefficient estimates with 95% bands, one frame per term.

    Each value is keyed by the window's final month. Bands come from classical
    rolling standard errors, since HAC is not available in the rolling
    estimator; they are indicative of parameter instability rather than a
    substitute for the full-sample inference reported elsewhere.
    """
    if window < 12:
        raise ValueError("window must be at least 12 months")
    if window > len(frame):
        raise ValueError(f"window of {window} exceeds the {len(frame)} available months")

    y = frame["excess"]
    X = sm.add_constant(frame.drop(columns="excess"), prepend=True).rename(
        columns={"const": ALPHA_LABEL}
    )

    results = RollingOLS(y, X, window=window).fit()
    params = results.params
    errors = results.bse

    estimates = {}
    for term in X.columns:
        estimate = params[term]
        error = errors[term]
        # Alpha is far easier to read annualized; betas are unitless already.
        scale = PERIODS_PER_YEAR if term == ALPHA_LABEL else 1.0
        estimates[term] = pd.DataFrame(
            {
                "estimate": estimate * scale,
                "lower": (estimate - Z_95 * error) * scale,
                "upper": (estimate + Z_95 * error) * scale,
            }
        ).dropna()

    return estimates


def stability_summary(estimates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How much each rolling coefficient moved, and whether it ever changed sign."""
    rows = []
    for term, frame in estimates.items():
        series = frame["estimate"]
        if series.empty:
            continue
        rows.append(
            {
                "term": term,
                "min": series.min(),
                "max": series.max(),
                "range": series.max() - series.min(),
                "changed_sign": bool((series > 0).any() and (series < 0).any()),
            }
        )
    return pd.DataFrame(rows).set_index("term")
