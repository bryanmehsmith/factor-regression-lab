"""Residual diagnostics and multicollinearity checks, each with a plain-language reading.

A coefficient table on its own invites the reader to trust it. These tests say
which OLS assumptions the data actually supports, and therefore which of the
standard errors in `regression.compare_standard_errors` deserves to be quoted.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from factor_lab.regression import ALPHA_LABEL, FactorRegression

SIGNIFICANCE = 0.05
# Above this, a regressor's variance is inflated enough that its own coefficient
# is hard to read even though the fit as a whole may be fine.
VIF_CONCERN = 5.0


@dataclass(frozen=True)
class DiagnosticTest:
    """One hypothesis test about the residuals, with its interpretation attached."""

    name: str
    statistic: float
    p_value: float | None
    null_hypothesis: str
    reading: str
    concerning: bool


def _jarque_bera(regression: FactorRegression) -> DiagnosticTest:
    statistic, p_value, skew, kurtosis = jarque_bera(regression.residuals)
    concerning = p_value < SIGNIFICANCE
    if concerning:
        reading = (
            f"Residuals are not normal (skew {skew:.2f}, kurtosis {kurtosis:.2f}). "
            f"At n = {regression.n_obs} this is the least worrying item on the list: OLS "
            "estimates stay unbiased and the t-statistics are asymptotically valid either "
            "way. It does mean exact small-sample inference and normal-theory prediction "
            "intervals are off, and fat tails are typical of monthly asset returns."
        )
    else:
        reading = (
            f"No evidence against normal residuals (skew {skew:.2f}, kurtosis {kurtosis:.2f}); "
            "normal-theory intervals are reasonable here."
        )
    return DiagnosticTest(
        name="Jarque-Bera (normality)",
        statistic=float(statistic),
        p_value=float(p_value),
        null_hypothesis="Residuals are normally distributed",
        reading=reading,
        concerning=concerning,
    )


def _breusch_pagan(regression: FactorRegression) -> DiagnosticTest:
    exog = sm.add_constant(regression.frame.drop(columns="excess"), prepend=True)
    statistic, p_value, _, _ = het_breuschpagan(regression.residuals, exog)
    concerning = p_value < SIGNIFICANCE
    if concerning:
        reading = (
            "Residual variance changes with the factor values (heteroskedasticity). "
            "Classical standard errors are biased here, so prefer White or Newey-West. "
            "Volatility clustering makes this the normal finding for asset returns."
        )
    else:
        reading = (
            "No evidence of heteroskedasticity; classical standard errors are not "
            "obviously wrong on this count."
        )
    return DiagnosticTest(
        name="Breusch-Pagan (heteroskedasticity)",
        statistic=float(statistic),
        p_value=float(p_value),
        null_hypothesis="Residual variance is constant (homoskedastic)",
        reading=reading,
        concerning=concerning,
    )


def _durbin_watson(regression: FactorRegression) -> DiagnosticTest:
    statistic = float(durbin_watson(regression.residuals))
    # 2 means no first-order autocorrelation; below 1.5 or above 2.5 is the usual flag.
    concerning = statistic < 1.5 or statistic > 2.5
    if statistic < 1.5:
        reading = (
            f"{statistic:.2f} is well below 2, indicating positive first-order autocorrelation. "
            "Classical standard errors understate uncertainty when this happens, which "
            "overstates the t-statistic on alpha. Newey-West is the appropriate correction."
        )
    elif statistic > 2.5:
        reading = (
            f"{statistic:.2f} is above 2, indicating negative first-order autocorrelation "
            "(month-to-month reversal in what the factors fail to explain)."
        )
    else:
        reading = (
            f"{statistic:.2f} is close to 2, so there is little first-order autocorrelation. "
            "Note this statistic only sees lag 1; check Ljung-Box for longer lags."
        )
    return DiagnosticTest(
        name="Durbin-Watson (lag-1 autocorrelation)",
        statistic=statistic,
        p_value=None,
        null_hypothesis="No first-order autocorrelation (statistic near 2)",
        reading=reading,
        concerning=concerning,
    )


def _ljung_box(regression: FactorRegression, lags: int | None = None) -> DiagnosticTest:
    residuals = regression.residuals
    if lags is None:
        lags = int(min(12, max(1, len(residuals) // 5)))
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    statistic = float(result["lb_stat"].iloc[0])
    p_value = float(result["lb_pvalue"].iloc[0])
    concerning = p_value < SIGNIFICANCE
    if concerning:
        reading = (
            f"Residual autocorrelation is present somewhere in the first {lags} lags. "
            "This is the specific condition Newey-West standard errors exist to handle, "
            "so quote the HAC column and set the lag length to at least this horizon."
        )
    else:
        reading = (
            f"Residuals look serially uncorrelated over {lags} lags, so the HAC correction "
            "should barely move the standard errors."
        )
    return DiagnosticTest(
        name=f"Ljung-Box (autocorrelation, {lags} lags)",
        statistic=statistic,
        p_value=p_value,
        null_hypothesis=f"No autocorrelation in residuals up to lag {lags}",
        reading=reading,
        concerning=concerning,
    )


def run_all(regression: FactorRegression) -> list[DiagnosticTest]:
    """The full battery, ordered so the autocorrelation evidence reads together."""
    return [
        _breusch_pagan(regression),
        _durbin_watson(regression),
        _ljung_box(regression),
        _jarque_bera(regression),
    ]


def diagnostics_table(tests: list[DiagnosticTest]) -> pd.DataFrame:
    """Test results as a display frame, without the long-form readings."""
    return pd.DataFrame(
        [
            {
                "test": test.name,
                "statistic": test.statistic,
                "p_value": test.p_value,
                "null_hypothesis": test.null_hypothesis,
                "flag": "review" if test.concerning else "ok",
            }
            for test in tests
        ]
    )


def variance_inflation_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF per regressor.

    A VIF of 5 means that regressor's coefficient variance is 5x what it would be
    if the regressors were orthogonal. High VIF does not bias the estimates or
    hurt the fit; it widens the individual standard errors, which is why a model
    can have a strong joint F-test and no individually significant coefficients.
    The intercept is excluded because its VIF is not interpretable.
    """
    exog = frame.drop(columns="excess", errors="ignore")
    design = sm.add_constant(exog, prepend=True)
    values = design.to_numpy(dtype=float)

    rows = []
    for position, name in enumerate(design.columns):
        if name == "const":
            continue
        rows.append(
            {
                "regressor": name,
                "vif": float(variance_inflation_factor(values, position)),
            }
        )

    table = pd.DataFrame(rows)
    table["flag"] = np.where(table["vif"] > VIF_CONCERN, "review", "ok")
    return table


def factor_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of the regressors, the raw material behind the VIFs."""
    exog = frame.drop(columns="excess", errors="ignore")
    return exog.corr()


def alpha_verdict(regression: FactorRegression, tests: list[DiagnosticTest]) -> str:
    """One sentence answering the question the demo poses."""
    significant = regression.alpha_pvalue < SIGNIFICANCE
    annual = regression.alpha_annualized
    autocorrelated = any(
        test.concerning and "autocorrelation" in test.name.lower() for test in tests
    )

    if significant:
        verdict = (
            f"Alpha is {annual:.2%} a year and is statistically distinguishable from zero "
            f"under {regression.se_type} standard errors "
            f"(t = {regression.alpha_tstat:.2f}, p = {regression.alpha_pvalue:.3f})."
        )
    else:
        verdict = (
            f"Alpha is {annual:.2%} a year but is not statistically distinguishable from zero "
            f"under {regression.se_type} standard errors "
            f"(t = {regression.alpha_tstat:.2f}, p = {regression.alpha_pvalue:.3f}); "
            f"the {regression.model} factors account for this asset's excess return."
        )

    if autocorrelated and regression.se_type != "Newey-West (HAC)":
        verdict += (
            " The residuals are autocorrelated, so this classical t-statistic is optimistic; "
            "read the Newey-West column instead."
        )
    return verdict
