"""Factor regression estimation and the standard-error comparison at the heart of the demo.

The point of the module is not that OLS is hard; statsmodels does the fitting.
The point is that the *same* point estimate of alpha carries three different
standard errors depending on what you are willing to assume about the residuals,
and that choice routinely decides whether alpha looks significant.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

PERIODS_PER_YEAR = 12
ALPHA_LABEL = "alpha"

# Display name -> (statsmodels cov_type, needs a lag length).
SE_TYPES: dict[str, tuple[str, bool]] = {
    "Classical (OLS)": ("nonrobust", False),
    "White (HC1)": ("HC1", False),
    "Newey-West (HAC)": ("HAC", True),
}
DEFAULT_SE_TYPE = "Newey-West (HAC)"


def default_hac_lags(n_obs: int) -> int:
    """Newey-West (1994) plug-in lag length: floor(4 * (n / 100) ** (2/9)).

    A rule of thumb rather than a truth. It is the common default in the
    literature, which is exactly why the app exposes the lag as a control: the
    reader should see that the HAC standard error moves with a choice the
    analyst makes.
    """
    if n_obs <= 0:
        return 0
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


@dataclass(frozen=True)
class FactorRegression:
    """One fitted factor regression, plus the metadata needed to describe it."""

    model: str
    se_type: str
    hac_lags: int | None
    frame: pd.DataFrame
    results: RegressionResultsWrapper

    @property
    def regressors(self) -> list[str]:
        return [column for column in self.frame.columns if column != "excess"]

    @property
    def n_obs(self) -> int:
        return int(self.results.nobs)

    @property
    def alpha_monthly(self) -> float:
        return float(self.results.params.iloc[0])

    @property
    def alpha_annualized(self) -> float:
        """Monthly intercept scaled by 12, the convention in the factor literature."""
        return self.alpha_monthly * PERIODS_PER_YEAR

    @property
    def alpha_tstat(self) -> float:
        return float(self.results.tvalues.iloc[0])

    @property
    def alpha_pvalue(self) -> float:
        return float(self.results.pvalues.iloc[0])

    @property
    def adj_r_squared(self) -> float:
        return float(self.results.rsquared_adj)

    @property
    def r_squared(self) -> float:
        return float(self.results.rsquared)

    @property
    def residuals(self) -> pd.Series:
        return self.results.resid

    @property
    def fitted(self) -> pd.Series:
        return self.results.fittedvalues


def _design(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Split an aligned frame into the dependent series and a design matrix with an intercept."""
    if "excess" not in frame.columns:
        raise ValueError("regression frame must contain an 'excess' column")
    y = frame["excess"]
    exog = frame.drop(columns="excess")
    if exog.empty:
        raise ValueError("regression frame contains no regressors")
    X = sm.add_constant(exog, prepend=True)
    return y, X.rename(columns={"const": ALPHA_LABEL})


def fit(
    frame: pd.DataFrame,
    model: str = "custom",
    se_type: str = DEFAULT_SE_TYPE,
    hac_lags: int | None = None,
) -> FactorRegression:
    """Fit `excess ~ factors` under one standard-error assumption.

    The point estimates are identical across `se_type`; only the standard errors,
    t-statistics, p-values and confidence intervals change.
    """
    if se_type not in SE_TYPES:
        raise KeyError(f"unknown se_type {se_type!r}; expected one of {list(SE_TYPES)}")

    y, X = _design(frame)
    cov_type, needs_lags = SE_TYPES[se_type]

    if needs_lags:
        lags = default_hac_lags(len(y)) if hac_lags is None else hac_lags
        results = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds={"maxlags": lags})
    else:
        lags = None
        results = sm.OLS(y, X).fit(cov_type=cov_type)

    return FactorRegression(
        model=model,
        se_type=se_type,
        hac_lags=lags,
        frame=frame,
        results=results,
    )


def coefficient_table(regression: FactorRegression, confidence: float = 0.95) -> pd.DataFrame:
    """Tidy per-coefficient inference table: estimate, SE, t, p, and a confidence interval."""
    results = regression.results
    intervals = results.conf_int(alpha=1 - confidence)
    lower_label = f"ci_lower_{confidence:.0%}"
    upper_label = f"ci_upper_{confidence:.0%}"

    table = pd.DataFrame(
        {
            "estimate": results.params,
            "std_error": results.bse,
            "t_stat": results.tvalues,
            "p_value": results.pvalues,
            lower_label: intervals.iloc[:, 0],
            upper_label: intervals.iloc[:, 1],
        }
    )
    table.index.name = "term"
    return table


def compare_standard_errors(
    frame: pd.DataFrame,
    hac_lags: int | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """The same regression under every standard-error assumption, side by side.

    Estimates repeat across the blocks by construction. The column worth reading
    is `t_stat`: serial correlation in residuals inflates classical t-statistics,
    so a HAC t-statistic below its classical counterpart is the normal case, and
    alpha crossing the conventional 1.96 threshold in one block but not another
    is the whole lesson of the demo.
    """
    blocks = {}
    for se_type in SE_TYPES:
        regression = fit(frame, se_type=se_type, hac_lags=hac_lags)
        table = coefficient_table(regression, confidence=confidence)
        blocks[se_type] = table[["estimate", "std_error", "t_stat", "p_value"]]

    combined = pd.concat(blocks, axis=1)
    combined.columns.names = ["standard_errors", "statistic"]
    return combined


def _restriction_matrix(regressor_names: list[str], restricted: list[str]) -> np.ndarray:
    """Rows selecting the coefficients to be jointly tested against zero."""
    matrix = np.zeros((len(restricted), len(regressor_names)))
    for row, name in enumerate(restricted):
        matrix[row, regressor_names.index(name)] = 1.0
    return matrix


@dataclass(frozen=True)
class NestedComparison:
    """Result of testing whether the regressors added by a richer model earn their place."""

    small_model: str
    large_model: str
    added: list[str]
    small_adj_r_squared: float
    large_adj_r_squared: float
    f_stat: float
    p_value: float
    df_num: int
    df_denom: int
    small_alpha_annualized: float
    large_alpha_annualized: float
    small_alpha_tstat: float
    large_alpha_tstat: float
    n_obs: int

    @property
    def incremental_adj_r_squared(self) -> float:
        return self.large_adj_r_squared - self.small_adj_r_squared


def compare_nested(
    frame: pd.DataFrame,
    small_regressors: list[str],
    small_model: str = "restricted",
    large_model: str = "full",
    se_type: str = DEFAULT_SE_TYPE,
    hac_lags: int | None = None,
) -> NestedComparison:
    """Test the regressors that the larger model adds, jointly, against zero.

    Both models are fitted on the rows of `frame`, so the comparison is on an
    identical sample; refitting the smaller model on its own longer history
    would make the test invalid. When `se_type` is robust the test statistic is
    a robust Wald statistic rather than a textbook F, which is the correct
    counterpart to the standard errors being reported alongside it.
    """
    large_regressors = [column for column in frame.columns if column != "excess"]
    unknown = [name for name in small_regressors if name not in large_regressors]
    if unknown:
        raise ValueError(f"{unknown} are not regressors of the larger model, so the models are not nested")

    added = [name for name in large_regressors if name not in small_regressors]
    if not added:
        raise ValueError("the two models have identical regressors, so there is nothing to test")

    large = fit(frame, model=large_model, se_type=se_type, hac_lags=hac_lags)
    small = fit(
        frame[["excess", *small_regressors]],
        model=small_model,
        se_type=se_type,
        hac_lags=hac_lags,
    )

    design_names = [ALPHA_LABEL, *large_regressors]
    test = large.results.f_test(_restriction_matrix(design_names, added))

    return NestedComparison(
        small_model=small_model,
        large_model=large_model,
        added=added,
        small_adj_r_squared=small.adj_r_squared,
        large_adj_r_squared=large.adj_r_squared,
        f_stat=float(np.squeeze(test.fvalue)),
        p_value=float(np.squeeze(test.pvalue)),
        df_num=len(added),
        df_denom=int(large.results.df_resid),
        small_alpha_annualized=small.alpha_annualized,
        large_alpha_annualized=large.alpha_annualized,
        small_alpha_tstat=small.alpha_tstat,
        large_alpha_tstat=large.alpha_tstat,
        n_obs=large.n_obs,
    )
