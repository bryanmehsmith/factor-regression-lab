"""Synthetic factor data with known parameters.

Every test builds returns from coefficients it chose itself, so a failure means
the wiring is wrong rather than that the market changed.
"""

import numpy as np
import pandas as pd
import pytest

N_MONTHS = 600
TRUE_ALPHA = 0.004  # 0.4% a month, roughly 4.8% a year
TRUE_BETAS = {"Mkt-RF": 1.10, "SMB": 0.35, "HML": -0.20}


def month_end_index(n_months: int = N_MONTHS) -> pd.DatetimeIndex:
    return pd.date_range("1975-01-31", periods=n_months, freq="ME")


def make_factors(rng: np.random.Generator, n_months: int = N_MONTHS) -> pd.DataFrame:
    """Independent, roughly realistic monthly factor returns.

    Covers every factor in `data.FACTOR_MODELS` so tests can add a regressor to a
    model and check whether it earns its place.
    """
    index = month_end_index(n_months)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.006, 0.045, n_months),
            "SMB": rng.normal(0.002, 0.030, n_months),
            "HML": rng.normal(0.003, 0.030, n_months),
            "RMW": rng.normal(0.003, 0.025, n_months),
            "CMA": rng.normal(0.003, 0.020, n_months),
            "Mom": rng.normal(0.006, 0.045, n_months),
        },
        index=index,
    )


def make_frame(
    rng: np.random.Generator,
    residuals: np.ndarray | None = None,
    alpha: float = TRUE_ALPHA,
    betas: dict[str, float] | None = None,
    n_months: int = N_MONTHS,
    demean_residuals: bool = True,
) -> pd.DataFrame:
    """A regression frame whose `excess` column is built from known parameters.

    Residuals are demeaned by default so the fitted intercept recovers `alpha`
    almost exactly. Without it, the sample mean of the noise is itself a random
    variable of roughly 0.0008, so a test asserting "alpha comes back as 0.004"
    would pass or fail on the seed rather than on the code. Tests that are
    specifically about sampling uncertainty pass `demean_residuals=False`.
    """
    betas = TRUE_BETAS if betas is None else betas
    factors = make_factors(rng, n_months)[list(betas)]
    if residuals is None:
        residuals = rng.normal(0, 0.02, n_months)
    if demean_residuals:
        residuals = residuals - residuals.mean()

    excess = alpha + factors.mul(pd.Series(betas)).sum(axis=1) + residuals

    frame = pd.DataFrame({"excess": excess})
    for name in betas:
        frame[name] = factors[name]
    return frame


def ar1_residuals(rng: np.random.Generator, rho: float = 0.7, n_months: int = N_MONTHS) -> np.ndarray:
    """Positively autocorrelated errors, the case Newey-West exists to handle."""
    shocks = rng.normal(0, 0.02, n_months)
    series = np.zeros(n_months)
    series[0] = shocks[0]
    for i in range(1, n_months):
        series[i] = rho * series[i - 1] + shocks[i]
    return series


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260726)


@pytest.fixture
def clean_frame(rng: np.random.Generator) -> pd.DataFrame:
    return make_frame(rng)
