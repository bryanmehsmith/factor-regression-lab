import pandas as pd
import pytest

from conftest import TRUE_ALPHA, TRUE_BETAS, make_frame
from factor_lab import regression, rolling


def test_rolling_estimates_cover_every_term_including_alpha(clean_frame):
    estimates = rolling.rolling_estimates(clean_frame, window=60)

    assert set(estimates) == {regression.ALPHA_LABEL, *TRUE_BETAS}
    for frame in estimates.values():
        assert list(frame.columns) == ["estimate", "lower", "upper"]


def test_rolling_windows_start_after_the_first_full_window(clean_frame):
    window = 60
    estimates = rolling.rolling_estimates(clean_frame, window=window)

    beta = estimates["Mkt-RF"]
    assert len(beta) == len(clean_frame) - window + 1
    assert beta.index[0] == clean_frame.index[window - 1]


def test_rolling_bands_bracket_the_estimate(clean_frame):
    estimates = rolling.rolling_estimates(clean_frame, window=60)

    for frame in estimates.values():
        assert (frame["lower"] <= frame["estimate"]).all()
        assert (frame["estimate"] <= frame["upper"]).all()


def test_rolling_estimates_match_a_direct_fit_on_the_same_window(clean_frame):
    """The first window must reproduce a standalone OLS fit on those same months.

    This pins the rolling machinery down exactly, rather than asserting that a
    noisy average of overlapping windows lands near the true parameter.
    """
    window = 120
    estimates = rolling.rolling_estimates(clean_frame, window=window)
    direct = regression.fit(clean_frame.iloc[:window], se_type="Classical (OLS)")

    for name, value in direct.results.params.items():
        scale = 12 if name == regression.ALPHA_LABEL else 1
        rolled = estimates[name]["estimate"].iloc[0]
        assert rolled == pytest.approx(value * scale, rel=1e-8)


def test_rolling_betas_are_stable_on_stable_data(clean_frame):
    """Data generated from a constant beta should not show a drifting rolling beta."""
    estimates = rolling.rolling_estimates(clean_frame, window=120)

    beta = estimates["Mkt-RF"]["estimate"]
    assert beta.std() < 0.15
    assert (beta - TRUE_BETAS["Mkt-RF"]).abs().max() < 0.5


def test_rolling_detects_a_structural_break_in_beta(rng):
    """A beta that genuinely changes halfway should show up as a wide rolling range."""
    first = make_frame(rng, betas={"Mkt-RF": 0.4}, n_months=300)
    second = make_frame(rng, betas={"Mkt-RF": 1.6}, n_months=300)
    second.index = second.index + pd.DateOffset(years=25)
    frame = pd.concat([first, second])

    estimates = rolling.rolling_estimates(frame, window=60)
    summary = rolling.stability_summary(estimates)

    assert summary.loc["Mkt-RF", "range"] > 0.8


def test_stability_summary_reports_sign_changes(clean_frame):
    estimates = rolling.rolling_estimates(clean_frame, window=60)
    summary = rolling.stability_summary(estimates)

    assert set(summary.index) == {regression.ALPHA_LABEL, *TRUE_BETAS}
    # A true beta of 1.10 should never flip sign in any window.
    assert not summary.loc["Mkt-RF", "changed_sign"]


def test_window_longer_than_the_sample_is_rejected(clean_frame):
    with pytest.raises(ValueError, match="exceeds"):
        rolling.rolling_estimates(clean_frame, window=len(clean_frame) + 1)


def test_absurdly_short_window_is_rejected(clean_frame):
    with pytest.raises(ValueError, match="at least 12"):
        rolling.rolling_estimates(clean_frame, window=3)
