import numpy as np
import pytest

from conftest import TRUE_ALPHA, TRUE_BETAS, ar1_residuals, make_frame
from factor_lab import regression


def test_recovers_known_alpha_and_betas(clean_frame):
    """Every coefficient should land within three standard errors of the truth.

    Tolerances are stated in standard errors rather than absolute numbers so the
    test measures the estimator against its own stated precision.
    """
    fitted = regression.fit(clean_frame, model="FF3", se_type="Classical (OLS)")

    assert fitted.alpha_monthly == pytest.approx(TRUE_ALPHA, abs=0.001)
    for name, true_beta in TRUE_BETAS.items():
        standard_error = fitted.results.bse[name]
        assert abs(fitted.results.params[name] - true_beta) < 3 * standard_error
    assert fitted.n_obs == len(clean_frame)


def test_alpha_confidence_intervals_have_close_to_nominal_coverage():
    """A 95% interval should contain the truth about 95% of the time.

    Checking coverage on a single dataset tests the seed, not the code: a correct
    95% interval misses one time in twenty by design. Repeating over independent
    samples tests the actual claim the interval makes.
    """
    trials = 300
    covered = 0
    for seed in range(trials):
        frame = make_frame(np.random.default_rng(seed), demean_residuals=False)
        fitted = regression.fit(frame, se_type="Classical (OLS)")
        row = regression.coefficient_table(fitted).loc[regression.ALPHA_LABEL]
        if row["ci_lower_95%"] <= TRUE_ALPHA <= row["ci_upper_95%"]:
            covered += 1

    coverage = covered / trials
    # Three binomial standard errors around 0.95 at n=300 is roughly +/- 0.038.
    assert 0.91 <= coverage <= 0.99, f"coverage was {coverage:.1%}"


def test_annualized_alpha_scales_the_monthly_intercept(clean_frame):
    fitted = regression.fit(clean_frame)
    assert fitted.alpha_annualized == pytest.approx(fitted.alpha_monthly * 12)


def test_point_estimates_are_identical_across_standard_error_types(clean_frame):
    """Only the uncertainty changes; if the estimates move, the comparison is meaningless."""
    comparison = regression.compare_standard_errors(clean_frame)
    estimates = comparison.xs("estimate", axis=1, level="statistic")

    first = estimates.iloc[:, 0]
    for column in estimates.columns[1:]:
        np.testing.assert_allclose(estimates[column], first, rtol=1e-10)


def test_hac_standard_errors_exceed_classical_under_autocorrelation(rng):
    """The demo's central claim, so it gets a test.

    With positively autocorrelated residuals, classical standard errors are too
    small and the classical t-statistic on alpha is correspondingly too large.
    """
    frame = make_frame(rng, residuals=ar1_residuals(rng, rho=0.7))

    classical = regression.fit(frame, se_type="Classical (OLS)")
    hac = regression.fit(frame, se_type="Newey-West (HAC)", hac_lags=6)

    classical_se = classical.results.bse[regression.ALPHA_LABEL]
    hac_se = hac.results.bse[regression.ALPHA_LABEL]

    assert hac_se > classical_se
    assert abs(hac.alpha_tstat) < abs(classical.alpha_tstat)


def test_hac_lag_default_follows_newey_west_rule():
    # floor(4 * (600 / 100) ** (2/9)) == 5
    assert regression.default_hac_lags(600) == 5
    assert regression.default_hac_lags(0) == 0


def test_nested_test_does_not_reject_an_irrelevant_added_factor(rng):
    """A factor with a true coefficient of zero should not earn its place."""
    betas = {**TRUE_BETAS, "CMA": 0.0}
    frame = make_frame(rng, betas=betas)

    comparison = regression.compare_nested(
        frame,
        small_regressors=list(TRUE_BETAS),
        se_type="Classical (OLS)",
    )

    assert comparison.added == ["CMA"]
    assert comparison.p_value > 0.05
    assert comparison.incremental_adj_r_squared < 0.01


def test_nested_test_rejects_a_genuinely_useful_added_factor(rng):
    betas = {**TRUE_BETAS, "RMW": 0.60}
    frame = make_frame(rng, betas=betas)

    comparison = regression.compare_nested(
        frame,
        small_regressors=list(TRUE_BETAS),
        se_type="Classical (OLS)",
    )

    assert comparison.added == ["RMW"]
    assert comparison.p_value < 0.01
    assert comparison.incremental_adj_r_squared > 0.01
    assert comparison.df_num == 1


def test_nested_test_uses_one_sample_for_both_models(rng):
    frame = make_frame(rng, betas={**TRUE_BETAS, "CMA": 0.0})
    comparison = regression.compare_nested(frame, small_regressors=list(TRUE_BETAS))

    assert comparison.n_obs == len(frame)


def test_nested_test_rejects_non_nested_models(clean_frame):
    with pytest.raises(ValueError, match="not nested"):
        regression.compare_nested(clean_frame, small_regressors=["Mkt-RF", "NotAFactor"])


def test_nested_test_rejects_identical_models(clean_frame):
    with pytest.raises(ValueError, match="nothing to test"):
        regression.compare_nested(clean_frame, small_regressors=list(TRUE_BETAS))


def test_unknown_standard_error_type_is_rejected(clean_frame):
    with pytest.raises(KeyError):
        regression.fit(clean_frame, se_type="bootstrapped")


def test_frame_without_excess_column_is_rejected(clean_frame):
    with pytest.raises(ValueError, match="excess"):
        regression.fit(clean_frame.drop(columns="excess"))
