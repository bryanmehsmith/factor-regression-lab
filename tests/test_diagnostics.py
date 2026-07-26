import numpy as np
import pytest

from conftest import N_MONTHS, TRUE_BETAS, ar1_residuals, make_frame
from factor_lab import diagnostics, regression


def _find(tests, fragment):
    return next(test for test in tests if fragment.lower() in test.name.lower())


def _heteroskedastic_frame(rng):
    """Residual spread rising with the market return, so constant variance is false.

    The pattern is monotone in the signed factor rather than its absolute value
    because Breusch-Pagan regresses squared residuals on the regressors linearly;
    a symmetric V-shape in |Mkt-RF| is real heteroskedasticity that this
    particular test is blind to.
    """
    frame = make_frame(rng, residuals=np.zeros(N_MONTHS))
    scale = 0.02 + 0.6 * (frame["Mkt-RF"] - frame["Mkt-RF"].min())
    frame["excess"] = frame["excess"] + rng.normal(0, 1, len(frame)) * scale
    return frame


def test_breusch_pagan_detects_injected_heteroskedasticity(rng):
    frame = _heteroskedastic_frame(rng)

    fitted = regression.fit(frame, se_type="Classical (OLS)")
    test = _find(diagnostics.run_all(fitted), "breusch-pagan")

    assert test.p_value < 0.05
    assert test.concerning


def test_breusch_pagan_passes_on_constant_variance_residuals(clean_frame):
    fitted = regression.fit(clean_frame, se_type="Classical (OLS)")
    test = _find(diagnostics.run_all(fitted), "breusch-pagan")

    assert test.p_value > 0.05
    assert not test.concerning


def test_robust_standard_errors_differ_from_classical_under_heteroskedasticity(rng):
    frame = _heteroskedastic_frame(rng)

    classical = regression.fit(frame, se_type="Classical (OLS)")
    white = regression.fit(frame, se_type="White (HC1)")

    classical_se = classical.results.bse["Mkt-RF"]
    white_se = white.results.bse["Mkt-RF"]

    assert white_se != pytest.approx(classical_se, rel=0.05)


def test_ljung_box_and_durbin_watson_flag_autocorrelated_residuals(rng):
    frame = make_frame(rng, residuals=ar1_residuals(rng, rho=0.7))
    fitted = regression.fit(frame, se_type="Classical (OLS)")
    tests = diagnostics.run_all(fitted)

    ljung_box = _find(tests, "ljung-box")
    durbin_watson = _find(tests, "durbin-watson")

    assert ljung_box.p_value < 0.05
    assert ljung_box.concerning
    assert durbin_watson.statistic < 1.5
    assert durbin_watson.concerning


def test_diagnostics_pass_on_well_behaved_residuals(clean_frame):
    fitted = regression.fit(clean_frame, se_type="Classical (OLS)")
    tests = diagnostics.run_all(fitted)

    assert not _find(tests, "ljung-box").concerning
    assert not _find(tests, "durbin-watson").concerning
    assert not _find(tests, "jarque-bera").concerning


def test_vif_flags_near_collinear_regressors(rng):
    """A duplicated factor plus noise: the fit is fine, the individual betas are not readable."""
    frame = make_frame(rng)
    frame["HML_twin"] = frame["HML"] + rng.normal(0, 0.0005, len(frame))

    table = diagnostics.variance_inflation_factors(frame).set_index("regressor")

    assert table.loc["HML", "vif"] > diagnostics.VIF_CONCERN
    assert table.loc["HML_twin", "vif"] > diagnostics.VIF_CONCERN
    assert table.loc["HML", "flag"] == "review"
    assert table.loc["Mkt-RF", "vif"] < diagnostics.VIF_CONCERN


def test_vif_is_near_one_for_independent_regressors(clean_frame):
    table = diagnostics.variance_inflation_factors(clean_frame)

    assert (table["vif"] < 1.5).all()
    assert (table["flag"] == "ok").all()


def test_vif_excludes_the_intercept(clean_frame):
    table = diagnostics.variance_inflation_factors(clean_frame)

    assert "const" not in table["regressor"].tolist()
    assert set(table["regressor"]) == set(TRUE_BETAS)


def test_factor_correlations_are_square_over_regressors_only(clean_frame):
    correlations = diagnostics.factor_correlations(clean_frame)

    assert list(correlations.columns) == list(TRUE_BETAS)
    assert correlations.shape == (len(TRUE_BETAS), len(TRUE_BETAS))
    np.testing.assert_allclose(np.diag(correlations), 1.0)


def test_verdict_reports_a_real_alpha_as_significant(rng):
    frame = make_frame(rng, alpha=0.01)  # 1% a month is far too large to miss
    fitted = regression.fit(frame)
    tests = diagnostics.run_all(fitted)

    verdict = diagnostics.alpha_verdict(fitted, tests)

    assert "is statistically distinguishable from zero" in verdict


def test_verdict_reports_a_zero_alpha_as_insignificant(rng):
    frame = make_frame(rng, alpha=0.0)
    fitted = regression.fit(frame, model="FF3")
    tests = diagnostics.run_all(fitted)

    verdict = diagnostics.alpha_verdict(fitted, tests)

    assert "not statistically distinguishable from zero" in verdict


def test_verdict_warns_when_classical_errors_are_quoted_despite_autocorrelation(rng):
    frame = make_frame(rng, alpha=0.01, residuals=ar1_residuals(rng, rho=0.8))
    fitted = regression.fit(frame, se_type="Classical (OLS)")
    tests = diagnostics.run_all(fitted)

    verdict = diagnostics.alpha_verdict(fitted, tests)

    assert "Newey-West" in verdict


def test_diagnostics_table_has_one_row_per_test(clean_frame):
    fitted = regression.fit(clean_frame)
    tests = diagnostics.run_all(fitted)
    table = diagnostics.diagnostics_table(tests)

    assert len(table) == len(tests)
    assert set(table.columns) == {"test", "statistic", "p_value", "null_hypothesis", "flag"}
