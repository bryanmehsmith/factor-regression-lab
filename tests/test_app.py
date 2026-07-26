"""Smoke tests for the Streamlit app via Streamlit's own headless harness.

The app script is where the modules are actually wired together, so a unit-tested
library can still ship a broken page. These run the real script and assert it
renders without raising, across the parameter combinations most likely to break:
the simplest model (no nested comparison available), the richest model, and a
short sample period.

Marked `slow` because each run re-executes the whole script.
"""

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = "app/streamlit_app.py"
TIMEOUT = 120

pytestmark = pytest.mark.slow


def _run(**widget_values) -> AppTest:
    """Render the app, optionally change sidebar widgets by key, and render again."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    app.run()
    assert not app.exception, f"app raised on first render: {[e.value for e in app.exception]}"

    for key, value in widget_values.items():
        app.session_state[key] = value
    if widget_values:
        app.run()
        assert not app.exception, f"app raised after setting {widget_values}: {[e.value for e in app.exception]}"

    return app


def test_app_renders_with_defaults():
    app = _run()

    assert not app.exception
    assert any("Factor Regression Lab" in str(title.value) for title in app.title)
    # Four headline metrics plus the nested-comparison trio.
    assert len(app.metric) >= 4


def test_app_renders_the_simplest_model():
    """CAPM has no simpler model to test against, so section 3 must not blow up."""
    app = _run(model="CAPM")

    assert not app.exception
    assert any("simplest model available" in str(caption.value) for caption in app.caption)


def test_app_renders_the_richest_model():
    app = _run(model="FF5+Mom")

    assert not app.exception


def test_app_renders_under_classical_standard_errors():
    app = _run(se_type="Classical (OLS)")

    assert not app.exception


def test_app_renders_with_the_longest_rolling_window():
    app = _run(window=180)

    assert not app.exception
