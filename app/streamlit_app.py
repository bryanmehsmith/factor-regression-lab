"""Interactive factor regression lab: is an asset's excess return alpha, or factor exposure?"""

import pandas as pd
import streamlit as st

from factor_lab import data, diagnostics, plots, regression, rolling

st.set_page_config(page_title="Factor Regression Lab", layout="wide")

INDUSTRY_SOURCE = "French industry portfolios"
TICKER_SOURCE = "Yahoo Finance ticker"


@st.cache_data(ttl=3600, show_spinner="Loading factor returns...")
def load_factors(force_refresh: bool) -> pd.DataFrame:
    return data.load_factors(force_refresh=force_refresh)


@st.cache_data(ttl=3600, show_spinner="Loading industry portfolios...")
def load_industries(force_refresh: bool) -> pd.DataFrame:
    return data.load_industry_portfolios(force_refresh=force_refresh)


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance...")
def load_ticker(ticker: str) -> pd.Series:
    return data.load_yfinance_asset(ticker)


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return data.load_test_asset_catalog()


st.title("Factor Regression Lab")
st.caption(
    "When an asset beats the market, is that skill or is it factor exposure? "
    "Regress its excess return on the Fama-French factors: the intercept is what "
    "the factors cannot explain, and the standard errors decide whether that "
    "intercept means anything."
)

with st.sidebar:
    st.header("Regression settings")

    # Widgets carry explicit keys so the app tests can address them by name
    # rather than by position in the element tree.
    source = st.radio("Test asset from", [INDUSTRY_SOURCE, TICKER_SOURCE], key="source")
    use_live = st.checkbox("Try a live data refresh", value=False, key="use_live")
    st.caption(
        "Off by default; falls back to a bundled snapshot if the data source is unavailable."
    )

    model = st.selectbox(
        "Factor model",
        list(data.FACTOR_MODELS),
        index=list(data.FACTOR_MODELS).index("FF5"),
        key="model",
        help="Each model's factors are a strict subset of the next, so they can be compared as nested models.",
    )
    se_type = st.selectbox(
        "Standard errors for the headline",
        list(regression.SE_TYPES),
        index=list(regression.SE_TYPES).index(regression.DEFAULT_SE_TYPE),
        key="se_type",
    )
    hac_lags = st.number_input(
        "HAC lag length (months)",
        min_value=0,
        max_value=36,
        value=6,
        key="hac_lags",
        help="Newey-West lag truncation. The rule of thumb is floor(4 * (n / 100) ** (2/9)).",
    )
    window = st.slider(
        "Rolling window (months)",
        min_value=24,
        max_value=180,
        value=60,
        step=12,
        key="window",
    )

try:
    factors = load_factors(use_live)
except Exception as exc:
    st.error(f"Could not load factor data: {exc}")
    st.stop()

if source == INDUSTRY_SOURCE:
    try:
        portfolios = load_industries(use_live)
    except Exception as exc:
        st.error(f"Could not load industry portfolios: {exc}")
        st.stop()
    with st.sidebar:
        asset_name = st.selectbox(
            "Industry portfolio", portfolios.columns.tolist(), index=0, key="industry"
        )
    asset_returns = portfolios[asset_name]
else:
    catalog = load_catalog()
    with st.sidebar:
        choice = st.selectbox(
            "Ticker",
            catalog["ticker"].tolist(),
            key="ticker",
            format_func=lambda ticker: f"{ticker} ({catalog.set_index('ticker').loc[ticker, 'name']})",
        )
        custom = st.text_input("Or type any ticker", value="", key="custom_ticker").strip().upper()
    asset_name = custom or choice
    try:
        asset_returns = load_ticker(asset_name)
    except Exception as exc:
        st.error(f"Could not load {asset_name} from Yahoo Finance: {exc}")
        st.info("Yahoo Finance rate-limits heavily. The industry portfolios need no live data.")
        st.stop()

frame = data.build_regression_frame(asset_returns, factors, model)
if frame.empty:
    st.warning("No overlapping months between this asset and the factor data.")
    st.stop()

with st.sidebar:
    years = sorted({timestamp.year for timestamp in frame.index})
    start_year, end_year = st.select_slider(
        "Sample period",
        options=years,
        value=(years[0], years[-1]),
        key="sample_period",
    )

frame = frame.loc[str(start_year) : f"{end_year}-12-31"]
minimum_months = len(data.FACTOR_MODELS[model]) + 24
if len(frame) < minimum_months:
    st.warning(
        f"Only {len(frame)} months in this sample; at least {minimum_months} are needed "
        f"to estimate {model} with any confidence. Widen the sample period."
    )
    st.stop()

fitted = regression.fit(frame, model=model, se_type=se_type, hac_lags=int(hac_lags))
tests = diagnostics.run_all(fitted)

st.subheader(f"{asset_name} on {model}")
st.caption(
    f"{frame.index[0]:%b %Y} to {frame.index[-1]:%b %Y}, {fitted.n_obs} monthly observations. "
    f"The dependent variable is the return in excess of the risk-free rate."
)

verdict = diagnostics.alpha_verdict(fitted, tests)
if fitted.alpha_pvalue < diagnostics.SIGNIFICANCE:
    st.success(verdict)
else:
    st.info(verdict)

first, second, third, fourth = st.columns(4)
first.metric("Annualized alpha", f"{fitted.alpha_annualized:.2%}")
second.metric(f"Alpha t-stat ({fitted.se_type.split(' ')[0]})", f"{fitted.alpha_tstat:.2f}")
third.metric("Adjusted R-squared", f"{fitted.adj_r_squared:.3f}")
fourth.metric("Months", f"{fitted.n_obs}")

st.divider()
st.subheader("1. Coefficients")
st.caption(
    "`alpha` is the monthly intercept: average excess return the factors leave unexplained. "
    "Each beta is the asset's exposure to that factor. The confidence interval is the range "
    "of values the data cannot rule out at 95%."
)
st.dataframe(
    regression.coefficient_table(fitted).style.format(
        {
            "estimate": "{:.4f}",
            "std_error": "{:.4f}",
            "t_stat": "{:.2f}",
            "p_value": "{:.4f}",
            "ci_lower_95%": "{:.4f}",
            "ci_upper_95%": "{:.4f}",
        }
    ),
    width="stretch",
)

st.divider()
st.subheader("2. The same regression, three assumptions about the residuals")
st.caption(
    "The estimates in these three blocks are identical; only the uncertainty around them "
    "changes. Classical standard errors assume residuals are independent with constant "
    "variance. White drops the constant-variance assumption. Newey-West additionally allows "
    "residuals to be correlated across months. Each weaker assumption is more honest and "
    "usually widens the interval, so a t-statistic that clears 1.96 in the first column and "
    "not the third was never really significant."
)
comparison = regression.compare_standard_errors(frame, hac_lags=int(hac_lags))
st.pyplot(plots.tstat_comparison(comparison))
with st.expander("Full comparison table"):
    st.dataframe(comparison.style.format("{:.4f}"), width="stretch")

st.divider()
st.subheader("3. Does the richer model earn its place?")
models = list(data.FACTOR_MODELS)
position = models.index(model)
if position == 0:
    st.caption(f"{model} is the simplest model available, so there is nothing simpler to test it against.")
else:
    smaller = models[position - 1]
    nested = regression.compare_nested(
        frame,
        small_regressors=data.FACTOR_MODELS[smaller],
        small_model=smaller,
        large_model=model,
        se_type=se_type,
        hac_lags=int(hac_lags),
    )
    st.caption(
        f"Adding {', '.join(nested.added)} to {smaller} raises R-squared mechanically, because "
        "extra regressors always do. The joint test asks whether the improvement is larger than "
        "chance would produce. Both models are fitted on the same months, which is what makes "
        "the comparison valid."
    )
    left, middle, right = st.columns(3)
    left.metric(
        "Joint test p-value",
        f"{nested.p_value:.4f}",
        help=f"F = {nested.f_stat:.2f} on {nested.df_num} and {nested.df_denom} degrees of freedom",
    )
    middle.metric(
        "Adjusted R-squared",
        f"{nested.large_adj_r_squared:.3f}",
        delta=f"{nested.incremental_adj_r_squared:+.3f} vs {smaller}",
    )
    right.metric(
        "Annualized alpha",
        f"{nested.large_alpha_annualized:.2%}",
        delta=f"{nested.large_alpha_annualized - nested.small_alpha_annualized:+.2%} vs {smaller}",
        delta_color="off",
    )
    if nested.p_value < diagnostics.SIGNIFICANCE:
        st.markdown(
            f"**{' and '.join(nested.added)} carry information {smaller} misses**, so the "
            f"alpha from {smaller} was partly just unmodelled factor exposure."
        )
    else:
        st.markdown(
            f"**{' and '.join(nested.added)} add nothing here.** The extra fit is within what "
            f"chance would produce, so {smaller} is the better description of this asset."
        )

st.divider()
st.subheader("4. Are the OLS assumptions actually met?")
st.caption(
    "These tests decide which column in section 2 deserves to be quoted. The residual plots "
    "show the same thing visually."
)
st.dataframe(
    diagnostics.diagnostics_table(tests).style.format({"statistic": "{:.3f}", "p_value": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)
for test in tests:
    with st.expander(f"{test.name} ({'review' if test.concerning else 'ok'})"):
        st.markdown(f"**Null hypothesis:** {test.null_hypothesis}")
        st.markdown(test.reading)
st.pyplot(plots.diagnostics_grid(fitted))

st.divider()
st.subheader("5. Can the individual betas be trusted?")
st.caption(
    "Correlated factors inflate each other's standard errors. A high VIF does not bias the "
    "estimates or spoil the fit; it means that particular coefficient is hard to pin down, "
    "which is how a model can be jointly strong with no individually significant factor."
)
vif_column, correlation_column = st.columns([1, 1])
with vif_column:
    st.dataframe(
        diagnostics.variance_inflation_factors(frame).style.format({"vif": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
with correlation_column:
    st.pyplot(plots.correlation_heatmap(diagnostics.factor_correlations(frame)))

st.divider()
st.subheader("6. Is one number per factor even the right summary?")
st.caption(
    f"Re-estimating over {window}-month windows shows how much the full-sample coefficients "
    "are averages of genuinely different regimes. Bands are indicative classical intervals."
)
if window > len(frame):
    st.warning(f"The rolling window ({window} months) is longer than the sample ({len(frame)} months).")
else:
    estimates = rolling.rolling_estimates(frame, window=window)
    summary = rolling.stability_summary(estimates)
    st.dataframe(
        summary.style.format({"min": "{:.3f}", "max": "{:.3f}", "range": "{:.3f}"}),
        width="stretch",
    )
    term = st.selectbox("Show rolling estimate for", list(estimates), key="rolling_term")
    full_sample = (
        fitted.alpha_annualized
        if term == regression.ALPHA_LABEL
        else float(fitted.results.params[term])
    )
    st.pyplot(plots.rolling_term(term, estimates[term], full_sample=full_sample))

st.divider()
st.subheader("7. What the model does and does not explain")
st.caption(
    "The gap between the two lines is cumulative alpha. It makes the intercept tangible, "
    "though a visible gap is not evidence on its own; that is what section 2 is for."
)
st.pyplot(plots.cumulative_fit(fitted))

st.caption(
    "Data: Ken French Data Library (Dartmouth) for factors and industry portfolios, "
    "Yahoo Finance via yfinance for individual tickers, with bundled fallback snapshots. "
    "For research and education only, not investment advice."
)
