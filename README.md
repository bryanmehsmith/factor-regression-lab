# Factor Regression Lab

Factor regression and statistical inference on US equity returns. Factors come from the Ken French Data Library; the test asset is either one of French's 17 industry portfolios or any Yahoo Finance ticker.

The demo answers one question as carefully as it can: **when an asset beats the market, is that skill, or is it factor exposure, and is whatever remains statistically distinguishable from zero?**

This README builds the two ideas the demo depends on, in order: **section 1** is why alpha only makes sense relative to a chosen model, **section 2** is why the standard-error method you pick can decide whether that alpha is real, and **section 3** is an honest list of what none of this claims to do. Each section opens with a plain paragraph; the formulas behind it are folded under **"Show the full derivation"** for when you want the rigor instead of the intuition. The [interactive frontend](frontend/index.html) walks through the same two ideas, then lets you actually run the regression and steps through the output in ten numbered sections of its own.

## 1. Why alpha only means something relative to a model

"This fund returned 14% a year" is not a finding. The question is always 14% *compared to what*, and the honest comparison is not a single index but a set of systematic risks the investor could have taken deliberately and cheaply. That comparison is a regression: take an asset's return in excess of the risk-free rate, and explain it with the returns of portfolios built to capture known systematic exposures. **Alpha, the intercept, is the average monthly return the factors fail to explain.** It is the only part that is a candidate for skill, and it is defined entirely by which factors sit on the right-hand side. Add a factor that genuinely drives the asset, and alpha shrinks; the "skill" was an exposure you had not named yet.

This is why the demo lets you switch between nested models:

- **CAPM**: the market alone. Everything else is alpha.
- **FF3**: adds size (SMB, small minus big) and value (HML, high minus low book-to-market).
- **FF5**: adds profitability (RMW, robust minus weak) and investment (CMA, conservative minus aggressive).
- **FF5+Mom**: adds momentum, the tendency of recent winners to keep winning.

Watching alpha decay as factors are added is the single most useful thing here. It is also a warning: with enough factors you can explain away almost any track record, so "controlling for" more is not automatically more rigorous.

<details>
<summary>Show the full derivation</summary>

```
excess_return = alpha + b1 * (Mkt-RF) + b2 * SMB + b3 * HML + ... + residual
```

Each beta says how much of the asset's behaviour is that exposure.

A note on construction. Mkt-RF, SMB and HML all come from the 5-factor file rather than the standalone 3-factor file, which defines SMB slightly differently and starts in a different year. Taking them from one source keeps FF3 a genuine subset of FF5 over an identical sample, which is what makes the nested test in the app's section 6 a real test rather than an apples-to-oranges comparison. Momentum has no such conflict to avoid, so it is sourced from its own separate French file.

</details>

## 2. Why the standard error decides the answer

Getting alpha is easy. Deciding whether it is distinguishable from zero is where the work is, and it depends on assumptions about the residuals that are rarely stated out loud. The point estimate never changes. What changes is the standard error, and with it the t-statistic. The app's **section 3** lets you choose which of the three below to trust for the headline number, and **section 5** puts all three side by side so you can watch the gap.

<details>
<summary>Show the full derivation</summary>

- **Classical (OLS)** assumes residuals are independent of each other and have constant variance. Under those assumptions it is the most efficient choice. Asset returns violate both.
- **White (HC1)** drops constant variance. Volatility clusters in markets, so residual spread is genuinely larger in some periods than others; this correction accounts for it.
- **Newey-West (HAC)** additionally allows residuals to be correlated across months. This matters most, because positive residual autocorrelation makes the several hundred monthly observations behave like far fewer independent ones. Classical standard errors treat every month as fresh evidence, understate uncertainty, and inflate the t-statistic accordingly.

Each step weakens an assumption, and usually widens the interval. **An alpha whose t-statistic clears 1.96 under classical errors but not under Newey-West was never significant; the first number was an artifact of assuming away serial correlation.** Section 5 of the app puts all three side by side for exactly this reason, and the residual diagnostics in section 7 tell you which one you are entitled to quote.

The HAC lag length is exposed as a control rather than hidden, because it is a judgment call. The common default is the Newey-West plug-in rule, `floor(4 * (n / 100) ** (2/9))`, but nothing makes it correct. Moving the slider and watching the t-statistic drift is the honest version of that story.

Two more things the app makes visible, both of which are routinely skipped:

- **Multicollinearity.** Correlated factors inflate each other's standard errors. The variance inflation factor quantifies it. High VIF does not bias the estimates or hurt the fit, which is how a model can be jointly significant with no individually significant coefficient.
- **Parameter stability.** A full-sample regression reports one beta and implies it held for sixty years. Re-estimating over rolling windows usually shows something much less stable. Utilities, for instance, show a market beta wandering between roughly 0.3 and 1.0 depending on the window.

</details>

## 3. What this does not do

- Alpha here is gross of costs, fees, and taxes. Real implementation would consume much of it.
- The industry portfolios and bundled ticker catalog are US only, because that is what French publishes and what the catalog was seeded with; the free-text ticker box will take any Yahoo Finance symbol, but there is no comparable free factor library for non-US markets like the JSE to properly evaluate one.
- The factors are treated as known and error-free. They are themselves estimated portfolios.
- Statistical significance is not economic significance, and this is a single regression on a single asset with no correction for the fact that you can click through dozens of them looking for a low p-value. That multiple-comparisons problem deserves its own demo.

## Setup

```bash
uv sync
```

## Usage

The interactive version of this is a static JS frontend (`frontend/`), not a
Streamlit app: it ports `regression.py`/`diagnostics.py`/`rolling.py` directly
into JS (`frontend/js/modules/`) and runs client-side against bundled monthly
snapshots. Open `frontend/index.html` through any static file server, e.g.:

```bash
cd frontend && python -m http.server
```

`app/api.py` is a small companion JSON API, used only for live French-data
refresh and free-text Yahoo Finance ticker lookups:

```bash
uv run python app/api.py --port=8000
# GET http://127.0.0.1:8000/factors?refresh=true
# GET http://127.0.0.1:8000/industries?refresh=true
# GET http://127.0.0.1:8000/ticker?symbol=AAPL
```

Print the same analysis as a text report:

```bash
uv run scripts/run_regression.py --asset Utils --model FF5
uv run scripts/run_regression.py --asset SPY --yfinance --model FF3
```

Run tests:

```bash
uv run pytest
```

Regenerate the public frontend's deterministic synthetic datasets:

```bash
uv run scripts/generate_synthetic_frontend_data.py
```

The JS port is checked against the Python reference implementation before
being trusted, not just eyeballed: `scripts/export_regression_fixtures.py`
fits a battery of synthetic frames (clean, autocorrelated, heteroskedastic,
varying sample sizes) with the real `regression.py`/`diagnostics.py`/`rolling.py`,
and `scripts/verify_stats_parity.mjs` recomputes everything in JS and diffs.
`scripts/export_dist_reference.py` + `scripts/verify_distributions.mjs` do the
same for the hand-rolled t/F/chi-square distribution functions against scipy.
Re-run both whenever either side changes:

```bash
uv run scripts/export_regression_fixtures.py
uv run scripts/export_dist_reference.py
cd scripts && node verify_stats_parity.mjs && node verify_distributions.mjs && cd ..
```

## Data

The public frontend defaults to deterministic synthetic factors and industry
returns generated by `scripts/generate_synthetic_frontend_data.py`. They are
designed to exercise the regression, diagnostics, and rolling-window mechanics,
not to reproduce market history. Live refresh requests factor and industry data
from the [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Individual live tickers come from Yahoo Finance
through `yfinance`, resampled to monthly total returns.

Both paths degrade rather than crash: fresh cache, then live download, then stale cache, then the bundled parquet snapshots in `assets/`. Live refresh is off by default in the UI, since Dartmouth and Yahoo both throttle.

## Project layout

- `src/factor_lab/data.py` - French CSV parsing, the caching and fallback chain, yfinance loading, and factor/asset alignment
- `src/factor_lab/regression.py` - OLS fitting, the coefficient table, the three-way standard-error comparison, and nested model tests
- `src/factor_lab/diagnostics.py` - Breusch-Pagan, Durbin-Watson, Ljung-Box, Jarque-Bera, VIF, each with a plain-language reading
- `src/factor_lab/rolling.py` - rolling-window estimation and a stability summary
- `src/factor_lab/plots.py` - matplotlib figures (used by the CLI report; the frontend has its own SVG charts)
- `src/factor_lab/cli.py` - the text report
- `app/api.py` - tiny JSON API backing the frontend's live-refresh and ticker lookup
- `frontend/` - the static JS demo: the above three `factor_lab` modules ported to `frontend/js/modules/{regression,diagnostics,rolling}.js`, plus hand-rolled t/F/chi-square distribution functions in `distributions.js`
- `scripts/generate_synthetic_frontend_data.py` - generates the public frontend's deterministic synthetic datasets
- `scripts/export_snapshot_json.py` - exports research snapshots for local personal use
- `scripts/export_regression_fixtures.py` / `export_dist_reference.py` + `scripts/verify_stats_parity.mjs` / `verify_distributions.mjs` - the JS-vs-Python parity check described above
- `tests/` - synthetic-data tests with known parameters, including a coverage simulation for the confidence intervals and a check that HAC standard errors exceed classical ones under injected autocorrelation
- `data/` - gitignored cache of downloaded data

## Scope

The coding, architecture, and infrastructure choices here are proof-of-concept appropriate, not indicative of an enterprise setup; a company environment would warrant a different strategy. For research and education only, not investment advice. Personal project using public data only; not affiliated with or representative of my employer.
