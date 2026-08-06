# Factor Regression Lab

Factor regression and statistical inference on US equity returns. Factors come from the Ken French Data Library; the test asset is either one of French's 17 industry portfolios or any Yahoo Finance ticker.

The demo answers one question as carefully as it can: **when an asset beats the market, is that skill, or is it factor exposure, and is whatever remains statistically distinguishable from zero?**

## Why alpha only means something relative to a model

"This fund returned 14% a year" is not a finding. The question is always 14% *compared to what*, and the honest comparison is not a single index but a set of systematic risks the investor could have taken deliberately and cheaply.

That comparison is a regression. Take an asset's return in excess of the risk-free rate, and explain it with the returns of portfolios built to capture known systematic exposures:

```
excess_return = alpha + b1 * (Mkt-RF) + b2 * SMB + b3 * HML + ... + residual
```

Each beta says how much of the asset's behaviour is that exposure. **Alpha, the intercept, is the average monthly return the factors fail to explain.** It is the only part that is a candidate for skill, and it is defined entirely by which factors sit on the right-hand side. Add a factor that genuinely drives the asset, and alpha shrinks; the "skill" was an exposure you had not named yet.

This is why the demo lets you switch between nested models:

- **CAPM**: the market alone. Everything else is alpha.
- **FF3**: adds size (SMB, small minus big) and value (HML, high minus low book-to-market).
- **FF5**: adds profitability (RMW, robust minus weak) and investment (CMA, conservative minus aggressive).
- **FF5+Mom**: adds momentum, the tendency of recent winners to keep winning.

Watching alpha decay as factors are added is the single most useful thing here. It is also a warning: with enough factors you can explain away almost any track record, so "controlling for" more is not automatically more rigorous.

A note on construction. Mkt-RF, SMB and HML all come from the 5-factor file rather than the standalone 3-factor file, which defines SMB slightly differently and starts in a different year. Taking them from one source keeps FF3 a genuine subset of FF5 over an identical sample, which is what makes the nested test in section 3 a real test rather than an apples-to-oranges comparison. Momentum has no such conflict to avoid, so it is sourced from its own separate French file.

## Why the standard error decides the answer

Getting alpha is easy. Deciding whether it is distinguishable from zero is where the work is, and it depends on assumptions about the residuals that are rarely stated out loud.

The point estimate never changes. What changes is the standard error, and with it the t-statistic:

- **Classical (OLS)** assumes residuals are independent of each other and have constant variance. Under those assumptions it is the most efficient choice. Asset returns violate both.
- **White (HC1)** drops constant variance. Volatility clusters in markets, so residual spread is genuinely larger in some periods than others; this correction accounts for it.
- **Newey-West (HAC)** additionally allows residuals to be correlated across months. This matters most, because positive residual autocorrelation makes the several hundred monthly observations behave like far fewer independent ones. Classical standard errors treat every month as fresh evidence, understate uncertainty, and inflate the t-statistic accordingly.

Each step weakens an assumption, and usually widens the interval. **An alpha whose t-statistic clears 1.96 under classical errors but not under Newey-West was never significant; the first number was an artifact of assuming away serial correlation.** Section 2 of the app puts all three side by side for exactly this reason, and the residual diagnostics in section 4 tell you which one you are entitled to quote.

The HAC lag length is exposed as a control rather than hidden, because it is a judgment call. The common default is the Newey-West plug-in rule, `floor(4 * (n / 100) ** (2/9))`, but nothing makes it correct. Moving the slider and watching the t-statistic drift is the honest version of that story.

Two more things the app makes visible, both of which are routinely skipped:

- **Multicollinearity.** Correlated factors inflate each other's standard errors. The variance inflation factor quantifies it. High VIF does not bias the estimates or hurt the fit, which is how a model can be jointly significant with no individually significant coefficient.
- **Parameter stability.** A full-sample regression reports one beta and implies it held for sixty years. Re-estimating over rolling windows usually shows something much less stable. Utilities, for instance, show a market beta wandering between roughly 0.3 and 1.0 depending on the window.

## What this does not do

- Alpha here is gross of costs, fees, and taxes. Real implementation would consume much of it.
- The industry portfolios and bundled ticker catalog are US only, because that is what French publishes and what the catalog was seeded with; the free-text ticker box will take any Yahoo Finance symbol, but there is no comparable free factor library for non-US markets like the JSE to properly evaluate one.
- The factors are treated as known and error-free. They are themselves estimated portfolios.
- Statistical significance is not economic significance, and this is a single regression on a single asset with no correction for the fact that you can click through dozens of them looking for a low p-value. That multiple-comparisons problem deserves its own demo.

## Setup

```bash
uv sync
```

## Usage

Run the interactive app:

```bash
uv run streamlit run app/streamlit_app.py
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

Rebuild the bundled fallback snapshots from a fresh download:

```bash
uv run scripts/refresh_snapshot.py
```

## Data

Factors and industry portfolios come from the [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) as zipped monthly CSVs, parsed straight from the download. Individual tickers come from Yahoo Finance via `yfinance`, resampled to monthly total returns.

Both paths degrade rather than crash: fresh cache, then live download, then stale cache, then the bundled parquet snapshots in `assets/`. Live refresh is off by default in the UI, since Dartmouth and Yahoo both throttle.

## Project layout

- `src/factor_lab/data.py` - French CSV parsing, the caching and fallback chain, yfinance loading, and factor/asset alignment
- `src/factor_lab/regression.py` - OLS fitting, the coefficient table, the three-way standard-error comparison, and nested model tests
- `src/factor_lab/diagnostics.py` - Breusch-Pagan, Durbin-Watson, Ljung-Box, Jarque-Bera, VIF, each with a plain-language reading
- `src/factor_lab/rolling.py` - rolling-window estimation and a stability summary
- `src/factor_lab/plots.py` - matplotlib figures
- `src/factor_lab/cli.py` - the text report
- `app/streamlit_app.py` - the interactive demo
- `tests/` - synthetic-data tests with known parameters, including a coverage simulation for the confidence intervals and a check that HAC standard errors exceed classical ones under injected autocorrelation
- `data/` - gitignored cache of downloaded data

## Scope

The coding, architecture, and infrastructure choices here are proof-of-concept appropriate, not indicative of an enterprise setup; a company environment would warrant a different strategy. For research and education only, not investment advice.
