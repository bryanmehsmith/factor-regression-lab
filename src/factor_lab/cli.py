"""Command-line report: the same analysis the Streamlit app renders, as text.

Useful for sanity-checking a result without the UI in the way, and for the
verification step in the project README.
"""

import argparse

import pandas as pd

from factor_lab import data, diagnostics, regression, rolling

DEFAULT_ASSET = "Utils"


def _format_frame(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    return frame.to_string(float_format=lambda value: floatfmt.format(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--asset",
        default=DEFAULT_ASSET,
        help=(
            "Industry portfolio name (for example Utils, Finan, Oil) or, with "
            "--yfinance, a ticker such as SPY"
        ),
    )
    parser.add_argument(
        "--yfinance",
        action="store_true",
        help="treat --asset as a yfinance ticker instead of a French industry portfolio",
    )
    parser.add_argument(
        "--model",
        default="FF5",
        choices=list(data.FACTOR_MODELS),
        help="factor model on the right-hand side (default: FF5)",
    )
    parser.add_argument(
        "--se",
        default=regression.DEFAULT_SE_TYPE,
        choices=list(regression.SE_TYPES),
        help="standard errors to quote in the headline output",
    )
    parser.add_argument("--hac-lags", type=int, default=None, help="HAC lag length (default: Newey-West rule)")
    parser.add_argument("--start", default=None, help="earliest month to include, as YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=rolling.DEFAULT_WINDOW, help="rolling window in months")
    parser.add_argument("--live", action="store_true", help="force a fresh download from Dartmouth")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    factors = data.load_factors(force_refresh=args.live)
    if args.yfinance:
        asset_returns = data.load_yfinance_asset(args.asset)
    else:
        portfolios = data.load_industry_portfolios(force_refresh=args.live)
        if args.asset not in portfolios.columns:
            raise SystemExit(
                f"{args.asset!r} is not an industry portfolio. Available: {', '.join(portfolios.columns)}"
            )
        asset_returns = portfolios[args.asset]

    frame = data.build_regression_frame(asset_returns, factors, args.model)
    if args.start:
        frame = frame.loc[args.start :]
    if frame.empty:
        raise SystemExit("no overlapping months between the asset and the factors")

    fitted = regression.fit(frame, model=args.model, se_type=args.se, hac_lags=args.hac_lags)
    tests = diagnostics.run_all(fitted)

    print(f"\n{args.asset} regressed on {args.model}")
    print(
        f"Sample: {frame.index[0]:%Y-%m} to {frame.index[-1]:%Y-%m} "
        f"({fitted.n_obs} months), standard errors: {fitted.se_type}"
        + (f", HAC lags: {fitted.hac_lags}" if fitted.hac_lags is not None else "")
    )

    print("\nVerdict")
    print(f"  {diagnostics.alpha_verdict(fitted, tests)}")

    print(f"\nHeadline: annualized alpha {fitted.alpha_annualized:.2%}, ", end="")
    print(f"adjusted R-squared {fitted.adj_r_squared:.3f}")

    print("\nCoefficients")
    print(_format_frame(regression.coefficient_table(fitted)))

    print("\nSame regression under three standard-error assumptions")
    comparison = regression.compare_standard_errors(frame, hac_lags=args.hac_lags)
    print(_format_frame(comparison.xs("t_stat", axis=1, level="statistic"), "{:.2f}"))

    smaller = _previous_model(args.model)
    if smaller:
        nested = regression.compare_nested(
            frame,
            small_regressors=data.FACTOR_MODELS[smaller],
            small_model=smaller,
            large_model=args.model,
            se_type=args.se,
            hac_lags=args.hac_lags,
        )
        print(f"\nDoes {args.model} beat {smaller}?")
        print(f"  Added factors: {', '.join(nested.added)}")
        print(
            f"  Joint test: F = {nested.f_stat:.2f} on {nested.df_num} and "
            f"{nested.df_denom} df, p = {nested.p_value:.4f}"
        )
        print(
            f"  Adjusted R-squared: {nested.small_adj_r_squared:.3f} -> "
            f"{nested.large_adj_r_squared:.3f} "
            f"({nested.incremental_adj_r_squared:+.3f})"
        )
        print(
            f"  Annualized alpha: {nested.small_alpha_annualized:.2%} -> "
            f"{nested.large_alpha_annualized:.2%}"
        )

    print("\nResidual diagnostics")
    print(_format_frame(diagnostics.diagnostics_table(tests).set_index("test")))
    for test in tests:
        print(f"\n  {test.name}\n    {test.reading}")

    print("\nMulticollinearity")
    print(_format_frame(diagnostics.variance_inflation_factors(frame).set_index("regressor"), "{:.2f}"))

    if args.window <= len(frame):
        estimates = rolling.rolling_estimates(frame, window=args.window)
        print(f"\nRolling stability ({args.window}-month windows)")
        print(_format_frame(rolling.stability_summary(estimates), "{:.3f}"))

    print("\nData: Ken French Data Library. For research and education only, not investment advice.\n")


def _previous_model(model: str) -> str | None:
    """The next-simplest model, for the nested comparison."""
    models = list(data.FACTOR_MODELS)
    position = models.index(model)
    return models[position - 1] if position > 0 else None


if __name__ == "__main__":
    main()
