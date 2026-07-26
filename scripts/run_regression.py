"""CLI entrypoint: print the full factor regression report for one asset.

    uv run scripts/run_regression.py --asset Utils --model FF5
    uv run scripts/run_regression.py --asset SPY --yfinance --model FF3
"""

from factor_lab.cli import main

if __name__ == "__main__":
    main()
