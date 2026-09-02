"""Generate deterministic synthetic factor and industry data for the public demo."""

from __future__ import annotations

import argparse
import calendar
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "frontend" / "js" / "data"
INDUSTRIES = [
    "Food",
    "Mines",
    "Oil",
    "Clths",
    "Durbl",
    "Chems",
    "Cnsum",
    "Cnstr",
    "Steel",
    "FabPr",
    "Machn",
    "Cars",
    "Trans",
    "Utils",
    "Rtail",
    "Finan",
    "Other",
]
CATALOG = [
    {"ticker": "SPY", "name": "S&P 500 ETF"},
    {"ticker": "IWM", "name": "Russell 2000 small-cap ETF"},
    {"ticker": "QQQ", "name": "Nasdaq 100 ETF"},
    {"ticker": "VTV", "name": "Vanguard Value ETF"},
    {"ticker": "VUG", "name": "Vanguard Growth ETF"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway"},
    {"ticker": "AAPL", "name": "Apple"},
    {"ticker": "KO", "name": "Coca-Cola"},
    {"ticker": "XOM", "name": "Exxon Mobil"},
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
]


def month_ends(start_year: int, end_year: int) -> list[str]:
    return [
        f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def rounded(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def build_datasets() -> tuple[dict, dict, list[dict[str, str]]]:
    dates = month_ends(2000, 2025)
    steps = range(len(dates))
    factors = {
        "Mkt-RF": rounded([0.006 + 0.032 * math.sin(t * 0.61) + 0.018 * math.cos(t * 0.17) for t in steps]),
        "SMB": rounded([0.002 + 0.016 * math.sin(t * 0.39 + 0.8) for t in steps]),
        "HML": rounded([0.001 + 0.018 * math.cos(t * 0.31 + 0.4) for t in steps]),
        "RMW": rounded([0.002 + 0.012 * math.sin(t * 0.27 + 1.7) for t in steps]),
        "CMA": rounded([0.001 + 0.011 * math.cos(t * 0.23 + 2.1) for t in steps]),
        "Mom": rounded([0.003 + 0.02 * math.sin(t * 0.47 + 2.6) for t in steps]),
        "RF": rounded([0.002 + 0.0008 * math.cos(t * 0.09) for t in steps]),
    }

    industries: dict[str, list[float]] = {}
    for index, name in enumerate(INDUSTRIES):
        beta_market = 0.75 + 0.06 * (index % 8)
        beta_size = -0.22 + 0.08 * (index % 7)
        beta_value = -0.18 + 0.07 * (index % 6)
        beta_profitability = -0.12 + 0.06 * (index % 5)
        beta_investment = -0.1 + 0.05 * (index % 5)
        beta_momentum = -0.08 + 0.04 * (index % 6)
        values = []
        for t in range(len(dates)):
            residual = 0.012 * math.sin(t * 1.13 + index * 0.73)
            value = (
                factors["RF"][t]
                + beta_market * factors["Mkt-RF"][t]
                + beta_size * factors["SMB"][t]
                + beta_value * factors["HML"][t]
                + beta_profitability * factors["RMW"][t]
                + beta_investment * factors["CMA"][t]
                + beta_momentum * factors["Mom"][t]
                + residual
            )
            values.append(value)
        industries[name] = rounded(values)

    metadata = {
        "kind": "synthetic",
        "description": "Deterministic synthetic monthly returns for teaching. Not observed market data.",
        "generator": "scripts/generate_synthetic_frontend_data.py",
        "version": 1,
    }
    return (
        {"metadata": metadata, "dates": dates, "series": factors},
        {"metadata": metadata, "dates": dates, "series": industries},
        CATALOG,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    factors, industries, catalog = build_datasets()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("factors.json", factors),
        ("industries.json", industries),
        ("catalog.json", catalog),
    ):
        path = args.output_dir / filename
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        print(f"Wrote synthetic frontend data: {path}")


if __name__ == "__main__":
    main()
