"""Exports the bundled French snapshots as JSON for the static frontend.

Everything here is already monthly (unlike momentum-factor's daily prices), so
this is a straight parquet -> JSON conversion, not a resample. Run whenever
refresh_snapshot.py updates the bundled parquet snapshots:

    uv run scripts/refresh_snapshot.py
    uv run scripts/export_snapshot_json.py
    git add frontend/js/data/*.json
    git commit -m "Refresh frontend data snapshots"
"""

import json
from pathlib import Path

import pandas as pd

from factor_lab.data import load_factors, load_industry_portfolios, load_test_asset_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "js" / "data"


def frame_to_json(frame: pd.DataFrame) -> dict:
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "series": {
            column: [None if pd.isna(v) else round(float(v), 6) for v in frame[column]]
            for column in frame.columns
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    factors = load_factors()
    (OUTPUT_DIR / "factors.json").write_text(json.dumps(frame_to_json(factors), separators=(",", ":")))
    print(f"Wrote factors.json ({len(factors)} months, columns {factors.columns.tolist()})")

    industries = load_industry_portfolios()
    (OUTPUT_DIR / "industries.json").write_text(json.dumps(frame_to_json(industries), separators=(",", ":")))
    print(f"Wrote industries.json ({len(industries)} months, {len(industries.columns)} portfolios)")

    catalog = load_test_asset_catalog()
    (OUTPUT_DIR / "catalog.json").write_text(json.dumps(catalog.to_dict(orient="records")))
    print(f"Wrote catalog.json ({len(catalog)} tickers)")


if __name__ == "__main__":
    main()
