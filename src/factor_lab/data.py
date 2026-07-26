"""Data loading: Ken French factor and portfolio CSVs, yfinance test assets, monthly alignment.

Ken French publishes both sides of the regression (factor returns and ready-made
test portfolios), so the core path needs no other data source. Everything is
monthly, in percent in the raw files, and converted to decimals here.
"""

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data"
SNAPSHOT_DIR = PROJECT_ROOT / "assets"
TEST_ASSETS_PATH = PROJECT_ROOT / "config" / "test_assets.csv"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
# French reposts the library monthly, so a week-old cache is still current enough.
CACHE_MAX_AGE = timedelta(days=7)
# Sentinels French uses for missing observations.
MISSING_VALUES = (-99.99, -999.0)

# Factor models available in the app, ordered simplest to richest. Each model's
# regressors are a strict subset of the next, which is what makes the nested
# F-tests in `regression.compare_nested` valid.
FACTOR_MODELS: dict[str, list[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


@dataclass(frozen=True)
class FrenchDataset:
    """One zipped CSV in the Ken French Data Library."""

    key: str
    zip_name: str
    description: str


# Note that Mkt-RF, SMB and HML all come from the 5-factor file rather than the
# standalone 3-factor file. The two files define SMB slightly differently (the
# 5-factor SMB comes from 2x3 sorts on all three of value, profitability and
# investment) and they start in different years. Taking every factor from one
# file keeps FF3 a genuine subset of FF5 over an identical sample, so comparing
# them is a clean nested test rather than an apples-to-oranges one.
DATASETS: dict[str, FrenchDataset] = {
    "factors": FrenchDataset(
        key="factors",
        zip_name="F-F_Research_Data_5_Factors_2x3_CSV.zip",
        description="Fama-French 5 factors (2x3 sorts) plus the risk-free rate, monthly",
    ),
    "momentum": FrenchDataset(
        key="momentum",
        zip_name="F-F_Momentum_Factor_CSV.zip",
        description="Fama-French momentum factor (Mom), monthly",
    ),
    "industries": FrenchDataset(
        key="industries",
        zip_name="17_Industry_Portfolios_CSV.zip",
        description="17 value-weighted industry portfolios, monthly",
    ),
}

_DATA_ROW = re.compile(r"^\s*(\d{6})\s*,")


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the monthly block out of a Ken French CSV.

    These files open with a free-text preamble of varying length, then a header
    row whose first field is blank, then monthly rows keyed YYYYMM. Files often
    continue with further sections (annual returns, equal-weighted returns), so
    locating rows structurally and stopping at the first non-monthly line is
    more robust than slicing at fixed offsets. The first monthly block is the
    one wanted in every dataset used here.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if _DATA_ROW.match(line))
    except StopIteration:
        raise ValueError("no monthly YYYYMM rows found in the French CSV")
    if start == 0:
        raise ValueError("French CSV has monthly rows but no header row above them")

    header = [field.strip() for field in lines[start - 1].split(",")]
    header[0] = "date"

    rows = []
    for line in lines[start:]:
        if not _DATA_ROW.match(line):
            break
        rows.append([field.strip() for field in line.split(",")])

    width = len(header)
    rows = [row[:width] for row in rows if len(row) >= width]
    frame = pd.DataFrame(rows, columns=header)

    month_ends = pd.PeriodIndex(frame["date"], freq="M").to_timestamp(how="end").normalize()
    # Drop the inferred MonthEnd freq: parquet does not round-trip it, so keeping
    # it would make a fresh download and a cache hit compare unequal, and it
    # would assert a regularity these files do not actually guarantee.
    index = pd.DatetimeIndex(month_ends.to_numpy(), name="date")

    values = frame.drop(columns="date").apply(pd.to_numeric, errors="coerce")
    values.index = index

    # Percent to decimal, with French's missing-data sentinels dropped first so
    # they don't become plausible-looking returns of -99% or -999%.
    values = values.mask(values.isin(MISSING_VALUES))
    return values / 100.0


def _download_dataset(dataset: FrenchDataset) -> pd.DataFrame:
    url = FRENCH_BASE + dataset.zip_name
    # Dartmouth returns 403 to the default urllib agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        blob = response.read()

    archive = zipfile.ZipFile(io.BytesIO(blob))
    inner_name = archive.namelist()[0]
    text = archive.read(inner_name).decode("latin-1")

    frame = _parse_french_csv(text)
    if frame.empty:
        raise ValueError(f"{dataset.zip_name} parsed to an empty frame")
    return frame


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < CACHE_MAX_AGE


def load_french_dataset(
    key: str,
    cache_dir: Path = CACHE_DIR,
    snapshot_dir: Path = SNAPSHOT_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """One French dataset as monthly decimal returns, indexed by month end.

    Cached to `cache_dir` and reused while fresh. If a live download is needed
    and fails (no network, Dartmouth unreachable), falls back to the on-disk
    cache even when stale, then to the bundled snapshot in `snapshot_dir`, so a
    public demo degrades instead of crashing.
    """
    dataset = DATASETS[key]
    cache_path = cache_dir / f"{key}.parquet"
    snapshot_path = snapshot_dir / f"{key}_snapshot.parquet"

    if not force_refresh and _is_cache_fresh(cache_path):
        return pd.read_parquet(cache_path)

    try:
        frame = _download_dataset(dataset)
    except Exception:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        if snapshot_path.exists():
            return pd.read_parquet(snapshot_path)
        raise

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)
    return frame


def load_factors(
    cache_dir: Path = CACHE_DIR,
    snapshot_dir: Path = SNAPSHOT_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Every factor plus RF in one frame: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF."""
    factors = load_french_dataset("factors", cache_dir, snapshot_dir, force_refresh)
    momentum = load_french_dataset("momentum", cache_dir, snapshot_dir, force_refresh)
    momentum = momentum.rename(columns={momentum.columns[0]: "Mom"})
    combined = factors.join(momentum[["Mom"]], how="left")
    ordered = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    return combined[[column for column in ordered if column in combined.columns]]


def load_industry_portfolios(
    cache_dir: Path = CACHE_DIR,
    snapshot_dir: Path = SNAPSHOT_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """17 value-weighted industry portfolios as monthly decimal returns."""
    return load_french_dataset("industries", cache_dir, snapshot_dir, force_refresh)


def load_test_asset_catalog(path: Path = TEST_ASSETS_PATH) -> pd.DataFrame:
    """Optional yfinance tickers offered in the UI (`ticker,name`)."""
    return pd.read_csv(path)


def monthly_returns_from_prices(prices: pd.Series) -> pd.Series:
    """Month-end total returns from a daily adjusted close series."""
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.index = returns.index.normalize()
    return returns


def load_yfinance_asset(ticker: str, start: str = "1990-01-01") -> pd.Series:
    """Monthly returns for a yfinance ticker, on the same month-end index as the factors.

    Imported lazily so the core French path does not pay for yfinance, and so a
    broken yfinance install cannot take the whole app down.
    """
    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise ValueError(f"yfinance returned no usable closes for {ticker!r}")

    returns = monthly_returns_from_prices(close)
    returns.name = ticker
    return returns


def build_regression_frame(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    """Align one asset against a factor model.

    Returns a frame whose first column `excess` is the asset's return in excess
    of the risk-free rate (the left-hand side of a factor regression; regressing
    a raw total return on excess-return factors would misstate alpha), followed
    by the model's factor columns. Rows with any missing value are dropped so
    every reported statistic comes from an identical sample.
    """
    if model not in FACTOR_MODELS:
        raise KeyError(f"unknown factor model {model!r}; expected one of {list(FACTOR_MODELS)}")

    regressors = FACTOR_MODELS[model]
    missing = [column for column in [*regressors, "RF"] if column not in factors.columns]
    if missing:
        raise ValueError(f"factor frame is missing columns: {missing}")

    aligned = pd.concat(
        [asset_returns.rename("asset"), factors[[*regressors, "RF"]]],
        axis=1,
        join="inner",
    ).dropna()

    frame = pd.DataFrame(index=aligned.index)
    frame["excess"] = aligned["asset"] - aligned["RF"]
    for column in regressors:
        frame[column] = aligned[column]
    return frame
