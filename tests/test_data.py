import numpy as np
import pandas as pd
import pytest

from factor_lab import data

# Mirrors the real layout: free-text preamble, a header row whose first field is
# blank, monthly YYYYMM rows, then a second section that must not be picked up.
FRENCH_CSV = """This file was created using the 202605 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RF
196307,   -0.39,   -0.48,   -0.81,   0.27
196308,    5.08,   -0.80,    1.70,   0.25
196309,   -1.57,  -99.99,    0.00,   0.27

  Annual Factors: January-December
1964,    2.27,    0.10,    1.63,   0.30
1965,    5.08,    0.80,    1.70,   0.25
"""


def test_parser_reads_only_the_monthly_block():
    frame = data._parse_french_csv(FRENCH_CSV)

    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(frame) == 3, "the annual section below the blank line must be ignored"


def test_parser_converts_percent_to_decimal():
    frame = data._parse_french_csv(FRENCH_CSV)

    assert frame["Mkt-RF"].iloc[1] == pytest.approx(0.0508)
    assert frame["RF"].iloc[0] == pytest.approx(0.0027)


def test_parser_treats_french_sentinels_as_missing():
    """-99.99 means "no observation", not a 99% monthly loss."""
    frame = data._parse_french_csv(FRENCH_CSV)

    assert np.isnan(frame["SMB"].iloc[2])
    assert frame["SMB"].notna().sum() == 2


def test_parser_indexes_on_month_end_dates():
    frame = data._parse_french_csv(FRENCH_CSV)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tolist() == [
        pd.Timestamp("1963-07-31"),
        pd.Timestamp("1963-08-31"),
        pd.Timestamp("1963-09-30"),
    ]


def test_parser_rejects_a_file_with_no_monthly_rows():
    with pytest.raises(ValueError, match="no monthly"):
        data._parse_french_csv("Just a preamble.\nAnd nothing else.\n")


def test_loader_falls_back_to_the_bundled_snapshot_when_the_download_fails(tmp_path, monkeypatch):
    """The public demo must degrade rather than crash when Dartmouth is unreachable."""
    cache_dir = tmp_path / "data"
    snapshot_dir = tmp_path / "assets"
    snapshot_dir.mkdir()

    expected = data._parse_french_csv(FRENCH_CSV)
    expected.to_parquet(snapshot_dir / "factors_snapshot.parquet")

    def explode(dataset):
        raise OSError("Dartmouth unreachable")

    monkeypatch.setattr(data, "_download_dataset", explode)

    frame = data.load_french_dataset(
        "factors",
        cache_dir=cache_dir,
        snapshot_dir=snapshot_dir,
        force_refresh=True,
    )

    pd.testing.assert_frame_equal(frame, expected)


def test_loader_prefers_a_stale_cache_over_the_snapshot(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    snapshot_dir = tmp_path / "assets"
    cache_dir.mkdir()
    snapshot_dir.mkdir()

    cached = data._parse_french_csv(FRENCH_CSV)
    cached.to_parquet(cache_dir / "factors.parquet")
    # A snapshot that is obviously different, so the assertion can tell them apart.
    (cached * 0).to_parquet(snapshot_dir / "factors_snapshot.parquet")

    monkeypatch.setattr(data, "_download_dataset", lambda dataset: (_ for _ in ()).throw(OSError()))

    frame = data.load_french_dataset(
        "factors",
        cache_dir=cache_dir,
        snapshot_dir=snapshot_dir,
        force_refresh=True,
    )

    pd.testing.assert_frame_equal(frame, cached)


def test_loader_raises_when_there_is_nothing_to_fall_back_to(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_download_dataset", lambda dataset: (_ for _ in ()).throw(OSError("no network")))

    with pytest.raises(OSError):
        data.load_french_dataset(
            "factors",
            cache_dir=tmp_path / "data",
            snapshot_dir=tmp_path / "assets",
            force_refresh=True,
        )


def _factor_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02, -0.01, 0.03, 0.00, 0.01],
            "SMB": [0.001] * 6,
            "HML": [0.002] * 6,
            "RMW": [0.003] * 6,
            "CMA": [0.004] * 6,
            "Mom": [0.005] * 6,
            "RF": [0.002] * 6,
        },
        index=index,
    )


def test_regression_frame_subtracts_the_risk_free_rate():
    """Regressing a total return on excess-return factors would misstate alpha."""
    factors = _factor_frame()
    asset = pd.Series(0.015, index=factors.index)

    frame = data.build_regression_frame(asset, factors, "FF3")

    assert frame["excess"].iloc[0] == pytest.approx(0.015 - 0.002)
    assert list(frame.columns) == ["excess", "Mkt-RF", "SMB", "HML"]


def test_regression_frame_selects_the_requested_model():
    factors = _factor_frame()
    asset = pd.Series(0.015, index=factors.index)

    assert list(data.build_regression_frame(asset, factors, "CAPM").columns) == ["excess", "Mkt-RF"]
    assert list(data.build_regression_frame(asset, factors, "FF5+Mom").columns) == [
        "excess",
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
        "Mom",
    ]


def test_regression_frame_intersects_on_overlapping_months():
    factors = _factor_frame()
    asset = pd.Series(0.015, index=factors.index[2:])

    frame = data.build_regression_frame(asset, factors, "FF3")

    assert len(frame) == 4
    assert frame.index[0] == factors.index[2]


def test_regression_frame_drops_months_with_missing_factors():
    factors = _factor_frame()
    factors.loc[factors.index[1], "HML"] = np.nan
    asset = pd.Series(0.015, index=factors.index)

    frame = data.build_regression_frame(asset, factors, "FF3")

    assert len(frame) == 5
    assert factors.index[1] not in frame.index


def test_regression_frame_rejects_an_unknown_model():
    factors = _factor_frame()
    asset = pd.Series(0.015, index=factors.index)

    with pytest.raises(KeyError, match="unknown factor model"):
        data.build_regression_frame(asset, factors, "FF7")


def test_monthly_returns_from_daily_prices():
    days = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    prices = pd.Series(100.0, index=days)
    prices.loc["2020-02-01":] = 110.0
    prices.loc["2020-03-01":] = 121.0

    returns = data.monthly_returns_from_prices(prices)

    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.index.tolist() == [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")]


def test_factor_models_are_strictly_nested():
    """The nested F-tests in the app depend on this, so it is worth asserting."""
    models = list(data.FACTOR_MODELS.values())
    for smaller, larger in zip(models, models[1:]):
        assert set(smaller).issubset(set(larger))
