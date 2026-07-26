"""Rebuild the bundled parquet snapshots from a fresh Dartmouth download.

The snapshots are the last line of the fallback chain in `data.load_french_dataset`,
so the public demo keeps working when Dartmouth is unreachable. Run this by hand
every so often and commit the result; the files are small.

    uv run scripts/refresh_snapshot.py
"""

from factor_lab import data


def main() -> None:
    data.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for key, dataset in data.DATASETS.items():
        frame = data._download_dataset(dataset)
        destination = data.SNAPSHOT_DIR / f"{key}_snapshot.parquet"
        frame.to_parquet(destination)
        size_kb = destination.stat().st_size / 1024
        print(
            f"{key}: {len(frame)} months, {frame.index[0]:%Y-%m} to {frame.index[-1]:%Y-%m}, "
            f"{len(frame.columns)} columns, {size_kb:.0f} KB -> {destination.name}"
        )


if __name__ == "__main__":
    main()
