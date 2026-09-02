# Codex Agent Instructions

This file provides guidance to Codex and other coding agents working in this repository.

## What this is

Factor regression / statistical inference lab: is an asset's excess return alpha, or just factor exposure? See `README.md` for the actual statistics content (why alpha is model-relative, why standard-error choice matters, multicollinearity, parameter stability) - read it before touching `regression.py`/`diagnostics.py`/`rolling.py`, since the demo's whole point is the statistical reasoning, not just the numbers.

## Two implementations that must stay in sync - but this one has a safety net

- **`src/factor_lab/`** - the reference implementation (`data.py`, `regression.py`, `diagnostics.py`, `rolling.py`, `plots.py`, `cli.py`), covered by `tests/` (pytest).
- **`frontend/js/modules/{regression,diagnostics,rolling}.js` + `distributions.js`** - the *actual deployed demo*: the same three modules hand-ported to vanilla JS, plus hand-rolled t/F/chi-square distribution functions (no JS stats library used), running client-side against bundled monthly snapshots.

Unlike the sibling `momentum-factor` repo, drift between these two is caught automatically: `scripts/export_regression_fixtures.py` + `scripts/export_dist_reference.py` generate fixtures from the real Python implementation, and `scripts/verify_stats_parity.mjs` + `scripts/verify_distributions.mjs` recompute the same thing in JS and diff. **Always re-run these after touching `regression.py`/`diagnostics.py`/`rolling.py`/`distributions.js`** - see `.claude/skills/verify` for the exact commands. Don't skip them because pytest passed; pytest only exercises the Python side.

**Working-directory gotcha**: the export scripts (`uv run scripts/export_*.py`) write their fixture JSON into `scripts/`, resolved relative to the script file itself, but the `node scripts/verify_*.mjs` scripts read that JSON relative to whatever directory `node` was invoked from. Run from the repo root as `README.md`'s usage section literally shows, they fail with `ENOENT`. Run them with `scripts/` as the working directory instead (`cd scripts && node verify_stats_parity.mjs`) - that's what actually works, and what CI does.

## `app/api.py` is not the main demo

`demos.json` in the parent demo-site repo lists this as `kind: "api"`, entrypoint `app/api.py`, port 8502 - accurate for what `launcher.py` starts as a process. But the endpoint itself only backs the frontend's live French-data refresh and free-text Yahoo Finance ticker lookup (`/factors`, `/industries`, `/ticker`). The core regression/diagnostics logic the demo showcases runs entirely client-side against bundled snapshots, not through this API.

## Data fallback chain

Both French factors and yfinance ticker data degrade through: fresh cache → live download → stale cache → bundled parquet snapshot in `assets/`. Live refresh is off by default in the UI (Dartmouth and Yahoo both throttle) - don't "fix" this by defaulting it on.

## Commands

- `uv sync` - install
- `uv run pytest` - Python test suite
- `cd frontend && python -m http.server` - serve the actual demo locally
- `uv run python app/api.py --port=8000` - the live-refresh/ticker-lookup companion API
- `uv run scripts/run_regression.py --asset Utils --model FF5` - CLI text report
- `uv run scripts/refresh_snapshot.py` then `uv run scripts/export_snapshot_json.py` - rebuild bundled snapshots from a fresh download
- JS-vs-Python parity check: see `.claude/skills/verify`

## Verification

See `.claude/skills/verify` - the parity-script step is the part most worth not skipping, since it's the only thing standing between "the Python side is right" and "the deployed JS demo is right."

While iterating on one module, run `uv run pytest -k <module>` instead of the full suite; run the full `uv run pytest` before calling verification done.

## Deployment

Pulled into `demo-site` as a git submodule at `apps/factor-regression`, sharing demo-site's root `uv` workspace venv with `momentum-factor` (bumped to the same pandas/pyarrow floor for that reason - the old `pyarrow<19` Streamlit-SIGSEGV cap no longer applies to either, see demo-site's `AGENTS.md`). A push to this repo's `main` triggers `.github/workflows/bump-demo-site.yml`, which bumps the submodule pointer in `demo-site` (requires a `DEMO_SITE_PAT` secret). `.github/workflows/test.yml` runs `uv run pytest` on push/PR and gates the bump.

## Scope

Proof-of-concept, not enterprise-grade. For research and education only, not investment advice. Personal project using public data only; not affiliated with or representative of my employer.
