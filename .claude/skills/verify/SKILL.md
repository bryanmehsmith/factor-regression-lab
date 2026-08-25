---
name: verify
description: Project-specific verification steps for factor-regression-lab - run pytest, then the JS-vs-Python parity scripts, then manually check the frontend.
---

# Verify

This repo ships a Python reference implementation (`src/factor_lab/`) and a hand-ported JS frontend (`frontend/js/modules/{regression,diagnostics,rolling}.js` + `distributions.js`) that reimplements the same statistics client-side. Unlike its sibling `momentum-factor`, this repo *does* have an automated way to catch drift between the two - use it, it's cheap and it exists for exactly this reason.

## Steps

1. **Python suite**: `uv sync && uv run pytest` (`testpaths = ["tests"]` per `pyproject.toml`). Covers OLS fitting, standard-error variants (classical/White/Newey-West), diagnostics, rolling estimation, and a coverage simulation for the confidence intervals.
2. **If you touched `src/factor_lab/regression.py`, `diagnostics.py`, or `rolling.py`** (or the hand-rolled t/F/chi-square distributions in `frontend/js/modules/distributions.js`): re-run the JS-vs-Python parity check -
   ```bash
   uv run scripts/export_regression_fixtures.py
   uv run scripts/export_dist_reference.py
   cd scripts && node verify_stats_parity.mjs && node verify_distributions.mjs && cd ..
   ```
   The export scripts write their fixture JSON into `scripts/` (relative to the script file, not CWD), and the verify scripts read it relative to *their own* CWD - so the `node` calls must run with `scripts/` as the working directory, not the repo root, even though `uv run scripts/export_*.py` runs fine from the root. (`README.md`'s usage section shows the flat `node scripts/verify_stats_parity.mjs` form; that fails with `ENOENT` run from the repo root - the `cd scripts &&` form above is the one that actually works.)

   These fit a battery of synthetic frames (clean, autocorrelated, heteroskedastic, varying sample sizes) with the real Python modules, then recompute the same thing in JS and diff. A change to either side that isn't ported to the other will fail here - don't skip this and rely on eyeballing.
3. **Manual frontend check**: `cd frontend && python -m http.server`, open it, run a regression against a couple of assets/models (CAPM through FF5+Mom), confirm the numbers look sane and the console is clean.
4. **If you touched `app/api.py`**: `uv run python app/api.py --port=8000`, hit `GET /factors?refresh=true`, `GET /industries?refresh=true`, and `GET /ticker?symbol=AAPL`, confirm each responds. This backs only the frontend's live French-data refresh and free-text ticker lookup, not the core regression logic.
5. **If checked out under demo-site**: `launcher.py` starts this via `app/api.py` on port 8502 per `demos.json` - see demo-site's own verify skill for the proxied end-to-end check.

## Scope note

Docs-only changes: `uv run pytest` is enough. A change confined to `app/api.py`, `plots.py`, or `cli.py` doesn't need the parity scripts (they don't touch the ported JS modules) - steps 1 and the relevant manual check (3 or 4) suffice.
