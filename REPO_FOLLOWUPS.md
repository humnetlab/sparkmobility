# sparkmobility repository follow-ups

Scope for a separate PR, independent of the TimeGeo refactor. Structural
hygiene across the whole package. Ordered by expected ROI.

## 1. Fix filename typos (breaking change — do it now)

- [sparkmobility/datatset.py](sparkmobility/datatset.py) → `dataset.py`
- [sparkmobility/utils/county_tesslation.py](sparkmobility/utils/county_tesslation.py) → `county_tessellation.py`

Both are public import paths (`from sparkmobility.datatset import MobilityDataset`).
Every user types the typo forever unless fixed. Do it while the package
has few external users; later it's a hard-to-justify breaking change.

Steps: `git mv`, grep for importers, update.

## 2. Move tests to repo root; expand beyond TimeGeo

Today the only tests live at
[sparkmobility/models/timegeo/tests/](sparkmobility/models/timegeo/tests/).
Standard layout is a top-level `tests/` mirroring `sparkmobility/`.

EPR, Gravity, `processing/`, `utils/` have zero tests. Minimum useful
additions:

- `tests/models/test_epr.py` — 30-line smoke test: tiny df → fit →
  generate → assert output columns and non-empty.
- `tests/models/test_gravity.py` — same pattern.
- `tests/processing/test_user_selection.py` — guard `filter_users`
  thresholds against regressions.

These catch ~80% of "refactor broke it" bugs at ~10 min each to write.

## 3. Remove build artifacts from source control

Currently tracked (or at least present):

- `build/` — setuptools leftover
- `dist/sparkmobility-0.1.0-*.whl` + `.tar.gz` — stale the moment you
  bump version
- `sparkmobility.egg-info/` — setuptools leftover
- `tmp/` — empty scratch dir

Add to `.gitignore`, `git rm -r --cached` them. These drift from the
real source and pollute grep results during refactors.

## 4. Slim `pyproject.toml` runtime deps

[pyproject.toml](pyproject.toml) currently pins Jupyter-kernel dependencies
as runtime deps: `ipykernel`, `ipython`, `jedi`, `debugpy`, `pexpect`,
`pickleshare`, `jupyter_client`, `jupyter_core`, `prompt-toolkit`,
`appnope`, `asttokens`, `comm`, `matplotlib-inline`, `nest-asyncio`,
`parso`, `executing`, `pure_eval`, `ptyprocess`.

Anyone who `pip install sparkmobility`s gets a 200MB+ Jupyter environment
they didn't ask for. Split:

```toml
[project.optional-dependencies]
notebook = ["ipykernel", "ipython", "jedi", "debugpy", ...]
dev = ["pytest", "ruff", "mypy", ...]
```

Keep the core `dependencies` list to what the library actually imports
at runtime: pandas, pyarrow, h3, h3spark, pyspark, geopandas, pyproj,
numpy, matplotlib, folium, tqdm, powerlaw, statsmodels, censusdis, geopy,
py4j.

## 5. Add a minimal `docs/` directory

A new user today can't answer "which model do I pick?" without reading
source. Three files solves it:

- `docs/quickstart.md` — 10 lines per model showing `fit` / `generate`
- `docs/data-format.md` — what columns each model's `fit()` expects
  (user_column, timestamp_column, location_column, type_column) and
  their dtypes
- `docs/models.md` — one-paragraph comparison: Gravity (aggregate flows,
  needs OD zones), EPR (individual trajectories, rho/gamma), TimeGeo
  (individual trajectories with commuter/non-commuter split, needs
  activity sequences)

Source material already exists in docstrings and
[sparkmobility/models/timegeo/PERF_FOLLOWUPS.md](sparkmobility/models/timegeo/PERF_FOLLOWUPS.md) —
just externalize.

## 6. Make `examples/` runnable

[examples/](examples/) contains six PNGs and zero `.py` files. Rename the
PNGs to `examples/figures/` and add:

- `examples/timegeo_quickstart.py` — cleaned version of
  `package_testing/run_timegeo.py` with paths parameterized via argparse
- `examples/epr_quickstart.py` — minimal EPR fit/generate
- `examples/gravity_quickstart.py` — minimal Gravity fit

These double as integration tests the moment you add `pytest` coverage
for "examples run without error."

## 7. CI + pre-commit (once tests exist)

No `.github/workflows/` exists. [autoformat.sh](autoformat.sh) runs
formatters locally but nothing enforces it on PRs.

Once items 2 and 6 are in place, minimal CI:

- `.github/workflows/test.yml` — `pytest tests/` on push + PR
- `.pre-commit-config.yaml` — ruff + black + a "no print() in src/"
  check (prevents regressing Phase 5's logging work)

Skip this until there are actual tests to run — CI'ing nothing is worse
than no CI.

## 8. Shrink the TimeGeo run.log — `--quiet` leaks

A real run on 45M stay points produced
[run.log](../package_testing/timegeo_results_0421/run.log) of **4,270
lines**. Breakdown (per `awk | sort | uniq -c`):

| Pattern (fires per row group × 2 C++ passes) | Occurrences |
|---|---:|
| `Table schema:` + 10 column descriptions | 249 × 11 |
| `Row group has N rows`                   | 249 |
| `Processing N rows in chunk N`           | 249 |
| `Number of chunks: N`                    | 249 |
| `Found columns: user_id=...`             | 249 |
| `Chunk N types - user_id: N, ...`        | 249 |
| `Processing final batch of N users...`   | 240 |

That's ~3,984 lines of repeated C++ startup diagnostics = **93% of the
log**. Only ~286 lines carry real signal (step banners, row-group
processing, stage summaries, the one bad_alloc summary).

Phase 6's `--quiet` flag was supposed to gate these but only covered
two patterns. Three-part fix:

**8a. C++ side — broaden `--quiet` gating in `module_2_3_1.cpp`.** Wrap
every per-row-group `cout` in `if (!g_quiet)`:
- Schema dump (currently gated but apparently still leaking — investigate
  whether the rebuilt binary actually ships with the gate)
- `Row group has N rows`
- `Processing N rows in chunk N`
- `Number of chunks: N`
- `Found columns: ...`
- `Chunk N types - ...`
- `Processing final batch of N users...`

These are startup diagnostics that belong on a `--verbose` path, not
default. The `--quiet`/`TIMEGEO_QUIET=1` plumbing already exists, just
needs to cover more print sites.

**8b. Python side — broaden `_LogFilter` as defense in depth.** In
[_native/runner.py](sparkmobility/models/timegeo/_native/runner.py),
currently swallows only `Processing row group N of M` and the bad_alloc
error. Add regex matches for the 7 leaked patterns above and drop them
silently. This way even an old unrebuilt C++ binary produces a clean log.

**8c. Fix misleading row-group summary.** Today:
```
processed 450/450 row groups in 201.3s
333 row groups failed with std::bad_alloc (groups 117–449)
```
Those are contradictory to a reader — "processed 450/450" counts
*attempts*, not *successes*. Change runner.py's flush to:
```
attempted 450 row groups in 201.3s; 117 succeeded, 333 failed with std::bad_alloc (groups 117-449)
```
Same two lines collapse to one and no longer mislead.

Expected result on a 45M-row run: log shrinks from ~4,270 to ~300 lines.
Arrow memory-pool fragmentation (the underlying cause of the 333 bad_allocs) is
separately tracked in [sparkmobility/models/timegeo/PERF_FOLLOWUPS.md](sparkmobility/models/timegeo/PERF_FOLLOWUPS.md)
as a real fix; item 8 is only log hygiene.

## 9. Fix `plots.py` schema mismatch with `TimeGeo.generate()` output

The DataFrame-facing wrappers in
[sparkmobility/models/timegeo/plots.py](sparkmobility/models/timegeo/plots.py)
materialize `traj_df` to parquet and delegate to `_pipeline/s5_aggregated_plots.py`'s
`plot_*_parquet` helpers. Those helpers expect `user_id` and `timeslot`
columns — but `TimeGeo.generate()` in
[core.py:333](sparkmobility/models/timegeo/core.py#L333) renames
`user_id`→`user` and derives a `datetime` column from `timeslot * slot_interval`.

Result: user-facing `plot_stay_durations(traj_df, ...)` KeyErrors on any
post-`generate()` DataFrame. This is why
[package_testing/run_timegeo.py](../package_testing/run_timegeo.py) currently
calls the `_pipeline` helpers directly against the raw
`Simulation/simulation_results.parquet` (which still has the original schema).

Two-line fix in each wrapper: before calling `to_parquet`, rename back
(`user`→`user_id`) and re-derive `timeslot` from `datetime` (`(hour*60 + minute) // slot_interval`).
Or: push the rename into `generate()` output as an opt-in, keeping the raw
schema by default.

## Out of scope

- Publishing to PyPI. Independent packaging concern.
- Rewriting Gravity/EPR around the same `configure_logging_if_needed`
  pattern TimeGeo uses — already done in Phase 5 of the TimeGeo PR.
- Vendoring the C++ binary via wheels. Tracked in
  [sparkmobility/models/timegeo/PERF_FOLLOWUPS.md](sparkmobility/models/timegeo/PERF_FOLLOWUPS.md).
- Fixing the underlying Arrow memory-pool fragmentation that causes the
  bad_alloc failures. Item 8 only reduces noise; the data-loss issue
  belongs in PERF_FOLLOWUPS.md.
