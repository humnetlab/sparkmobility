# TimeGeo performance follow-ups

Scope for a separate PR after the fit/generate refactor lands. Ranked by
expected wall-time impact on the 46M-row sparkmobility test input.

## 1. Vectorize `iterrows` loops in stage 2 (biggest win)

**File:** [_pipeline/s2_simulation_prep.py:61-118](/data_1/albert/sparkmobility/sparkmobility/models/timegeo/_pipeline/s2_simulation_prep.py#L61-L118)

`_generate_simulation_input_aligned_coords` iterates every row of the
aligned parquet in pure Python. On the test input that's ~45M Python-loop
iterations. Rewrite as a groupby + agg → dict build per user.

Same pattern in [_pipeline/s4_postprocessing.py:195-245](/data_1/albert/sparkmobility/sparkmobility/models/timegeo/_pipeline/s4_postprocessing.py#L195-L245)
(`export_simulation_results_to_parquet`) — parses `Mapped/*.txt` line-by-line
in Python; can be a `pd.read_csv(..., sep=' ')` per file + vectorized
location/timeslot assignment.

Expected: stage 2 & stage 4 drop from minutes to seconds.

## 2. Parallelize the two C++ passes

**File:** [core.py `fit()` stage 2](/data_1/albert/sparkmobility/sparkmobility/models/timegeo/core.py)

`run_parameter_generation` is invoked serially for commuters and
non-commuters over the same input parquet. Two `Popen`s running in parallel
would nearly halve the C++ wall time.

Tradeoff: 2× peak memory. The C++ already `setrlimit`s itself to 2 GB
per process, so the combined cap is 4 GB — bounded and acceptable on the
48-core machine.

## 3. Skip the aligned-parquet re-read in `fit()`

**File:** [core.py](/data_1/albert/sparkmobility/sparkmobility/models/timegeo/core.py) (the `aligned_df = pd.read_parquet(aligned_path)` block before the C++ call)

Currently the full aligned parquet is loaded into pandas just to filter by
FA user IDs and re-write as `FilteredForParameters.parquet`. Replace with a
`pyarrow.dataset` scan + `pc.field("caid").isin(...)` predicate, or feed
the C++ binary the aligned parquet directly (same schema — the filter is
only needed when the FA step actually removed users, which is skipped
when `skip_frequent_user_extraction=True`).

Expected: ~1 min saved on the 46M-row input + RAM relief.

## 4. Spark-native stage 1 (optional, bigger lift)

Stage 1 (align, remove_redundant, FA filter, clean_and_format) is exactly
where Spark beats single-node pandas — the current input is already 1000
Spark-written parquet parts, so we're collapsing a partitioned dataset into
a pandas DataFrame only to re-partition it ourselves.

Option A: add `TimeGeo.fit_spark(spark_df, ...)` that runs stage 1 on Spark
and writes the parquet the C++ binary consumes. Keep `fit(stay_df)` as-is
for pandas inputs, matching EPR/Gravity.

Option B: accept a `pd.DataFrame | pyspark.sql.DataFrame` union in
`fit()`. Simpler surface, but the runtime dependency on pyspark becomes
mandatory rather than the current optional-via-sparkmobility-root.

Stages 2-5 stay on single-node: the C++ binary is single-process, and
`simulate_all_parallel` already uses `multiprocessing` with per-user
locality — Spark's partition model doesn't add anything there, and the
text intermediates (`simulation_parameter.txt`, `Mapped/*.txt`) aren't
Spark-friendly.

Expected: stage 1 drops from ~N minutes (pandas, single-core) to
roughly N/num_cpus — but only pays off for inputs already in Spark, which
is the common sparkmobility case.

## Out of scope (still)

- C++ row-group OOM (Arrow memory-pool fragmentation). Orthogonal; not a
  speed issue on the happy path.
- Packaging the C++ binary via wheels. Build convenience, not runtime perf.
