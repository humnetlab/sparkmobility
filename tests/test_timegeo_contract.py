"""Regression tests for the TimeGeo input contract.

The C++ parameter binary resolves its columns by name but casts them with
dynamic_pointer_cast, which returns null rather than raising when the Arrow
type is wrong. Every mismatch below is therefore silent: the pipeline still
"succeeds", it just calibrates fewer users or different parameters. These
tests pin the dtypes so that can't regress unnoticed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h3.api.basic_int as h3int
import pandas as pd
import pyarrow.parquet as pq
import pytest

from sparkmobility.models.timegeo._pipeline import data_alignment
from sparkmobility.models.timegeo.paths import TimeGeoPaths

BINARY = Path(data_alignment.__file__).resolve().parents[1] / "_native" / "module_2_3_1"

RES = 8
N_USERS = 6
N_DAYS = 21


def _stays() -> pd.DataFrame:
    """Synthetic stays in the schema the Scala backend emits."""
    rows = []
    for u in range(N_USERS):
        home = h3int.latlng_to_cell(34.05 + u * 0.01, -118.25, RES)
        work = h3int.latlng_to_cell(34.10 + u * 0.01, -118.35, RES)
        others = [
            h3int.latlng_to_cell(34.0 + 0.01 * k, -118.1 - 0.01 * k, RES)
            for k in range(20)
        ]
        for day in range(N_DAYS):
            ts = 1546300800 + day * 86400
            dow = (day % 7) + 1
            for cell, typ, hour in (
                (home, 1, 2),
                (work, 2, 9),
                (others[day % len(others)], 0, 19),
            ):
                rows.append(
                    (
                        f"u{u}",
                        ts + hour * 3600,
                        cell,
                        typ,
                        format(home, "X"),
                        format(work, "X"),
                        dow,
                    )
                )
    df = pd.DataFrame(
        rows,
        columns=[
            "caid",
            "stay_start_timestamp",
            "h3_id_region",
            "type",
            "home_h3_index",
            "work_h3_index",
            "day_of_week",
        ],
    )
    return df.astype({"h3_id_region": "int64", "type": "int32", "day_of_week": "int32"})


@pytest.fixture(scope="module")
def aligned(tmp_path_factory) -> Path:
    paths = TimeGeoPaths(tmp_path_factory.mktemp("tg"))
    paths.ensure()
    return data_alignment.align(_stays(), paths)


def test_align_keeps_type_as_int32(aligned):
    """`type` must stay int32: the C++ reads it as an Int32Array only.

    When it arrives as a string the cast fails silently, location_type stays
    0 for every row, and the "not a work location" guard never fires -- which
    inflates avg_loc_count rather than producing any visible error.
    """
    assert pq.read_schema(aligned).field("type").type == "int32"


def test_align_normalizes_string_labels_to_ints(tmp_path):
    """String labels are accepted on input but converted, not passed through."""
    df = _stays()
    df["type"] = df["type"].map({0: "other", 1: "home", 2: "work"})
    paths = TimeGeoPaths(tmp_path)
    paths.ensure()
    out = data_alignment.align(df, paths)
    table = pq.read_table(out)
    assert table.schema.field("type").type == "int32"
    assert set(table.column("type").to_pylist()) == {0, 1, 2}


def test_home_work_h3_stay_strings(aligned):
    """home/work H3 must reach the C++ as strings; ints make it skip the user."""
    schema = pq.read_schema(aligned)
    for col in ("home_h3_index", "work_h3_index"):
        assert schema.field(col).type == "string", col


def test_h3_id_region_is_integral(aligned):
    """`h3_id_region` is cast to int64 and fed to h3.cell_to_boundary()."""
    assert pq.read_schema(aligned).field("h3_id_region").type in ("int64", "int32")


@pytest.mark.skipif(not BINARY.exists(), reason="native binary not built")
def test_every_user_survives_to_calibration(aligned, tmp_path):
    """End-to-end: N users in, N users out.

    This is the assertion that catches a dtype regression. The C++ returns
    early for any user whose home or work H3 it failed to read, so a silent
    cast failure shows up here as zero calibrated users while every other
    stage still reports success.
    """
    outdir = tmp_path / "params"
    outdir.mkdir()
    subprocess.run(
        [
            str(BINARY),
            str(aligned),
            str(outdir),
            "0",
            "2",
            "3000",
            "1.0",
            "600",
            "0.6",
            "-0.21",
            "--quiet",
        ],
        cwd=str(outdir),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    produced = sorted(outdir.rglob("Parameters*.txt"))
    assert produced, "C++ wrote no parameter file"
    calibrated = sum(1 for p in produced for line in p.open() if line.strip())
    assert (
        calibrated == N_USERS
    ), f"expected {N_USERS} users calibrated, got {calibrated}"
