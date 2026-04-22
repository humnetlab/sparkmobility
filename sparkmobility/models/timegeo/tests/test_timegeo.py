"""Smoke tests for the TimeGeo model's public API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Locate the package root so the subprocess test can find it from any cwd.
_PKG_ROOT = Path(__file__).resolve().parents[4]


def test_import_from_any_cwd(tmp_path: Path) -> None:
    """`from sparkmobility.models.timegeo import TimeGeo` must work from /tmp.

    Guards against the old requirement to `os.chdir(timegeo_organized/)` and
    `sys.path.insert`.
    """
    script = "from sparkmobility.models.timegeo import TimeGeo; print(TimeGeo().__class__.__name__)"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_PKG_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, res.stderr
    assert "TimeGeo" in res.stdout


def test_constructor_defaults() -> None:
    from sparkmobility.models.timegeo import TimeGeo

    model = TimeGeo()
    assert model.num_days == 1
    assert model.min_unique_locations == 15
    assert model.skip_frequent_user_extraction is False
    assert model.fitted_ is False


def test_generate_before_fit_raises() -> None:
    from sparkmobility.models.timegeo import TimeGeo

    model = TimeGeo()
    with pytest.raises(RuntimeError, match="fit"):
        model.generate()


def test_cleanup_idempotent() -> None:
    from sparkmobility.models.timegeo import TimeGeo

    model = TimeGeo()
    model.cleanup()  # no work_dir allocated yet
    model.cleanup()  # second call must be safe


def test_log_filter_collapses_noise() -> None:
    import logging

    from sparkmobility.models.timegeo._native.runner import _LogFilter

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("timegeo.test")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    f = _LogFilter(logger)
    for line in [
        "Starting program...",
        "Processing row group 1 of 3",
        "Processing row group 2 of 3",
        "Error processing row group 3: std::bad_alloc",
        "Ending program...",
    ]:
        f.feed(line)
    f.flush()

    msgs = [r.getMessage() for r in records]
    # "Processing row group" per-line output must not leak through.
    assert not any("Processing row group 1" in m for m in msgs)
    # Summary line must appear.
    assert any("processed 2/3 row groups" in m for m in msgs)
    # bad_alloc collapsed into a single error.
    assert sum(1 for m in msgs if "std::bad_alloc" in m) == 1
