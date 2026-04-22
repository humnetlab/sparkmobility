"""
Runner for the C++ parameter-estimation binary (module_2_3_1).

Streams the binary's stdout/stderr line-by-line through a log filter that
collapses two known-noisy patterns:

- "Processing row group N of M" (up to hundreds per run) — suppressed;
  replaced by a single end-of-run summary.
- "Error processing row group N: std::bad_alloc" — suppressed individually;
  replaced by a single summary with the count and the first/last group ids.

Everything else passes through as INFO.

Binary lives next to this file. The `--quiet` flag and TIMEGEO_QUIET=1 env
var both silence the per-row-group chatter on the C++ side as well.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_BINARY = _HERE / "module_2_3_1"

_RE_ROWGROUP = re.compile(r"^Processing row group (\d+) of (\d+)$")
_RE_BADALLOC = re.compile(r"^Error processing row group (\d+): std::bad_alloc$")


class _LogFilter:
    """Line-by-line filter that collapses noisy patterns into summaries."""

    def __init__(self, log: logging.Logger):
        self.log = log
        self._row_group_count = 0
        self._row_group_total: Optional[int] = None
        self._row_group_started: Optional[float] = None
        self._bad_alloc_groups: List[int] = []

    def feed(self, line: str) -> None:
        m = _RE_ROWGROUP.match(line)
        if m:
            self._row_group_count += 1
            self._row_group_total = int(m.group(2))
            if self._row_group_started is None:
                self._row_group_started = time.monotonic()
            return
        m = _RE_BADALLOC.match(line)
        if m:
            self._bad_alloc_groups.append(int(m.group(1)))
            return
        if line:
            self.log.info("module_2_3_1: %s", line)

    def flush(self) -> None:
        if self._row_group_count:
            elapsed = (
                time.monotonic() - self._row_group_started
                if self._row_group_started
                else 0.0
            )
            self.log.info(
                "module_2_3_1: processed %d/%d row groups in %.1fs",
                self._row_group_count,
                self._row_group_total or self._row_group_count,
                elapsed,
            )
        if self._bad_alloc_groups:
            g = self._bad_alloc_groups
            self.log.error(
                "module_2_3_1: %d row groups failed with std::bad_alloc (groups %d\u2013%d)",
                len(g),
                g[0],
                g[-1],
            )


class CppModuleHandler:
    """Thin wrapper that invokes the C++ binary via subprocess."""

    def __init__(self, module_path: Path | str | None = None):
        self.module_path = Path(module_path) if module_path else _BINARY
        if not self.module_path.exists():
            raise FileNotFoundError(
                f"C++ binary not found at {self.module_path}. "
                f"Reinstall sparkmobility (`pip install sparkmobility`) or, "
                f"for an editable checkout, run `pip install -e .` from the repo root."
            )
        if not os.access(self.module_path, os.X_OK):
            raise PermissionError(f"C++ binary at {self.module_path} is not executable")

    def run_parameter_generation(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        commuter_mode: bool = False,
        min_num_stay: int = 2,
        max_num_stay: int = 3000,
        nw_thres: float = 1.0,
        slot_interval: int = 600,
        rho: float = 0.6,
        gamma: float = -0.21,
        quiet: bool = True,
        timeout: int = 3600,
    ) -> bool:
        """Run parameter generation and write outputs under ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(input_path).resolve()

        cmd = [
            str(self.module_path.resolve()),
            str(input_path),
            str(output_dir.resolve()),
            "1" if commuter_mode else "0",
            str(min_num_stay),
            str(max_num_stay),
            str(nw_thres),
            str(slot_interval),
            str(rho),
            str(gamma),
        ]
        if quiet:
            cmd.append("--quiet")

        logger.info(
            "running module_2_3_1 (%s) input=%s output_dir=%s",
            "commuters" if commuter_mode else "non-commuters",
            input_path.name,
            output_dir,
        )

        # cwd=output_dir: some C++ writes land in cwd via relative paths
        # (e.g. NonComm_pt_daily.txt). Setting cwd here keeps them next to
        # the other parameter files instead of the caller's working directory.
        env = os.environ.copy()
        if quiet:
            env["TIMEGEO_QUIET"] = "1"

        filt = _LogFilter(logger)
        start = time.monotonic()
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(output_dir),
                env=env,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    filt.feed(line.rstrip("\n"))
                    if time.monotonic() - start > timeout:
                        proc.kill()
                        logger.error("module_2_3_1 timed out after %ds", timeout)
                        filt.flush()
                        return False
                rc = proc.wait()
        except OSError as exc:
            logger.error("failed to launch module_2_3_1: %s", exc)
            return False

        filt.flush()
        if rc != 0:
            logger.error("module_2_3_1 failed (exit %d)", rc)
            return False
        return True

    def check_module_availability(self) -> Dict[str, Any]:
        return {
            "binary_path": str(self.module_path),
            "binary_exists": self.module_path.exists(),
            "binary_executable": os.access(self.module_path, os.X_OK),
            "binary_size": (
                self.module_path.stat().st_size if self.module_path.exists() else 0
            ),
        }
