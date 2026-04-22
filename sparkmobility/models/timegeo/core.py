"""
Public entry point for the TimeGeo mobility simulator.

Matches the fit/generate surface used by `Gravity` and `EPR`:

    from sparkmobility.models.timegeo import TimeGeo
    model = TimeGeo(num_days=1, num_cpus=8)
    model.fit(stay_df)
    traj_df = model.generate()
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from sparkmobility.logging_setup import configure_if_needed
from sparkmobility.models.timegeo._native.runner import CppModuleHandler
from sparkmobility.models.timegeo._pipeline import data_alignment
from sparkmobility.models.timegeo._pipeline.s1_sr_to_siminput import (
    clean_and_format_fa_users_parquet,
    decode_and_write_parameters,
    extract_frequent_users_parquet,
    extract_stay_regions_for_frequent_users_parquet,
    remove_redundant_stays_parquet,
)
from sparkmobility.models.timegeo._pipeline.s2_simulation_prep import (
    activeness,
    generate_simulation_input_parquet,
    generate_simulation_parameters,
    otherLocations,
    split_simulation_inputs,
)
from sparkmobility.models.timegeo._pipeline.s3_simulation_mapper import (
    simulate_all_parallel,
)
from sparkmobility.models.timegeo._pipeline.s4_postprocessing import (
    compress_and_export_simulation_results,
)
from sparkmobility.models.timegeo.paths import TimeGeoPaths

logger = logging.getLogger(__name__)

_DEFAULT_B_ARRAY = list(range(1, 21))


class TimeGeo:
    """TimeGeo mobility simulator with a fit/generate API."""

    def __init__(
        self,
        num_days: int = 1,
        num_cpus: int = 8,
        sample_fraction: float = 1.0,
        min_unique_locations: int = 15,
        skip_frequent_user_extraction: bool = False,
        min_num_stay: int = 1,
        max_num_stay: int = 100000,
        rho: float = 0.6,
        gamma: float = -0.21,
        nw_thres: float = 1.0,
        slot_interval: int = 600,
        work_prob_weekday: float = 0.829,
        work_prob_weekend: float = 0.354,
        reg_prob: float = 0.846,
        gmm_group_index: int = 0,
        random_state: Optional[int] = None,
        name: str = "TimeGeo model",
        configure_logging_if_needed: bool = True,
    ):
        if configure_logging_if_needed:
            configure_if_needed()
        self.num_days = num_days
        self.num_cpus = num_cpus
        self.sample_fraction = sample_fraction
        self.min_unique_locations = min_unique_locations
        self.skip_frequent_user_extraction = skip_frequent_user_extraction
        self.min_num_stay = min_num_stay
        self.max_num_stay = max_num_stay
        self.rho = rho
        self.gamma = gamma
        self.nw_thres = nw_thres
        self.slot_interval = slot_interval
        self.work_prob_weekday = work_prob_weekday
        self.work_prob_weekend = work_prob_weekend
        self.reg_prob = reg_prob
        self.gmm_group_index = gmm_group_index
        self.random_state = random_state
        self.name = name

        self._paths: Optional[TimeGeoPaths] = None
        self._work_dir_is_temp: bool = False
        self._cpp = CppModuleHandler()
        self.fitted_: bool = False
        self.fit_stats_: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return (
            f'TimeGeo(name="{self.name}", num_days={self.num_days}, '
            f"num_cpus={self.num_cpus}, rho={self.rho}, gamma={self.gamma})"
        )

    @property
    def work_dir(self) -> Optional[Path]:
        return self._paths.base if self._paths else None

    def fit(
        self,
        stay_df: pd.DataFrame,
        user_column: str = "caid",
        timestamp_column: str = "stay_start_timestamp",
        location_column: str = "h3_id_region",
        type_column: str = "type",
        work_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Estimate parameters from a stay-point DataFrame.

        Writes intermediates under ``work_dir`` (tempdir by default) and
        returns a stats dict. Call :meth:`generate` afterwards.
        """
        if work_dir is None:
            self._paths = TimeGeoPaths(Path(tempfile.mkdtemp(prefix="timegeo_")))
            self._work_dir_is_temp = True
        else:
            self._paths = TimeGeoPaths(Path(work_dir))
            self._work_dir_is_temp = False
        self._paths.ensure()
        logger.info("TimeGeo work_dir=%s", self._paths.base)

        # Stage 1a: DataFrame → aligned parquet
        aligned_path = data_alignment.align(
            stay_df,
            self._paths,
            user_column=user_column,
            timestamp_column=timestamp_column,
            location_column=location_column,
            type_column=type_column,
        )

        # Stage 1b: drop consecutive same-location stays
        filtered_path = self._paths.srfiltered / "FilteredStayRegions_set.parquet"
        remove_redundant_stays_parquet(str(aligned_path), str(filtered_path))

        # Stage 1c: frequent-active-user filter (optional)
        if self.skip_frequent_user_extraction or self.min_unique_locations <= 0:
            logger.info(
                "skipping FA-user filter (min_unique_locations=%d, skip=%s)",
                self.min_unique_locations,
                self.skip_frequent_user_extraction,
            )
            fa_stay_regions_path = filtered_path
        else:
            fa_users_path = self._paths.srfiltered / "FAUsers.parquet"
            extract_frequent_users_parquet(
                input_path=str(filtered_path),
                output_path=str(fa_users_path),
                num_stays_threshold=self.min_unique_locations,
            )
            fa_stay_regions_path = (
                self._paths.srfiltered / "FAUsers_StayRegions.parquet"
            )
            extract_stay_regions_for_frequent_users_parquet(
                fa_users_path=str(fa_users_path),
                input_path=str(filtered_path),
                output_path=str(fa_stay_regions_path),
            )

        # Stage 1d: label home/work/other + sequential "other" indices
        cleaned_path = self._paths.srfiltered / "FAUsers_Cleaned_Formatted.parquet"
        clean_and_format_fa_users_parquet(
            input_path=str(fa_stay_regions_path),
            output_path=str(cleaned_path),
        )

        # Stage 2: C++ parameter estimation. It reads the aligned parquet
        # (has Latitude/Longitude + string `type`) filtered by the FA user set.
        cleaned_df = pd.read_parquet(cleaned_path)
        frequent_user_ids = cleaned_df["caid"].unique().tolist()
        num_users = len(frequent_user_ids)
        logger.info("parameter generation over %d users", num_users)

        # When FA was skipped, the aligned parquet already has exactly the
        # users that cleaned_df covers — hand C++ the aligned path directly.
        # Otherwise scan-and-filter with an Arrow predicate so we never
        # materialize the full aligned parquet in pandas.
        if self.skip_frequent_user_extraction or self.min_unique_locations <= 0:
            param_input_path = aligned_path
        else:
            param_input_path = self._paths.srfiltered / "FilteredForParameters.parquet"
            filtered_table = ds.dataset(str(aligned_path), format="parquet").to_table(
                filter=ds.field("caid").isin(frequent_user_ids)
            )
            pq.write_table(
                filtered_table, str(param_input_path), row_group_size=100_000
            )

        # Non-commuters and commuters share the input but write distinct
        # subdirs (parameters_commuters/, parameters_noncommuters/), so the
        # two passes are independent. Run them concurrently — each C++
        # process self-caps to 2 GB via setrlimit, so combined peak is
        # bounded at 4 GB.
        def _run(commuter_mode: bool) -> bool:
            return self._cpp.run_parameter_generation(
                input_path=param_input_path,
                output_dir=self._paths.parameters,
                commuter_mode=commuter_mode,
                min_num_stay=self.min_num_stay,
                max_num_stay=self.max_num_stay,
                nw_thres=self.nw_thres,
                slot_interval=self.slot_interval,
                rho=self.rho,
                gamma=self.gamma,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_nc = pool.submit(_run, False)
            fut_c = pool.submit(_run, True)
            ok_nc = fut_nc.result()
            ok_c = fut_c.result()
        if not ok_nc:
            raise RuntimeError("C++ parameter generation (non-commuters) failed")
        if not ok_c:
            raise RuntimeError("C++ parameter generation (commuters) failed")

        # Stage 3: decode indexed (b1, b2) → numeric values
        commuter_raw = self._paths.parameters_commuters / "ParametersCommuters.txt"
        noncommuter_raw = (
            self._paths.parameters_noncommuters / "ParametersNonCommuters.txt"
        )
        commuter_dec = self._paths.parameters / "ParametersCommuters.txt"
        noncommuter_dec = self._paths.parameters / "ParametersNonCommuters.txt"
        num_commuters = _count_lines(commuter_raw)
        num_noncommuters = _count_lines(noncommuter_raw)
        if num_commuters == 0 and num_noncommuters == 0:
            raise RuntimeError(
                "parameter estimation produced no commuter or non-commuter records"
            )
        decode_and_write_parameters(
            b1_array=_DEFAULT_B_ARRAY,
            b2_array=_DEFAULT_B_ARRAY,
            commuter_input_path=str(commuter_raw),
            noncommuter_input_path=str(noncommuter_raw),
            commuter_output_path=str(commuter_dec),
            noncommuter_output_path=str(noncommuter_dec),
        )

        # Stage 4: simulation-prep artifacts
        sim_loc = self._paths.simulation / "simulation_location.txt"
        sim_par = self._paths.simulation / "simulation_parameter.txt"
        generate_simulation_input_parquet(
            input_path=str(cleaned_path), output_path=str(sim_loc)
        )
        generate_simulation_parameters(
            commuter_path=str(commuter_dec),
            noncommuter_path=str(noncommuter_dec),
            output_path=str(sim_par),
            work_prob_weekday=self.work_prob_weekday,
            work_prob_weekend=self.work_prob_weekend,
            num_days=self.num_days,
            reg_prob=self.reg_prob,
            gmm_group_index=self.gmm_group_index,
        )
        split_simulation_inputs(
            parameter_path=str(sim_par),
            location_path=str(sim_loc),
            formatted_user_path=str(cleaned_path),
            output_dir=str(self._paths.simulation),
            num_cpus=self.num_cpus,
        )
        activeness(
            noncomm_daily_path=str(self._paths.parameters / "NonComm_pt_daily.txt"),
            noncomm_weekly_path=str(self._paths.parameters / "NonComm_pt_weekly.txt"),
            comm_daily_path=str(self._paths.parameters / "Comm_pt_daily.txt"),
            comm_weekly_path=str(self._paths.parameters / "Comm_pt_weekly.txt"),
            output_path=str(self._paths.simulation / "activeness.txt"),
        )
        otherLocations(
            input_path=str(sim_loc),
            output_path=str(self._paths.simulation / "otherlocation.txt"),
            sample_fraction=self.sample_fraction,
        )

        num_locations = (
            int(cleaned_df.groupby("caid")["location_index"].nunique().sum())
            if "location_index" in cleaned_df.columns
            else 0
        )
        self.fit_stats_ = {
            "num_users": num_users,
            "num_commuters": num_commuters,
            "num_noncommuters": num_noncommuters,
            "num_locations": num_locations,
            "work_dir": str(self._paths.base),
            "parameter_file": str(sim_par),
        }
        self.fitted_ = True
        logger.info(
            "fit complete: %d users (%d commuters, %d non-commuters)",
            num_users,
            num_commuters,
            num_noncommuters,
        )
        return self.fit_stats_

    def generate(
        self,
        n_agents: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Simulate trajectories from the fitted model.

        Returns a DataFrame with columns
        ``[user, datetime, location_type, longitude, latitude]`` where
        ``datetime`` is derived from slot index × ``slot_interval``.
        """
        if not self.fitted_ or self._paths is None:
            raise RuntimeError("TimeGeo.generate() requires a prior fit() call")

        sim_dir = self._paths.simulation
        simulate_all_parallel(
            num_cpus=self.num_cpus,
            other_locations_file=str(sim_dir / "otherlocation.txt"),
            activeness_file=str(sim_dir / "activeness.txt"),
            num_days=self.num_days,
            start_slot=0,
            users_locations_dir=str(self._paths.simulation_locations),
            users_parameters_dir=str(self._paths.simulation_parameters),
            output_dir=str(self._paths.simulation_mapped),
        )

        parquet_file = sim_dir / "simulation_results.parquet"
        compress_and_export_simulation_results(
            input_folder=str(self._paths.simulation_mapped) + "/",
            compressed_folder=str(self._paths.simulation_compressed) + "/",
            parquet_file=str(parquet_file),
            analysis_dir=str(self._paths.analysis) + "/",
        )

        df = pd.read_parquet(parquet_file)
        if n_agents is not None and n_agents > 0:
            users = df["user_id"].drop_duplicates().head(n_agents)
            df = df[df["user_id"].isin(users)]

        # Slot → relative datetime (epoch + slot * slot_interval).
        base = pd.Timestamp("1970-01-01")
        df = df.rename(columns={"user_id": "user"})
        df["datetime"] = base + pd.to_timedelta(
            df["timeslot"].astype("int64") * self.slot_interval, unit="s"
        )
        out_cols = ["user", "datetime", "location_type", "longitude", "latitude"]
        return df[out_cols].reset_index(drop=True)

    def cleanup(self) -> None:
        """Remove the work directory if it was a tempdir (idempotent)."""
        if self._paths is None:
            return
        if self._work_dir_is_temp and self._paths.base.exists():
            shutil.rmtree(self._paths.base, ignore_errors=True)
            logger.info("cleaned up temp work_dir %s", self._paths.base)
        self._paths = None
        self._work_dir_is_temp = False


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)
