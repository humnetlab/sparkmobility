"""
Work-directory layout for a TimeGeo run.

`TimeGeoPaths` encapsulates the on-disk scratch layout. Step entrypoints
accept a `TimeGeoPaths` (or individual `Path` subdirs derived from it) and
never hardcode `./results/...` literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeGeoPaths:
    base: Path

    def __post_init__(self) -> None:
        # Allow str or Path; coerce to Path.
        object.__setattr__(self, "base", Path(self.base))

    @property
    def data_cdr(self) -> Path:
        return self.base / "Data_CDR_to_SRFiltered"

    @property
    def srfiltered(self) -> Path:
        return self.base / "SRFiltered_to_SimInput"

    @property
    def parameters(self) -> Path:
        return self.base / "Parameters"

    @property
    def parameters_commuters(self) -> Path:
        return self.parameters / "Commuters"

    @property
    def parameters_noncommuters(self) -> Path:
        return self.parameters / "NonCommuters"

    @property
    def simulation(self) -> Path:
        return self.base / "Simulation"

    @property
    def simulation_locations(self) -> Path:
        return self.simulation / "Locations"

    @property
    def simulation_parameters(self) -> Path:
        return self.simulation / "Parameters"

    @property
    def simulation_mapped(self) -> Path:
        return self.simulation / "Mapped"

    @property
    def simulation_compressed(self) -> Path:
        return self.simulation / "Compressed"

    @property
    def analysis(self) -> Path:
        return self.base / "Analysis"

    @property
    def figs(self) -> Path:
        return self.base / "figs"

    def ensure(self) -> None:
        """Create every subdirectory. Idempotent."""
        for d in (
            self.data_cdr,
            self.srfiltered,
            self.parameters,
            self.parameters_commuters,
            self.parameters_noncommuters,
            self.simulation,
            self.simulation_locations,
            self.simulation_parameters,
            self.simulation_mapped,
            self.simulation_compressed,
            self.analysis,
            self.figs,
        ):
            d.mkdir(parents=True, exist_ok=True)
