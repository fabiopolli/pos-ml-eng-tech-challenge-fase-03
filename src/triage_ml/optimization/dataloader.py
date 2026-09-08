"""Dataset sizing helpers (Fase 2, Etapa 5).

Wraps ``triage_ml.data.prepare.prepare_dataset`` so that the optimization
pipeline can run the same preparation algorithm (with the documented exclusion
rules, seed handling and ``PreparationReport`` semantics) on alternative
``sample_size`` values without duplicating business logic.

Two env vars and one config knob steer the slice catalog at runtime:

* ``configs/training.yaml`` declares an optional ``dataset_sizing`` list.
* ``TRIAGE_OPTIMIZATION_ENABLED`` is the master switch read by the DAG factory;
  when ``false`` (the default to keep the Etapa 7 DAG working unchanged) the
  helpers still operate but the DAG does not iterate.
* ``TRIAGE_DATASET_SLICES`` (comma separated integers) overrides the YAML
  catalog at runtime so operators can probe a single slice in isolation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from triage_ml.data.prepare import PreparationReport, prepare_dataset


@dataclass(frozen=True)
class DatasetSlice:
    """A single (slice_name, sample_size) pair to materialize."""

    name: str
    sample_size: int

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive; got {self.sample_size}")
        if not self.name or not self.name.replace("-", "_").replace("_", "").isalnum():
            raise ValueError(f"slice name must be alphanumeric; got {self.name!r}")


def _resolve_slice_catalog(config: dict[str, Any]) -> list[int]:
    """Return the ordered list of ``sample_size`` integers to iterate on."""

    env_value = os.environ.get("TRIAGE_DATASET_SLICES")
    if env_value:
        catalog: list[int] = []
        for raw in env_value.split(","):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                catalog.append(int(stripped))
            except ValueError as exc:
                raise ValueError(
                    "TRIAGE_DATASET_SLICES must be a comma-separated list of integers; "
                    f"got {env_value!r}"
                ) from exc
        if not catalog:
            raise ValueError("TRIAGE_DATASET_SLICES parsed to an empty list")
        return sorted(set(catalog))

    raw = config.get("dataset_sizing")
    if raw is None:
        return [int(config.get("sample_size", 5_000))]
    if not isinstance(raw, list) or not raw:
        raise ValueError("configs/training.yaml: dataset_sizing must be a non-empty list")
    catalog = []
    for entry in raw:
        if isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0:
            raise ValueError(f"configs/training.yaml: invalid dataset_sizing entry {entry!r}")
        catalog.append(entry)
    return sorted(set(catalog))


def slice_name_for(sample_size: int) -> str:
    """Stable, human-readable slice name keyed on the sample size."""

    if sample_size >= 1_000:
        return f"slice_{sample_size // 1000}k"
    return f"slice_{sample_size}"


def iter_dataset_slices(
    config: dict[str, Any],
    *,
    base_csv: str | Path | None = None,
) -> Iterator[DatasetSlice]:
    """Yield ``DatasetSlice`` descriptors from the given training config.

    ``base_csv`` validation is eager: any path passed in must resolve to an
    existing file before the iterator yields. The check is done once so the
    CLI listing (``TRIAGE_DATASET_SLICES``) can still build its catalog even
    when the operator has not yet downloaded the raw CSV.
    """

    base_csv_path = Path(base_csv) if base_csv is not None else None
    if base_csv_path is not None and not base_csv_path.is_file():
        raise FileNotFoundError(f"base_csv not found: {base_csv_path}")
    for sample_size in _resolve_slice_catalog(config):
        yield DatasetSlice(name=slice_name_for(sample_size), sample_size=sample_size)


def load_canonical_for_slice(
    config: dict[str, Any],
    *,
    base_csv: str | Path,
    sample_size: int,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, PreparationReport]:
    """Run the canonical preparation pipeline on ``base_csv`` for one slice.

    Reproducibility: the seed is taken from ``config.random_state`` unless
    ``random_state`` is explicitly provided (Airflow passes the same seed for
    every slice).
    """

    seed = random_state if random_state is not None else int(config["random_state"])
    raw = pd.read_csv(Path(base_csv))
    prepared, report = prepare_dataset(raw, sample_size=sample_size, random_state=seed)
    return prepared, report
