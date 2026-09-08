"""Smoke tests for the Fase 2 / Etapa 5 optimization DAG.

The Airflow container imports the DAG lazily (at task scheduler startup),
not on package import, so the simplest sanity check is compile + ``ast.parse``
of the source file plus a few shape assertions on the configured tasks.
"""

from __future__ import annotations

from pathlib import Path


def test_optimization_dag_source_compiles_and_declares_expected_policy() -> None:
    dag_path = Path(__file__).parents[1] / "airflow" / "dags" / "triage_retraining_optimization.py"
    source = dag_path.read_text(encoding="utf-8")

    compile(source, str(dag_path), "exec")
    assert 'dag_id="triage_ml_retraining_optimization"' in source
    assert "schedule=None" in source
    assert "catchup=False" in source
    assert "max_active_runs=1" in source
    assert "TRIAGE_OPTIMIZATION_ENABLED" in source
    assert "compare_slices" in source


def test_optimization_dag_resolves_helpers() -> None:
    """The DAG must import the helpers from the orchestration module only."""

    dag_path = Path(__file__).parents[1] / "airflow" / "dags" / "triage_retraining_optimization.py"
    source = dag_path.read_text(encoding="utf-8")
    expected_imports = {
        "train_with_sample_size",
        "export_onnx_for_version",
        "benchmark_for_version",
    }
    for name in expected_imports:
        assert name in source, f"{name} must be imported in the optimization DAG"
