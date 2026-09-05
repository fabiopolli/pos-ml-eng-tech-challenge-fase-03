import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist() -> None:
    required = {
        "README.md",
        "AGENTS.md",
        "pyproject.toml",
        "docs/CHECKLIST.md",
        "docs/WORKFLOW_AGENTICO.md",
        ".agents/workflow.md",
        ".agents/operating-model.md",
        ".github/workflows/ci.yml",
        "Dockerfile",
        "docker-compose.yml",
    }

    missing = sorted(path for path in required if not (ROOT / path).is_file())
    assert not missing, f"Required files are missing: {missing}"


def test_required_directories_exist() -> None:
    required = {
        "airflow/dags",
        "configs",
        "data/raw",
        "data/processed",
        "infra",
        "models",
        "monitoring/grafana/dashboards",
        "monitoring/grafana/provisioning",
        "monitoring/prometheus",
        "notebooks",
        "reports/figures",
        "scripts",
        "src/triage_ml/dev_api",
        "src/triage_ml/data",
        "src/triage_ml/models",
        "src/triage_ml/monitoring",
    }

    missing = sorted(path for path in required if not (ROOT / path).is_dir())
    assert not missing, f"Required directories are missing: {missing}"


def test_repository_does_not_track_data_artifacts_or_large_files() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    entries = [line.split(maxsplit=3) for line in result.stdout.splitlines()]
    tracked = [(fields[1], Path(fields[3])) for fields in entries]
    data_suffixes = {".csv", ".jsonl", ".parquet", ".tsv", ".xlsx"}
    prohibited = [
        path
        for _, path in tracked
        if (
            (path.parts[0] == "data" and path.suffix in data_suffixes)
            or path.suffix in {".joblib", ".onnx", ".pkl"}
            or path.name.startswith(("credentials", "service-account"))
            or path.name == ".env"
        )
    ]
    size_result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        input="\n".join(object_id for object_id, _ in tracked),
    )
    sizes = {
        object_id: int(size)
        for object_id, size in (line.split() for line in size_result.stdout.splitlines())
    }
    oversized = [path for object_id, path in tracked if sizes[object_id] > 5 * 1024 * 1024]

    assert not prohibited, f"Prohibited files are tracked: {prohibited}"
    assert not oversized, f"Files larger than 5 MiB are tracked: {oversized}"


def test_packaged_training_config_matches_project_config() -> None:
    project_config = ROOT / "configs" / "training.yaml"
    packaged_config = ROOT / "src" / "triage_ml" / "training.yaml"

    assert packaged_config.read_bytes() == project_config.read_bytes()
