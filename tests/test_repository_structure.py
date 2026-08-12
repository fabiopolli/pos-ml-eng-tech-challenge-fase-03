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
        "src/triage_ml/api",
        "src/triage_ml/data",
        "src/triage_ml/models",
        "src/triage_ml/monitoring",
    }

    missing = sorted(path for path in required if not (ROOT / path).is_dir())
    assert not missing, f"Required directories are missing: {missing}"
