from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_api_dockerfile_has_reproducible_runtime_guards() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "AS portal-runtime" in dockerfile
    assert "AS dev-dashboard-runtime" in dockerfile
    assert "AS runtime" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert "COPY models" not in dockerfile


def test_api_compose_mounts_models_read_only_and_requires_secrets() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["api-prod"]
    environment = service["environment"]

    assert service["volumes"] == ["./models:/models:ro"]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert environment["MODEL_PATH"].startswith("${API_MODEL_PATH:?")
    for name in (
        "TRIAGE_ML_API_KEY_SERVICE",
        "TRIAGE_ML_API_KEY_DOCTOR",
        "TRIAGE_ML_API_KEY_PATIENT",
    ):
        assert environment[name].startswith(f"${{{name}:?")


def test_front_containers_are_isolated_and_depend_on_healthy_api() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    for name, target in (
        ("portal-prod", "portal-runtime"),
        ("dashboard-dev", "dev-dashboard-runtime"),
    ):
        service = compose["services"][name]
        assert service["build"]["target"] == target
        assert service["depends_on"]["api-prod"]["condition"] == "service_healthy"
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]

    portal_environment = compose["services"]["portal-prod"]["environment"]
    assert portal_environment["TRIAGE_ML_PROD_API_URL"] == "http://api-prod:8000"
    dashboard_environment = compose["services"]["dashboard-dev"]["environment"]
    assert dashboard_environment["TRIAGE_ML_DEV_API_URL"] == "http://api-prod:8000"
