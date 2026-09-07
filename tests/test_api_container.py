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
    # The fronts (Streamlit) and the API share the same ``triage_ml``
    # package; setting PYTHONPATH at runtime-base lets ``streamlit run
    # front/app_dev.py`` resolve ``from triage_ml.url_validation import
    # ...`` without packaging ``src/`` as a wheel inside the image.
    assert "PYTHONPATH=/app/src" in dockerfile


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

    portal = compose["services"]["portal-prod"]
    assert portal["build"]["target"] == "portal-runtime"
    assert portal["depends_on"]["api-prod"]["condition"] == "service_healthy"
    assert portal["read_only"] is True
    assert portal["cap_drop"] == ["ALL"]
    assert portal["security_opt"] == ["no-new-privileges:true"]
    portal_environment = portal["environment"]
    assert portal_environment["TRIAGE_ML_PROD_API_URL"] == "http://api-prod:8000"

    dashboard = compose["services"]["dashboard-dev"]
    assert dashboard["profiles"] == ["dev"]
    assert dashboard["build"]["target"] == "dev-dashboard-runtime"
    assert "depends_on" not in dashboard or "api-prod" not in dashboard.get("depends_on", {})
    assert dashboard["read_only"] is True
    assert dashboard["cap_drop"] == ["ALL"]
    assert dashboard["security_opt"] == ["no-new-privileges:true"]
    dashboard_environment = dashboard["environment"]
    assert "TRIAGE_ML_DEV_API_URL" in dashboard_environment
    assert "api-prod" not in dashboard_environment["TRIAGE_ML_DEV_API_URL"]
    assert "TRIAGE_ML_DEV_API_KEY_DOCTOR" in dashboard_environment
    assert dashboard_environment["TRIAGE_ML_DEV_API_KEY_DOCTOR"].startswith(
        "${TRIAGE_ML_DEV_API_KEY_DOCTOR:?"
    )

    # The ``dashboard-dev`` default URL points at ``http://api-dev:8000``,
    # so the profile ``dev`` must also define a matching ``api-dev``
    # service — otherwise the dashboard would never connect.
    api_dev = compose["services"].get("api-dev")
    assert api_dev is not None, "compose must define api-dev under profiles: [dev]"
    assert api_dev["profiles"] == ["dev"]
    assert api_dev["read_only"] is True
    assert api_dev["cap_drop"] == ["ALL"]
    assert api_dev["security_opt"] == ["no-new-privileges:true"]
