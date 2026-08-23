"""Manually exercise the local dev API and persist sanitized evidence.

This script is invoked by the developer to validate the API locally
(``uv run python scripts/validate_api.py``). It is not part of the
runtime: it exists to satisfy the F1.T5 evidence requirement in
``docs/plans/PLAN-text-classifier.md``.

The script writes a sanitized JSON file under ``reports/evidence/`` so
that no clinical text ever appears in version control.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from triage_ml.dev_api import app as app_module
from triage_ml.dev_api import config as api_config
from triage_ml.dev_api import language as api_language

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "reports" / "evidence"
EVIDENCE_FILE = EVIDENCE_DIR / "api-dev.json"

# Tiny fixtures only. No clinical abstracts are referenced or persisted.
FIXTURES = [
    "Tumor growth in the liver of an elderly patient with metastases detected.",
    "Acute myocardial infarction in a 62-year-old after chest pain.",
    "Brain tumor removed from a young patient.",
    "Diabetic neuropathy causing chronic foot ulcer.",
    "Pneumonia in a 70 year old patient with fever.",
]

# Sentinels used to force language-detection verdicts on specific inputs.
# Text bodies are kept generic — they exist only to drive the policy.
LANG_FIXTURES = {
    "short_text": "liver tumor",
    "indeterminate_text": (
        "The study cohort included patients with mixed-language clinical notes that the "
        "detector could not classify reliably."
    ),
    "unsupported_text": (
        "Relatamos um paciente de 62 anos com infarto agudo do miocárdio após dor torácica "
        "e dispneia progressiva, submetido a angiocoronariografia que confirmou oclusão arterial."
    ),
}


def _require_error(response, *, status_code: int, error_code: str) -> None:
    body = response.json()
    if response.status_code != status_code or body.get("error_code") != error_code:
        raise RuntimeError(
            f"expected HTTP {status_code} {error_code}, got "
            f"HTTP {response.status_code} {body.get('error_code')}"
        )


def _assert_sanitized(value: object) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    forbidden_values = [*FIXTURES, *LANG_FIXTURES.values()]
    if any(text in serialized for text in forbidden_values):
        raise RuntimeError("evidence contains an input fixture")

    def has_forbidden_key(item: object) -> bool:
        if isinstance(item, dict):
            return any(
                key in {"text", "input"} or has_forbidden_key(child) for key, child in item.items()
            )
        if isinstance(item, list):
            return any(has_forbidden_key(child) for child in item)
        return False

    if has_forbidden_key(value):
        raise RuntimeError("evidence contains a forbidden input field")


@contextmanager
def _strict_lang_config():
    """Temporarily raise ``min_language_score`` so the indeterminate branch triggers."""

    original = app_module.get_api_config
    api_config.reset_api_config_cache()
    app_module.get_api_config = lambda: api_config.ApiConfig(
        supported_languages={"en"},
        min_text_chars_for_language_check=20,
        min_language_score=0.5,
    )
    try:
        yield
    finally:
        app_module.get_api_config = original
        api_config.reset_api_config_cache()


def _run_language_checks(client: TestClient) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    short_resp = client.post("/predict", json={"text": LANG_FIXTURES["short_text"]})
    _require_error(
        short_resp,
        status_code=422,
        error_code="text_too_short_for_language_check",
    )
    records.append(
        {
            "scenario": "short_text",
            "status_code": short_resp.status_code,
            "body": short_resp.json(),
            "header_server_timing": short_resp.headers.get("server-timing"),
        }
    )

    with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("en", 0.1)):
        indeterminate_resp = client.post(
            "/predict", json={"text": LANG_FIXTURES["indeterminate_text"]}
        )
    _require_error(
        indeterminate_resp,
        status_code=422,
        error_code="indeterminate_language",
    )
    records.append(
        {
            "scenario": "indeterminate_language",
            "status_code": indeterminate_resp.status_code,
            "body": indeterminate_resp.json(),
            "header_server_timing": indeterminate_resp.headers.get("server-timing"),
        }
    )

    # The normalized probability clears the strict threshold, but Portuguese
    # remains outside the configured allow-list.
    with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("pt", 0.9)):
        unsupported_resp = client.post("/predict", json={"text": LANG_FIXTURES["unsupported_text"]})
    _require_error(
        unsupported_resp,
        status_code=422,
        error_code="unsupported_language",
    )
    records.append(
        {
            "scenario": "unsupported_language",
            "status_code": unsupported_resp.status_code,
            "body": unsupported_resp.json(),
            "header_server_timing": unsupported_resp.headers.get("server-timing"),
        }
    )

    return records


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Plain-mode run: ``min_language_score = 0.0`` (default) so the real
    # langid detector accepts all five English fixtures.
    api_config.reset_api_config_cache()
    with TestClient(app_module.app) as client:
        health_resp = client.get("/health")
        health_resp.raise_for_status()
        health_body = health_resp.json()
        if not health_body.get("model_loaded"):
            raise RuntimeError("dev API did not load the model")

        model_info_resp = client.get("/model-info")
        model_info_resp.raise_for_status()
        model_info_body = model_info_resp.json()

        predict_records: list[dict[str, object]] = []
        for text in FIXTURES:
            resp = client.post("/predict", json={"text": text})
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("request_id") != resp.headers.get("x-request-id"):
                raise RuntimeError("prediction request ID does not match X-Request-ID")
            timing = resp.headers.get("server-timing", "")
            if "detect;dur=" not in timing or "predict;dur=" not in timing:
                raise RuntimeError("prediction response is missing Server-Timing stages")
            predict_records.append(
                {
                    "label": payload.get("label"),
                    "label_name": payload.get("label_name"),
                    "score": payload.get("score"),
                    "model_version": payload.get("model_version"),
                    "latency_ms": payload.get("latency_ms"),
                    "request_id": payload.get("request_id"),
                    "warnings": payload.get("warnings"),
                    "headers_x_request_id": resp.headers.get("x-request-id"),
                    "header_server_timing": resp.headers.get("server-timing"),
                }
            )

        empty_resp = client.post("/predict", json={"text": ""})
        if empty_resp.status_code != 422:
            raise RuntimeError(f"empty text returned HTTP {empty_resp.status_code}, expected 422")
        empty_record = {
            "status_code": empty_resp.status_code,
            "body": empty_resp.json(),
            "headers_x_request_id": empty_resp.headers.get("x-request-id"),
        }

        models_resp = client.get("/models")
        models_resp.raise_for_status()
        models_body = models_resp.json()

        reload_target = next(
            (
                version
                for version in models_body["versions"]
                if version != models_body.get("current")
            ),
            None,
        )
        if reload_target is None:
            reload_success_body = {"skipped": "no alternate valid model version"}
        else:
            reload_success_resp = client.post("/reload", json={"model_version": reload_target})
            if reload_success_resp.status_code != 200:
                raise RuntimeError(
                    f"reload to alternate version returned HTTP "
                    f"{reload_success_resp.status_code}, expected 200"
                )
            reload_success_body = reload_success_resp.json()

        reload_not_found_resp = client.post(
            "/reload",
            json={"model_version": "99999999T999999Z-deadbeef0000"},
        )
        if reload_not_found_resp.status_code != 404:
            raise RuntimeError(
                f"reload to unknown version returned HTTP "
                f"{reload_not_found_resp.status_code}, expected 404"
            )
        reload_not_found_body = reload_not_found_resp.json()

    # Strict-mode run uses deterministic normalized probabilities so both
    # policy rejection branches remain reproducible.
    with _strict_lang_config(), TestClient(app_module.app) as strict_client:
        language_records = _run_language_checks(strict_client)

    evidence = {
        "endpoint": "/predict (dev API)",
        "health": health_body,
        "model_info_summary": {
            "model_version": model_info_body.get("model_version"),
            "model_name": model_info_body.get("model_name"),
            "task_type": model_info_body.get("task_type"),
            "language": model_info_body.get("language"),
            "n_train": model_info_body.get("n_train"),
            "n_test": model_info_body.get("n_test"),
            "selected_classifier": model_info_body.get("selection", {}).get("selected_classifier"),
            "metric": model_info_body.get("selection", {}).get("metric"),
            "metrics": {
                "accuracy": model_info_body.get("metrics", {}).get("accuracy"),
                "balanced_accuracy": model_info_body.get("metrics", {}).get("balanced_accuracy"),
                "macro_f1": model_info_body.get("metrics", {}).get("macro_f1"),
                "weighted_f1": model_info_body.get("metrics", {}).get("weighted_f1"),
            },
            "git_commit": model_info_body.get("git_commit"),
            "git_dirty": model_info_body.get("git_dirty"),
            "created_at": model_info_body.get("created_at"),
        },
        "models": models_body,
        "reload_success": reload_success_body,
        "reload_not_found": reload_not_found_body,
        "predictions": predict_records,
        "language_checks": language_records,
        "empty_text_request": empty_record,
    }
    _assert_sanitized(evidence)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=EVIDENCE_DIR,
        prefix=".api-dev-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(evidence, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(EVIDENCE_FILE)
    print(f"health: {health_body}")
    print(f"wrote {EVIDENCE_FILE} ({EVIDENCE_FILE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
