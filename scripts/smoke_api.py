"""Manually exercise the smoke API and persist sanitized evidence.

This script is invoked by the developer to validate the API locally
(``uv run python scripts/smoke_api.py``). It is not part of the
runtime: it exists to satisfy the F1.T5 evidence requirement in
``docs/plans/PLAN-text-classifier.md``.

The script writes a sanitized JSON file under ``reports/evidence/`` so
that no clinical text ever appears in version control.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from triage_ml.api.app import app

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "reports" / "evidence"
EVIDENCE_FILE = EVIDENCE_DIR / "api-smoke.json"

# Tiny fixtures only. No clinical abstracts are referenced or persisted.
FIXTURES = [
    "Tumor growth in the liver of an elderly patient with metastases detected.",
    "Acute myocardial infarction in a 62-year-old after chest pain.",
    "Brain tumor removed from a young patient.",
    "Diabetic neuropathy causing chronic foot ulcer.",
    "Pneumonia in a 70 year old patient with fever.",
]


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    with client:
        health_resp = client.get("/health")
        health_resp.raise_for_status()
        health_body = health_resp.json()
        if not health_body.get("model_loaded"):
            raise RuntimeError("smoke API did not load the model")
        predict_records: list[dict[str, object]] = []
        for text in FIXTURES:
            resp = client.post("/predict", json={"text": text})
            resp.raise_for_status()
            payload = resp.json()
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
    evidence = {
        "endpoint": "/predict (smoke API)",
        "health": health_body,
        "predictions": predict_records,
        "empty_text_request": empty_record,
    }
    EVIDENCE_FILE.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"health: {health_body}")
    print(f"wrote {EVIDENCE_FILE} ({EVIDENCE_FILE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
