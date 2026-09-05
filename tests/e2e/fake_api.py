"""Deterministic API double used only by browser tests."""

from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()
prediction_calls = 0


class PredictIn(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": True, "model_version": "e2e-model"}


@app.post("/predict")
def predict(payload: PredictIn, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    global prediction_calls
    if x_api_key != "doc-e2e-key":
        raise HTTPException(status_code=403, detail="forbidden")
    prediction_calls += 1
    return {
        "label": 2,
        "label_name": "cardiovascular diseases",
        "score": None,
        "latency_ms": 1.25,
        "model_version": "e2e-model",
        "request_id": "e2e-request",
    }


@app.get("/_test/stats")
def stats() -> dict[str, int]:
    return {"prediction_calls": prediction_calls}


@app.post("/_test/reset")
def reset() -> dict[str, int]:
    global prediction_calls
    prediction_calls = 0
    return {"prediction_calls": prediction_calls}
