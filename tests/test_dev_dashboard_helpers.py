"""Hermetic tests for the dev dashboard helpers.

The dashboard itself runs inside Streamlit, which is heavy to spin up in
unit tests. We cover the deterministic helpers (``_check_health``,
``_post_predict``, ``_render_response`` and the preset table) by mocking
``requests.request`` so we don't need a live API.

Streamlit is imported lazily inside the module (see ``front/app_dev.py``
import guard below) — when not available, only the pure helpers execute.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "front" / "app_dev.py"


@pytest.fixture
def dashboard_module():
    """Import the dashboard module bypassing the Streamlit runtime."""

    spec = importlib.util.spec_from_file_location("app_dev", DASHBOARD_PATH)
    assert spec is not None and spec.loader is not None, "could not load app_dev"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass decorators find the module name.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - streamlit missing
        pytest.skip(f"streamlit not installed: {exc}")
    return module


class _FakeResponse:
    """Mimics ``requests.Response`` enough for the dashboard helpers."""

    def __init__(
        self,
        *,
        status_code: int,
        body: object,
        headers: dict[str, str] | None = None,
        elapsed_seconds: float = 0.123,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.elapsed = type("elapsed", (), {"total_seconds": lambda self: elapsed_seconds})()
        self.text = json.dumps(body, ensure_ascii=False)

    def json(self) -> object:
        return self._body


class _EnterableMock(MagicMock):
    """MagicMock that is also a no-op context manager.

    Streamlit's ``st.expander(...)`` returns a container that supports
    both attribute access (for streamlit writes) and the context-manager
    protocol. Returning ``self`` from ``__enter__`` keeps ``with`` blocks
    alive so the wrapped body actually executes.
    """

    def __enter__(self) -> _EnterableMock:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def test_health_calls_endpoint_and_parses_response(dashboard_module):
    response = _FakeResponse(
        status_code=200,
        body={
            "status": "ok",
            "model_version": "20260823T135811Z-bed2194376bc",
            "model_loaded": True,
        },
        headers={
            "X-Request-ID": "abc123",
            "Server-Timing": "detect;dur=0.500, predict;dur=1.234",
        },
    )
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        result = dashboard_module._check_health("http://api.example.com")
    assert mocked.call_count == 1
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "GET"
    assert url == "http://api.example.com/health"
    assert mocked.call_args.kwargs["timeout"] == dashboard_module.REQUEST_TIMEOUT_SECONDS
    assert result.status_code == 200
    assert result.body["model_loaded"] is True
    assert result.server_timing == "detect;dur=0.500, predict;dur=1.234"
    assert result.request_id == "abc123"
    assert result.elapsed_ms == pytest.approx(123.0, abs=0.01)


def test_predict_posts_json_payload(dashboard_module):
    response = _FakeResponse(
        status_code=200,
        body={
            "label": 1,
            "label_name": "neoplasms",
            "score": None,
            "model_version": "20260823T135811Z-bed2194376bc",
            "latency_ms": 1.5,
            "request_id": "deadbeef",
            "warnings": [],
        },
    )
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        result = dashboard_module._post_predict("http://api.example.com/", "Tumor growth")
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "POST"
    assert url == "http://api.example.com/predict"
    assert mocked.call_args.kwargs["json"] == {"text": "Tumor growth"}
    assert result.status_code == 200
    assert result.body["label_name"] == "neoplasms"


def test_invalid_json_body_is_wrapped_in_raw(dashboard_module):
    response = _FakeResponse(status_code=500, body={})
    response.json = MagicMock(side_effect=ValueError("not json"))
    response.text = "<html>oops</html>"
    with patch.object(dashboard_module.requests, "request", return_value=response):
        result = dashboard_module._post_predict("http://api.example.com", "anything")
    assert result.status_code == 500
    assert result.body == {"raw": "<html>oops</html>"}


def test_request_exception_surfaces_to_caller(dashboard_module):
    with patch.object(
        dashboard_module.requests,
        "request",
        side_effect=dashboard_module.requests.RequestException("boom"),
    ):
        with pytest.raises(dashboard_module.requests.RequestException):
            dashboard_module._check_health("http://api.example.com")


def test_language_presets_have_expected_error_codes(dashboard_module):
    presets = dashboard_module.LANGUAGE_PRESETS
    assert presets["Texto curto (<20 chars)"]["expected_error_code"] == (
        "text_too_short_for_language_check"
    )
    assert presets["Confiança baixa (mock)"]["expected_error_code"] == ("indeterminate_language")
    assert presets["Idioma fora do allow-list"]["expected_error_code"] == ("unsupported_language")
    assert presets["Inglês válido"]["expected_error_code"] is None
    # Texts must satisfy the minimum length except for the short preset.
    for name, payload in presets.items():
        if name == "Texto curto (<20 chars)":
            assert len(payload["text"]) < 20
        else:
            assert len(payload["text"]) >= 20


def test_url_stripping_handles_trailing_slash(dashboard_module):
    response = _FakeResponse(status_code=200, body={"status": "ok", "model_loaded": True})
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        dashboard_module._check_health("http://api.example.com///")
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "GET"
    assert url == "http://api.example.com/health"


def test_documentation_shortcuts_point_to_existing_files(dashboard_module):
    """The sidebar shortcuts resolve to real markdown files via file:// URIs.

    Regression for the broken relative links (they used to point to
    ``/docs/...`` inside the Streamlit server, which 404s).
    """

    expected = {
        dashboard_module.DOC_PLAN: "Plan do classificador",
        dashboard_module.DOC_CHECKLIST: "Checklist oficial",
        dashboard_module.DOC_REPORT_FASE_1: "Relatório Fase 1",
    }
    for path, label in expected.items():
        assert path.is_file(), f"{label} target missing: {path}"
        uri = path.as_uri()
        assert uri.startswith("file:///"), f"{label} uri malformed: {uri}"
        assert path.name in uri, f"{label} uri should mention the filename"


def test_repo_root_is_above_dashboard(dashboard_module):
    """``REPO_ROOT`` walks one level up from ``front/app_dev.py`` to the repo root."""

    # This test lives at ``tests/test_dev_dashboard_helpers.py`` so its
    # parent is the project root; the dashboard lives at
    # ``<root>/front/app_dev.py`` and ``REPO_ROOT`` should match.
    expected_root = Path(__file__).resolve().parents[1]
    assert dashboard_module.REPO_ROOT.resolve() == expected_root.resolve()


# ---------------------------------------------------------------------------
# ``/model-info`` sidebar
# ---------------------------------------------------------------------------


def test_get_model_info_hits_endpoint(dashboard_module):
    response = _FakeResponse(
        status_code=200,
        body={
            "model_version": "20260823T135811Z-bed2194376bc",
            "model_name": "tfidf_linear_svc_baseline",
            "task_type": "multiclass_text_classification",
            "language": "en",
            "classes": [1, 2, 3],
            "label_mapping": {"1": "neoplasms", "2": "digestive", "3": "cardiovascular"},
            "random_state": 42,
            "n_train": 14438,
            "n_test": 3610,
            "metrics": {},
            "preprocessing": {},
            "selection": {},
            "dependency_versions": {"python": "3.12"},
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_dirty": False,
            "created_at": "2026-08-23T13:58:11+00:00",
        },
    )
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        result = dashboard_module._get_model_info("http://api.example.com/")
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "GET"
    assert url == "http://api.example.com/model-info"
    assert result.status_code == 200
    assert result.body["model_version"].startswith("2026")


def test_format_pct_rounds_to_two_decimals(dashboard_module):
    assert dashboard_module._format_pct(0.87654) == "87.65%"
    assert dashboard_module._format_pct(0) == "0.00%"
    assert dashboard_module._format_pct(1) == "100.00%"
    assert dashboard_module._format_pct(None) == "—"


def test_render_model_sidebar_invokes_all_panels(dashboard_module):
    """Every section of the sidebar renderer must be exercised without error."""

    info = {
        "model_version": "20260823T135811Z-bed2194376bc",
        "model_name": "tfidf_linear_svc_baseline",
        "task_type": "multiclass_text_classification",
        "language": "en",
        "classes": [1, 2],
        "label_mapping": {"1": "neoplasms", "2": "digestive"},
        "random_state": 42,
        "n_train": 14438,
        "n_test": 3610,
        "metrics": {
            "accuracy": 0.91,
            "balanced_accuracy": 0.88,
            "macro_f1": 0.87,
            "weighted_f1": 0.9,
            "per_class": {
                "1": {"precision": 0.9, "recall": 0.91, "f1": 0.9, "support": 100},
                "2": {"precision": 0.85, "recall": 0.83, "f1": 0.84, "support": 80},
            },
        },
        "selection": {
            "selected_classifier": "linear_svc",
            "metric": "macro_f1",
            "folds": 5,
            "test_set_used_for_selection": False,
            "candidates": {
                "logreg": {"mean_macro_f1": 0.83, "std_macro_f1": 0.01},
                "linear_svc": {"mean_macro_f1": 0.86, "std_macro_f1": 0.02},
            },
        },
        "dependency_versions": {
            "python": "3.12",
            "numpy": "1.26",
            "scipy": "1.11",
            "scikit_learn": "1.4",
            "joblib": "1.3",
        },
        "preprocessing": {"classifier": "linear_svc"},
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "git_dirty": True,
        "created_at": "2026-08-23T13:58:11+00:00",
    }
    calls = {"metric": 0, "markdown": 0, "json": 0, "dataframe": 0, "columns": 0}

    import streamlit as real_st

    def _columns(spec, **kwargs):
        calls["columns"] += 1
        width = spec if isinstance(spec, int) else len(spec)
        cols = [MagicMock(name=f"col_{i}") for i in range(width)]
        for col in cols:
            # Route column-level metric/markdown/json calls to the patched module
            # so we can count them through the same ``calls`` dict.
            col.metric = MagicMock(side_effect=lambda *a, **k: _bump("metric"))
            col.markdown = MagicMock(side_effect=lambda *a, **k: _bump("markdown"))
            col.dataframe = MagicMock(side_effect=lambda *a, **k: _bump("dataframe"))
        return cols

    def _bump(key: str) -> None:
        calls[key] = calls[key] + 1

    mocks = {
        "metric": MagicMock(side_effect=lambda *args, **kwargs: _bump("metric")),
        "columns": MagicMock(side_effect=_columns),
        "markdown": MagicMock(side_effect=lambda *args, **kwargs: _bump("markdown")),
        "json": MagicMock(side_effect=lambda *args, **kwargs: _bump("json")),
        "dataframe": MagicMock(side_effect=lambda *args, **kwargs: _bump("dataframe")),
        "caption": MagicMock(),
        "info": MagicMock(),
        "expander": MagicMock(return_value=_EnterableMock()),
    }
    with patch.multiple(real_st, **mocks):
        dashboard_module._render_model_sidebar(info)

    # Five expanders (Identidade, Treinamento, Seleção, Métricas, Classes).
    assert mocks["expander"].call_count == 5
    # Two ``st.metric`` calls in Treinamento + four in Métricas — six total.
    assert calls["metric"] == 6
    # Two dataframes (per-class metrics, label mapping).
    assert calls["dataframe"] == 2
    # JSON block for dependency_versions.
    assert calls["json"] == 1


def test_render_model_sidebar_handles_empty_metrics(dashboard_module):
    """An empty manifest must not crash — fallbacks surface ``—``."""

    import streamlit as real_st

    mocks = {
        "metric": MagicMock(),
        "columns": MagicMock(return_value=[MagicMock(), MagicMock()]),
        "markdown": MagicMock(),
        "json": MagicMock(),
        "dataframe": MagicMock(),
        "caption": MagicMock(),
        "expander": MagicMock(return_value=_EnterableMock()),
    }
    with patch.multiple(real_st, **mocks):
        dashboard_module._render_model_sidebar({})  # type: ignore[arg-type]

    # Five sections rendered even with an empty payload.
    assert mocks["expander"].call_count == 5
    # No dataframe when there's no per_class / label_mapping to show.
    assert mocks["dataframe"].call_count == 0
