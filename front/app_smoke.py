"""Streamlit smoke dashboard for the triage_ml FastAPI service.

The dashboard is a developer-facing tool for exercising the ``/health``
and ``/predict`` endpoints against any reachable instance of the API
(local, container, or cloud). It is **not** an observability surface
— latency, error rate and request volume belong to Prometheus/Grafana
(see ``monitoring/``). It does not render model metrics either: the
classifier is fixed and the dataset evaluation is captured by the
notebooks and the implementation report.

Abas:
  1. **Health & versão** — chama ``GET /health`` e mostra ``status``,
     ``model_version`` e ``model_loaded`` retornados pela API.
  2. **Predição** — área de texto + botão "Executar predição" envia
     ``POST /predict`` e exibe ``label``/``label_name``/``score``/
     ``latency_ms``/``request_id`` e os headers ``X-Request-ID`` /
     ``Server-Timing``.
  3. **Política de idioma** — quatro cenários canônicos (texto curto,
     score baixo, idioma fora do allow-list, sucesso em inglês) com
     presets prontos para colar payloads e validar a resposta de erro.

A comunicação é puramente HTTP contra a URL configurada na sidebar.
O dashboard nunca toca o artefato do modelo nem o detector ``langid``
diretamente — o que garante paridade com qualquer ambiente (local,
Docker, cloud) e respeita o contrato da API oficial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

# --- Constantes visuais (dark mode premium, identidade da Fase 02) ---
PAGE_TITLE = "triage_ml — Smoke API"
PAGE_ICON = "🧪"
DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10.0
SUCCESS_STATUS = 200

LANGUAGE_PRESETS: dict[str, dict[str, str]] = {
    "Texto curto (<20 chars)": {
        "expected_error_code": "text_too_short_for_language_check",
        "text": "liver tumor",
        "note": "11 caracteres — o detector nem é chamado; a API rejeita direto.",
    },
    "Confiança baixa (mock)": {
        "expected_error_code": "indeterminate_language",
        "text": (
            "The study cohort included patients with mixed-language clinical "
            "notes that the detector could not classify reliably."
        ),
        "note": (
            "Texto real em inglês; no smoke script a fixture mocka "
            "``langid`` com log-prob saturado em ``-1000`` (score 0.0)."
        ),
    },
    "Idioma fora do allow-list": {
        "expected_error_code": "unsupported_language",
        "text": (
            "Relatamos um paciente de 62 anos com infarto agudo do miocárdio "
            "após dor torácica e dispneia progressiva."
        ),
        "note": (
            "Texto em pt-BR; passa o detector (score alto) mas cai fora de "
            "``supported_languages: [en]``."
        ),
    },
    "Inglês válido": {
        "expected_error_code": None,
        "text": (
            "We report a 62-year-old patient with an aggressive liver tumor "
            "that required urgent surgical resection and histopathological "
            "evaluation."
        ),
        "note": "Texto clínico real do corpus; deve retornar 200 com label_name.",
    },
}


@dataclass(frozen=True)
class ApiResponse:
    """Container with both the parsed body and the headers."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]
    elapsed_ms: float

    def _header(self, name: str) -> str | None:
        # HTTP headers are case-insensitive; ``requests.Response.headers``
        # is a ``CaseInsensitiveDict`` while tests may pass plain dicts.
        lower = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lower:
                return value
        return None

    @property
    def server_timing(self) -> str | None:
        return self._header("server-timing")

    @property
    def request_id(self) -> str | None:
        return self._header("x-request-id")


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> ApiResponse:
    """Wrapper que centraliza timeout, headers e parsing JSON."""

    response = requests.request(
        method=method,
        url=url,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"Accept": "application/json"},
    )
    elapsed_ms = response.elapsed.total_seconds() * 1000.0
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text}
    return ApiResponse(
        status_code=response.status_code,
        body=body,
        headers={k: v for k, v in response.headers.items()},
        elapsed_ms=elapsed_ms,
    )


def _check_health(api_url: str) -> ApiResponse:
    return _request_json("GET", f"{api_url.rstrip('/')}/health")


def _post_predict(api_url: str, text: str) -> ApiResponse:
    return _request_json(
        "POST",
        f"{api_url.rstrip('/')}/predict",
        payload={"text": text},
    )


def _render_response(response: ApiResponse, *, expected_error_code: str | None) -> None:
    """Render the API response with the same shape regardless of status."""

    status_label = "✅ sucesso" if response.status_code == SUCCESS_STATUS else "❌ erro"
    st.markdown(
        f"**HTTP {response.status_code}** · {status_label} · "
        f"`{response.elapsed_ms:.2f} ms` (round-trip do client)"
    )

    if expected_error_code is not None and response.status_code != SUCCESS_STATUS:
        actual = response.body.get("error_code", "—")
        if actual == expected_error_code:
            st.success(f"`error_code` esperado: `{expected_error_code}` — bate com a resposta.")
        else:
            st.warning(
                f"`error_code` esperado: `{expected_error_code}` — resposta retornou `{actual}`."
            )

    st.markdown("**Body**")
    st.code(json.dumps(response.body, indent=2, ensure_ascii=False), language="json")

    st.markdown("**Headers relevantes**")
    headers_view = {
        k: v
        for k, v in response.headers.items()
        if k.lower() in {"x-request-id", "server-timing", "content-type", "content-length"}
    }
    if response.server_timing or response.request_id:
        if response.server_timing:
            st.code(f"Server-Timing: {response.server_timing}", language="text")
        if response.request_id:
            st.code(f"X-Request-ID: {response.request_id}", language="text")
    if headers_view:
        with st.expander("Todos os headers"):
            st.json(headers_view)


def _render_css() -> None:
    """Dark mode premium, alinhado com a Fase 02."""

    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117 !important; }
        [data-testid="stAppViewContainer"] { background-color: #0e1117 !important; }
        h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown { color: #e6edf3 !important; }
        div[data-testid="stMetricValue"] {
            color: #58a6ff !important;
            font-weight: 800 !important;
            text-shadow: 0px 0px 10px rgba(88, 166, 255, 0.3);
        }
        div[data-testid="stMetricLabel"] {
            color: #8b949e !important;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #0e1117; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #161b22;
            border-radius: 8px 8px 0px 0px;
            color: #8b949e;
            border: 1px solid #30363d;
            border-bottom: none;
            padding: 0 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #21262d !important;
            color: #ffffff !important;
            border-top: 3px solid #58a6ff !important;
        }
        .stCodeBlock { border: 1px solid #30363d; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    _render_css()

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown(
        "Dashboard de smoke para a API FastAPI do classificador de triagem. "
        "Use para validar manualmente `/health` e `/predict` em qualquer "
        "ambiente — local, container ou cloud. Métricas de produção "
        "continuam a viver no stack **Prometheus + Grafana**."
    )

    with st.sidebar:
        st.markdown("### 🔌 Conexão")
        api_url = st.text_input(
            "URL base da API",
            value=DEFAULT_API_URL,
            help="Ex.: http://127.0.0.1:8000 (local), http://api.cloud.run.app",
        )
        st.caption(
            "Suba a API local com `PYTHONPATH=src uv run uvicorn triage_ml.api.app:app --reload`."
        )

        if st.button("🔄 Atualizar health", use_container_width=True):
            st.session_state["force_health"] = True

        st.markdown("---")
        st.markdown("### 📚 Atalhos")
        st.markdown("- [Plan do classificador](docs/plans/PLAN-text-classifier.md)")
        st.markdown("- [Checklist oficial](docs/CHECKLIST.md)")
        st.markdown("- [Relatório Fase 1](docs/reports/IMPLEMENTATION-REPORT-FASE-1.md)")

    tab_health, tab_predict, tab_language = st.tabs(
        ["🩺 Health", "🎯 Predição", "🌐 Política de idioma"]
    )

    with tab_health:
        st.header("🩺 Health check")
        st.markdown(
            "Executa `GET /health` e exibe o resultado. Útil para confirmar "
            "que o artefato subiu e o modelo está carregado."
        )
        if st.button("Chamar /health", key="call_health"):
            with st.spinner("Chamando /health ..."):
                try:
                    response = _check_health(api_url)
                except requests.RequestException as exc:
                    st.error(f"Falha ao chamar /health: {exc}")
                else:
                    _render_response(response, expected_error_code=None)
                    if response.status_code == SUCCESS_STATUS:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("status", str(response.body.get("status", "—")))
                        col2.metric(
                            "model_version",
                            str(response.body.get("model_version", "—"))[:24],
                        )
                        col3.metric(
                            "model_loaded",
                            "sim" if response.body.get("model_loaded") else "não",
                        )

    with tab_predict:
        st.header("🎯 Predição")
        st.markdown(
            "Cole um texto clínico e dispare `POST /predict`. A resposta "
            "completa aparece logo abaixo com `label`, `label_name`, "
            "`score`, `latency_ms`, `request_id` e os headers de telemetria."
        )

        default_text = LANGUAGE_PRESETS["Inglês válido"]["text"]
        text = st.text_area(
            "Texto para classificar",
            value=default_text,
            height=180,
            help="Texto livre; será enviado como `text` no payload JSON.",
        )

        col_btn, col_meta = st.columns([1, 3])
        with col_btn:
            run_clicked = st.button("▶ Executar predição", type="primary")
        with col_meta:
            st.caption(
                "Endpoint: `POST /predict` · "
                'Payload: `{"text": "..."}` · '
                "Resposta: `PredictOut` ou `ErrorOut`."
            )

        if run_clicked:
            with st.spinner("Chamando /predict ..."):
                try:
                    response = _post_predict(api_url, text)
                except requests.RequestException as exc:
                    st.error(f"Falha ao chamar /predict: {exc}")
                else:
                    _render_response(response, expected_error_code=None)
                    if response.status_code == SUCCESS_STATUS:
                        latency_ms = response.body.get("latency_ms")
                        if latency_ms is not None:
                            st.metric("latency_ms (servidor)", f"{latency_ms:.3f}")

    with tab_language:
        st.header("🌐 Política de idioma")
        st.markdown(
            "Quatro cenários canônicos da política de idioma configurada em "
            "`configs/api.yaml` (`supported_languages`, `min_text_chars`, "
            "`min_language_score`). Selecione um preset, ajuste o texto se "
            "quiser e dispare."
        )

        preset_names = list(LANGUAGE_PRESETS.keys())
        preset_name = st.selectbox("Preset", preset_names)
        preset = LANGUAGE_PRESETS[preset_name]

        text = st.text_area(
            "Texto",
            value=preset["text"],
            height=140,
            key=f"lang_text_{preset_name}",
        )
        st.caption(preset["note"])

        if st.button("▶ Disparar cenário", key="run_language"):
            with st.spinner("Chamando /predict ..."):
                try:
                    response = _post_predict(api_url, text)
                except requests.RequestException as exc:
                    st.error(f"Falha ao chamar /predict: {exc}")
                else:
                    _render_response(response, expected_error_code=preset["expected_error_code"])

    st.markdown("---")
    st.caption(
        "Smoke dashboard v1 · triage_ml · sem persistência de payloads. "
        "Latência, taxa de erro e volume pertencem ao Prometheus/Grafana."
    )


if __name__ == "__main__":
    main()
