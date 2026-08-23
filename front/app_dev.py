"""Streamlit dev dashboard for the triage_ml FastAPI service.

The dashboard is a developer-facing tool for exercising the ``/health``,
``/model-info`` and ``/predict`` endpoints against any reachable
instance of the API (local, container, or cloud). It is **not** an
observability surface — latency, error rate and request volume belong
to Prometheus/Grafana (see ``monitoring/``). It does, however, render
the validated artifact manifest in the sidebar so developers can see at
a glance which model is being served, how it was selected and how it
performed on the held-out test split.

Abas:
  1. **Health & versão** — chama ``GET /health`` e mostra ``status``,
     ``model_version`` e ``model_loaded`` retornados pela API.
  2. **Predição** — área de texto + botão "Executar predição" envia
     ``POST /predict`` e exibe ``label``/``label_name``/``score``/
     ``latency_ms``/``request_id`` e os headers ``X-Request-ID`` /
     ``Server-Timing``.
   3. **Política de idioma** — três cenários reproduzíveis (texto curto,
      idioma fora do allow-list, sucesso em inglês) com
     presets prontos para colar payloads e validar a resposta de erro.

Sidebar:
  * **Conexão** — URL base da API + botão para revalidar ``/health``.
  * **🔁 Trocar modelo** — consome ``GET /models`` e ``POST /reload``
    para listar versões imutáveis e trocar o holder da API em runtime
    (após re-validar manifesto + checksum).
  * **🧠 Modelo** — bloco que consome ``GET /model-info`` e mostra
    identidade, métricas de treino, seleção do classificador, métricas
    globais/per-classe e mapeamento de labels.

A comunicação é puramente HTTP contra a URL configurada na sidebar.
O dashboard nunca toca o artefato do modelo nem o detector ``langid``
diretamente — o que garante paridade com qualquer ambiente (local,
Docker, cloud) e respeita o contrato da API oficial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# --- Constantes visuais (dark mode premium, identidade da Fase 02) ---
PAGE_TITLE = "triage_ml — Dev API"
PAGE_ICON = "🧪"
DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10.0
SUCCESS_STATUS = 200

# Absolute path to the repo root. Used by the dashboard tests to confirm
# the dashboard is rooted one level under the project tree.
REPO_ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_PRESETS: dict[str, dict[str, Any]] = {
    "Texto curto (<20 chars)": {
        "expected_error_code": "text_too_short_for_language_check",
        "text": "liver tumor",
        "note": "11 caracteres — o detector nem é chamado; a API rejeita direto.",
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
        "note": "Texto sintético em inglês; deve retornar 200 com label_name.",
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
        allow_redirects=False,
    )
    elapsed_ms = response.elapsed.total_seconds() * 1000.0
    try:
        parsed = response.json()
        body = parsed if isinstance(parsed, dict) else {"raw_json": parsed}
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


def _get_model_info(api_url: str) -> ApiResponse:
    """Return ``GET /model-info`` against the configured API.

    The endpoint exposes the validated artifact manifest so the dashboard
    can render metrics, training details and dependency versions without
    touching the filesystem directly. Returns ``503 model_not_ready`` if
    the API has not loaded its artifact yet.
    """

    return _request_json("GET", f"{api_url.rstrip('/')}/model-info")


def _list_models(api_url: str) -> ApiResponse:
    """Return ``GET /models`` against the configured API.

    Pure read-only endpoint listing every immutable artifact version
    available under ``models/`` (newest first) plus the version the
    API is currently serving.
    """

    return _request_json("GET", f"{api_url.rstrip('/')}/models")


def _reload_model(api_url: str, model_version: str) -> ApiResponse:
    """Return ``POST /reload`` against the configured API.

    Body: ``{"model_version": "<version>"}``. On success the API swaps
    the holder to the new version (re-validating manifest + checksum)
    and returns the new ``model_version``. Errors map to ``404
    model_not_found`` or ``500 model_incompatible``.
    """

    return _request_json(
        "POST",
        f"{api_url.rstrip('/')}/reload",
        payload={"model_version": model_version},
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


def _render_model_sidebar(info: dict[str, Any]) -> None:
    """Render the loaded model manifest in the sidebar.

    The manifest comes straight from ``GET /model-info`` and mirrors the
    validated ``metadata.json``. The renderer is split into five panels so
    every relevant property of the inference model is visible at a glance:

    1. **Identidade** — version, name, task and supported language.
    2. **Treinamento** — split sizes, random seed, git commit, dirty flag,
       creation timestamp and dependency versions.
    3. **Seleção do classificador** — chosen classifier and the
       training-only cross-validation summary of both candidates.
    4. **Métricas** — overall accuracy/balanced accuracy/macro F1/weighted
       F1 plus the per-class precision/recall/F1/support table.
    5. **Classes** — integer label set and the human-readable mapping.
    """

    st.markdown("---")
    st.markdown("### 🧠 Modelo carregado")
    st.caption(f"Endpoint: `GET /model-info` · versão `{info.get('model_version', '—')}`.")

    with st.expander("Identidade", expanded=True):
        st.markdown(f"- **model_version**: `{info.get('model_version', '—')}`")
        st.markdown(f"- **model_name**: `{info.get('model_name', '—')}`")
        st.markdown(f"- **task_type**: `{info.get('task_type', '—')}`")
        st.markdown(f"- **language**: `{info.get('language', '—')}`")

    with st.expander("Treinamento", expanded=True):
        col_a, col_b = st.columns(2)
        col_a.metric("n_train", f"{info.get('n_train', 0):,}".replace(",", "."))
        col_b.metric("n_test", f"{info.get('n_test', 0):,}".replace(",", "."))
        random_state = info.get("random_state", "—")
        st.markdown(f"- **random_state**: `{random_state}`")
        git_commit = info.get("git_commit", "unknown")
        git_dirty = info.get("git_dirty", False)
        dirty_marker = " *(dirty)*" if git_dirty else ""
        st.markdown(f"- **git_commit**: `{git_commit}`{dirty_marker}")
        st.markdown(f"- **created_at**: `{info.get('created_at', '—')}`")
        deps = info.get("dependency_versions", {}) or {}
        if deps:
            st.markdown("**dependency_versions**")
            st.json(deps)

    selection = info.get("selection", {}) or {}
    candidates = selection.get("candidates", {}) or {}
    with st.expander("Seleção do classificador", expanded=False):
        st.markdown(f"- **selected_classifier**: `{selection.get('selected_classifier', '—')}`")
        st.markdown(f"- **metric**: `{selection.get('metric', '—')}`")
        st.markdown(f"- **folds**: `{selection.get('folds', '—')}`")
        st.markdown(
            "- **test_set_used_for_selection**: "
            f"`{selection.get('test_set_used_for_selection', '—')}`"
        )
        if candidates:
            st.markdown("**Candidatos (cross-validation no treino)**")
            for name, payload in candidates.items():
                mean = float(payload.get("mean_macro_f1", 0.0))
                std = float(payload.get("std_macro_f1", 0.0))
                marker = " ← escolhido" if name == selection.get("selected_classifier") else ""
                st.markdown(f"- `{name}`{marker}: mean macro F1 = **{mean:.4f}** (± {std:.4f})")

    metrics = info.get("metrics", {}) or {}
    per_class = metrics.get("per_class", {}) or {}
    with st.expander("Métricas", expanded=True):
        overall = metrics.get("overall") or {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
        }
        col1, col2 = st.columns(2)
        col1.metric("accuracy", _format_pct(overall.get("accuracy")))
        col2.metric("balanced_accuracy", _format_pct(overall.get("balanced_accuracy")))
        col3, col4 = st.columns(2)
        col3.metric("macro_f1", _format_pct(overall.get("macro_f1")))
        col4.metric("weighted_f1", _format_pct(overall.get("weighted_f1")))
        if per_class:
            st.markdown("**Per-class**")
            rows = [
                {
                    "class": class_label,
                    "precision": _format_pct(payload.get("precision")),
                    "recall": _format_pct(payload.get("recall")),
                    "f1": _format_pct(payload.get("f1")),
                    "support": payload.get("support", 0),
                }
                for class_label, payload in per_class.items()
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)

    label_mapping = info.get("label_mapping", {}) or {}
    classes = info.get("classes", []) or []
    with st.expander("Classes & mapeamento", expanded=False):
        st.markdown(f"- **classes**: `{classes}`")
        if label_mapping:
            rows = [{"label": key, "name": value} for key, value in sorted(label_mapping.items())]
            st.dataframe(rows, hide_index=True, use_container_width=True)


def _format_pct(value: float | int | None) -> str:
    """Format a 0..1 metric as a fixed-precision percentage string."""

    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


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
        "Dashboard de desenvolvimento para a API FastAPI do classificador de triagem. "
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
        api_url = api_url.rstrip("/")
        previous_api_url = st.session_state.get("cached_api_url")
        if previous_api_url != api_url:
            for key in ("models_payload", "model_info_payload", "model_picker_selection"):
                st.session_state.pop(key, None)
            st.session_state["cached_api_url"] = api_url
            st.session_state["force_models_list"] = True
            st.session_state["force_model_info"] = True
            st.session_state["force_health"] = True
        st.caption(
            "Suba a API local com "
            "`PYTHONPATH=src uv run uvicorn triage_ml.dev_api.app:app --reload`."
        )

        if st.button("🔄 Atualizar health", use_container_width=True):
            st.session_state["force_health"] = True

        st.markdown("---")
        st.markdown("### 🔁 Trocar modelo")
        st.caption(
            "Lista as versões imutáveis disponíveis em `models/` e troca "
            "o holder global do processo da API via `POST /reload`. Use apenas localmente."
        )

        if st.button("🔄 Listar versões", use_container_width=True, key="refresh_models"):
            st.session_state["force_models_list"] = True

        versions_error: str | None = None
        versions_payload: dict[str, Any] | None = st.session_state.get("models_payload")
        if versions_payload is None or st.session_state.get("force_models_list"):
            try:
                with st.spinner("Chamando /models ..."):
                    response = _list_models(api_url)
            except requests.RequestException as exc:
                versions_error = f"Falha ao chamar /models: {exc}"
            else:
                if response.status_code == SUCCESS_STATUS and isinstance(response.body, dict):
                    versions_payload = response.body
                    st.session_state["models_payload"] = response.body
                    st.session_state["force_models_list"] = False
                else:
                    versions_error = (
                        f"/models respondeu HTTP {response.status_code}: "
                        f"{response.body.get('error_code', '—')}"
                    )
                st.session_state["force_models_list"] = False

        if versions_error is not None:
            st.error(versions_error)
        elif isinstance(versions_payload, dict):
            versions_list: list[str] = list(versions_payload.get("versions") or [])
            current_version: str | None = versions_payload.get("current")
            if not versions_list:
                st.info("Nenhuma versão disponível em `models/`.")
            else:
                # Default the picker to the currently loaded version so the
                # dashboard never starts with an empty selection; otherwise
                # default to the first (newest) version.
                default_index = 0
                if current_version and current_version in versions_list:
                    default_index = versions_list.index(current_version)
                # ``st.session_state`` keeps the picker choice stable across
                # reruns triggered by unrelated widgets (e.g. the text input).
                picker_key = "model_picker_selection"
                if (
                    picker_key not in st.session_state
                    or st.session_state[picker_key] not in versions_list
                ):
                    st.session_state[picker_key] = versions_list[default_index]
                selected_version = st.selectbox(
                    "Versão para servir",
                    versions_list,
                    key=picker_key,
                    help="A versão atual está destacada pelo botão 'Servir esta versão'.",
                )
                if current_version:
                    st.caption(f"Atualmente servindo: `{current_version}`")
                if st.button(
                    "🚀 Servir esta versão",
                    type="primary",
                    use_container_width=True,
                    key="serve_version",
                ):
                    try:
                        with st.spinner(f"Chamando POST /reload ({selected_version}) ..."):
                            reload_response = _reload_model(api_url, selected_version)
                    except requests.RequestException as exc:
                        st.error(f"Falha ao chamar /reload: {exc}")
                    else:
                        if reload_response.status_code == SUCCESS_STATUS:
                            new_version = reload_response.body.get(
                                "model_version", selected_version
                            )
                            st.success(f"API agora está servindo `{new_version}`.")
                            # Force both downstream blocks to refresh on the
                            # next rerun so the sidebar reflects the swap.
                            st.session_state["force_model_info"] = True
                            st.session_state["force_models_list"] = True
                            st.session_state["force_health"] = True
                            st.rerun()
                        else:
                            error_code = reload_response.body.get("error_code", "—")
                            st.error(
                                f"Reload falhou (HTTP {reload_response.status_code}, "
                                f"`error_code={error_code}`)."
                            )
        else:
            st.info("Lista de versões indisponível. Clique em 'Listar versões'.")

        st.markdown("---")
        st.markdown("### 🧠 Modelo")
        st.caption(
            f"Chama `GET /model-info` em **{api_url}** para inspecionar "
            "o artefato carregado pela API."
        )
        if st.button("🔄 Atualizar info do modelo", use_container_width=True, key="refresh_model"):
            st.session_state["force_model_info"] = True

        model_info_error: str | None = None
        model_info_payload: dict[str, Any] | None = st.session_state.get("model_info_payload")
        if model_info_payload is None or st.session_state.get("force_model_info"):
            try:
                with st.spinner("Chamando /model-info ..."):
                    response = _get_model_info(api_url)
            except requests.RequestException as exc:
                model_info_error = f"Falha ao chamar /model-info: {exc}"
            else:
                if response.status_code == SUCCESS_STATUS and isinstance(response.body, dict):
                    model_info_payload = response.body
                    st.session_state["model_info_payload"] = response.body
                    st.session_state["force_model_info"] = False
                else:
                    model_info_error = (
                        f"/model-info respondeu HTTP {response.status_code}: "
                        f"{response.body.get('error_code', '—')}"
                    )
                st.session_state["force_model_info"] = False

        if model_info_error is not None:
            st.error(model_info_error)
        elif isinstance(model_info_payload, dict):
            _render_model_sidebar(model_info_payload)
        else:
            st.info("Sem dados do modelo. Atualize para carregar o manifesto.")

    tab_health, tab_predict, tab_language = st.tabs(
        ["🩺 Health", "🎯 Predição", "🌐 Política de idioma"]
    )

    with tab_health:
        st.header("🩺 Health check")
        st.markdown(
            "Executa `GET /health` e exibe o resultado. Útil para confirmar "
            "que o artefato subiu e o modelo está carregado."
        )
        health_clicked = st.button("Chamar /health", key="call_health")
        if health_clicked or st.session_state.pop("force_health", False):
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
            "Três cenários reproduzíveis da política de idioma configurada em "
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
        "Dev dashboard v1 · triage_ml · sem persistência de payloads em disco. "
        "Latência, taxa de erro e volume pertencem ao Prometheus/Grafana; "
        "informações do modelo carregado ficam na sidebar via /model-info."
    )


if __name__ == "__main__":
    main()
