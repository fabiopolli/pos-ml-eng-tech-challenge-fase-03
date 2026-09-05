"""Streamlit production dashboard with a role-specific clinical experience.

This interface is deliberately separate from ``app_dev.py``.  It demonstrates
the production RBAC policy without exposing the doctor's API key to the browser:
the key stays in the Streamlit server process and is only used after a doctor
has authenticated.  Patients never invoke ``POST /predict`` and never receive
model labels, scores, or the clinical text submitted by a doctor.

The username/password login is a local demonstration boundary for the Tech
Challenge.  A cloud deployment must replace it with an identity provider and
server-side secret manager.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit

import requests
import streamlit as st

PAGE_TITLE = "triage_ml — Portal Clínico"
PAGE_ICON = "🩺"
DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_ERROR_BODY_CHARS = 2_000

ROLE_DOCTOR = "doctor"
ROLE_PATIENT = "patient"
ROLE_LABELS = {ROLE_DOCTOR: "Médico", ROLE_PATIENT: "Paciente"}


@dataclass(frozen=True)
class DashboardConfig:
    """Runtime-only configuration; none of these values belongs in Git."""

    api_url: str
    doctor_username: str
    doctor_password: str = field(repr=False)
    patient_username: str
    patient_password: str = field(repr=False)
    doctor_api_key: str = field(repr=False)


@dataclass(frozen=True)
class ApiResponse:
    """Small HTTP response representation safe to render in the dashboard."""

    status_code: int
    body: dict[str, object]
    elapsed_ms: float
    request_id: str | None
    server_timing: str | None


def _normalize_api_url(url: str) -> str:
    """Accept an explicit HTTP(S) URL without embedded credentials."""

    candidate = url.strip().rstrip("/")
    parsed = urlsplit(candidate)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("API URL has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("API URL must be HTTP(S) without credentials, query, or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def load_config(environ: Mapping[str, str] | None = None) -> DashboardConfig:
    """Read all secrets from environment variables and fail closed if absent."""

    source = os.environ if environ is None else environ
    names = (
        "TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME",
        "TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD",
        "TRIAGE_ML_DASHBOARD_PATIENT_USERNAME",
        "TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD",
        "TRIAGE_ML_API_KEY_DOCTOR",
    )
    missing = [name for name in names if not source.get(name)]
    if missing:
        raise RuntimeError(f"Missing dashboard configuration: {', '.join(missing)}")

    return DashboardConfig(
        api_url=_normalize_api_url(source.get("TRIAGE_ML_PROD_API_URL", DEFAULT_API_URL)),
        doctor_username=source["TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME"],
        doctor_password=source["TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD"],
        patient_username=source["TRIAGE_ML_DASHBOARD_PATIENT_USERNAME"],
        patient_password=source["TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD"],
        doctor_api_key=source["TRIAGE_ML_API_KEY_DOCTOR"],
    )


def authenticate(username: str, password: str, config: DashboardConfig) -> str | None:
    """Map credentials to a role without trusting a role submitted by the UI."""

    doctor_match = hmac.compare_digest(username, config.doctor_username) and hmac.compare_digest(
        password, config.doctor_password
    )
    patient_match = hmac.compare_digest(username, config.patient_username) and hmac.compare_digest(
        password, config.patient_password
    )
    if doctor_match:
        return ROLE_DOCTOR
    if patient_match:
        return ROLE_PATIENT
    return None


def can_request_prediction(role: str) -> bool:
    """Return whether a dashboard role may invoke the protected endpoint."""

    return role == ROLE_DOCTOR


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    api_key: str | None = None,
) -> ApiResponse:
    """Issue a server-side API request without storing input text locally."""

    headers = {"Accept": "application/json"}
    if api_key is not None:
        headers["X-API-Key"] = api_key

    start = perf_counter()
    response = requests.request(
        method=method,
        url=url,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=False,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    try:
        parsed = response.json()
        body = parsed if isinstance(parsed, dict) else {"raw": response.text[:MAX_ERROR_BODY_CHARS]}
    except ValueError:
        body = {"raw": response.text[:MAX_ERROR_BODY_CHARS]}

    return ApiResponse(
        status_code=response.status_code,
        body=body,
        elapsed_ms=elapsed_ms,
        request_id=response.headers.get("X-Request-ID"),
        server_timing=response.headers.get("Server-Timing"),
    )


def check_health(config: DashboardConfig) -> ApiResponse:
    """Call the public health endpoint, available to both dashboard roles."""

    return _request_json("GET", f"{config.api_url}/health")


def doctor_predict(config: DashboardConfig, text: str) -> ApiResponse:
    """Call the protected prediction route from the Streamlit server only."""

    return _request_json(
        "POST",
        f"{config.api_url}/predict",
        payload={"text": text},
        api_key=config.doctor_api_key,
    )


def _render_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0e1117; }
        .role-card { padding: 1rem 1.25rem; border-radius: .75rem;
                     border: 1px solid #30363d; background: #161b22; }
        .patient-card { border-left: 4px solid #58a6ff; }
        .doctor-card { border-left: 4px solid #3fb950; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_health(config: DashboardConfig) -> None:
    try:
        response = check_health(config)
    except requests.RequestException as exc:
        st.error(f"Não foi possível consultar a API: {exc}")
        return

    if response.status_code != 200:
        st.warning(f"Health retornou HTTP {response.status_code}.")
        return

    col_status, col_model = st.columns(2)
    col_status.metric("Status da API", str(response.body.get("status", "—")))
    col_model.metric("Modelo carregado", "Sim" if response.body.get("model_loaded") else "Não")
    st.caption(f"Versão do modelo: `{response.body.get('model_version', '—')}`")


def _render_doctor_dashboard(config: DashboardConfig) -> None:
    st.markdown('<div class="role-card doctor-card">', unsafe_allow_html=True)
    st.subheader("Área médica")
    st.write("Envie um texto clínico em inglês para apoiar a triagem. A decisão final é humana.")
    st.warning(
        "Uso profissional: a classe produzida pelo modelo não constitui diagnóstico e não deve "
        "ser comunicada ao paciente sem avaliação clínica."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    _render_health(config)

    with st.form("doctor_prediction", clear_on_submit=True):
        text = st.text_area(
            "Texto clínico para triagem",
            placeholder="We report a 62-year-old patient with persistent chest pain...",
            max_chars=20_000,
        )
        submitted = st.form_submit_button("Executar predição", type="primary")

    if not submitted:
        return
    if not text.strip():
        st.warning("Informe um texto clínico antes de enviar.")
        return

    try:
        response = doctor_predict(config, text)
    except requests.RequestException as exc:
        st.error(f"Falha de comunicação com a API: {exc}")
        return

    if response.status_code != 200:
        st.error(f"A predição não foi concluída (HTTP {response.status_code}).")
        st.code(json.dumps(response.body, ensure_ascii=False, indent=2), language="json")
        return

    st.success("Predição concluída. Interprete o resultado somente no contexto clínico.")
    left, middle, right = st.columns(3)
    left.metric("Classe", str(response.body.get("label_name", "—")))
    middle.metric("Score", _format_score(response.body.get("score")))
    right.metric("Latência do servidor", f"{float(response.body.get('latency_ms', 0)):.2f} ms")
    st.caption(
        f"Modelo `{response.body.get('model_version', '—')}` · "
        f"Request ID `{response.body.get('request_id', response.request_id or '—')}`"
    )
    if response.server_timing:
        st.caption(f"Server-Timing: `{response.server_timing}`")


def _format_score(value: object) -> str:
    if value is None:
        return "Não disponível"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "Não disponível"


def _render_patient_dashboard(config: DashboardConfig) -> None:
    st.markdown('<div class="role-card patient-card">', unsafe_allow_html=True)
    st.subheader("Área do paciente")
    st.write("Acompanhe as etapas de uma avaliação com revisão obrigatória de um profissional.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning(
        "Este portal não fornece diagnóstico. Resultados automáticos, nomes de doenças e "
        "pontuações são restritos à equipe médica para evitar interpretações inseguras."
    )

    step_understand, step_review, step_guidance = st.tabs(
        ["1. Entenda o processo", "2. Revisão médica", "3. Próximos passos"]
    )
    with step_understand:
        st.subheader("Sua segurança vem primeiro")
        st.write(
            "O modelo é uma ferramenta de apoio e pode errar. Por isso, qualquer classificação "
            "precisa ser analisada em conjunto com sintomas, histórico e exames."
        )
        st.info("Nenhuma classificação automática é exibida nesta área.")

    with step_review:
        st.subheader("Status: aguardando avaliação profissional")
        st.write(
            "Um profissional habilitado deve revisar as informações antes de conversar com você "
            "sobre hipóteses, diagnóstico ou tratamento."
        )
        st.progress(2 / 3, text="Etapa de revisão médica")

    with step_guidance:
        st.subheader("Como buscar atendimento")
        st.write(
            "Converse com sua equipe de saúde para receber orientação individualizada. "
            "Não tome decisões médicas com base apenas em sistemas automatizados."
        )
        st.error(
            "Em caso de sintomas graves, piora rápida ou risco imediato, procure um serviço de "
            "emergência."
        )

    with st.expander("Estado técnico do serviço"):
        _render_health(config)
    st.caption(
        "Proteção de acesso ativa: a sessão do paciente não possui chave médica e não chama "
        "o endpoint de predição."
    )


def _render_login(config: DashboardConfig) -> None:
    st.subheader("Entrar no Portal Clínico")
    st.write("Use seu usuário e senha. O perfil de acesso é definido pelas credenciais.")
    st.caption("Ambiente demonstrativo: não use credenciais pessoais ou dados clínicos reais.")
    with st.form("login"):
        username = st.text_input("Usuário", autocomplete="username")
        password = st.text_input("Senha", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Entrar", type="primary")

    if submitted:
        role = authenticate(username, password, config)
        if role is None:
            st.error("Credenciais inválidas.")
            return
        st.session_state["dashboard_role"] = role
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    _render_css()
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Demonstração de RBAC clínico — dados não são persistidos pelo dashboard.")

    try:
        config = load_config()
    except (RuntimeError, ValueError) as exc:
        st.error(f"Configuração indisponível: {exc}")
        st.stop()

    role = st.session_state.get("dashboard_role")
    if role not in ROLE_LABELS:
        _render_login(config)
        return

    with st.sidebar:
        st.success(f"Sessão: {ROLE_LABELS[role]}")
        st.caption(f"API configurada: `{config.api_url}`")
        if st.button("Sair", use_container_width=True):
            st.session_state.pop("dashboard_role", None)
            st.rerun()

    if can_request_prediction(role):
        _render_doctor_dashboard(config)
    else:
        _render_patient_dashboard(config)


if __name__ == "__main__":
    main()
