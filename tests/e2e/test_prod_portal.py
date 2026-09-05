"""Browser-level evidence for the production portal RBAC journey."""

from __future__ import annotations

import pytest
import requests
from playwright.sync_api import Page, expect

PORTAL_URL = "http://127.0.0.1:8501"
API_URL = "http://127.0.0.1:8765"

pytestmark = pytest.mark.e2e


def _login(page: Page, username: str, password: str) -> None:
    page.goto(PORTAL_URL)
    page.get_by_label("Usuário").fill(username)
    page.get_by_label("Senha").fill(password)
    page.get_by_role("button", name="Entrar").click()


def test_invalid_credentials_keep_the_login_boundary(page: Page) -> None:
    _login(page, "desconhecido", "senha-incorreta")

    expect(page.get_by_text("Credenciais inválidas.")).to_be_visible()
    expect(page.get_by_role("heading", name="Entrar no Portal Clínico")).to_be_visible()


def test_patient_has_guided_journey_without_prediction_access(page: Page) -> None:
    requests.post(f"{API_URL}/_test/reset", timeout=5).raise_for_status()
    _login(page, "paciente-e2e", "paciente-senha")

    expect(page.get_by_role("heading", name="Área do paciente")).to_be_visible()
    expect(page.get_by_text("Este portal não fornece diagnóstico", exact=False)).to_be_visible()
    page.get_by_role("tab", name="2. Revisão médica").click()
    expect(page.get_by_text("Status: aguardando avaliação profissional")).to_be_visible()
    page.get_by_role("tab", name="3. Próximos passos").click()
    expect(page.get_by_text("Em caso de sintomas graves", exact=False)).to_be_visible()
    expect(page.get_by_role("button", name="Executar predição")).to_have_count(0)

    stats = requests.get(f"{API_URL}/_test/stats", timeout=5).json()
    assert stats["prediction_calls"] == 0

    page.get_by_role("button", name="Sair").click()
    expect(page.get_by_role("heading", name="Entrar no Portal Clínico")).to_be_visible()


def test_doctor_can_request_clinical_support(page: Page) -> None:
    requests.post(f"{API_URL}/_test/reset", timeout=5).raise_for_status()
    _login(page, "medico-e2e", "medico-senha")

    expect(page.get_by_role("heading", name="Área médica")).to_be_visible()
    expect(page.get_by_text("não constitui diagnóstico", exact=False)).to_be_visible()
    page.get_by_label("Texto clínico para triagem").fill(
        "A patient presents with persistent chest pain and shortness of breath."
    )
    page.get_by_role("button", name="Executar predição").click()

    expect(page.get_by_text("Predição concluída", exact=False)).to_be_visible()
    expect(page.get_by_text("cardiovascular diseases", exact=True)).to_be_visible()
    stats = requests.get(f"{API_URL}/_test/stats", timeout=5).json()
    assert stats["prediction_calls"] == 1
