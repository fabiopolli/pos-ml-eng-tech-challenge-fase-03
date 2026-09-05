# Front — Dashboards Streamlit

Ferramentas de interface Streamlit. Elas não fazem parte do runtime da API:
existem para validação manual e demonstração do Tech Challenge.

Para uma sequência reproduzível de demonstração e gravação, consulte o
[`Guia de uso dos fronts`](../docs/guides/GUIA-USO-FRONTS.md).

Use o dashboard somente em localhost. Ele aceita uma URL informada pelo
usuário e envia textos para esse destino; não é uma interface pública.

## `app_dev.py`

Dashboard Streamlit (tema dark premium fixo via CSS) que fala HTTP
com qualquer instância da API (local, container ou cloud) e expõe
três abas + uma sidebar fixa:

**Abas**:

1. **🩺 Health** — chama `GET /health` e mostra `status`, `model_version`
   e `model_loaded`. Status `degraded` aparece em amarelo (`model_loaded=false`)
   e `ok` em verde.
2. **🎯 Predição** — área de texto + `POST /predict` com leitura completa
   do body, `latency_ms`, `request_id` e os headers `X-Request-ID` /
   `Server-Timing`. Mostra também `label`, `label_name` e `score`
   quando a API devolve o label escalar (`LinearSVC` → `score=null`).
3. **🌐 Política de idioma** — três cenários reproduzíveis da política
   definida em `configs/api.yaml` (texto curto, idioma fora do allow-list
   e inglês válido) com validação automática do `error_code`. O branch de
   probabilidade baixa exige mock no processo da API e fica nos testes/script.

**Sidebar**:

- **🔌 Conexão** — URL base da API + botão "Atualizar health" para
  revalidar `/health` sem reiniciar o Streamlit. A troca de URL limpa
  os caches de modelos/manifesto, e requests não seguem redirects.
- **🔁 Trocar modelo** — consome `GET /models` para listar as versões
  imutáveis disponíveis em `models/` (newest-first), mostra a versão
  atualmente em uso, deixa você escolher outra via `<selectbox>` (com
  default = versão corrente) e dispara `POST /reload`. Resposta
  bem-sucedida força o refresh dos blocos abaixo; resposta `404
  model_not_found` ou `500 model_incompatible` mostra o erro em
  vermelho sem alterar o holder. O picker fica em memória na sessão,
  mas o reload altera o holder global do processo da API; use apenas localmente.
- **🧠 Modelo** — consome `GET /model-info` e mostra, em cinco
  expanders:
  - **Identidade** — `model_version`, `model_name`, `task_type`, `language`.
  - **Treinamento** — `n_train` / `n_test` via `st.metric`,
    `random_state`, `git_commit` (com marcação `(dirty)` quando
    aplicável), `created_at` e `dependency_versions` em código.
  - **Seleção do classificador** — `selected_classifier`, métrica,
    folds, `test_set_used_for_selection=False`, mais tabela inline
    com `mean_macro_f1 ± std_macro_f1` dos candidatos `logreg` ×
    `linear_svc` (o escolhido fica marcado com `← escolhido`).
  - **Métricas** — quatro `st.metric` (`accuracy`, `balanced_accuracy`,
    `macro_f1`, `weighted_f1`) e tabela per-classe (`precision`,
    `recall`, `f1`, `support`).
  - **Classes & mapeamento** — lista de classes e tabela
    `label` ↔ `label_name`.

A conexão é HTTP contra a URL configurada na sidebar (default
`http://127.0.0.1:8000`). O dashboard não toca o artefato do modelo
nem o detector `langid` — qualquer divergência com a resposta da API é
um sinal de bug. Helpers HTTP (`_check_health`, `_post_predict`,
`_get_model_info`, `_list_models`, `_reload_model`, `_request_json`)
são cobertos por testes herméticos em `tests/test_dev_dashboard_helpers.py`.

### Como rodar

```bash
# 1. Suba a API em outro terminal
uv run uvicorn triage_ml.dev_api.app:app --host 127.0.0.1 --port 8000

# 2. Suba o dashboard
uv run streamlit run front/app_dev.py
```

Acesse `http://localhost:8501`.

### O que **não** é objetivo deste dashboard

- Métricas de produção (latência p95, taxa de erro, comparação
  baseline vs otimizado) → moram no stack **Prometheus + Grafana**
  (`monitoring/`).
- Avaliação do modelo → notebooks `01_eda.ipynb` / `02_model_baseline.ipynb`
  e `docs/reports/Etapa_2_Modelo_baseline_e_serialização.md`.
- Privacidade clínica → o dashboard não armazena payloads nem textos
  em disco; widgets ainda mantêm seus valores em memória durante a sessão.
- Persistência entre sessões — o model picker é por-design em
  memória; ao reiniciar o Streamlit, a escolha volta ao default
  (versão atualmente em uso na API).

## `app_prod.py` — portal clínico com RBAC

Dashboard separado para demonstrar a API oficial (`src/triage_ml/api/`)
no vídeo STAR. Tem uma tela de login e duas experiências distintas:

- **Médico**: o processo Streamlit chama `POST /predict` com a chave de médico
  configurada no ambiente e apresenta a resposta da API para apoio à triagem.
- **Paciente**: não recebe chave médica e **nunca chama** `/predict`; vê somente
  uma jornada educativa em três etapas (processo, revisão médica e próximos
  passos). Não recebe diagnóstico, nome de doença, classe ou score; o estado
  técnico da API fica recolhido em uma seção secundária.

O login é deliberadamente uma demonstração local baseada em credenciais de
ambiente. Ele não substitui um provedor de identidade (OIDC/Identity Platform)
em nuvem. As senhas e a API key ficam no processo Streamlit, nunca no código,
na URL, no estado da sessão ou no navegador.

### Como rodar localmente (PowerShell)

Em um primeiro terminal, suba a API oficial com as chaves configuradas:

```powershell
$env:TRIAGE_ML_API_KEY_SERVICE = "srv-..."
$env:TRIAGE_ML_API_KEY_DOCTOR = "doc-..."
$env:TRIAGE_ML_API_KEY_PATIENT = "pat-..."
$env:MODEL_PATH = "C:\caminho\para\models\<versao>\model.joblib"
uv run uvicorn triage_ml.api.app:app --host 127.0.0.1 --port 8000
```

Em outro terminal, use a **mesma chave de médico** e defina credenciais de
demonstração que não devem ser versionadas:

```powershell
$env:TRIAGE_ML_PROD_API_URL = "http://127.0.0.1:8000"
$env:TRIAGE_ML_API_KEY_DOCTOR = "doc-..."
$env:TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME = "medico-demo"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD = "uma-senha-local-forte"
$env:TRIAGE_ML_DASHBOARD_PATIENT_USERNAME = "paciente-demo"
$env:TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD = "outra-senha-local-forte"

uv run streamlit run front/app_prod.py
```

Abra `http://localhost:8501`. Para o vídeo, entre uma vez como paciente para
mostrar o bloqueio clínico e outra como médico para executar uma predição com
texto sintético em inglês. Não use laudos reais na demonstração.

### Testes de navegador

Os testes Playwright em `tests/e2e/test_prod_portal.py` iniciam o portal contra
uma API determinística exclusiva de teste e validam pelo Chromium:

- rejeição de credenciais inválidas;
- login, navegação pelas três etapas e logout do paciente;
- ausência do botão de predição e **zero chamadas** a `POST /predict` durante a
  sessão do paciente;
- login médico, aviso de uso profissional e predição autenticada server-side.

O job `front-e2e` do GitHub Actions instala o Chromium, sobe os dois processos e
publica logs, screenshots e traces de falha no artefato `front-e2e-evidence` por
14 dias. Para reproduzir localmente, instale o navegador com
`uv run playwright install chromium`, suba a API de teste e o Streamlit com as
variáveis demonstrativas descritas acima e execute:

```powershell
uv run pytest tests/e2e -m e2e --browser chromium --output test-results/playwright `
  --screenshot only-on-failure --tracing retain-on-failure
```
