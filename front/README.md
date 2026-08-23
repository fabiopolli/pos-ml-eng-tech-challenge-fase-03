# Front — Dashboard de desenvolvimento

Ferramentas de interface voltadas para o desenvolvedor. Não fazem parte
do runtime da API; existem apenas para acelerar validações manuais da
API de desenvolvimento (`src/triage_ml/dev_api/`).

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
   quando a API devolve o array de classes (`LinearSVC` → `score=null`).
3. **🌐 Política de idioma** — quatro cenários canônicos da política
   definida em `configs/api.yaml` (texto curto, confiança baixa,
   idioma fora do allow-list, inglês válido) com validação automática
   do `error_code` retornado pela API. Os cenários embutidos
   (EN/CURTO/BAIXA/PT) moram em `LANGUAGE_PRESETS`.

**Sidebar**:

- **🔌 Conexão** — URL base da API + botão "Atualizar health" para
  revalidar `/health` sem reiniciar o Streamlit. O campo aceita
  qualquer URL (local, container, cloud) — `Path.as_uri()` /
  `requests` resolvem o resto.
- **🔁 Trocar modelo** — consome `GET /models` para listar as versões
  imutáveis disponíveis em `models/` (newest-first), mostra a versão
  atualmente em uso, deixa você escolher outra via `<selectbox>` (com
  default = versão corrente) e dispara `POST /reload`. Resposta
  bem-sucedida força o refresh dos blocos abaixo; resposta `404
  model_not_found` ou `500 model_incompatible` mostra o erro em
  vermelho sem alterar o holder. Estado apenas em memória — não
  persiste em arquivo nem entre sessões Streamlit.
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
PYTHONPATH=src uv run uvicorn triage_ml.dev_api.app:app --host 127.0.0.1 --port 8000

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
  em disco. O estado do picker vive só em `st.session_state`.
- Persistência entre sessões — o model picker é por-design em
  memória; ao reiniciar o Streamlit, a escolha volta ao default
  (versão atualmente em uso na API).
