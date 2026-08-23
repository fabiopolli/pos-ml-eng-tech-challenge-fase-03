# Front — Dashboard de desenvolvimento

Ferramentas de interface voltadas para o desenvolvedor. Não fazem parte
do runtime da API; existem apenas para acelerar validações manuais da
API de desenvolvimento (`src/triage_ml/dev_api/`).

## `app_dev.py`

Dashboard Streamlit que fala HTTP com qualquer instância da API
(local, container ou cloud) e expõe três abas + uma sidebar fixa:

**Abas**:

1. **Health** — chama `GET /health`.
2. **Predição** — área de texto + `POST /predict` com leitura completa
   do body, `latency_ms`, `request_id` e `Server-Timing`.
3. **Política de idioma** — quatro cenários canônicos da política
   definida em `configs/api.yaml`, com validação do `error_code`
   retornado pela API.

**Sidebar**:

- **🔌 Conexão** — URL base da API + botão para revalidar `/health`.
- **🧠 Modelo** — consome `GET /model-info` e mostra, em expanders:
  identidade do artefato (`model_version`, `model_name`, `task_type`,
  `language`), treinamento (`n_train`, `n_test`, `random_state`,
  `git_commit`, `git_dirty`, `created_at`, `dependency_versions`),
  seleção do classificador (`selected_classifier`, folds, candidatos
  `logreg` × `linear_svc` com mean ± std do cross-validation),
  métricas globais (`accuracy`, `balanced_accuracy`, `macro_f1`,
  `weighted_f1`) e métricas per-classe (`precision`, `recall`, `f1`,
  `support`) em formato tabular.
- **📚 Atalhos** — links `file://` para a documentação versionada
  (Plan do classificador, Checklist oficial, Relatório Fase 1).

A conexão é HTTP contra a URL configurada na sidebar (default
`http://127.0.0.1:8000`). O dashboard não toca o artefato do modelo
nem o detector `langid` — qualquer divergência com a resposta da API é
um sinal de bug.

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
  baseline vs otimizado) → moram no stack **Prometheus + Grafana**.
- Avaliação do modelo → notebooks `01_eda.ipynb` / `02_model_baseline.ipynb`
  e `docs/reports/IMPLEMENTATION-REPORT-FASE-1.md`.
- Privacidade clínica → o dashboard não armazena payloads nem textos.
