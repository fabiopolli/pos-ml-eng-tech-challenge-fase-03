# Front — Smoke dashboard

Ferramentas de interface voltadas para o desenvolvedor. Não fazem parte
do runtime da API; existem apenas para acelerar smoke tests manuais.

## `app_smoke.py`

Dashboard Streamlit que fala HTTP com qualquer instância da API
(local, container ou cloud) e expõe três áreas:

1. **Health** — chama `GET /health`.
2. **Predição** — área de texto + `POST /predict` com leitura completa
   do body, `latency_ms`, `request_id` e `Server-Timing`.
3. **Política de idioma** — quatro cenários canônicos da política
   definida em `configs/api.yaml`, com validação do `error_code`
   retornado pela API.

A conexão é HTTP contra a URL configurada na sidebar (default
`http://127.0.0.1:8000`). O dashboard não toca o artefato do modelo
nem o detector `langid` — qualquer divergência com a resposta da API é
um sinal de bug.

### Como rodar

```bash
# 1. Suba a API em outro terminal
PYTHONPATH=src uv run uvicorn triage_ml.api.app:app --host 127.0.0.1 --port 8000

# 2. Suba o dashboard
uv run streamlit run front/app_smoke.py
```

Acesse `http://localhost:8501`.

### O que **não** é objetivo deste dashboard

- Métricas de produção (latência p95, taxa de erro, comparação
  baseline vs otimizado) → moram no stack **Prometheus + Grafana**.
- Avaliação do modelo → notebooks `01_eda.ipynb` / `02_model_baseline.ipynb`
  e `docs/reports/IMPLEMENTATION-REPORT-FASE-1.md`.
- Privacidade clínica → o dashboard não armazena payloads nem textos.
