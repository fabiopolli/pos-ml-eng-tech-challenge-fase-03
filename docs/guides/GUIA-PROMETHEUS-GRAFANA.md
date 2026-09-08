# Guia de uso do Prometheus e Grafana

Este guia cobre a stack de observabilidade (Fase 2 — Etapa 6), do scrape do Prometheus à leitura do dashboard Grafana.

## Arquitetura da stack

```text
                          ┌─────────────────────┐
                          │ api-prod (FastAPI)  │──┐
                          │ :8000, model=sklearn│  │
                          └─────────────────────┘  │
                                                  │ /metrics
                                                  ▼
                          ┌─────────────────────┐  ┌──────────────────┐
                          │ api-onnx (FastAPI)  │─►│ Prometheus :9090 │
                          │ :8000, model=onnx   │  └──────────────────┘
                          └─────────────────────┘         │
                                                          │ PromQL
                                                          ▼
                                                  ┌──────────────────┐
                                                  │ Grafana :3000    │
                                                  │ dashboard 4 painéis│
                                                  └──────────────────┘
```

- O Prometheus **scrapa** `/metrics` a cada 5 s em ambas as APIs, com timeout de 4 s.
- O Grafana **consome** o Prometheus via datasource provisionado em [`monitoring/grafana/provisioning/datasources/datasource.yml`](../../monitoring/grafana/provisioning/datasources/datasource.yml).
- O dashboard é **provisionado automaticamente** em [`monitoring/grafana/dashboards/triage_ml.json`](../../monitoring/grafana/dashboards/triage_ml.json).
- Cópia física do dashboard em [`reports/figures/triage_ml_dashboard.{json,png}`](../../reports/figures/) (geradas por [`scripts/render_observability_dashboard.py`](../../scripts/render_observability_dashboard.py)).

## Subir a stack

```bash
# 1. Gerar .env a partir de um único comando (detecta o modelo mais recente em
#    models/, gera secrets aleatórios via secrets.token_urlsafe(32), falha
#    rápido se você não tiver modelo treinado). Equivalente a:
#    MODEL_VERSION=...   TRIAGE_ML_API_KEY_*=...   GRAFANA_ADMIN_PASSWORD=...
uv run python scripts/bootstrap_observability_overlay.py

# 2. Subir overlay de observabilidade (api-prod + api-onnx + prometheus + grafana)
docker compose -f infra/docker-compose.yml up -d --wait

# 3. Conferir o estado dos serviços
docker compose -f infra/docker-compose.yml ps

# 4. Gerar tráfego benigno para popular os painéis
uv run python scripts/generate_observability_traffic.py \
  --sklearn-url http://127.0.0.1:8001 \
  --onnx-url   http://127.0.0.1:8002 \
  --api-key "$(grep '^TRIAGE_ML_API_KEY_DOCTOR=' .env | cut -d= -f2)" \
  --iterations 5

# 5. Acessar
#    Prometheus: http://localhost:9090
#    Grafana:    http://localhost:3000  (admin / $GRAFANA_ADMIN_PASSWORD)

# 6. Encerrar
docker compose -f infra/docker-compose.yml down
```

> **Por que dois containers da API?** Para isolar a comparação de latência sklearn vs ONNX no mesmo probe de carga, sem precisar de flags na API de produção. `api-sklearn` mantém `TRIAGE_ML_MODEL_VARIANT=sklearn` (default) e `api-onnx` fixa `TRIAGE_ML_MODEL_VARIANT=onnx`.
>
> **Armadilha frequente: `cat > .env <<EOF ... EOF` no shell.** Em alguns shells interativos o heredoc termina assim que você digita a primeira `EOF` solta, e o comando subsequente (`docker compose up`) lê um `.env` truncado. O `docker compose` então reclama `required variable MODEL_VERSION is missing a value`. Use sempre o helper [`scripts/bootstrap_observability_overlay.py`](../../scripts/bootstrap_observability_overlay.py) que escreve o arquivo de forma atômica e detecta automaticamente a versão do modelo em `models/`.

## Métricas expostas

A API mantém um `CollectorRegistry` privado — não vaza do registry global do `prometheus_client`. As três séries são:

### `triage_ml_requests_total`

Counter rotulado por rota, método, status HTTP e variante do modelo.

```
triage_ml_requests_total{
  route="/predict",
  method="POST",
  status="200",
  model_variant="sklearn"
} 1234
```

### `triage_ml_request_latency_seconds`

Histograma com buckets `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]` segundos, rotulado por `route`, `method` e `model_variant`. Soma em `_sum` e contagem em `_count`.

```
triage_ml_request_latency_seconds_bucket{
  route="/predict", method="POST", model_variant="sklearn", le="0.025"
} 800
triage_ml_request_latency_seconds_bucket{
  route="/predict", method="POST", model_variant="sklearn", le="0.05"
} 1100
triage_ml_request_latency_seconds_sum{...} 45.32
triage_ml_request_latency_seconds_count{...} 1234
```

### `triage_ml_prediction_errors_total`

Counter rotulado por rota, `error_code` (allow-list público) e variante. Apenas `/predict` incrementa.

```
triage_ml_prediction_errors_total{
  route="/predict", error_code="unsupported_language", model_variant="sklearn"
} 12
```

Allow-list público de `error_code` (válido como label Prometheus):

```
validation_failed, text_too_short_for_language_check, indeterminate_language,
unsupported_language, language_config_incompatible, model_not_ready,
prediction_failed, unauthorized, forbidden, clinician_review_required, request_failed
```

`internal_error` é deliberadamente **fora** da allow-list — não vaza como label Prometheus.

## PromQL — exemplos prontos

### Latência p95 por variante

```promql
histogram_quantile(
  0.95,
  sum by (le, model_variant) (
    rate(triage_ml_request_latency_seconds_bucket{route="/predict", method="POST"}[5m])
  )
)
```

### Taxa de requisições por status code

```promql
sum by (status) (rate(triage_ml_requests_total[1m]))
```

### Total de erros de predição por `error_code` (últimos 5 min)

```promql
sum by (error_code) (
  increase(triage_ml_prediction_errors_total[5m])
)
```

### Throughput por variante

```promql
sum by (model_variant) (rate(triage_ml_requests_total{route="/predict"}[1m]))
```

### Comparativo direto sklearn vs ONNX (latência p95)

```promql
histogram_quantile(0.95, sum by (le, model_variant) (
  rate(triage_ml_request_latency_seconds_bucket{route="/predict", method="POST"}[5m])
))
```

### SLO: % de predições abaixo de 100 ms

```promql
sum(rate(triage_ml_request_latency_seconds_bucket{route="/predict", method="POST", le="0.1"}[5m]))
/
sum(rate(triage_ml_request_latency_seconds_count{route="/predict", method="POST"}[5m]))
```

## Painéis do dashboard

O dashboard [`triage_ml`](../../monitoring/grafana/dashboards/triage_ml.json) tem 4 painéis na ordem:

### 1. `Requests by route/status`

Time series empilhado de `rate(triage_ml_requests_total[1m])` agrupado por `route` e `status`. Útil para:

- Verificar volume absoluto (RPS).
- Detectar picos de 4xx/5xx.
- Confirmar se `/predict` domina o tráfego.

### 2. `Latency p95`

`histogram_quantile(0.95, sum by (le, model_variant) (...))` com duas séries (sklearn/onnx). Mostra a curva de p95 ao longo do tempo.

Para alternar para p50 ou p99, edite o PromQL direto no painel.

### 3. `Prediction error rate`

Razão entre `rate(triage_ml_prediction_errors_total[5m])` e o total de requisições de `/predict`, agrupada por variante e `error_code`.

Atenção: rotas `/health`, `/model-info`, `/models` e `/metrics` não entram nesse painel (não são rotas de predição).

### 4. `Baseline vs optimized (p95)`

Tabela com p50/p95/p99 HTTP por `model_variant`, calculada pelo Prometheus apenas sobre `POST /predict`. O comparativo offline com macro-F1 permanece em `reports/benchmarks/dataset_sizing.json`.

Quando o benchmark controlado rodar, a tabela reflete o `Δ macro-F1` aceitável (≤ 1 pp). Veja [Etapa_5_Otimizacao_do_modelo.md](../reports/Etapa_5_Otimizacao_do_modelo.md).

## Provisionamento automático

| Componente | Arquivo | Como é aplicado |
|---|---|---|
| Datasource Prometheus | `monitoring/grafana/provisioning/datasources/datasource.yml` | Carregado pelo Grafana no startup via volume mount |
| Dashboard | `monitoring/grafana/dashboards/triage_ml.json` | Provisionado por `monitoring/grafana/provisioning/dashboards.yml` |
| Scrape config | `monitoring/prometheus/prometheus.yml` | Carregado pelo Prometheus no startup |
| Rede privada | `infra/docker-compose.yml::networks` | Isola Prometheus, Grafana e APIs de outros serviços |

Para adicionar um dashboard novo sem rebuildar a imagem, basta montar o JSON em `monitoring/grafana/dashboards/`.

## Privacidade e segurança

- **`text` jamais vira label** — a allow-list de labels é `route`, `method`, `status`, `model_variant`, `error_code` (e `le` para buckets do histograma). Tentativas de incluir `text`, `label_name` ou `request_id` quebram `tests/test_observability_privacy.py` e o teste de cardinalidade `tests/test_metrics_labels_respect_allowed_cardinality`.
- **Canário de privacidade**: `PRIVACY-CANARY-CARDIOVASCULAR-RESPIRATORY 2025 with severe stenosis and arrhythmia` é procurado em `/metrics`, body de `/predict` e logs capturados por `tests/test_observability_privacy.py`. Se aparecer em qualquer um desses lugares, o teste falha.
- **`/metrics` sem autenticação** por padrão (Fase 2). A Etapa 8 (cloud) deve colocar o endpoint atrás de reverse proxy autenticado ou movê-lo para uma porta privada.

## Troubleshooting

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Grafana abre sem o dashboard `triage_ml` | Provisionamento falhou | Conferir `monitoring/grafana/provisioning/dashboards.yml` e os logs do contêiner Grafana |
| Painéis sem dados | Prometheus não está scrapeando | `curl http://localhost:9090/api/v1/targets` deve listar `api-prod` e `api-onnx` como `up` |
| `error_code=unknown` aparece | Código fora da allow-list | Inspecionar logs da API; pode ser um bug na rota de exceção |
| Latência p95 muito alta mesmo com `model.onnx` | ONNX ainda carregando singleton na 1ª request | Aguardar warmup de ~5 requests (esperado pelo `benchmark_predictor`) |
| Prometheus "out of memory" | Cardinalidade explodiu | Conferir se algum label novo foi adicionado sem passar pela allow-list |
| Grafana mostra "Data source not found" | Datasource não provisionou | Conferir volume mount de `monitoring/grafana/provisioning/datasources/datasource.yml` |

## Boas práticas

- **Não usar `histogram_quantile` em série pequena** — a estimativa degrada com baixo volume; o painel de p95 só é confiável após ~100 samples.
- **Não remover `le`** dos PromQL de histograma; filtre `route="/predict"` e `method="POST"` para não misturar health checks e scrapes.
- **Não adicionar labels além da allow-list** sem atualizar `tests/test_observability_privacy.py` e `tests/test_metrics_labels_respect_allowed_cardinality`.
- **Não expor `/metrics` em produção final sem reverse proxy** — o endpoint está aberto na Fase 2 para inspeção; a Etapa 8 vai endurecer.
- **Sempre correlacionar com `Server-Timing`** — para regressões finas de latência (ex.: `predict;dur` crescendo isolado), o header HTTP é mais granular que o histograma agregado do Prometheus.
