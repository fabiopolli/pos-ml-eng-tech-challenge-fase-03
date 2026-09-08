# Relatório de implementação — Etapa 6 (Observabilidade e stack Prometheus/Grafana)

| Campo | Valor |
|---|---|
| Integrante | Bill |
| Etapa do checklist | Etapa 6 — Observabilidade e stack Prometheus/Grafana (`docs/CHECKLIST.md`) |
| Período desta entrega | 2026-09-08 (implementação inicial + dois ciclos de revisão cruzada) |
| Última revisão | 2026-09-08 — auditoria executável corrigiu imagem com extras, readiness ONNX, datasource, queries, cardinalidade e gerador de tráfego |
| Status | Stack declarativa validada, dashboard canônico com 4 painéis, privacidade de `text` coberta na API real e suíte completa verde com extras |

Este relatório documenta a entrega da Etapa 6: middleware Prometheus no `/predict`, métricas privadas (`CollectorRegistry` dedicado) com cardinalidade controlada, dashboard Grafana provisionado automaticamente, Compose overlay com `api-sklearn` + `api-onnx` + Prometheus + Grafana em rede privada, gerador de carga sintética benigna e política de privacidade de `text` enforçada por teste automatizado.

## 1. Resumo executivo

- `src/triage_ml/observability/metrics.py` define `REQUESTS_TOTAL`, `REQUEST_LATENCY_SECONDS` e `PREDICTION_ERRORS_TOTAL` em `CollectorRegistry` **privado** (não vaza do registry global do `prometheus_client`), buckets `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5]`s e degrade gracioso quando o `[observability]` extra está ausente.
- `src/triage_ml/observability/middleware.py` — ASGI middleware route-aware que mapeia `path → /predict|/reload|/health|/model-info|/models|/metrics|/"` via templates, mantendo a label `route` de cardinalidade baixa (qualquer path fora dos templates cai em `/other`).
- API oficial (`src/triage_ml/api/app.py`): `add_middleware(PrometheusMiddleware)`, novo endpoint `GET /metrics` (sem RBAC nesta fase; revisitar na Etapa 8), `TRIAGE_ML_MODEL_VARIANT={sklearn,onnx}` resolvido em `app.state.model_variant` no `lifespan` (single source of truth). `predict` consulta o adapter ONNX quando a variante ativa é `onnx` (LinearSVC cai para `decision_function`).
- Erros de `POST /predict` gravam `request.state.error_code` em allow-list público para `prediction_errors_total{route, error_code, model_variant}`; erros de outras rotas não poluem essa métrica.
- `infra/docker-compose.yml` overlay roda `api-sklearn:8000`, `api-onnx:8000`, `prometheus:9090`, `grafana:3000` em rede privada `observability`, com `read_only: true`, `cap_drop: ALL`, `no-new-privileges` (espelha o compose de produção).
- `monitoring/prometheus/prometheus.yml` scrape em `api-sklearn:8000/metrics` e `api-onnx:8000/metrics` (intervalo 5s); datasource + provider YAML em `monitoring/grafana/provisioning/`; dashboard JSON canônico `monitoring/grafana/dashboards/triage_ml.json` com 4 painéis (`Requests by route/status`, `Latency p95` por `model_variant`, `Prediction error rate`, tabela `Baseline vs optimized (p50/p95/p99)`).
- `Dockerfile` usa um builder dedicado para `runtime-observability` com os extras lockados; `runtime` permanece sem extras e é o target explícito das APIs padrão.
- `scripts/generate_observability_traffic.py` — gerador de carga sintética benigna que intercala as variantes, aceita `--iterations`, exige credencial compatível com doctor e falha se o scrape não confirmar as observações.
- 13 novos testes em `tests/test_observability_metrics.py` e `tests/test_observability_privacy.py` — verifica `ALLOWED_LABELS`, cardinalidade do payload Prometheus (com `le` reservado para buckets de histograma), ausência do fixture de texto `"PRIVACY-CANARY-CARDIOVASCULAR..."` em logs/resposta/métricas, e `render_metrics()`.
- Política de privacidade publicada em [`README.md`](../../README.md) e em [`.agents/contracts/README.md`](../../.agents/contracts/README.md): texto é classificado e descartado; nunca persistido, logado, copiado para label ou retornado em erro. Labels Prometheus permitidas: `route`, `method`, `status`, `model_variant`, `error_code` (+ `le` reservado pelo Prometheus para buckets de histograma).
- Suíte final com extras: **276 testes verdes**, 1 skip do cenário que exige ambiente sem extras.

## 2. Escopo e alinhamento com o plano

Itens concluídos da Etapa 6:

- [x] Expor métricas com `prometheus_client` no middleware da API oficial.
- [x] Medir total de requisições por rota/status.
- [x] Medir latência/tempo de resposta (reaproveitando `Server-Timing` da Etapa 2).
- [x] Medir total/taxa de erros.
- [x] Evitar labels de alta cardinalidade e conteúdo clínico. Teste automatizado varre labels aceitos.
- [x] Configurar Compose com API, Prometheus e Grafana.
- [x] Provisionar dashboard reprodutível em JSON com pelo menos quatro painéis: requisições, latência p95, erros e comparação baseline vs otimizado.
- [x] Salvar print e JSON do dashboard em `reports/figures/`.
- [x] Garantir que `text` nunca aparece em logs, payloads de erro ou labels de métrica (teste de fumaça).
- [x] Documentar a política de não retenção do `text` após a resposta.

O aceite oficial (20% junto com Etapa 5) — stack completa no Compose e dashboard exibindo as métricas propostas, incluindo o comparativo baseline vs otimizado — foi comprovado na execução local de 2026-09-08 (vide § 8).

## 3. Arquitetura e reuso

```text
                       ┌─────────────────────────────────────┐
                       │      Grafana (provisionado)         │
                       │   dashboards/triage_ml.json (4 p.)  │
                       └────────────────┬────────────────────┘
                                        │ datasource=Prometheus
                                        ▼
                       ┌─────────────────────────────────────┐
                       │   Prometheus (scrape_interval=5s)   │
                       │   targets: api-sklearn + api-onnx   │
                       └────────────────┬────────────────────┘
                                        │ GET /metrics (allowed_codes union)
                                        ▼
   ┌────────────────────────┐   ┌─────────────────────────────────────┐
   │ infra/docker-compose   │──▶│  Dockerfile (target=runtime-obs.)   │
   │   api-sklearn (8001)   │   │  instala [observability,optimization]│
   │   api-onnx   (8002)    │   │                                     │
   │   prometheus  (9090)   │   │                                     │
   │   grafana     (3000)   │   │                                     │
   └────────────┬───────────┘   └────────────┬────────────────────────┘
                ▼                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ src/triage_ml/observability/                                │
   │   metrics.py      CollectorRegistry privado + degrade no-op │
   │   middleware.py   PrometheusMiddleware (route-aware)        │
   └────────────┬────────────────────────────────────────────────┘
                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ src/triage_ml/api/app.py  (Fase 2 — produção)               │
   │   add_middleware(PrometheusMiddleware)                      │
   │   app.state.model_variant (single source via lifespan)      │
   │   /metrics → render_metrics()                               │
   └────────────┬────────────────────────────────────────────────┘
                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ src/triage_ml/optimization/onnx_adapter.py  (Etapa 5)       │
   │   OnnxModelAdapter.__call__ → (labels, proba, kinds)         │
   └─────────────────────────────────────────────────────────────┘
```

| Componente | Responsabilidade | Reuso |
|---|---|---|
| `src/triage_ml/observability/metrics.py` | Definição dos 3 collectors no registry privado e constante `ALLOWED_LABELS`. | `prometheus_client` opcional. |
| `src/triage_ml/observability/middleware.py` | Middleware ASGI route-aware, normalização de `error_code` e `route`, integração com `_model_variant` (single source). | `ALLOWED_ERROR_CODES`/`LANGUAGE_ERROR_CODES` extraídos de `triage_ml.dev_api.app`. |
| `src/triage_ml/api/app.py` (Fase 2) | `add_middleware`, `GET /metrics`, branch ONNX em `/predict`, `app.state.model_variant`. | `ModelHolder.reload_to`/`load`, `_assert_language_consistency`, `read_runtime_variant` da Etapa 5. |
| `infra/docker-compose.yml` | Overlay com `api-sklearn`, `api-onnx`, Prometheus e Grafana. | `Dockerfile` (`runtime-observability`), `monitoring/prometheus/prometheus.yml`, dashboards/provisioning YAML. |
| `monitoring/prometheus/prometheus.yml` | Scrape dos dois alvos na rede `observability`. | Image `prom/prometheus:v2.55.1`. |
| `monitoring/grafana/dashboards/triage_ml.json` | 4 painéis canônicos (requests, latency p95, error rate, baseline vs optimized table). | Provisioning YAML (`dashboards.yml` + `datasource.yml`). |
| `scripts/generate_observability_traffic.py` | Carga sintética benigna para popular o dashboard. | `urllib.request` stdlib (sem dependências extras). |

## 4. Contrato das métricas

| Métrica | Tipo | Labels | Observação |
|---|---|---|---|
| `triage_ml_requests_total` | Counter | `route`, `method`, `status`, `model_variant` | Uma observação por request HTTP, independente do resultado. |
| `triage_ml_request_latency_seconds` | Histogram | `route`, `method`, `model_variant` | Buckets `(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)`. |
| `triage_ml_prediction_errors_total` | Counter | `route`, `error_code`, `model_variant` | Incrementado apenas em `/predict` para `status >= 400` e `error_code` no allow-list público. |

`ALLOWED_LABELS = frozenset({"route", "method", "status", "model_variant", "error_code"})` é a constante de cardinalidade — qualquer outra label no payload Prometheus falha no teste `test_metric_payload_respects_allowed_labels`.

`FORBIDDEN_LABEL_HINTS = ("text", "label_name", "request_id")` é a heurística de defesa contra vazamento de `text` clínico em labels (validado por `tests/test_observability_privacy.py::test_metric_payload_excludes_clinical_text`).

## 5. Contrato de privacidade (teste de fumaça)

- `text` nunca aparece em: response body de `/predict`, `tests/.../conftest.py`, fixtures, payloads de erro, logs estruturados, métricas Prometheus.
- O canário `"PRIVACY-CANARY-CARDIOVASCULAR..."` é procurado no payload de `/metrics`, no body de `/predict` (sucesso e erro) e nos logs capturados; qualquer hit falha o teste.
- O dashboard Grafana usa apenas labels derivadas (`route`, `model_variant`, `error_code`); nenhum painel recebe `text` como input.
- Política textual publicada em `README.md` ("Política de privacidade") e em `.agents/contracts/README.md` ("Sem persistência de `text` após a resposta").

## 6. Endurecimento aplicado nos 2 ciclos de revisão cruzada (2026-09-08)

Após a implementação inicial, Bill executou **dois ciclos** de revisão estática cruzada (alto risco por tocar em código de produção com métricas Prometheus, RBAC e dependências opcionais) com cross-validação por dois sub-agentes. Pontos consolidados:

### 6.1 Primeiro ciclo (commit `6151c87`)

1. **`general_handler` engole traceback** — `JSONRenderer` do structlog sem `format_exc_info`; `StackInfoRenderer` + `format_exc_info` em `logging_config.py`, trocado `logger.error` por `logger.exception`.
2. **`StarletteHTTPException` aceitava qualquer string como `error_code`** — criado `_resolve_error_code` que filtra contra `ALLOWED_ERROR_CODES` extraído como constante compartilhada em `dev_api/app.py` (adicionados `unauthorized`, `forbidden`, `clinician_review_required`).
3. **`validation_failed` divergia** entre Etapas 2 e 3 — padronizado para `"Request body is invalid."` em ambos.
4. **`request_id` fallback `"unknown"`** — helper `_request_id_for` retorna `None` quando o middleware não rodou.
5. **`latency_ms` incluía detecção de idioma** — renomeado para `predict_latency_ms`; `Server-Timing` agora sempre emite `total;dur=` + `detect;dur=` + `predict;dur=` quando disponíveis.
6. **`setup_logging` reconfigurava structlog em cada `create_app`** — idempotente via flag `_logging_configured`.
7. **`/predict` RBAC inline vs `RequireRole`** — extraída `RequirePredictRole` em `auth.py`.
8. **`/reload` perdia `model_version` no log de erro** — agora loga `model_version=payload.model_version`.
9. **`Server-Timing` 3 casas vs `latency_ms` float cru** — `latency_ms` agora `round(x, 3)`.
10. **`/model-info` e `/models` públicos** — restritos a `RequireRole(["service","doctor"])`.
11. **Fingerprint SHA-256 simples da chave API** — substituído por `HMAC-SHA-256` com sal aleatório de 32 bytes em import-time.
12. **`front/app_prod.py` renderizava `json.dumps(response.body)`** — agora exibe apenas `request_id` para correlação.
13. **`RequireRole.allowed_roles` mutável** — convertido para `frozenset`.

### 6.2 Segundo ciclo (commit `6cbed53`)

1. **`ModelHolder.reload_to` não invalidava `_onnx_predictor`** — agora zera dentro de `self._lock` antes do swap para evitar que `/predict` concorrente recicle o adapter da versão anterior.
2. **`create_app` setava `app.state.model_variant` duas vezes** — `lifespan` é o único setter; helper do middleware cai para `"sklearn"` quando ausente (cobre `TestClient` sem lifespan).
3. **`/predict` ONNX chamava `session.run` 2×** (em `predict` + `predict_proba`) — `OnnxModelAdapter.__call__` emite **uma** chamada via `_run_once`, retornando `(labels, proba, kinds)` em tuple.
4. **`_resolve_label_index` faltava em `OnnxModelAdapter.predict`** — fallback chain (index → class match → argmax → best-effort cast) evita `IndexError` em LinearSVC.
5. **`_load_onnx` aceitava `classes=None`/`"a","b"` silenciosamente** — `ValueError` quando ausente ou não-inteiro.
6. **`_slice_identity_fields` retornava silenciosamente `"logreg"`** — `ValueError` quando nem `selected_classifier` parâmetro nem `selection_overrides.classifier` está presente.
7. **`_find_reusable_for_size` reusava órfão** — checa `joblib_path.is_file()` antes do reuso; cross-checa `manifest.get("selected_classifier")` vs config ativa.
8. **`export_onnx_for_version` sempre `reused=False`** — agora `reused=True` quando `model.onnx` checksum on-disk bate com `existing_optimization.onnx_checksum_sha256`.
9. **`benchmark_for_version` swallow de `validate_variant_metadata`** — agora `structlog.get_logger().warning(...)` (não silencioso).
10. **`_METRIC_ERROR_CODES` estático fora de sincronia** — `frozenset(ALLOWED_ERROR_CODES | LANGUAGE_ERROR_CODES | {"request_failed"})` derivado das constantes do `dev_api/app.py` (single source). `internal_error` foi **deliberadamente removido** do allow-list público (não vaza como label Prometheus).
11. **`production_client` fixture não pegava env vars** — `get_settings.cache_clear()` adicionado para evitar que outro teste da suíte congele uma instância stale.
12. **`test_load_onnx_rejects_*` com bundle sintético inválido** — fixtures migradas para `monkeypatch` em `load_artifact`/`_validated_model_path` (não monta o bundle completo via `validate_artifact_bundle`, que exige 16 chaves + per_class + fingerprints + git_commit + created_at + dependency_versions).

### 6.3 Testes adicionados no segundo ciclo

- `tests/test_post_fase2_review_round2.py::test_reload_to_invalidates_onnx_predictor_cache` — `_onnx_predictor` zera após `reload_to`.
- `tests/test_post_fase2_review_round2.py::test_load_onnx_rejects_missing_classes` / `test_load_onnx_rejects_non_int_classes` — `_load_onnx` defensivo.
- `tests/test_post_fase2_review_round2.py::test_onnx_adapter_reuses_session_across_calls` / `test_onnx_adapter_call_runs_session_once` — singleton + single `session.run` via `__call__`.
- `tests/test_post_fase2_review_round2.py::test_validation_failed_appears_in_metrics` — integração `TestClient → PREDICTION_ERRORS_TOTAL` no 422.
- `tests/test_post_fase2_review_round2.py::test_model_not_ready_returns_503_when_onnx_missing` — branch ONNX sem `model.onnx` vira 503.
- `tests/test_post_fase2_review_round2.py::test_metric_error_codes_includes_full_union` — `_METRIC_ERROR_CODES` espelha `ALLOWED_ERROR_CODES | LANGUAGE_ERROR_CODES | {"request_failed"}`.

`tests/test_post_fase2_review.py` foi atualizado: o allow-list de `internal_error` foi removido (vazamento de label Prometheus) e os códigos LANGUAGE (`text_too_short_for_language_check`, `indeterminate_language`, `language_config_incompatible`) foram adicionados ao parametrize.

## 7. Configuração

O arquivo `.env.example` documenta as variáveis para a stack de observabilidade:

```dotenv
# Compose overlay (infra/docker-compose.yml)
MODEL_VERSION=20260905T171611Z-f2cb6f23f9cd
TRIAGE_ML_MODEL_VARIANT=sklearn   # sklearn | onnx
TRIAGE_ML_LOG_LEVEL=INFO
TRIAGE_ML_RATELIMIT_DEFAULT=120/minute
TRIAGE_ML_RATELIMIT_PREDICT=120/minute

# Grafana provisioning
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=set-a-strong-password

# (Opcional) afrouxa a guarda de loopback em generate_observability_traffic.py
TRIAGE_ALLOW_PUBLIC_OBSERVABILITY=false
```

A imagem do overlay usa `runtime-observability`, cuja virtualenv é resolvida no builder com `--extra observability --extra optimization`. A produção continua consumindo explicitamente `target=runtime`.

## 8. Comandos reproduzíveis

```bash
# Validação estática e testes da Etapa 6
uv run ruff format --check src/ tests/ scripts/ airflow/
uv run ruff check src/ tests/ scripts/ airflow/
uv run pytest tests/test_observability_metrics.py tests/test_observability_privacy.py -v
uv run --extra optimization --extra observability pytest tests/

# Subir a stack overlay (api-sklearn + api-onnx + prometheus + grafana)
docker compose -f infra/docker-compose.yml build
docker compose -f infra/docker-compose.yml up -d --wait
docker compose -f infra/docker-compose.yml ps

# Verificar /metrics em ambos os serviços
curl -s http://127.0.0.1:8001/metrics | grep -E '^triage_ml_'
curl -s http://127.0.0.1:8002/metrics | grep -E '^triage_ml_'

# Gerar tráfego sintético benigno (32 amostras por variante)
python scripts/generate_observability_traffic.py \
  --sklearn-url http://127.0.0.1:8001 \
  --onnx-url http://127.0.0.1:8002 \
  --api-key "$TRIAGE_ML_API_KEY_DOCTOR" \
  --iterations 5

# Conferir o dashboard
open http://127.0.0.1:3000/d/triage-ml-observability   # admin / $GRAFANA_ADMIN_PASSWORD
```

## 9. Evidência de execução 2026-09-08

Execução local do overlay Compose (artefato `20260905T171611Z-f2cb6f23f9cd` montado em `/models:ro`):

| Verificação | Resultado |
|---|---|
| `docker compose -f infra/docker-compose.yml build` | sucesso para `triage-ml-api-observability:local`, `prom/prometheus:v2.55.1`, `grafana/grafana:11.3.1`. |
| `docker compose ... up -d --wait` | 4 contêineres `healthy` em ≤ 20s (`api-sklearn`, `api-onnx`, `prometheus`, `grafana`). |
| `GET /health` em `api-sklearn` (8001) | `{"status":"ok","model_version":"20260905T171611Z-f2cb6f23f9cd","model_variant":"sklearn"}`. |
| `GET /health` em `api-onnx` (8002) | `{"status":"ok","model_version":"20260905T171611Z-f2cb6f23f9cd","model_variant":"onnx"}`. |
| `GET /metrics` em ambos | expõe `triage_ml_requests_total`, `triage_ml_request_latency_seconds_*`, `triage_ml_prediction_errors_total` sem labels proibidas. |
| `python scripts/generate_observability_traffic.py` | Com `--iterations 5`, produz 320 chamadas (5 × 32 × 2 variantes), intercaladas e confirmadas pelo scrape. |
| Painel "Requests by route/status" | série temporal por `(route, status)` mostrando `/predict` em ambos os variants. |
| Painel "Latency p95" | séries separadas por `model_variant` (`sklearn`, `onnx`). |
| Painel "Prediction error rate" | vazio até erro proposital; após `POST /predict` com chave inválida aparece `error_code="unauthorized"`. |
| Painel "Baseline vs optimized (p95)" | tabela com `p50`, `p95`, `p99` por `model_variant`. |
| Grafana provisioning | dashboard provisionado automaticamente a partir de `/var/lib/grafana/dashboards/triage_ml.json` (sem upload manual). |
| Logs do Prometheus | scrape `up{job="triage-ml-api",instance="api-sklearn:8000"} = 1` e `= 1` para `api-onnx`. |

Print e JSON do dashboard versionáveis em `reports/figures/triage_ml_dashboard.png` e `reports/figures/triage_ml_dashboard.json` (captura via `curl -u admin:$GRAFANA_ADMIN_PASSWORD http://127.0.0.1:3000/api/dashboards/uid/triage-ml-observability`).

## 10. Validação final pós 2 ciclos de revisão cruzada

| Verificação | Resultado |
|---|---|
| `uv run ruff format --check .` | aprovado (67 arquivos unchanged) |
| `uv run ruff check .` | aprovado (All checks passed!) |
| `uv run --extra optimization --extra observability pytest tests/` | **276 aprovados**, 1 skip do cenário que exige ambiente sem extras, 0 falhas |
| Política de privacidade | `"PRIVACY-CANARY-CARDIOVASCULAR..."` ausente de logs, body e métricas (assertions em `test_observability_privacy.py`) |
| Cardinalidade Prometheus | apenas `route`, `method`, `status`, `model_variant`, `error_code` (+ `le` para buckets) |
| `app.state.model_variant` | single-source via `lifespan`; helper `_model_variant` cai para `"sklearn"` em `TestClient` sem lifespan |
| `OnnxModelAdapter.__call__` | emite exatamente **1** `session.run` por chamada (verificado por spy em `test_onnx_adapter_call_runs_session_once`) |
| `_METRIC_ERROR_CODES` | derivado de `ALLOWED_ERROR_CODES \| LANGUAGE_ERROR_CODES \| {"request_failed"}`; `internal_error` intencionalmente fora |

A Etapa 6 está concluída. A stack Prometheus + Grafana + API oficial está pronta para a Etapa 8 consolidar a documentação final e o comparativo baseline vs otimizado apresentado no dashboard.
