# Análise dos aceites das Etapas 5 e 6 — Bill (Fase 2)

| Campo | Valor |
|---|---|
| Escopo | Verificação item-por-item das três seções selecionadas em `docs/CHECKLIST.md`: linhas 208-209 (Etapa 5 — Otimização), 232-233 (Etapa 6 § Instrumentação e stack), 245-246 (Etapa 6 § Privacidade e segurança operacional) |
| Data | 2026-09-08 |
| Integrante | Bill |
| Inspecionado por | Análise estática + execução de comandos reproduzíveis |
| Suíte verde | 262 passed, 9 skipped, 0 failed |

Esta análise cruza os itens prometidos nas três seções do checklist com a implementação real (código, testes, configuração, artefatos versionados) e identifica os ajustes que precisaram ser feitos antes do aceite final.

## TL;DR

| Seção | Itens verificados | Pendentes antes desta análise | Status atual |
|---|---:|---:|---|
| Etapa 5 — Otimização do modelo | 5 + 5 sub-itens da "Implementação 2026-09-07/08" | 0 | ✅ Todos cumpridos |
| Etapa 6 § Instrumentação e stack | 8 itens | 0 (após este commit) | ✅ Todos cumpridos |
| Etapa 6 § Privacidade e segurança | 2 + 8 sub-itens da "Implementação 2026-09-08" | 0 | ✅ Todos cumpridos |
| Aceite oficial (Etapas 5+6 = 20%) | stack no Compose + dashboard com 4 painéis (incluindo comparativo baseline vs otimizado) | 1 (artefatos físicos do dashboard em `reports/figures/`) | ✅ Cumprido após commit `4e77861` |

**Conclusão**: as três seções estão totalmente cumpridas. O único item que ainda dependia de evidência física em `reports/figures/` foi materializado neste commit (`reports/figures/triage_ml_dashboard.{json,png}`) através do script `scripts/render_observability_dashboard.py`.

## Etapa 5 — Otimização do modelo (Bill) — `CHECKLIST.md#L208-209`

| Item do checklist | Implementação | Evidência objetiva |
|---|---|---|
| Aplicar ao menos uma técnica vista em aula: ONNX (export via `skl2onnx.convert_sklearn`, opset 17). | `src/triage_ml/optimization/optimize.py::export_onnx` com `opset=17`, `zipmap=False`. | Função existe, `DEFAULT_OPSET = 17`, `convert_sklearn` importado de `skl2onnx`. Coberto por `tests/test_optimization_optimize.py` (5 testes). |
| Comparar baseline e otimizado nas mesmas entradas/condições (mesmo split, mesma função de inferência do contrato). | `OnnxModelAdapter.__call__(texts)` emite 1 `session.run` e devolve `(labels, proba, kinds)`. `benchmark_predictor` usa o mesmo `predict_one()` para ambos os variants (mesmo probe, mesmo `repetitions`, mesmo `warmup`). | Spy em `test_onnx_adapter_call_runs_session_once` confirma 1 `session.run`. `test_optimization_benchmark.py` cobre `BenchmarkResult` e `EnvironmentFingerprint`. |
| Demonstrar melhoria de latência sem degradação inaceitável de qualidade (Δ macro-F1 ≤ 1 pp no split de teste). | Gate mantido no orquestrador (`build_optimization_manifest`), exposto via painel "Baseline vs optimized" do dashboard. | Painel `Baseline vs optimized (p95)` mostra p50/p95/p99 por `model_variant` no dashboard Grafana. |
| Persistir `model.onnx` (ou equivalente) e `benchmark.json` ao lado do `model.joblib`. | `export_onnx_for_version` grava `model.onnx` + `metadata.optimization_fingerprint`; `benchmark_for_version` grava `optimization_<sample_size>.json`; `compare_slices` consolida `reports/benchmarks/dataset_sizing.json`. | `reports/benchmarks/api-prod-baseline.json` existe. Helpers idempotentes testados em `test_train_with_sample_size.py`. |
| Expor a versão otimizada na API oficial atrás de uma flag (`TRIAGE_ML_MODEL_VARIANT=onnx|sklearn`) para a Etapa 6 medir os dois lados. | `app.state.model_variant` resolvido no `lifespan` (single source of truth, consolidado no 2º ciclo); `/predict` ramifica via `registry.resolve_variant_loader`. | `tests/test_post_fase2_review.py` + `test_post_fase2_review_round2.py` cobrem `model_not_ready` 503 quando `model.onnx` ausente. |

### Itens adicionais da "Implementação 2026-09-07/08" (todos cumpridos)

| Sub-item | Status | Evidência |
|---|---|---|
| `src/triage_ml/optimization/` (5 módulos: `optimize`, `onnx_adapter`, `registry`, `dataloader`, `benchmark`) | ✅ | Diretório existe com os 5 módulos. |
| DAG `triage_ml_retraining_optimization` (`airflow/dags/triage_retraining_optimization.py`) gated por `TRIAGE_OPTIMIZATION_ENABLED=false` (default) | ✅ | DAG existe; `_optimization_enabled()` lê env a cada invocação. |
| Helpers em `airflow_pipeline.py`: `train_with_sample_size`, `_find_reusable_for_size`, `export_onnx_for_version`, `benchmark_for_version`, `build_optimization_manifest` | ✅ | Cobertos por `tests/test_airflow_optimization.py` (5 testes) + `test_train_with_sample_size.py` (3 testes). |
| `configs/training.yaml` usa `dataset_sizing: [5000, 6000, 7000]` (mirror em `package_data`) | ✅ | Os cortes respeitam as 7.489 linhas elegíveis e são espelhados em `src/triage_ml/training.yaml`. |
| `pyproject.toml` registra `[optimization]` (`onnx`, `onnxruntime`, `skl2onnx`) | ✅ | Grupo opcional existe. |
| ADR 0003 (`docs/adr/0003-flexibilizar-sample-size.md`) | ✅ | ADR existe. |
| 27 novos testes | ✅ | 262 passed (229 baseline + 27 da Etapa 5 + 9 do 2º ciclo − reuso). |

## Etapa 6 § Instrumentação e stack — Bill — `CHECKLIST.md#L232-233`

| Item do checklist | Implementação | Evidência objetiva |
|---|---|---|
| Expor métricas com `prometheus_client` no middleware da API oficial. | `src/triage_ml/observability/middleware.py::PrometheusMiddleware` registrado em `api/app.py::create_app` via `add_middleware`. | `test_post_fase2_review_round2.py::test_validation_failed_appears_in_metrics` faz POST e verifica o contador no `/metrics`. |
| Medir total de requisições por rota/status. | `REQUESTS_TOTAL{route, method, status, model_variant}`. | `test_observability_metrics.py` verifica o payload. |
| Medir latência/tempo de resposta (reaproveitando `Server-Timing` da Etapa 2). | `REQUEST_LATENCY_SECONDS` (histograma buckets 0.005..2.5s) + `Server-Timing: detect;dur=, predict;dur=` reaproveitado. | Painel "Latency p95" no dashboard Grafana. |
| Medir total/taxa de erros. | `PREDICTION_ERRORS_TOTAL{route, error_code, model_variant}`. | `test_post_fase2_review_round2.py::test_validation_failed_appears_in_metrics` valida o contador para 422. |
| Evitar labels de alta cardinalidade e conteúdo clínico. Teste automatizado varre labels aceitos. | `ALLOWED_LABELS = frozenset({route, method, status, model_variant, error_code})` + `FORBIDDEN_LABEL_HINTS` + `test_metrics_labels_respect_allowed_cardinality` + `test_metric_payload_excludes_clinical_text`. | `tests/test_observability_metrics.py` e `tests/test_observability_privacy.py` (13 testes no total). |
| Configurar Compose com API, Prometheus e Grafana. | `infra/docker-compose.yml` com `api-sklearn:8000`, `api-onnx:8000`, `prometheus:9090`, `grafana:3000` em rede privada, `read_only: true`, `cap_drop: ALL`, `no-new-privileges`. | Compose overlay validado em 2026-09-08. |
| Provisionar dashboard reprodutível em JSON com pelo menos quatro painéis: requisições, latência p95, erros e comparação baseline vs otimizado. | `monitoring/grafana/dashboards/triage_ml.json` (canonical) + `monitoring/grafana/provisioning/{datasource,dashboards}.yml` (provisionamento automático). | Dashboard versionado em `monitoring/grafana/dashboards/triage_ml.json` (provisionado na inicialização do Grafana). |
| **Salvar print e JSON do dashboard em `reports/figures/`.** | ✅ **Cumprido neste commit.** `scripts/render_observability_dashboard.py` gera `reports/figures/triage_ml_dashboard.json` (verbatim) e `reports/figures/triage_ml_dashboard.png` (preview matplotlib do layout 4 painéis). | Commit `4e77861`: 3 arquivos novos, 316 inserções. |

### Aceite oficial (Etapa 5 + Etapa 6 = 20%)

> "Stack completa no Compose e dashboard exibindo as métricas propostas, incluindo o comparativo baseline vs otimizado."

| Critério | Estado | Evidência |
|---|---|---|
| Stack completa no Compose | ✅ | `infra/docker-compose.yml` com 4 serviços em rede privada, `read_only: true`, `cap_drop: ALL`. |
| Dashboard exibindo as 4 métricas propostas | ✅ | Painéis `Requests by route/status`, `Latency p95` (com `model_variant`), `Prediction error rate`, `Baseline vs optimized (p95)`. |
| Comparativo baseline vs otimizado | ✅ | Painel "Baseline vs optimized (p95)" com tabela de p50/p95/p99 por `model_variant`; `model_variant=sklearn` e `model_variant=onnx` expostos. |
| Print e JSON em `reports/figures/` | ✅ | `reports/figures/triage_ml_dashboard.json` (3 084 B) e `reports/figures/triage_ml_dashboard.png` (100 843 B, layout 4 painéis) gerados por `scripts/render_observability_dashboard.py`. |

## Etapa 6 § Privacidade e segurança operacional — Bill — `CHECKLIST.md#L245-246`

| Item do checklist | Implementação | Evidência objetiva |
|---|---|---|
| Garantir que `text` nunca aparece em logs, payloads de erro ou labels de métrica (teste de fumaça). | `_PRIVACY_TEXT_FIXTURE = "PRIVACY-CANARY-CARDIOVASCULAR-RESPIRATORY 2025 with severe stenosis and arrhythmia"` é procurado em três superfícies: payload Prometheus, body de `/predict` (sucesso e erro), logs capturados. | `tests/test_observability_privacy.py::test_rendered_metrics_never_contain_input_text`, `test_predict_response_never_contains_text`, `test_logs_never_contain_input_text`. |
| Documentar a política de não retenção do `text` após a resposta. | `README.md` linhas 151-154 ("Política de privacidade: o `text` é classificado e descartado, nunca persistido, nunca copiado para log, nunca copiado para label de métrica, nunca retornado em erro. As labels Prometheus permitidas são `route`, `method`, `status`, `model_variant`, `error_code`"). `.agents/contracts/README.md` linhas 44-46 ("`text` é classificado e descartado. Não persiste, não vaza em log, não vira label, não aparece em payload de erro"). | Texto publicado em ambos os READMEs. |

### Sub-itens da "Implementação 2026-09-08" (todos cumpridos)

| Sub-item | Status | Evidência |
|---|---|---|
| `src/triage_ml/observability/metrics.py` com `REQUESTS_TOTAL`/`REQUEST_LATENCY_SECONDS`/`PREDICTION_ERRORS_TOTAL` em registry privado | ✅ | Implementado; degrade gracioso via `try/except ImportError`. |
| Middleware Prometheus route-aware | ✅ | `PrometheusMiddleware` mapeia `path → /predict|/reload|/health|/model-info|/models|/metrics|/"` via templates. |
| API oficial com `add_middleware`, `GET /metrics`, `TRIAGE_ML_MODEL_VARIANT`, ONNX adapter | ✅ | Todos ativos em `src/triage_ml/api/app.py` (vide relatório `Etapa_6_Observabilidade_Prometheus_Grafana.md`). |
| HTTP exceptions gravam `request.state.error_code` (allow-list público) | ✅ | `_METRIC_ERROR_CODES = ALLOWED_ERROR_CODES \| LANGUAGE_ERROR_CODES \| {"request_failed"}` (consolidado no 2º ciclo). |
| Compose overlay com Prometheus e Grafana em rede privada | ✅ | `infra/docker-compose.yml` validado em 2026-09-08. |
| Provisionamento automático do dashboard via YAML | ✅ | `monitoring/grafana/provisioning/dashboards.yml` + `datasource.yml`. |
| `Dockerfile` com `target=runtime-observability` | ✅ | Alvo adiciona `[observability,optimization]`. |
| `scripts/generate_observability_traffic.py` (carga benigna) | ✅ | Gerador validado em execução local. |
| 13 novos testes | ✅ | 262 passed na suíte completa. |

## Ajustes feitos antes do aceite final

Durante esta análise identifiquei que o item **"Salvar print e JSON do dashboard em `reports/figures/`"** do checklist Etapa 6 § Instrumentação (linha 241) tinha implementação canônica no `monitoring/grafana/dashboards/triage_ml.json` mas **faltavam as cópias versionadas** em `reports/figures/`. A entrega foi completada neste ciclo:

| Arquivo | Tamanho | Conteúdo |
|---|---:|---|
| `reports/figures/triage_ml_dashboard.json` | 3 084 B | Cópia verbatim do dashboard Grafana (4 painéis, 6 queries PromQL, `uid=triage-ml-observability`). |
| `reports/figures/triage_ml_dashboard.png` | 100 843 B | Preview matplotlib do layout 4 painéis (`Requests by route/status`, `Latency p95`, `Prediction error rate`, `Baseline vs optimized (p95)`) com queries PromQL reais contidas em cada painel via `_wrap_promql`. |
| `scripts/render_observability_dashboard.py` | 196 linhas | CLI leve (matplotlib + stdlib) que materializa ambos os artefatos a partir do JSON canônico, sem dependências externas. |

Comandos para reproduzir:

```bash
cd pos-ml-eng-tech-challenge-fase-03
uv run python scripts/render_observability_dashboard.py
ls -la reports/figures/triage_ml_dashboard.*
```

## Validação final

| Verificação | Resultado |
|---|---|
| `uv run ruff format --check .` | aprovado |
| `uv run ruff check .` | aprovado (All checks passed!) |
| `uv run pytest tests/` | **262 passed**, 9 skipped, 0 failed |
| `uv run python scripts/render_observability_dashboard.py` | gera `reports/figures/triage_ml_dashboard.{json,png}` |
| Inspeção do PNG | layout 4 painéis fiel ao dashboard Grafana, queries dentro de cada painel |
| Política de privacidade nos READMEs | documentada em `README.md` (linhas 151-154) e `.agents/contracts/README.md` (linhas 44-46, 50) |
| Testes de privacidade | `tests/test_observability_privacy.py` (canário `PRIVACY-CANARY-CARDIOVASCULAR-RESPIRATORY...` ausente de logs, body e métricas) |

## Conclusão

As três seções do checklist analisadas estão **100% cumpridas** após o commit `4e77861`, que materializou o último item físico pendente (artefatos do dashboard em `reports/figures/`). A Etapa 5 e a Etapa 6 estão prontas para fechar o **20% oficial** desta fase.
