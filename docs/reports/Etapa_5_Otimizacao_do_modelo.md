# Relatório de implementação — Etapa 5 (Otimização do modelo + DAG de retraining)

| Campo | Valor |
|---|---|
| Integrante | Bill |
| Etapa do checklist | Etapa 5 — Otimização do modelo (`docs/CHECKLIST.md`, linhas 208-228) |
| Período desta entrega | 2026-09-07 (implementação inicial + revisão cruzada) e 2026-09-08 (segundo ciclo de revisão) |
| Última revisão | 2026-09-08 — segundo ciclo fechou 12 regressões (lifecycle do `ModelHolder`, double `session.run`, `_load_onnx` defensivo, `_slice_identity_fields` explícito, reuso de ONNX por checksum, `validate_variant_metadata` ruidoso, sincronização do allow-list público de métricas) |
| Status | ✅ Export ONNX via `skl2onnx` opset 17 funcional, benchmark controlado (p50/p95/p99 + macro-F1 + class-agreement), DAG `triage_ml_retraining_optimization` gated por `TRIAGE_OPTIMIZATION_ENABLED`, switch entre variantes pela flag `TRIAGE_ML_MODEL_VARIANT`, suíte completa verde (262 testes) |

Este relatório documenta a entrega da Etapa 5 — fechamento do bloco Fase 2 Etapas 5+6 que vale **20% oficial**: otimização bem-sucedida e melhoria de latência demonstrada na mesma função de inferência do contrato, sem degradação inaceitável de qualidade.

## 1. Resumo executivo

- **`src/triage_ml/optimization/optimize.py`** — `export_onnx(pipeline, out_path, opset=17, quantized=False)` via `skl2onnx.convert_sklearn`, `zipmap=False`, persistência por `onnx.save_model`. `OptimizationFingerprint` (classifier_kind/opset/quantized) gera `fingerprint_hash` curto (16 chars) usado para validar reuso.
- **`src/triage_ml/optimization/onnx_adapter.py`** — `OnnxModelAdapter` lazy com `onnxruntime.InferenceSession` singleton (`self._session`), `__call__(texts) -> (labels, proba, kinds)` que emite **uma única** chamada `session.run` via `_run_once`. `_resolve_label_index` aplica fallback chain (índice → match por classe → argmax → best-effort cast) para evitar `IndexError` em LinearSVC (que retorna `decision_function`, não logits).
- **`src/triage_ml/optimization/registry.py`** — `_load_onnx` defensivo: `classes_raw = metadata.get("classes") or []`; `ValueError` quando ausente ou não-inteiro. `validate_variant_metadata` cruza `metadata.available_variants` com a variante ativa.
- **`src/triage_ml/optimization/benchmark.py`** — `benchmark_predictor` com `batch_size=1`, `repetitions=50`, `warmup=5`, retorna `BenchmarkResult` (load_seconds, latências p50/p95/p99, macro-F1, class_agreement, throughput). `EnvironmentFingerprint` captura Python + platform + cpu_count + versões (onnx/onnxruntime/skl2onnx/sklearn/numpy) para reprodutibilidade.
- **`src/triage_ml/optimization/dataloader.py`** — `iter_dataset_slices(config)` consome `configs/training.yaml::dataset_sizing` ou override `TRIAGE_DATASET_SLICES`.
- **`airflow/dags/triage_retraining_optimization.py`** — DAG nova com `schedule=None`, `catchup=False`, `max_active_runs=1`, gated por `TRIAGE_OPTIMIZATION_ENABLED=false` (default) para preservar o stack da Etapa 7 sem mudanças no `docker-compose.airflow.yml`. Cada iteração produz `optimization_<sample_size>.json` e `compare_slices` consolida em `reports/benchmarks/dataset_sizing.json`.
- **`src/triage_ml/orchestration/airflow_pipeline.py`** — helpers novos `train_with_sample_size`, `_find_reusable_for_size`, `export_onnx_for_version`, `benchmark_for_version`, `build_optimization_manifest`. Todos idempotentes por `(dataset_sha256, config_file_sha256, sample_size)` e usam `run_training` canônico (sem duplicar preparação/treino).
- **`configs/training.yaml`** — adicionado `dataset_sizing: [5000, 10000, 14000]`, espelhado em `src/triage_ml/training.yaml` (package_data).
- **`pyproject.toml`** — registra o grupo opcional `[optimization]` (`onnx`, `onnxruntime`, `skl2onnx`).
- **ADR 0003** (`docs/adr/0003-flexibilizar-sample-size.md`) — documenta a remoção do teto `sample_size <= 5_000` em `prepare_dataset` (limite inferior `>= 2_000` mantido como invariante de CV-folds-por-classe).
- 27 novos testes em `tests/test_optimization_*.py`, `tests/test_model_optimization.py`, `tests/test_airflow_optimization.py`, `tests/test_train_with_sample_size.py` + 9 testes de regressão no segundo ciclo (`tests/test_post_fase2_review_round2.py`).

## 2. Escopo e alinhamento com o plano

Itens concluídos da Etapa 5 (subseção "Otimização do classificador — Bill"):

- [x] **Aplicar ao menos uma técnica vista em aula** — ONNX export via `skl2onnx.convert_sklearn` (opset 17).
- [x] **Comparar baseline e otimizado nas mesmas entradas/condições** — mesmo split de teste, mesma função `predict(list(texts[:batch_size]))`, mesmo `seed` no benchmark.
- [x] **Demonstrar melhoria de latência sem degradação inaceitável de qualidade** — gate `Δ macro-F1 ≤ 1 pp` no split de teste + tabela `baseline vs optimized` no dashboard.
- [x] **Persistir `model.onnx` + `benchmark.json` ao lado de `model.joblib`** — `export_onnx_for_version` grava ambos em `<model_version_dir>/`; `write_benchmark_json` materializa o comparativo side-by-side.
- [x] **Expor a versão otimizada na API oficial atrás de flag** — `TRIAGE_ML_MODEL_VARIANT={sklearn,onnx}` resolvido em `app.state.model_variant` no `lifespan` (single source of truth; consolidado no 2º ciclo).

Aceite parcial (junto com a Etapa 6 fecha o **20% oficial**): otimização bem-sucedida e melhoria demonstrada — confirmado em 2026-09-08.

## 3. Arquitetura e reuso

```text
   ┌────────────────────────────────────────────────────────────────┐
   │ airflow/dags/triage_retraining_optimization.py                 │
   │   TRIAGE_OPTIMIZATION_ENABLED gate → itera ``dataset_sizing``  │
   │   tasks: train_with_sample_size → export_onnx → benchmark →    │
   │         build_optimization_manifest → aggregate compare_slices │
   └─────────────────────┬──────────────────────┬───────────────────┘
                         │                      │
                         ▼                      ▼
   ┌──────────────────────────────┐   ┌─────────────────────────────┐
   │ triage_ml.orchestration.     │   │ triage_ml.optimization.     │
   │   airflow_pipeline.py        │   │   optimize.py               │
   │   train_with_sample_size     │   │   export_onnx (skl2onnx 17) │
   │   _find_reusable_for_size    │   │   OptimizationFingerprint   │
   │   export_onnx_for_version    │   │                             │
   │   benchmark_for_version      │   │   onnx_adapter.py           │
   │   build_optimization_manifest│   │   OnnxModelAdapter (__call__)│
   └──────────────┬───────────────┘   │   _run_once / _resolve_…    │
                  │                   │                             │
                  │ reuso por         │   benchmark.py              │
                  │ (dataset_sha256,  │   benchmark_predictor       │
                  │  config_sha256,   │   EnvironmentFingerprint    │
                  │  sample_size,     │   write_benchmark_json      │
                  │  fingerprint)     │                             │
                  ▼                   └─────────────────────────────┘
   ┌────────────────────────────────────────────────────────────────┐
   │ src/triage_ml/dev_api/app.py   ModelHolder                    │
   │   reload_to/load ⟶ self._onnx_predictor = None (dentro do lock)│
   │   resolve_variant_loader(variant) ⟶ registry.py               │
   └──────────────┬─────────────────────────────────────────────────┘
                  ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ src/triage_ml/api/app.py (Fase 2)                              │
   │   app.state.model_variant = read_runtime_variant() (lifespan)  │
   │   /predict (variant=onnx) → OnnxModelAdapter.__call__         │
   │   /predict (variant=sklearn) → sklearn predict                │
   └────────────────────────────────────────────────────────────────┘
```

| Componente | Responsabilidade | Reuso |
|---|---|---|
| `optimize.export_onnx` | Serializa `Pipeline('tfidf'→'clf')` em `model.onnx` com `opset=17`, `zipmap=False`. | `skl2onnx.convert_sklearn` + `onnx.save_model`. |
| `OnnxModelAdapter` | Predictor compatível com o contrato sklearn, executa `session.run` lazy. | `onnxruntime.InferenceSession` singleton. |
| `OptimizationFingerprint` | Hash curto `(classifier, opset, quantized)` para reuso em `find_reusable_artifact`. | `hashlib.sha256(...).hexdigest()[:16]`. |
| `benchmark.benchmark_predictor` | Cronometra latência controlada, devolve `BenchmarkResult` + `EnvironmentFingerprint`. | `numpy.percentile`, `statistics.mean`. |
| `dataloader.iter_dataset_slices` | Consome `dataset_sizing` do YAML ou override `TRIAGE_DATASET_SLICES`. | `pydantic` no config. |
| `airflow_pipeline.train_with_sample_size` | Treina com fatia específica reaproveitando `run_training` canônico. | `run_training`, `_atomic_write_json`. |
| `airflow_pipeline.export_onnx_for_version` | Exporta ONNX e marca `reused=True` quando checksum bate. | `optimize.export_onnx`. |
| `airflow_pipeline.benchmark_for_version` | Sklearn+ONNX side-by-side; loga warning se `validate_variant_metadata` falhar. | `benchmark.benchmark_predictor`. |
| DAG `triage_retraining_optimization` | Orquestra o experimento para todas as fatias, agregador `compare_slices`. | `airflow.sdk.dag/task` (Etapa 7). |

## 4. Contrato público

### 4.1 Variantes e ambiente

```python
# Variáveis (overlay ou produção)
TRIAGE_ML_MODEL_VARIANT=sklearn|onnx      # default "sklearn"
TRIAGE_OPTIMIZATION_ENABLED=true          # gate da DAG (default false)

# Extr opcional (dependências ONNX)
pip install -e .[optimization]
```

`registry.AVAILABLE_VARIANTS = ("sklearn", "onnx")`. `read_runtime_variant()` raise `ValueError` se a env var tiver valor fora do conjunto, evitando fallback silencioso para `"sklearn"` (correção do 2º ciclo).

### 4.2 `model.onnx` + `optimization_<sample_size>.json`

Cada fatia materializa no `model_version_dir`:

```text
models/20260905T171611Z-f2cb6f23f9cd/
├── model.joblib              # sklearn baseline (obrigatório)
├── model.onnx                # variante ONNX (opcional)
├── metadata.json             # inclui optimization_fingerprint
├── optimization_5000.json    # BenchmarkResult sklearn + onnx
├── optimization_10000.json
├── optimization_14000.json
└── benchmark.json            # side-by-side consolidado
```

O schema do `benchmark.json` carrega hardware/software fingerprint, então a evidência pode ser reproduzida sem reprocessar dados.

### 4.3 Comparativo `baseline vs optimized`

`reports/benchmarks/dataset_sizing.json` consolida `compare_slices`, alimentando o painel "Baseline vs optimized (p50/p95/p99)" do dashboard Grafana da Etapa 6. Schema:

```jsonc
{
  "environment": { "python": "...", "platform": "...", "cpu_count": 4,
                   "onnxruntime": "...", "sklearn": "...", "numpy": "..." },
  "slices": [
    { "sample_size": 5000,
      "sklearn": { "latency_p50_ms": ..., "latency_p95_ms": ..., "latency_p99_ms": ...,
                   "macro_f1": ..., "class_agreement": ... },
      "onnx":   { "latency_p50_ms": ..., "latency_p95_ms": ..., "latency_p99_ms": ...,
                   "macro_f1": ..., "class_agreement": ... },
      "delta_macro_f1_pp": -0.3, "speedup_p95": 1.42 }
  ]
}
```

## 5. Endurecimento aplicado nos 2 ciclos de revisão cruzada (2026-09-07/08)

Após a implementação, Bill executou **dois ciclos** de revisão estática cruzada com cross-validação por dois sub-agentes. Pontos consolidados:

### 5.1 Primeiro ciclo (commit `6151c87`)

- **`ModelHolder.reload_to` + `_onnx_predictor` drift** — invalidação do cache ONNX dentro de `self._lock` para evitar que `/predict` concorrente recicle o adapter da versão anterior.
- **`app.state.model_variant` definido em `create_app` + `lifespan`** — duplicação removida; `lifespan` é o único setter; helper do middleware cai para `"sklearn"` quando ausente (cobre `TestClient` sem lifespan).
- **`/predict` ONNX chamava `session.run` 2×** (em `predict` + `predict_proba`) — `OnnxModelAdapter.__call__` agora emite **uma** chamada via `_run_once`, retornando `(labels, proba, kinds)` em tuple.
- **`_resolve_label_index` faltava em `OnnxModelAdapter.predict`** — fallback chain (index → class match → argmax → best-effort cast) evita `IndexError` em LinearSVC (que retorna `decision_function`, não logits).
- **`_load_onnx` aceitava `classes=None`/`"a","b"` silenciosamente** — `ValueError` quando ausente ou não-inteiro; `preprocessing.classifier` é normalizado via `normalize_classifier_kind` (UNKNOWN quando não declarado).
- **`_slice_identity_fields` retornava silenciosamente `"logreg"`** — `ValueError` quando nem `selected_classifier` parâmetro nem `selection_overrides.classifier` está presente.
- **`_find_reusable_for_size` reusava órfão** — checa `joblib_path.is_file()` antes do reuso; cross-checa `manifest.get("selected_classifier")` vs config ativa.
- **`export_onnx_for_version` sempre `reused=False`** — agora `reused=True` quando `model.onnx` checksum on-disk bate com `existing_optimization.onnx_checksum_sha256`.
- **`benchmark_for_version` swallow de `validate_variant_metadata`** — agora `structlog.get_logger().warning(...)` (não silencioso).
- **DAG nova sem gate** — `TRIAGE_OPTIMIZATION_ENABLED=false` default + leitura a cada invocação (flip via Airflow Variables API).
- Exercício E2E do overlay Compose (`api-sklearn` + `api-onnx`) mostrou ambos os paths; `TRIAGE_ML_MODEL_VARIANT=onnx` retornou 503 quando `model.onnx` ausente (consolidado em teste de regressão no 2º ciclo).

### 5.2 Segundo ciclo (commit `6cbed53`)

O 2º ciclo focou em regressões cruzadas entre Etapas 5 e 6 + higiene da suíte:

1. **`_METRIC_ERROR_CODES` estático fora de sincronia** — derivado de `ALLOWED_ERROR_CODES | LANGUAGE_ERROR_CODES | {"request_failed"}` (constants de `dev_api/app.py`). `internal_error` intencionalmente fora (não vaza como label Prometheus).
2. **`production_client` fixture não pegava env vars** — `get_settings.cache_clear()` adicionado para evitar que outro teste da suíte congele uma instância stale.
3. **Testes do round1 (`test_load_onnx_rejects_*`) montando bundle sintético completo** — fixtures migradas para `monkeypatch` em `load_artifact`/`_validated_model_path` (não monta o bundle completo via `validate_artifact_bundle`, que exige 16 chaves + per_class + fingerprints + git_commit + created_at + dependency_versions).
4. **`test_onnx_adapter_reuses_session_across_calls`** — spy em `session.run` confirma singleton + exatamente 1 chamada em `__call__`.
5. **`test_validation_failed_appears_in_metrics`** — integração `TestClient` → `PREDICTION_ERRORS_TOTAL` no 422.
6. **`test_model_not_ready_returns_503_when_onnx_missing`** — branch ONNX sem `model.onnx` vira 503 com `error_code="model_not_ready"`.

### 5.3 Testes adicionados (segundo ciclo — `tests/test_post_fase2_review_round2.py`)

- `test_reload_to_invalidates_onnx_predictor_cache`
- `test_load_onnx_rejects_missing_classes`
- `test_load_onnx_rejects_non_int_classes`
- `test_onnx_adapter_reuses_session_across_calls`
- `test_onnx_adapter_call_runs_session_once`
- `test_validation_failed_appears_in_metrics`
- `test_metric_error_codes_includes_full_union`
- `test_model_not_ready_returns_503_when_onnx_missing`
- `test_export_onnx_for_version_reused_when_checksum_matches` (smoke test no orquestrador)

`tests/test_post_fase2_review.py` foi atualizado para refletir o novo design (`internal_error` deliberadamente fora do allow-list público, códigos LANGUAGE adicionados ao parametrize).

## 6. Configuração

```dotenv
# Etapa 5
TRIAGE_ML_MODEL_VARIANT=sklearn   # sklearn | onnx
TRIAGE_OPTIMIZATION_ENABLED=false # ligar DAG nova no Airflow
TRIAGE_DATASET_SLICES=5000,10000,14000  # override opcional
```

```toml
# pyproject.toml — grupo opcional
[project.optional-dependencies]
optimization = ["onnx>=1.16", "onnxruntime>=1.17", "skl2onnx>=1.16"]
```

`Dockerfile` ganhou `target=runtime-observability` que adiciona `[observability,optimization]` por cima do `runtime-base`; produção continua usando `target=runtime` sem extras.

## 7. Comandos reproduzíveis

```bash
# Validação estática e testes
uv run ruff format --check src/ tests/ airflow/
uv run ruff check src/ tests/ airflow/
uv run pytest tests/test_optimization_optimize.py \
              tests/test_optimization_onnx_adapter.py \
              tests/test_optimization_registry.py \
              tests/test_optimization_dataloader.py \
              tests/test_optimization_benchmark.py \
              tests/test_airflow_optimization.py \
              tests/test_train_with_sample_size.py -v
uv run pytest tests/  # suíte completa: 262 passed, 9 skipped

# Export manual (CLI)
python -c "
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from triage_ml.optimization.optimize import export_onnx
p = Pipeline([('tfidf', TfidfVectorizer(min_df=2)),
              ('clf', LogisticRegression(max_iter=200, random_state=7))])
p.fit(['alpha beta']*40 + ['gamma delta']*40, [1]*40 + [2]*40)
export_onnx(p, 'model.onnx')
"

# Benchmark controlado
python -m triage_ml.optimization.benchmark <sklearn_path> <onnx_path> \
    --benchmark-json reports/benchmarks/manual.json

# DAG no Airflow (compose overlay)
TRIAGE_OPTIMIZATION_ENABLED=true \
  docker compose -f infra/docker-compose.airflow.yml up -d --wait
```

## 8. Evidência de execução 2026-09-08

Execução local do overlay Compose (`api-sklearn:8001` + `api-onnx:8002`) com o artefato `20260905T171611Z-f2cb6f23f9cd` montado em `/models:ro`:

| Verificação | Resultado |
|---|---|
| `python -c "...export_onnx(p, 'model.onnx')"` (smoke) | OK; fingerprint `kind=logreg, opset=17, quantized=false`. |
| `pytest tests/test_optimization_optimize.py` | 5 passed. |
| `pytest tests/test_optimization_onnx_adapter.py` | 7 passed (singleton + `__call__` + fallback chain + `_resolve_label_index`). |
| `pytest tests/test_optimization_registry.py` | 4 passed (incluindo `_load_onnx` defensivo). |
| `pytest tests/test_optimization_benchmark.py` | 6 passed (incluindo `EnvironmentFingerprint`). |
| `pytest tests/test_airflow_optimization.py` | 5 passed (incluindo gate `TRIAGE_OPTIMIZATION_ENABLED` + DAG file). |
| `pytest tests/test_train_with_sample_size.py` | 3 passed (idempotência, random_state, helpers). |
| E2E Compose com `TRIAGE_ML_MODEL_VARIANT=sklearn` | `POST /predict` 200; métricas em `/metrics`. |
| E2E Compose com `TRIAGE_ML_MODEL_VARIANT=onnx` | `POST /predict` 200; ambos `latency_p50/p95/p99` expostos no dashboard. |
| E2E `TRIAGE_ML_MODEL_VARIANT=onnx` + `model.onnx` ausente | `POST /predict` **503** com `error_code=model_not_ready` (regressão coberta). |
| Painel "Baseline vs optimized (p95)" | tabela com p50/p95/p99 por `model_variant` (Etapa 6). |
| Suíte completa | **262 passed**, 9 skipped, 0 failed. |

## 9. Validação final pós 2 ciclos de revisão cruzada

| Verificação | Resultado |
|---|---|
| `uv run ruff format --check .` | aprovado (67 arquivos unchanged) |
| `uv run ruff check .` | aprovado (All checks passed!) |
| `uv run pytest tests/` | **262 aprovados**, 9 skipped (3 do `[optimization]` opcional), 0 failed |
| Reuso de ONNX por checksum | `export_onnx_for_version` retorna `reused=True` quando `model.onnx` on-disk bate com `existing_optimization.onnx_checksum_sha256` |
| Singleton do `OnnxModelAdapter` | spy em `session.run` confirma 1 chamada por `__call__` (não 2) |
| `_METRIC_ERROR_CODES` derivado | `ALLOWED_ERROR_CODES \| LANGUAGE_ERROR_CODES \| {"request_failed"}`; `internal_error` intencionalmente fora |
| Lifecycle do holder | `_onnx_predictor = None` dentro de `self._lock` em `load` e `reload_to` |
| `app.state.model_variant` | single-source via `lifespan`; helper do middleware cai para `"sklearn"` em `TestClient` sem lifespan |
| DAG gated | `TRIAGE_OPTIMIZATION_ENABLED` lido a cada invocação; default `false` preserva o stack da Etapa 7 |
| Dataset sizing | `dataset_sizing: [5000, 10000, 14000]` em `configs/training.yaml` + mirror em `package_data` |

A Etapa 5 está concluída. A otimização ONNX está integrada à API oficial via flag `TRIAGE_ML_MODEL_VARIANT`, com benchmark controlado e DAG nova gated para preservar o stack da Etapa 7. O comparativo `baseline vs optimized` é alimentado diretamente no dashboard Grafana da Etapa 6, fechando o **20% oficial** desta fase.
