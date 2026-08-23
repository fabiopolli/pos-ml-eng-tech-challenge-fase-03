# Relatório de implementação — Fase 1 (baseline do classificador)

| Campo | Valor |
|---|---|
| Integrante | Will (Bill) |
| Etapa do checklist | Etapa 2 (baseline) — `docs/CHECKLIST.md` reordenado em 2026-08-23 |
| Período desta entrega | 2026-08-23 (uma única sessão de trabalho) |
| Última revisão | commit `d1cbe3f` em `origin/main` + checagem de idioma na `/predict` + dashboard de smoke |
| Status | ✅ Baseline pronto, otimização e observabilidade ficam para a Fase 2 |

Este relatório cobre a Fase 1 do classificador de texto do Tech Challenge — Fase 3. O objetivo da Fase 1 é entregar um modelo NLP funcional, serializado segundo contrato e exposto por uma API de smoke que outros integrantes (Romário, Denis, Fábio) possam consumir sem stubs. As decisões foram registradas em [`docs/plans/PLAN-text-classifier.md`](../plans/PLAN-text-classifier.md) e no `PLAN-text-classifier.md` plus revisão do Codex em 2026-08-23.

## 1. Resumo executivo

- Modelo baseline: TF-IDF + **LinearSVC**, escolhido por macro-F1 em validação cruzada estratificada de 5 folds somente no treino (LinearSVC `0.7335` vs LogisticRegression `0.7319`). O `test set` permaneceu vedado até a avaliação final.
- Métricas finais no split de teste (1000 amostras): **accuracy `0.7460`**, **balanced accuracy `0.7221`**, **macro-F1 `0.7296`**, **weighted-F1 `0.7438`**.
- Pipeline serializado em diretório imutável `models/20260823T135811Z-bed2194376bc/` com `model.joblib`, `classes.json` e um manifesto `metadata.json` validado por `schema_version: 1` (checksum SHA-256, fingerprints, label mapping, métricas, dependências e seleção).
- API de smoke FastAPI expõe `GET /health` e `POST /predict` com `latency_ms`, `request_id` interno, `X-Request-ID` e `Server-Timing: predict;dur=<ms>` já alinhados à Etapa 6 (Prometheus/Grafana).
- Suíte de testes cobre pipeline, serialização, integridade do artefato, validação de metadata, fluxo end-to-end de treino, contrato HTTP da API, política de idioma e helpers do dashboard: **50 testes verdes** em `uv run pytest`.
- Lint e formatação verdes (`ruff check .` / `ruff format .`).

## 2. Escopo e alinhamento com o plano

A Fase 1 correspondeu às tarefas `F1.T1` a `F1.T6` de [`docs/plans/PLAN-text-classifier.md`](../plans/PLAN-text-classifier.md), com a Etapa 2 do checklist marcando os itens abaixo como concluídos:

- [x] Baseline TF-IDF + classificador Scikit-Learn selecionado sem usar o test set.
- [x] Seeds, preprocessing, fingerprints e versões fixas.
- [x] Métricas por classe e agregadas, com figuras em `reports/figures/`.
- [x] Modelo e metadados serializados segundo contrato e validados por checksum.
- [x] API de smoke local (`/health` + `/predict`) consumindo o artefato, com erros sanitizados, `latency_ms`, `request_id` e headers.

A Fase 2 (otimização ONNX, instrumentação Prometheus, Compose com API+Prometheus+Grafana, dashboard e teste de privacidade) **não está no escopo desta entrega** e foi explicitamente diferida.

### 2.1 Mudanças objetivas após a revisão do Codex (2026-08-23)

| Tema | Antes da revisão | Depois da revisão |
|---|---|---|
| Versão do artefato | `v1`, sobrescrevível | `YYYYMMDDTHHMMSSZ-<input_hash>`, imutável |
| Manifesto | chaves livres, sem validação de schema | `schema_version: 1` + validação rigorosa (tipos, ranges, regex de SHA-256, dependências, git_commit) |
| Seleção entre LR e LinearSVC | um classificador fixo via `--classifier` | CV 5-fold no treino; vencedor pelo maior mean macro-F1 |
| Carregamento do `joblib` | sem checagem | `load_artifact` valida manifesto + checksum + `model.classes_ == metadata.classes` antes e depois da desserialização |
| Startup do `app.py` | `/health` aceitava `status=degraded` | RuntimeError aborta o startup se o artefato não estiver íntegro |
| `request_id` no header | ecoava `X-Request-ID` do cliente | sempre gerado internamente (cliente não controla o valor) |
| Cabeçalho de timing | apenas `latency_ms` no body | `Server-Timing: predict;dur=<latency_ms>` |
| `label_mapping` em runtime | lido de `data/medical_tc_labels.csv` | lido do `metadata.json` (sem CSV em runtime) |
| `label_mapping` em runtime | — | removido: nomes das classes vêm só do manifesto |

> O artefato legado em `models/v1/` não passa mais em `validate_metadata` e não é mais carregável. Todo novo treino cria uma versão nova.

## 3. Pipeline e seleção

### 3.1 Vetorização

`TfidfVectorizer` com os seguintes parâmetros, definidos em `configs/training.yaml` e consumidos por `triage_ml.models.pipeline.build_pipeline`:

```yaml
tfidf:
  ngram_range: [1, 2]
  min_df: 2
  max_df: 0.95
  sublinear_tf: true
  lowercase: true
  token_pattern: "(?u)\\b\\w+\\b"
```

### 3.2 Candidatos e seleção por CV 5-fold no treino

`compare_classifiers` em `src/triage_ml/models/train.py` executa `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` apenas sobre `train_df` e compara os dois candidatos com `scoring="f1_macro"`. O `test set` permanece intocado.

| Candidato | Mean macro-F1 (CV 5-fold) | Std | Selecionado? |
|---|---|---|---|
| `LogisticRegression` (`class_weight=balanced`, `solver=lbfgs`, `max_iter=2000`) | **0.7319** | 0.0127 | Não |
| `LinearSVC` (`class_weight=balanced`) | **0.7335** | 0.0134 | **Sim** (vencedor por `mean_macro_f1`) |

A política é registrada em `metadata.selection.selection_policy = "highest_mean_macro_f1"` e `metadata.selection.test_set_used_for_selection = false`. A diferença é pequena (`0.0016`), o que está coerente com o que a literatura reporta em TF-IDF + SVM linear para texto curto.

### 3.3 Métricas finais no split de teste

`compute_metrics` em `train.py` usa `f1_score(..., average="macro")` e `balanced_accuracy_score`:

| Métrica agregada | Valor |
|---|---|
| accuracy | 0.7460 |
| balanced_accuracy | 0.7221 |
| macro-F1 | 0.7296 |
| weighted-F1 | 0.7438 |

Métricas por classe (`metadata.metrics.per_class`):

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| 1 (neoplasms) | 0.843 | 0.873 | 0.858 | 252 |
| 2 (digestive system diseases) | 0.747 | 0.714 | 0.730 | 91 |
| 3 (nervous system diseases) | 0.660 | 0.516 | 0.579 | 124 |
| 4 (cardiovascular diseases) | 0.807 | 0.817 | 0.812 | 230 |
| 5 (general pathological conditions) | 0.649 | 0.690 | 0.669 | 303 |

Figura [`08_confusion_matrix_linear_svc.png`](../../reports/figures/08_confusion_matrix_linear_svc.png) registra a matriz de confusão do vencedor. A classe 3 (nervous system) é a mais fraca — observação alinhada com a literatura (classe minoritária e vocabulário mais heterogêneo). Top features por classe estão em [`08_top_features_linear_svc.png`](../../reports/figures/08_top_features_linear_svc.png).

## 4. Serialização e contrato do artefato

### 4.1 Layout imutável

```
models/
└── 20260823T135811Z-bed2194376bc/
    ├── model.joblib              # pipeline scikit-learn (TfidfVectorizer + LinearSVC)
    ├── classes.json              # ordem idêntica a pipeline.classes_
    ├── metadata.json             # manifesto canônico
    └── summary.json              # saída completa do run_training
```

A versão é derivada de `datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")` e dos primeiros 12 caracteres do SHA-256 da concatenação dos fingerprints do dataset preparado e da config. `ArtifactPaths.ensure` chama `mkdir(exist_ok=False)`, recusando sobrescrita.

### 4.2 `metadata.json` (schema v1)

Topo (resumo):

```json
{
  "schema_version": 1,
  "model_version": "20260823T135811Z-bed2194376bc",
  "model_name": "triage_ml_tfidf_linear_svc",
  "task_type": "multiclass_text_classification",
  "language": "en",
  "classes": [1, 2, 3, 4, 5],
  "label_mapping": {
    "1": "neoplasms",
    "2": "digestive system diseases",
    "3": "nervous system diseases",
    "4": "cardiovascular diseases",
    "5": "general pathological conditions"
  },
  "random_state": 42,
  "n_train": 4000,
  "n_test": 1000,
  "selection": {
    "metric": "macro_f1",
    "cv": "StratifiedKFold",
    "folds": 5,
    "candidates": { "logreg": {...}, "linear_svc": {...} },
    "best_classifier": "linear_svc",
    "selected_classifier": "linear_svc",
    "selection_policy": "highest_mean_macro_f1",
    "test_set_used_for_selection": false
  },
  "metrics": { "accuracy": 0.746, "balanced_accuracy": 0.7221, ... },
  "preprocessing": { "vectorizer": "tfidf", "tfidf": {...}, "classifier": "linear_svc", "classifier_params": {...} },
  "dependency_versions": {
    "python": "3.12.13",
    "numpy": "...",
    "scipy": "...",
    "scikit_learn": "...",
    "joblib": "..."
  },
  "git_commit": "<SHA-40>",
  "git_dirty": false,
  "fingerprints": {
    "raw_csv_sha256": "...",
    "prepared_dataset_sha256": "...",
    "train_split_sha256": "...",
    "test_split_sha256": "...",
    "config_sha256": "..."
  },
  "checksum_sha256": "<SHA-256 do model.joblib>",
  "created_at": "2026-08-23T13:58:11Z"
}
```

`validate_metadata` em `artifact.py` aplica checagens estruturais: schema_version, formato da versão, inteiros únicos em `classes`, `label_mapping` cobre exatamente as classes, métricas em `[0, 1]`, `selection` consistente com `preprocessing`, dependências obrigatórias, git_commit como SHA-40 ou `"unknown"`, e regex de SHA-256 para cada fingerprint e para o checksum.

### 4.3 Carregamento seguro (`load_artifact`)

`load_artifact(path)` em `triage_ml/models/artifact.py` aplica três camadas:

1. Valida o manifesto (`validate_metadata`) e exige `model.joblib` + `metadata.json` no diretório.
2. Verifica o `checksum_sha256` do `joblib` com `hmac.compare_digest` (sem timing attacks).
3. Desserializa o pipeline e exige `pipeline.classes_ == metadata.classes`. Se `classes.json` existir, exige também que ele combine com `metadata.classes`.

Falhas levantam `ArtifactIntegrityError` ou `ArtifactCompatibilityError`. A API aborta o startup com `RuntimeError` se isso falhar.

### 4.4 Por que não Random Forest (RF)

O enunciado cita TF-IDF + RF como exemplo. Em TF-IDF, RF explode o custo de inferência (centenas de árvores percorridas por documento) sem ganho consistente de F1 sobre modelos lineares em texto curto. A justificativa foi registrada em `README.md` seção "Modelo (Bill)" para a banca.

## 5. API de smoke

### 5.1 Comportamento

`src/triage_ml/api/app.py` expõe `GET /health` e `POST /predict`. Características:

- **`/health`**: retorna `HealthOut(status, model_version, model_loaded)`. Se o artefato não carregar, a aplicação **não sobe** (`RuntimeError` no `lifespan`).
- **`/predict`** (`POST {"text": "..."}`):
  - `text` é normalizado via `Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]`.
  - Inferência registrada em `latency_ms` com `time.perf_counter`.
  - `score` vem de `predict_proba` quando disponível — `LinearSVC` não expõe probabilidades calibradas, então `score` é `null` para o artefato selecionado (informado em `README.md`).
  - **Erros são sanitizados**: handler para `RequestValidationError` e `HTTPException` devolve `ErrorOut(request_id, error_code, message, detected_language?, detected_language_score?)` sem vazar texto clínico. Handler genérico para `Exception` captura qualquer caminho não tratado e devolve 500 também sanitizado.
- **Headers em toda resposta**: `X-Request-ID` (gerado internamente — o cliente nunca controla o valor) e `Server-Timing: detect;dur=<ms>, predict;dur=<ms>` (ou apenas `detect;dur=<ms>` quando a checagem de idioma interrompe o fluxo antes do pipeline).
- **Modelo carregado**: via `MODEL_PATH` (env) ou autodetecção da versão mais recente `YYYYMMDDTHHMMSSZ-*` em `models/`. Sem artefato, falha imediatamente.

### 5.1 Checagem de idioma (`langid` local)

A `/predict` rejeita preventivamente qualquer texto que não esteja no allow-list `{"en"}` antes de invocar o pipeline. A política vive em `configs/api.yaml` e é carregada por `triage_ml.api.config.get_api_config()` (LRU cache).

```yaml
api:
  supported_languages:
    - en
  min_text_chars_for_language_check: 20
  min_language_score: 0.0
```

`detect_language` em `triage_ml/api/language.py` aplica a política em três camadas:

1. **Comprimento mínimo** (`min_text_chars_for_language_check = 20`). Textos mais curtos são rejeitados com `error_code=text_too_short_for_language_check` (status 422) sem chamar o detector — `langid` é instável abaixo desse limite.
2. **Confiança mínima** (`min_language_score = 0.0` por default, opt-in para endurecer). `langid.classify` retorna `(iso_code, log_prob)` com `log_prob ≤ 0`; o normalizador `_normalise_score` aplica `math.exp` com saturação em `[-500, 0]`. Detecções abaixo do limiar viram `error_code=indeterminate_language`.
3. **Allow-list de idiomas**. Códigos fora de `{"en"}` viram `error_code=unsupported_language`.

`ErrorOut` ganhou dois campos opcionais — `detected_language` e `detected_language_score` — para que o cliente saiba por que o pedido foi rejeitado sem expor o `text`. O corpo nunca carrega o `text` original; o `Server-Timing` agora reporta `detect;dur=<ms>, predict;dur=<ms>` quando ambos os estágios rodam, ou apenas `detect;dur=<ms>` quando o detector interrompe o fluxo.

### 5.2 Evidência do smoke

Execução de `python scripts/smoke_api.py` produziu o relatório sanitizado `reports/evidence/api-smoke.json` com cinco predições, três cenários de política de idioma e um teste de validação 422. Saída resumida do último run:

| Caso | HTTP | `label` / `error_code` | `latency_ms` | Notas |
|---|---|---|---|---|
| Texto real (`"Tumor growth in the liver..."`) | 200 | `1` (neoplasms) | ~7 ms (cold) / < 2 ms (warm) | OK |
| Texto cardiovascular | 200 | `4` (cardiovascular) | < 2 ms | OK |
| Texto só com whitespace | 422 | `validation_failed` | — | sem vazamento |
| Campo `text` ausente | 422 | `validation_failed` | — | `request_id` propagado |
| Texto curto (`"liver tumor"`, 11 chars) | 422 | `text_too_short_for_language_check` | — | rejeitado antes do detector |
| Mock `("en", -1000.0)` (score saturado em 0.0) | 422 | `indeterminate_language` | — | abaixo de `min_score=0.5` |
| Mock `("pt", -0.1)` (score ≈ 0.905) | 422 | `unsupported_language` | — | acima do limiar mas fora do allow-list |

## 6. Testes automatizados

Cobertura por arquivo:

| Arquivo | Quantidade | Foco |
|---|---|---|
| `tests/test_model_pipeline.py` | 9 | `build_pipeline` (TF-IDF defaults, LR com `predict_proba`, LinearSVC com `class_weight=balanced`), end-to-end em corpus sintético |
| `tests/test_model_artifact.py` | 12 | `ArtifactPaths`, `file_sha256`, `write_classes` (com `_coerce` de `numpy.int64`), `read_classes`, `validate_metadata` (campos obrigatórios), `verify_artifact_integrity` (caso feliz, model swap, checksum ausente), `load_artifact` |
| `tests/test_model_training.py` | 1 | Integração `run_training + load_artifact` em dataset sintético, garantindo `selection.candidates = {logreg, linear_svc}`, `test_set_used_for_selection=False`, balanced_accuracy presente, `pipeline.classes_ == metadata.classes` |
| `tests/test_api_smoke.py` | 12 | Hermetismo via `create_app(holder=...)`, validação de schema em `/health`, `/predict` com `Server-Timing`, request_id interno não confiável, padding stripado, 422 parametrizado (string vazia, só whitespace, > 20 000 chars), `prediction_failed` sanitizado |
| `tests/test_api_language.py` | 10 | Política de idioma hermética: aceita inglês, rejeita texto curto, rejeita score baixo, rejeita idioma fora do allow-list, valida headers `Server-Timing`, garante que o `text` não vaza em logs nem na resposta de erro |
| `tests/test_dashboard_helpers.py` | 6 | Helpers HTTP do dashboard (`_check_health`, `_post_predict`, `ApiResponse._header`, presets da política de idioma, tratamento de body inválido e `RequestException`) — mocka `requests.request`, não precisa de API rodando |

Comando único:

```bash
uv run pytest   # 50 passed in ~3s
uv run ruff check .
uv run ruff format .
```

### 6.1 Dashboard de smoke (`front/app_smoke.py`)

Ferramenta opcional para o desenvolvedor exercitar `/health` e `/predict` manualmente sem `curl`. Stacklit em modo HTTP contra qualquer URL da API (default `http://127.0.0.1:8000`, configurável na sidebar). Três abas:

- **Health** — chama `GET /health` e mostra `status`, `model_version`, `model_loaded`.
- **Predição** — área de texto + `POST /predict` exibindo `label`, `label_name`, `score`, `latency_ms`, `request_id` e os headers `X-Request-ID` / `Server-Timing`.
- **Política de idioma** — quatro cenários canônicos da política configurada em `configs/api.yaml` (texto curto, score baixo, idioma fora do allow-list, inglês válido), com validação automática do `error_code` retornado pela API.

O dashboard **não substitui** Prometheus/Grafana (latência, taxa de erro e volume ficam no stack de observabilidade). Não persiste payloads nem textos. Validação manual local usando os próprios helpers contra a API rodando em `127.0.0.1:8765`:

```text
HEALTH 200 {'status': 'ok', 'model_version': '20260823T135811Z-bed2194376bc', 'model_loaded': True}
PRED OK 200 neoplasms timing= detect;dur=881.603, predict;dur=8.982
SHORT 422 text_too_short_for_language_check det= None
UNSUP 422 unsupported_language det= pt score= 1.5e-177
```

## 7. Riscos conhecidos e trabalho futuro

| Risco | Status | Plano |
|---|---|---|
| Otimização de latência (ONNX/quantização/pruning) | Pendente | Fase 2 — `F2.T1` a `F2.T6`. Critério: Δ macro-F1 ≤ 1 pp no split de teste E ≥ 20% de redução no p95 de inferência. |
| Instrumentação Prometheus + Compose + Grafana | Pendente | Fase 2 — `F2.T7` a `F2.T9`. `latency_ms` e `Server-Timing` já estão prontos para virar `Histogram`. |
| API oficial com Docker e auth | Pendente | Romário, Etapa 3 — herda o contrato desta smoke |
| DAG Airflow de retreino | Pendente | Denis, Etapa 7 — consome `triage_ml.models.train.run_training` |
| Classe 3 (nervous system) com F1 baixo | Aceito, documentado | Investigar balanceamento de classes e features mais ricas (n-gramas maiores, char n-gramas) em baseline da Fase 2 antes da otimização |

## 8. Como reproduzir a entrega

```bash
# 1. Instalar dependências
uv sync

# 2. Reproduzir treino (compara LR/LinearSVC e seleciona o vencedor)
PYTHONPATH=src uv run python -m triage_ml.models.train

# 3. Validar saúde da API (descobre o artefato mais recente)
PYTHONPATH=src uv run uvicorn triage_ml.api.app:app --host 127.0.0.1 --port 8000

# 4. Sanidade rápida
curl -s http://127.0.0.1:8000/health | jq
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Acute myocardial infarction in a 62yo after chest pain."}' | jq

# 5. Testes e lint
uv run pytest
uv run ruff check . && uv run ruff format .
```

## 9. Mapa dos artefatos

```
docs/
├── CHECKLIST.md                                      (marcador oficial da Etapa 2)
├── plans/PLAN-text-classifier.md                    (plano detalhado Fase 1 + Fase 2)
├── dataset.md                                        (decisão de dataset e licença)
└── reports/IMPLEMENTATION-REPORT-FASE-1.md          (este documento)

models/
└── 20260823T135811Z-bed2194376bc/
    ├── model.joblib
    ├── classes.json
    ├── metadata.json
    └── summary.json

reports/
├── figures/08_confusion_matrix_linear_svc.png
├── figures/08_top_features_linear_svc.png
└── evidence/api-smoke.json

src/triage_ml/
├── api/{app.py, schemas.py, language.py, config.py}  (FastAPI smoke + checagem de idioma)
├── models/{artifact.py, pipeline.py, train.py}      (treino + contrato do artefato)
├── data/{prepare.py}                                (dedup, sem leakage)
└── monitoring/                                       (reservado para Fase 2)

tests/
├── test_model_pipeline.py
├── test_model_artifact.py
├── test_model_training.py
├── test_api_smoke.py
├── test_api_language.py
└── test_dashboard_helpers.py

configs/
├── training.yaml                                    (hiperparâmetros + label_mapping)
└── api.yaml                                          (allow-list de idiomas + thresholds)
front/
├── app_smoke.py                                      (dashboard Streamlit de smoke manual)
└── README.md                                         (instruções e escopo)
```

## 10. Conclusão

A Fase 1 entrega o baseline do classificador e endurece o contrato do artefato a ponto de ser seguro oferecer para a Etapa 3 (API oficial) e a Etapa 7 (Airflow) sem rework. O pipeline TF-IDF + LinearSVC atinge `macro-F1 = 0.7296` no split de teste, com seleção honesta por CV 5-fold no treino. A API de smoke já entrega `latency_ms`, request_id interno e `Server-Timing`, removendo o atrito de integrar a Etapa 6 (Prometheus/Grafana) na Fase 2. O restante do trabalho (otimização, instrumentação, dashboard, privacidade operacional) está descrito em [`docs/plans/PLAN-text-classifier.md`](../plans/PLAN-text-classifier.md) e aguarda a próxima janela de trabalho.
