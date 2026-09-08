# Guia de treinamento de modelos

Este guia cobre o ciclo completo de treinamento: preparação do dataset, comparação de candidatos, serialização do artefato versionado, integração com a API e retreino orquestrado.

## Visão geral

```text
   ┌─────────────────────────┐
   │ dataset (CSV/parquet)   │
   └─────────┬───────────────┘
             │
             ▼
   ┌─────────────────────────┐
   │ prepare.py (Denis/Et.1) │ → PreparationReport
   └─────────┬───────────────┘
             │
             ▼
   ┌────────────────────────────────────────────┐
   │ train.py — CV estratificada (5-fold)       │
   │   candidatos: logreg × linear_svc          │
   │   métrica: macro-F1                        │
   │   seleção: melhor média em CV              │
   └─────────┬──────────────────────────────────┘
             │
             ▼
   ┌────────────────────────────────────────────┐
   │ serialização imutável                      │
   │ models/YYYYMMDDTHHMMSSZ-<12hex>/          │
   │   model.joblib, classes.json,              │
   │   metadata.json, summary.json              │
   └─────────┬──────────────────────────────────┘
             │
             ▼
   ┌────────────────────────────────────────────┐
   │ (opcional Fase 2) optimize.export_onnx     │
   │   model.onnx ao lado de model.joblib       │
   └─────────┬──────────────────────────────────┘
             │
             ▼
   ┌────────────────────────────────────────────┐
   │ API oficial: api-prod / api-onnx           │
   │   MODEL_PATH=/models/<versão>/model.joblib │
   └────────────────────────────────────────────┘
```

## Pré-requisitos

- Python 3.12 com `uv` (`pip install uv`).
- Dependências: `uv sync --dev`.
- Para a Fase 2 (export ONNX): `uv sync --dev --extra optimization`.
- Dataset preparado em `data/processed/medical_tc_train.parquet` (Etapa 1).

## Passo 1 — Preparar o dataset (uma vez)

Se você ainda não rodou a Etapa 1:

```bash
# 1. Baixar dataset bruto (Medical Abstracts TC Corpus, CC BY-SA 3.0)
#    URL: https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus
#    Coloque em data/raw/medical_tc_train.csv

# 2. Rodar a preparação canônica
uv run python -m triage_ml.data.prepare \
  --input data/raw/medical_tc_train.csv \
  --output data/processed/medical_tc_train.parquet \
  --report reports/preparation.json

# Resultado: PreparationReport com contagens verificáveis
cat reports/preparation.json
```

Resultado esperado em uma execução típica:

```json
{
  "input_rows": 11550,
  "after_language_filter": 7489,
  "after_dedup": 5000,
  "sample_size": 5000,
  "n_train": 4000,
  "n_test": 1000,
  "fingerprint_sha256": "..."
}
```

Para detalhes do contrato de preparação, leia [Etapa_1_Fundacao_dados_e_contratos.md](../reports/Etapa_1_Fundacao_dados_e_contratos.md) e [ADR 0001](../adr/0001-escolha-recorte-dataset.md).

## Passo 2 — Treinar (CLI canônico)

A CLI `triage-ml-train` faz a seleção por CV e gera uma versão imutável:

```bash
# Padrão: comparar LR × LinearSVC no treino, selecionar por macro-F1
uv run triage-ml-train

# Override explícito de candidato
uv run triage-ml-train --classifier logreg
uv run triage-ml-train --classifier linear_svc

# Forçar um sample_size fora do default (limite mínimo 2 000 — vide ADR 0003)
uv run triage-ml-train --sample-size 8000

# Forçar random_state e versionamento custom
uv run triage-ml-train --random-state 7 --input-hash-suffix "experiment-A"

# Ajuda completa
uv run triage-ml-train --help
```

### O que a CLI faz

1. Carrega `data/processed/medical_tc_train.parquet` (ou outro dataset configurado em `configs/training.yaml`).
2. Aplica split 80/20 estratificado com `random_state=42`.
3. Executa `GridSearchCV` ou `cross_val_score` (5-fold, `scoring="f1_macro"`) sobre o **treino apenas** para cada candidato em `configs/training.yaml::selection.candidates`.
4. Seleciona o vencedor por maior média de macro-F1.
5. Re-treina o vencedor no treino inteiro.
6. Avalia no split de teste (accuracy, balanced_accuracy, macro-F1, weighted-F1, matriz de confusão, top features).
7. Serializa em `models/YYYYMMDDTHHMMSSZ-<12hex>/`.
8. Grava `reports/figures/<model_version>/` com confusion matrix e top features.

### Exemplo de saída

```text
$ uv run triage-ml-train
[prepare] loading data/processed/medical_tc_train.parquet (5000 rows)
[cv] logreg        cv_macro_f1=0.7319 ± 0.0098
[cv] linear_svc    cv_macro_f1=0.7335 ± 0.0102  ← winner
[train] linear_svc on 4000 samples
[eval]  test_macro_f1=0.7296
[save]  models/20260905T171611Z-f2cb6f23f9cd/
         model.joblib classes.json metadata.json summary.json
[figs]  reports/figures/20260905T171611Z-f2cb6f23f9cd/
         08_confusion_matrix_linear_svc.png
         08_top_features_linear_svc.png
```

### Modelo final selecionado

A escolha do `LinearSVC` foi feita na Etapa 2 com base em macro-F1 no split de teste:

| Modelo | macro-F1 (CV treino) | macro-F1 (teste) |
|---|---:|---:|
| `LogisticRegression(class_weight="balanced")` | 0.7319 ± 0.0098 | 0.7280 |
| `LinearSVC(class_weight="balanced")` | **0.7335 ± 0.0102** | **0.7296** |

Justificativa completa em [Etapa_2_Modelo_baseline_e_serialização.md](../reports/Etapa_2_Modelo_baseline_e_serialização.md).

## Passo 3 — Inspecionar o artefato serializado

```bash
# Listar versões
ls -la models/

# Inspecionar manifesto de uma versão
cat models/20260905T171611Z-f2cb6f23f9cd/metadata.json | jq
```

Schema resumido do `metadata.json` (validado por `schema_version: 1`):

```json
{
  "schema_version": 1,
  "model_version": "20260905T171611Z-f2cb6f23f9cd",
  "model_name": "tfidf_linear_svc",
  "task_type": "text_classification",
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
  "metrics": {
    "accuracy": 0.7460,
    "balanced_accuracy": 0.7221,
    "macro_f1": 0.7296,
    "weighted_f1": 0.7438
  },
  "preprocessing": {
    "tfidf": { "ngram_range": [1, 2], "min_df": 2, "max_df": 0.95, "sublinear_tf": true }
  },
  "selection": {
    "selected_classifier": "linear_svc",
    "candidates": [
      { "name": "logreg", "mean_macro_f1": 0.7319, "std_macro_f1": 0.0098 },
      { "name": "linear_svc", "mean_macro_f1": 0.7335, "std_macro_f1": 0.0102 }
    ]
  },
  "fingerprint_sha256": "...",
  "fingerprint_file_sha256": "...",
  "dependency_versions": { "scikit-learn": "1.6.0", "numpy": "1.26.4", "joblib": "1.4.0" },
  "git_commit": "abc1234...",
  "git_dirty": false,
  "created_at": "2026-09-05T17:16:11Z"
}
```

Campos obrigatórios validados pelo loader (`validate_artifact_bundle`):

- Estrutura: presença de `model.joblib`, `classes.json`, `metadata.json`.
- Manifesto: `schema_version == 1`, classes inteiras, label_mapping coerente.
- Checksum: SHA-256 do `model.joblib` confere.
- Parâmetros: `random_state`, `n_train`, `n_test` declarados.
- Dependências: versões de `scikit-learn`, `numpy`, `joblib` registradas.

## Passo 4 — Apontar a API para a nova versão

### API oficial (Docker)

```bash
# 1. Conferir integridade da versão desejada
curl -s -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" http://localhost:8000/models | jq

# 2. Trocar a versão em inferência sem reiniciar
curl -s -X POST http://localhost:8000/reload \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" \
  -d '{"model_version": "20260905T171611Z-f2cb6f23f9cd"}' | jq

# 3. Confirmar via /health
curl -s http://localhost:8000/health | jq
```

### API de desenvolvimento (uvicorn)

```bash
export MODEL_PATH=models/20260905T171611Z-f2cb6f23f9cd/model.joblib
uv run uvicorn triage_ml.dev_api.app:app --host 127.0.0.1 --port 8000
```

Detalhes completos do contrato HTTP em [GUIA-USO-API.md](./GUIA-USO-API.md).

## Passo 5 — Exportar ONNX (Fase 2 — Etapa 5)

Para comparar latência sklearn vs ONNX no dashboard Grafana:

```bash
# 1. Instalar extra de otimização (uma vez)
uv sync --dev --extra optimization

# 2. Exportar (substitui model.onnx ao lado de model.joblib)
uv run python -c "
from joblib import load
from triage_ml.optimization.optimize import export_onnx
p = load('models/20260905T171611Z-f2cb6f23f9cd/model.joblib')
export_onnx(p, 'models/20260905T171611Z-f2cb6f23f9cd/model.onnx', opset=17)
"

# 3. Conferir fingerprint em metadata.json
jq '.optimization_fingerprint' models/20260905T171611Z-f2cb6f23f9cd/metadata.json
# {"classifier_kind": "linear_svc", "opset": 17, "quantized": false,
#  "fingerprint_hash": "abc1234567890def"}

# 4. Trocar a variante na API oficial
curl -s -X POST http://localhost:8000/reload \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" \
  -d '{"model_version": "20260905T171611Z-f2cb6f23f9cd", "variant": "onnx"}' | jq
```

## Passo 6 — Automatizar via DAG (Etapa 7)

A DAG `triage_ml_retraining` orquestra o ciclo completo (ingestão → validação → treino → publicação no DagsHub). Para a Fase 2, a DAG `triage_ml_retraining_optimization` itera sobre `dataset_sizing: [5000, 10000, 14000]` e materializa `optimization_<sample_size>.json` por fatia.

```bash
# Habilitar a DAG nova (default: false para preservar o stack da Etapa 7)
export TRIAGE_OPTIMIZATION_ENABLED=true

# Subir o Airflow
docker compose -f docker-compose.airflow.yml up -d --wait
```

Detalhes operacionais em [Etapa_7_Orquestração_de_retreino.md](../reports/Etapa_7_Orquestração_de_retreino.md).

## Configuração de hiperparâmetros

Arquivo: [configs/training.yaml](../../configs/training.yaml)

```yaml
preprocessing:
  tfidf:
    ngram_range: [1, 2]
    min_df: 2
    max_df: 0.95
    sublinear_tf: true

selection:
  scoring: f1_macro
  cv_folds: 5
  candidates:
    - name: logreg
      estimator: sklearn.linear_model.LogisticRegression
      params:
        class_weight: balanced
        max_iter: 200
        random_state: 42
    - name: linear_svc
      estimator: sklearn.svm.LinearSVC
      params:
        class_weight: balanced
        random_state: 42

acceptance:
  max_quality_drop_pp: 1.0   # Δ macro-F1 aceitável entre baseline e otimizado
```

## Boas práticas

- **Não treinar fora do diretório imutável** — todos os artefatos vão em `models/<versão>/` e nunca são sobrescritos.
- **Não versionar o `model.onnx` no Git** — `.gitignore` já exclui `models/` e `data/`. Para reproduzir o `model.onnx`, basta re-exportar (idempotente).
- **Não confiar só em `git_commit` do manifesto** — sempre conferir o checksum do `model.joblib` antes de promover uma versão para produção (`validate_artifact_bundle` faz isso).
- **Não afrouxar `max_quality_drop_pp` sem revisão clínica** — 1 ponto percentual de queda no macro-F1 já pode representar dezenas de amostras mal classificadas por dia em produção.
- **Sempre rodar `uv run ruff check .` e `uv run pytest tests/` antes do push** — a suíte cobre o ciclo inteiro de treino, serialização, validação de bundle e inferência.
