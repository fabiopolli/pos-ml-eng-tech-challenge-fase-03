# Guia de treinamento de modelos

Este guia cobre o ciclo completo de treinamento: configuração do ambiente (`.env` + credenciais DagsHub), três caminhos oficiais para treinar (CLI local, DagsHub + CLI, DAG Airflow), comparação de candidatos, serialização do artefato versionado, integração com a API e retreino orquestrado.

> Para o contexto do projeto, consulte o [README](../../README.md) e a [CHECKLIST](../CHECKLIST.md).

## Visão geral

```text
   ┌─────────────────────────┐
   │ dataset (CSV do DagsHub)│
   └─────────┬───────────────┘
             │
             ▼
   ┌────────────────────────────────────────────┐
   │ ingestion (DAG: git clone + copy local)    │
   │   data/medical_tc_train.csv                │
   └─────────┬──────────────────────────────────┘
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
   │   metadata.json, summary.json,             │
   │   airflow_run.json (quando via DAG)        │
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

- Python 3.12 com [`uv`](https://docs.astral.sh/uv/) (`pip install uv`).
- Dependências: `uv sync --dev`.
- Para a Fase 2 (export ONNX): `uv sync --dev --extra optimization`.
- Para a Fase 2 (observabilidade): `uv sync --dev --extra observability`.
- Para rodar a DAG com ingestão do DagsHub: token de leitura (40 chars) em `DAGSHUB_USER_TOKEN` no `.env`.

## Passo 0 — Configurar `.env` e credenciais DagsHub

Copie o template e preencha apenas o que for usar:

```bash
cp .env.example .env
```

| Variável | Quando preencher | Observação |
|---|---|---|
| `DAGSHUB_USERNAME` | Sempre que for rodar a DAG `triage_ml_retraining` | Usuário do repo DagsHub (ex.: `fabiopolli`). |
| `DAGSHUB_USER_TOKEN` | Sempre que `TRIAGE_REQUIRE_AUTH=true` | Token de leitura do DagsHub (`Settings → Tokens`); **nunca** use a senha da conta. |
| `TRIAGE_INGEST_MODE` | `git` (clone oficial) ou `local` (fallback offline) | Default `git`. |
| `TRIAGE_INGEST_FALLBACK_LOCAL` | `true` (default) recomendado | Se o `git clone` falhar com 401/403, a DAG cai automaticamente em `local`. |
| `TRIAGE_DAG_RETRIES` | `0` em dev, `2` em produção | Em `dev` com `airflow dags test`, use `0` para falhar rápido. |

Como gerar o token:

1. Acesse `https://dagshub.com/user/settings/tokens`.
2. **Generate New Token** com escopo **read-only**.
3. Cole no `.env` (40 caracteres hex). Confirme:

```bash
awk -F= '/^DAGSHUB_USER_TOKEN=/{print length($2)}' .env   # esperado: 40
git check-ignore .env                                       # esperado: ".env"
```

Para `api-prod`, `portal-prod` e `dashboard-dev` **não** é necessário preencher as variáveis do DagsHub — eles consomem o modelo localmente em `models/<versão>/`.

## Passo 1 — Obter o dataset

O dataset oficial fica versionado no DagsHub: `data/medical_tc_train.csv` (`condition_label,medical_abstract`, 11.550 linhas após filtragem). Você pode obtê-lo por três caminhos, do mais “oficial” ao mais direto:

### 1.1 — Pela DAG `triage_ml_retraining` (recomendado para CI/CD e produção)

Pré-requisito: `.env` com credenciais DagsHub.

```bash
# Permita escrita pelo uid 50000 do container Airflow (apenas em dev)
chmod 777 data models reports

# Suba o stack
docker compose -f docker-compose.airflow.yml --project-name triage-airflow up -d --wait

# Dispare a DAG headless
docker compose -f docker-compose.airflow.yml --project-name triage-airflow exec -T airflow \
  airflow dags test triage_ml_retraining $(date -u +%Y-%m-%d)

# Acompanhe
docker compose -f docker-compose.airflow.yml --project-name triage-airflow exec -T airflow \
  airflow dags test triage_ml_retraining $(date -u +%Y-%m-%d) 2>&1 | \
  grep -E "Task succeeded|Task failed|model_version|reused"

# Desligue a stack
docker compose -f docker-compose.airflow.yml --project-name triage-airflow down
```

Saída esperada (com `TRIAGE_DAG_RETRIES=0`):

| Task | Estado | Observação |
|---|---|---|
| `ingest` | ✅ success | `git clone` autenticado da `main` do DagsHub → `data/medical_tc_train.csv`. |
| `validate` | ✅ success | `prepared_rows=5000`, `classes=[1..5]`. |
| `train` | ✅ success | Artefato novo em `models/2026…-<input_hash>/` com `macro_f1=0.7335`. Em uma segunda execução idêntica, retorna `reused=True`. |
| `verify` | ✅ success | Checksum do `model.joblib` validado. |

O `airflow_run.json` produzido pela DAG registra a proveniência:

```json
{
  "config_file_sha256": "88567105badae4ec07660e2e5cabe3f8b36669fec5d93f7c78ecace46bdb3b4b",
  "dataset_sha256": "ad53aebc682d6b87a5647f619a079bb446d286fdc93bf0159b812418f5758609",
  "source_commit": "069dc330e8f5c478a82c893cc224d63734781f6f"
}
```

### 1.2 — Baixar do DagsHub e treinar via CLI (sem subir Airflow)

Útil para dev local:

```bash
# API raw do DagsHub (mesmo arquivo da DAG, sem `git clone`)
curl -sSL -o data/medical_tc_train.csv \
  "https://dagshub.com/api/v1/repos/<owner>/<repo>/raw/main/data/medical_tc_train.csv"

# Conferir
wc -l data/medical_tc_train.csv    # esperado: 11551
sha256sum data/medical_tc_train.csv  # esperado: ad53aebc682d6b87a5647f619a079bb446d286fdc93bf0159b812418f5758609
```

Alternativa com sparse-checkout (se preferir Git):

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://dagshub.com/<owner>/<repo>.git /tmp/dataset-src
cd /tmp/dataset-src
git sparse-checkout set data/medical_tc_train.csv
cp data/medical_tc_train.csv /home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/data/
```

### 1.3 — CLI no host (CSV já presente)

Se o CSV já está em `data/medical_tc_train.csv` (de uma Opção 1.1 ou 1.2 anterior):

```bash
uv run triage-ml-train
```

Se o CSV estiver ausente, a CLI emite mensagem clara apontando para `--raw-csv` ou para a DAG.

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

# Apontar para outro CSV (caminho arbitrário)
uv run triage-ml-train --raw-csv /caminho/do/seu.csv

# Ajuda completa
uv run triage-ml-train --help
```

### O que a CLI faz

1. Carrega o CSV via `validate_dataset_file` (caminho padrão `data/medical_tc_train.csv`, configurável em [`configs/training.yaml`](../../configs/training.yaml)).
2. Aplica split 80/20 estratificado com `random_state=42`.
3. Executa `cross_val_score` (5-fold, `scoring="f1_macro"`) sobre o **treino apenas** para cada candidato em `configs/training.yaml::selection.candidates`.
4. Seleciona o vencedor por maior média de macro-F1.
5. Re-treina o vencedor no treino inteiro.
6. Avalia no split de teste (accuracy, balanced_accuracy, macro-F1, weighted-F1, matriz de confusão, top features).
7. Serializa em `models/YYYYMMDDTHHMMSSZ-<12hex>/` com `validate_artifact_bundle` (schema_version=1).
8. Grava `reports/figures/<model_version>/` com confusion matrix e top features (quando `report_outputs=true`).

### Exemplo de saída real

```text
$ uv run triage-ml-train
version: 20260909T234701Z-f2cb6f23f9cd (linear_svc)
n_train=4000 n_test=1000
accuracy=0.7520
balanced_accuracy=0.7281
macro_f1=0.7335
weighted_f1=0.7494
artifact: models/20260909T234701Z-f2cb6f23f9cd/model.joblib
metadata: models/20260909T234701Z-f2cb6f23f9cd/metadata.json
```

### Modelo final selecionado

A escolha do `LinearSVC` foi feita na Etapa 2 com base em macro-F1 no split de teste:

| Modelo | macro-F1 (CV treino) | macro-F1 (teste) |
|---|---:|---:|
| `LogisticRegression(class_weight="balanced")` | 0.7319 ± 0.0098 | 0.7280 |
| `LinearSVC(class_weight="balanced")` | **0.7335 ± 0.0102** | **0.7335** |

Justificativa completa em [Etapa_2_Modelo_baseline_e_serialização.md](../reports/Etapa_2_Modelo_baseline_e_serialização.md).

## Passo 3 — Inspecionar o artefato serializado

```bash
# Listar versões
ls -la models/

# Inspecionar manifesto de uma versão
jq . models/20260909T234701Z-f2cb6f23f9cd/metadata.json

# Validação programática
uv run python -c "
from pathlib import Path
from triage_ml.models.artifact import validate_artifact_bundle
m = validate_artifact_bundle(Path('models/20260909T234701Z-f2cb6f23f9cd/model.joblib'))
print(m['model_version'], m['selection']['best_classifier'], m['metrics']['macro_f1'])
"
```

Schema resumido do `metadata.json` (validado por `schema_version: 1`):

```json
{
  "schema_version": 1,
  "model_version": "20260909T234701Z-f2cb6f23f9cd",
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
    "accuracy": 0.7520,
    "balanced_accuracy": 0.7281,
    "macro_f1": 0.7335,
    "weighted_f1": 0.7494
  },
  "preprocessing": {
    "tfidf": { "ngram_range": [1, 2], "min_df": 2, "max_df": 0.95, "sublinear_tf": true },
    "classifier": "linear_svc"
  },
  "selection": {
    "best_classifier": "linear_svc",
    "candidates": {
      "logreg":     { "mean_macro_f1": 0.7319, "std_macro_f1": 0.0098 },
      "linear_svc": { "mean_macro_f1": 0.7335, "std_macro_f1": 0.0102 }
    }
  },
  "fingerprints": {
    "raw_csv_sha256": "ad53aebc...",
    "prepared_dataset_sha256": "ad53aebc...",
    "train_split_sha256": "...",
    "test_split_sha256": "...",
    "config_sha256": "..."
  },
  "checksum_sha256": "...",
  "checksum_file_sha256": "...",
  "dependency_versions": { "scikit-learn": "1.6.0", "numpy": "1.26.4", "joblib": "1.4.0" },
  "git_commit": "...",
  "git_dirty": false,
  "created_at": "2026-09-09T23:47:01Z"
}
```

Campos obrigatórios validados pelo loader (`validate_artifact_bundle`):

- **Estrutura**: presença de `model.joblib`, `classes.json`, `metadata.json`.
- **Manifesto**: `schema_version == 1`, classes inteiras, label_mapping coerente.
- **Checksum**: SHA-256 do `model.joblib` confere com `checksum_sha256`.
- **Parâmetros**: `random_state`, `n_train`, `n_test` declarados.
- **Dependências**: versões de `scikit-learn`, `numpy` (apenas `[:2]`), `scipy` (apenas `[:2]`), `joblib` registradas.

## Passo 4 — Apontar a API para a nova versão

### API oficial (Docker)

```bash
# 1. Conferir integridade da versão desejada
curl -s -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" http://localhost:8000/models | jq

# 2. Trocar a versão em inferência sem reiniciar
curl -s -X POST http://localhost:8000/reload \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" \
  -d '{"model_version": "20260909T234701Z-f2cb6f23f9cd"}' | jq

# 3. Confirmar via /health
curl -s http://localhost:8000/health | jq
```

### API de desenvolvimento (uvicorn)

```bash
export MODEL_PATH=models/20260909T234701Z-f2cb6f23f9cd/model.joblib
uv run uvicorn triage_ml.dev_api.app:app --host 127.0.0.1 --port 8000
```

Detalhes completos do contrato HTTP em [GUIA-USO-API.md](./GUIA-USO-API.md).

## Passo 5 — Exportar ONNX (Fase 2 — Etapa 5)

Para comparar latência sklearn vs ONNX no dashboard Grafana:

```bash
# 1. Instalar extra de otimização (uma vez)
uv sync --dev --extra optimization

# 2. Re-exportar o modelo com token_pattern re2-compatível
uv run python scripts/reexport_onnx.py --version 20260909T234701Z-f2cb6f23f9cd

# 3. Conferir fingerprint em metadata.json
jq '.optimization.optimization_fingerprint' \
  models/20260909T234701Z-f2cb6f23f9cd/metadata.json
# {"classifier": "linear_svc", "opset": 17, "quantized": false,
#  "fingerprint_hash": "abc1234567890def"}

# 4. Recarregar uma API iniciada com TRIAGE_ML_MODEL_VARIANT=onnx
curl -s -X POST http://localhost:8000/reload \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" \
  -d '{"model_version": "20260909T234701Z-f2cb6f23f9cd"}' | jq
```

> O tokenizador nativo do `onnxruntime` usa `re2` e rejeita o modificador `(?u)` que o sklearn embute no `token_pattern`. O `scripts/reexport_onnx.py` faz a troca in-memory (para `r"\b\w+\b"`) **sem** alterar `metadata.preprocessing.tfidf.token_pattern` (a divergência fica documentada apenas em `metadata.optimization.current_token_pattern`).

## Passo 6 — Automatizar via DAG (Etapa 7)

A DAG `triage_ml_retraining` orquestra o ciclo completo (ingestão → validação → treino → verificação). Para a Fase 2, a DAG `triage_ml_retraining_optimization` itera sobre `dataset_sizing: [5000, 6000, 7000]` e materializa `optimization_<sample_size>.json` por corte.

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

## Solução de problemas

| Sintoma | Causa provável | Mitigação |
|---|---|---|
| `FileNotFoundError: data/medical_tc_train.csv` | CSV ausente ao usar `triage-ml-train` | Use a DAG (Opção 1.1) ou `curl` da API raw do DagsHub (Opção 1.2), ou `--raw-csv <path>`. |
| `PermissionError` no `data/`, `models/` ou `reports/` ao subir Airflow | uid 50000 do container sem escrita no host | `chmod 777 data models reports` em dev. |
| DAG em `up_for_retry` por minutos | `retries=2` × `retry_delay=2min` | `TRIAGE_DAG_RETRIES=0` no `.env` para dev. |
| `401/403` no `git clone` da DAG | Token inválido/expirado | `awk -F= '/^DAGSHUB_USER_TOKEN=/{print length($2)}' .env` → esperado: 40. Como mitigação de smoke test, `TRIAGE_INGEST_FALLBACK_LOCAL=true` cai em `local`. |
| `re2: Error parsing '(?u)\b\w+\b'` no ONNX | Tokenizer nativo do `onnxruntime` rejeita `(?u)` | Rode `scripts/reexport_onnx.py` antes de servir a variante `onnx`. |