# 0003 — Flexibilizar o limite superior de `sample_size` para suportar `dataset_sizing`

- **Status**: Aceito (2026-09-08)
- **Responsável**: Bill

## Contexto

A Etapa 1 fixou `prepare_dataset(raw, *, sample_size, random_state)` com limite
hard `2_000 <= sample_size <= 5_000`. O limite superior foi escolhido para
manter o experimento leve e reduzir o tempo de treino (≈ minutos no TF-IDF +
LogReg/LinearSVC). A Etapa 1 salvou esse limite também em
[`configs/training.yaml`](../../configs/training.yaml) (`sample_size: 5000`).

A Fase 2 do [PLAN-text-classifier.md](../plans/PLAN-text-classifier.md) introduz
a Etapa 5 do checklist, que pede a comparação baseline vs. otimizado **e** a
exploração de **diferentes tamanhos de dataset** (`5K`, `10K`, `14K`). Os 14K
são a totalidade das linhas de `data/medical_tc_train.csv` após a
`PreparationReport` eliminar duplicatas/confitos.

Quando a DAG de otimização foi redesenhada para chamar `prepare_dataset`
repetidamente (uma vez por `sample_size`), a cláusula hard `sample_size > 5_000
=> ValueError` bloqueou o uso do dataset bruto fora do teto histórico.

## Decisão

Alterar `prepare_dataset` para aceitar qualquer `sample_size >= 2_000`, mantendo
o limite inferior e eliminando o teto superior. A constante `5_000` permanece
como default em `configs/training.yaml` e em `train.run_training`, preservando
o comportamento existente em todas as Etapas 1, 2, 3, 4 e 7 que dependem do
default. A Etapa 5 e a DAG `triage_ml_retraining_optimization` configuram
explicitamente outros valores via `dataset_sizing` ou `TRIAGE_DATASET_SLICES`.

A constante `MAX_DATASET_BYTES` (200 MiB) aplicada em `validate_dataset_file`
permanece como defesa contra OOM.

## Consequências

**Positivas**

- A Etapa 5 pode comparar treinamento em 5K / 10K / 14K sem duplicar lógica de
  preparação.
- `dataset_sizing` em `configs/training.yaml` ganha utilidade real.
- Limite inferior (`2_000`) preserva a invariante de que cada classe fica acima
  de `cv_folds` (5) entradas após a deduplicação.

**Trade-offs / Riscos**

- Treinos maiores ficam mais lentos e podem estourar o tempo limite da DAG
  (mitigado por `execution_timeout=timedelta(hours=1)` em `train_slice`).
- A função passa a confiar mais na checagem `eligible_rows >= sample_size`
  quando o CSV tem menos linhas que o solicitado — já era assim, agora é a
  única barreira.

## Como validar

- `uv run pytest tests/test_data_preparation.py` deve passar sem mudanças —
  todos os testes pré-existentes usam `sample_size` dentro do novo intervalo
  também.
- O teste de fronteira em
  [`tests/test_optimization_dataloader.py`](../../tests/test_optimization_dataloader.py)
  cobre um catálogo de slices `[5_000, 10_000, 14_000]`.
- A DAG pode ser executada localmente com
  `TRIAGE_DATASET_SLICES=5000,10000 uv run airflow dags test
  triage_ml_retraining_optimization 2026-09-08` para validar o fluxo end-to-end.

## Plano de rollback

Reintroduzir a checagem `<= 5_000` em `prepare_dataset` e remover o catálogo
`dataset_sizing`. A Etapa 5 voltaria a operar somente com o default atual.
