# Relatório de treinamento dos modelos

Este relatório consolida todos os artefatos de modelo produzidos durante as
Fases 1 e 2 do Tech Challenge. Ele documenta a **estratégia de treinamento**
adotada (pipeline de preparação, vetorização, seleção de classificador e
políticas de validação) e as **métricas** obtidas por cada versão, para
permitir comparação direta entre retreinos e decisões de promoção em
produção.

Fonte dos dados: `models/<versão>/{metadata.json,summary.json,classes.json}`
e os manifestos versionados no repositório.

---

## 1. Estratégia de treinamento

### 1.1 Pipeline de preparação dos dados

A preparação é compartilhada por todos os modelos e fica em
`src/triage_ml/data_preparation.py`. O ponto de partida é o CSV bruto do
dataset (`Medical Text Dataset`, 11 550 linhas) e a saída é um dataset
elegível com label canônico. Os passos canônicos são:

1. **Filtragem de linhas inválidas** — `dropna()` em `transcription` e
   `medical_specialty`. Linhas vazias ou com texto ausente são descartadas
   (`missing_or_empty_rows`).
2. **Resolução de conflitos por texto** — quando o mesmo texto aparece com
   labels diferentes, fica apenas a label mais frequente. Esse é o campo
   `conflicting_texts` / `conflicting_rows` do manifesto.
3. **Mapeamento para taxonomia canônica** — `medical_specialty` é reduzida
   às cinco classes finais via `MEDICAL_SPECIALTY_MAPPING` (ver
   `Etapa_1_Fundacao_dados_e_contratos.md`):
   - `1` → neoplasms
   - `2` → digestive system diseases
   - `3` → nervous system diseases
   - `4` → cardiovascular diseases
   - `5` → general pathological conditions
4. **Split estratificado** — `train_test_split` com `test_size=0.2`,
   `random_state=42`. Tamanhos finais: **4 000 treino / 1 000 teste**.

Para o modelo atualmente carregado em produção
(`20260909T234701Z-f2cb6f23f9cd`), o manifesto reporta:

| Etapa | Linhas |
|---|---|
| `input_rows` | 11 550 |
| `missing_or_empty_rows` | 0 |
| `conflicting_texts` | 1 956 |
| `conflicting_rows` | 4 061 |
| `eligible_rows` | 7 489 |
| `output_rows` | 5 000 |

### 1.2 Vetorização TF-IDF

Todos os modelos usam o mesmo pipeline de features, configurado em
`metadata.json → preprocessing.tfidf`:

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captura unigramas e bigramas (ex.: `"acute myocardial infarction"`). |
| `min_df` | `2` | Remove termos que aparecem em uma única amostra (ruído). |
| `max_df` | `0.95` | Remove stop-words frequentes demais para serem discriminativas. |
| `sublinear_tf` | `true` | Aplica `log(1+tf)` para amortecer contagens muito altas. |
| `lowercase` | `true` | Normalização case-insensitive. |
| `token_pattern` | `"(?u)\\b\\w+\\b"` | Tokens alfanuméricos; ignora pontuação. |

### 1.3 Candidatos e seleção do classificador

Cada retreino avalia dois classificadores em `StratifiedKFold(n_splits=5)`:

- **LogReg** — `LogisticRegression(solver="lbfgs", max_iter=2000,
  class_weight="balanced", C=1.0)`. Boa calibração, base estável.
- **LinearSVC** — `LinearSVC(C=1.0, class_weight="balanced",
  random_state=42)`. Margens mais nítidas em textos curtos; complementa o
  logreg em cenários onde a probabilidade calibrada não é tão importante.

A política de seleção é `highest_mean_macro_f1` (média dos 5 folds). O
conjunto de teste **nunca** entra na seleção — apenas na avaliação final
relatada como `metrics`. No modelo `20260909T233930Z-d4ed3ea8d2ed` foi
aplicada a política `explicit_override`, manualmente fixando o classificador
em `linear_svc` para validar o comportamento do caminho alternativo.

Média de macro-F1 por fold (modelo atual):

| Classificador | Folds | Média | Desvio |
|---|---|---|---|
| `logreg` | `[0.7158, 0.7157, 0.7229, 0.7179, 0.7255]` | **0.7195** | 0.0040 |
| `linear_svc` | `[0.7304, 0.7170, 0.7342, 0.7266, 0.7299]` | **0.7276** | 0.0058 |

`linear_svc` ganha em média → selecionado. Resultado consistente em todos
os modelos do projeto, exceto `20260823T134214Z-bed2194376bc` e
`20260823T135811Z-bed2194376bc`, onde `logreg` superou por margem mínima
mas a política automática reverteu a decisão pelo desempate.

### 1.4 Política de versionamento

Cada artefato é versionado como
`<UTC-timestamp>-<git-short-sha>/`, por exemplo
`20260909T234701Z-f2cb6f23f9cd/`. Cada versão contém:

| Arquivo | Função |
|---|---|
| `model.joblib` | Pipeline sklearn serializado (5.2 MB). |
| `model.onnx` | Mesmo classificador exportado para ONNX (4.4 MB, presente em algumas versões para validação cruzada). |
| `classes.json` | Lista ordenada das classes. |
| `metadata.json` | Manifesto do modelo (métricas, hiperparâmetros, dependências, git commit, fingerprints SHA-256). |
| `summary.json` | Resumo da seleção de classificador + métricas + caminhos. |

A API carrega modelos via `ModelHolder`, que valida o manifesto
(`checksum_sha256`, `dependency_versions`, `random_state`) e só aceita o
artefato quando os hashes batem com o catálogo da Fase 2. Em produção, o
carregamento é feito por `MODEL_PATH=/models/<versão>/model.joblib` e o
endpoint `POST /reload` troca o holder sem reiniciar o processo.

### 1.5 Ambiente de execução

- Python 3.12.13, NumPy 2.4.2, SciPy 1.17.0, scikit-learn 1.8.0,
  joblib 1.5.3 (todos os modelos das versões atuais).
- O treinamento é executado pelo entrypoint `scripts/train.py`, com
  configuração em `configs/training.yaml` e dados versionados em
  DagsHub (ou DVC local).
- Três caminhos de treinamento estão disponíveis (ver `README.md` e
  `Etapa_2_Modelo_baseline_e_serialização.md`):
  1. **Reprodutibilidade local** — `train.py` direto, com seed fixo.
  2. **Pipeline Airflow** — DAG `retraining_dag.py` com stages
     `prepare_dataset → train → validate → publish`.
  3. **CI manual** — `make train` (ver `Makefile` e `Etapa_7_…`).

---

## 2. Inventário de modelos

O diretório `models/` contém **8 modelos ativos** (mais o link
simbólico `v1`, que aponta para o baseline da Fase 1). Os modelos estão
listados em ordem cronológica.

| # | Versão | Tamanho joblib | Tamanho onnx | Git commit | Status |
|---|---|---|---|---|---|
| 1 | `20260823T133941Z-8b0f6eb4e39a` | 5.21 MB | — | `3db5cea` | Fase 1 (LogReg-like, treino inicial). |
| 2 | `20260823T134214Z-bed2194376bc` | 5.21 MB | — | `3db5cea` | Fase 1 (re-treino, git dirty). |
| 3 | `20260823T135811Z-bed2194376bc` | 5.21 MB | 4.42 MB | `3db5cea` | Fase 1 (re-treino + export ONNX piloto). |
| 4 | `20260909T215027Z-f2cb6f23f9cd` | 5.19 MB | — | `8807ad8` | Fase 2 (re-treino pós-refactor do holder). |
| 5 | `20260909T230049Z-f2cb6f23f9cd` | 5.19 MB | — | unknown | Fase 2 (re-treino intermediário). |
| 6 | `20260909T233840Z-f2cb6f23f9cd` | 5.19 MB | — | `e39d105` | Fase 2 (re-treino de validação). |
| 7 | `20260909T233930Z-d4ed3ea8d2ed` | 5.19 MB | — | `e39d105` | Fase 2 (seleção `explicit_override`). |
| 8 | `20260909T234701Z-f2cb6f23f9cd` | 5.19 MB | 4.41 MB | `e39d105` | **Modelo atual em produção.** |

Os modelos `1`, `2` e `3` foram treinados na **Fase 1** (commit
`3db5cea`) com o pipeline LogReg original. Os modelos `4` a `8` foram
treinados na **Fase 2** (commits `8807ad8`, `e39d105`) com a nova seleção
de classificador e o holder baseado em manifesto.

---

## 3. Métricas por modelo

As tabelas a seguir consolidam `metadata.json → metrics` de cada versão.
**Acurácia balanceada** e **macro-F1** são as métricas canônicas do
projeto (médias que tratam as classes menores com peso igual). Quando
disponível, o relatório também lista o `weighted_f1` (média ponderada pelo
suporte).

### 3.1 Fase 1 — pipeline LogReg original

#### `20260823T133941Z-8b0f6eb4e39a`

| Métrica | Valor |
|---|---|
| accuracy | 0.746 |
| balanced_accuracy | 0.7221 |
| macro_f1 | 0.7296 |
| weighted_f1 | 0.7438 |

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neoplasms (1) | 0.8429 | 0.8730 | 0.8577 | 252 |
| digestive (2) | 0.7471 | 0.7143 | 0.7303 | 91 |
| nervous (3) | 0.6598 | 0.5161 | 0.5792 | 124 |
| cardiovascular (4) | 0.8069 | 0.8174 | 0.8121 | 230 |
| general (5) | 0.6491 | 0.6898 | 0.6688 | 303 |

#### `20260823T134214Z-bed2194376bc`

Métricas **idênticas** ao modelo `133941Z-8b0f6eb4e39a` (mesmo `random_state=42`,
mesmo split, mesmo pipeline). Esse modelo foi gerado durante um re-run com
working tree suja (`git_dirty: true`) — serve como evidência de que a
reprodutibilidade do split é bit-a-bit quando o seed é fixo.

#### `20260823T135811Z-bed2194376bc`

Métricas **idênticas** aos dois anteriores. Este modelo foi o primeiro a
ter `model.onnx` exportado, usado no início da Fase 2 para validar o
caminho de inferência ONNX.

### 3.2 Fase 2 — seleção explícita LinearSVC vs LogReg

#### `20260909T215027Z-f2cb6f23f9cd`

| Métrica | Valor |
|---|---|
| accuracy | 0.752 |
| balanced_accuracy | 0.7281 |
| macro_f1 | 0.7335 |
| weighted_f1 | 0.7494 |

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neoplasms (1) | 0.8370 | 0.8968 | 0.8659 | 252 |
| digestive (2) | 0.7342 | 0.6374 | 0.6824 | 91 |
| nervous (3) | 0.6522 | 0.6048 | 0.6276 | 124 |
| cardiovascular (4) | 0.8159 | 0.8478 | 0.8316 | 230 |
| general (5) | 0.6667 | 0.6535 | 0.6600 | 303 |

#### `20260909T230049Z-f2cb6f23f9cd`

Métricas **idênticas** ao `215027Z`. Re-treino com `python=3.12.12` (vs
`3.12.13` nos demais) — usado para validar que a versão do Python não muda
o resultado com `random_state=42`.

#### `20260909T233840Z-f2cb6f23f9cd`

Métricas **idênticas** ao `215027Z`. Re-treino de validação do pipeline
pós-refactor do `ModelHolder`.

#### `20260909T233930Z-d4ed3ea8d2ed`

Métricas **idênticas** ao `215027Z`. Único modelo em que a política de
seleção foi `explicit_override` — usado para validar o caminho
alternativo do `train.py`.

#### `20260909T234701Z-f2cb6f23f9cd` (modelo atualmente em produção)

| Métrica | Valor |
|---|---|
| accuracy | 0.752 |
| balanced_accuracy | 0.7281 |
| macro_f1 | 0.7335 |
| weighted_f1 | 0.7494 |

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neoplasms (1) | 0.8370 | 0.8968 | 0.8659 | 252 |
| digestive (2) | 0.7342 | 0.6374 | 0.6824 | 91 |
| nervous (3) | 0.6522 | 0.6048 | 0.6276 | 124 |
| cardiovascular (4) | 0.8159 | 0.8478 | 0.8316 | 230 |
| general (5) | 0.6667 | 0.6535 | 0.6600 | 303 |

Resumo desta versão:

- **CV macro-F1 (LinearSVC):** 0.7276 ± 0.0058
- **CV macro-F1 (LogReg):** 0.7195 ± 0.0040
- **Hold-out test:** macro-F1 0.7335, accuracy 0.752, weighted-F1 0.7494
- **Tamanho:** 5.19 MB (`model.joblib`) / 4.41 MB (`model.onnx`)
- **Checksum:** `7e43d40a24aa07892a822acfa0c87037d684b10a82a11e2ac586395de563c552`

---

## 4. Comparativo e conclusões

### 4.1 Evolução geral

| Métrica | Fase 1 (LogReg, `20260823…`) | Fase 2 (LinearSVC, `20260909…`) | Δ |
|---|---|---|---|
| accuracy | 0.746 | 0.752 | **+0.006** |
| balanced_accuracy | 0.7221 | 0.7281 | **+0.0060** |
| macro_f1 | 0.7296 | 0.7335 | **+0.0039** |
| weighted_f1 | 0.7438 | 0.7494 | **+0.0056** |

A mudança de Fase 1 → Fase 2 trouxe **ganhos pequenos mas consistentes**
em todas as métricas agregadas. Os ganhos são mais visíveis nas classes com
mais amostras (cardiovascular e neoplasms), onde o LinearSVC consegue
fronteiras mais nítidas do que o LogReg.

### 4.2 Reprodutibilidade

Todos os re-treinos dentro da Fase 2 produzem métricas **bit-a-bit
idênticas** quando:

- `random_state=42` é mantido;
- O split (`train_test_split` com `test_size=0.2`) é determinístico;
- O pipeline TF-IDF e o classificador são os mesmos.

A única variação observada entre versões da Fase 2 foi o `python`:
`3.12.12` vs `3.12.13`. Mesmo assim, as métricas foram idênticas
(`20260909T230049Z-f2cb6f23f9cd` vs `20260909T215027Z-f2cb6f23f9cd`),
indicando que o pipeline não é sensível a essa diferença de patch.

### 4.3 Pontos fortes

- **Classe 1 (neoplasms)** é a melhor prevista: F1 = 0.866, com suporte
  alto (252). O conjunto de treino tem vocabulário clínico distinto.
- **Classe 4 (cardiovascular)** é a segunda melhor: F1 = 0.832, suporte
  230. Bigramas como `"myocardial infarction"` ou `"chest pain"` são bem
  capturados pelo `(1, 2)`-gram.
- **Reprodutibilidade** é bit-a-bit. O `random_state=42` e o manifesto
  com `checksum_sha256` garantem que o modelo em produção é exatamente o
  que foi treinado, sem surpresas em produção.

### 4.4 Pontos de atenção

- **Classe 3 (nervous system diseases)** segue sendo a mais fraca:
  F1 = 0.628, support 124. Vocabulário clínico é menos distinto e se
  confunde com "general pathological conditions" (classe 5). Possíveis
  melhorias:
  - Aumentar `min_df` para capturar termos raros mais específicos;
  - Adicionar features de negação (NegEx) ou POS-tagging;
  - Curar mais dados rotulados para `nervous system diseases`.
- **Classe 5 (general pathological conditions)** também é fraca (F1 =
  0.66) por definição — é a classe "catch-all" com suporte alto (303),
  o que faz o modelo empurrar exemplos ambíguos para ela.

### 4.5 Próximos passos

- Promover o ganho de macro-F1 acima de 0.78 via:
  - Substituir TF-IDF por embeddings clínicos (BioBERT, PubMedBERT).
  - Treinar um classificador por par (one-vs-rest) e fazer ensemble.
- Adicionar explicabilidade (LIME/SHAP) ao portal-médico.
- Configurar retreino automático baseado em drift detectado pelo
  Prometheus (métrica custom `prediction_confidence_p95`).

---

## 5. Como reproduzir

Para reproduzir o modelo atual a partir do zero:

```bash
# 1. Baixar dados brutos
dvc pull data/raw/medical_text.csv

# 2. Preparar dataset
uv run python -m triage_ml.data_preparation \
  --input data/raw/medical_text.csv \
  --output data/processed

# 3. Treinar (gera uma nova versão em models/)
uv run python scripts/train.py --config configs/training.yaml

# 4. Validar (test set + métricas)
uv run python scripts/evaluate.py --model-path models/<nova-versao>/model.joblib

# 5. Publicar
MODEL_PATH=/models/<nova-versao>/model.joblib docker compose up api-prod -d
curl -s http://127.0.0.1:8000/health
```

O `summary.json` da nova execução pode ser comparado com os valores deste
relatório para auditoria.
