# Resumo da Etapa 2 — Classificador NLP (Bill)

> Material de **1 página** para apresentar aos colegas da equipe. Linguagem descomplicada, foco em **decisões e porquês**, sem entrar em jargão.

---

## 1. O problema

O Tech Challenge pediu um classificador de textos médicos em inglês. O objetivo prático é triar abstracts (resumos) de artigos em **5 categorias clínicas** e expor isso via uma **API HTTP**, com latência baixa e sem expor o texto clínico em logs ou respostas.

> **Por que categorias clínicas, e não "normal / atenção / urgente"?** O dataset público aprovado (Medical Abstracts TC) traz categorias médicas, não níveis de urgência. Combinamos com os professores usar essas 5 categorias e registrar a decisão em `docs/dataset.md` e `docs/CHECKLIST.md`. O modelo não diz se um caso é urgente — ele classifica o tema.

---

## 2. Os dados

- **Fonte:** [Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus/data?select=medical_tc_train.csv) (licença pública, dataset selecionado por Denis na Etapa 1).
- **Total bruto:** ~14k abstracts em inglês.
- **Recorte usado no treino:** **5.000 amostras** (governado por `configs/training.yaml::sample_size`) para mantermos dentro do limite do enunciado (entre 2.000 e 5.000) e com custo computacional baixo.
- **Split:** 80/20 → **4.000 treino + 1.000 teste**, com `random_state=42` fixo (mesmo split sempre).
- **Classes (5):** `1 neoplasms` · `2 digestive system diseases` · `3 nervous system diseases` · `4 cardiovascular diseases` · `5 general pathological conditions`.
- **Por que tão pequeno?** TF-IDF + scikit-learn não precisa de 100k amostras para um baseline forte em texto; um recorte menor é reproduzível, cabe em memória e ainda dá espaço para a Etapa 5 (otimização de latência).

> **Paper de referência:** Schopf, Braun & Matthes (NLPIR 2022) — *Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches*. Eles comparam abordagens não-supervisionadas e zero-shot exatamente nesse corpus (LSA 31,6 · SBERT MiniLM 46,5 · DeBERTa zero-shot 57,3 em F1 micro). A leitura sustentou a escolha de **não usar embeddings no caminho crítico** (PLMs maiores não compensam o custo em inferência). Detalhes em [`docs/papers/README.md`](../papers/README.md).

---

## 3. O modelo

Pipeline: **TF-IDF + classificador linear** (sugestão do próprio enunciado).

```
TF-IDF (1-2 gramas, min_df=2, max_df=0.95, sublinear_tf)
        ↓
LinearSVC (class_weight="balanced")
```

### 3.1 Por que esse pipeline?

- **TF-IDF** é leve, determinístico, sem dependência externa (não precisa baixar modelos grandes).
- **Comparamos Logistic Regression × LinearSVC** dentro do treino via **validação cruzada 5-fold** (mesma semente) e escolhemos pelo **macro-F1 médio**:

| Modelo | Macro-F1 (CV 5-fold, só treino) |
|---|---|
| Logistic Regression | 0.7319 |
| **LinearSVC** | **0.7335** ← escolhido |

A diferença é pequena (`0.0016`), mas LinearSVC ganhou. A escolha ficou **carimbada antes** de olharmos o teste — isso evita o vício clássico de "escolher pelo test set".

### 3.2 Métricas finais (split de teste, 1.000 amostras)

| Métrica | Valor |
|---|---|
| accuracy | **0.7460** |
| balanced accuracy | **0.7221** |
| **macro-F1** | **0.7296** |
| weighted-F1 | **0.7438** |

Por classe (resumo):

| Classe | precision | recall | F1 |
|---|---|---|---|
| 1 neoplasms | 0.843 | 0.873 | 0.858 |
| 2 digestive | 0.747 | 0.714 | 0.730 |
| 3 nervous | 0.660 | 0.516 | **0.579** ← pior |
| 4 cardiovascular | 0.807 | 0.817 | 0.812 |
| 5 general | 0.649 | 0.690 | 0.669 |

> A classe 3 (*nervous system*) tem o pior F1 — vocabulário mais heterogêneo. Aceito e documentado; investigação fica para a Fase 2.

### 3.3 Por que **não** Random Forest?

O enunciado cita RF como exemplo. Em TF-IDF, RF fica **caro** na inferência (centenas de árvores por documento) sem ganho consistente de F1 sobre modelos lineares em texto curto. **Modelo leve** era requisito, e isso se alinha com a operação real-time.

### 3.4 Por que **não** uma camada de tradução pt→en?

Três razões, todas registradas:

1. **LGPD / privacidade** — chamar uma API de tradução externa com texto clínico vaza dado sensível para terceiros. Mesma razão para não usar APIs de LLM no caminho crítico.
2. **Latência** — adicionar uma chamada de rede derruba o p95 sem ganho de qualidade (o modelo treinado em inglês não "aprende" melhor com tradução automática).
3. **Custo** — cada request vira 2 chamadas (tradução + inferência) e cada uma é cobrada/em rede.

**Solução escolhida:** a `/predict` aplica uma política local de **detecção de idioma** com `langid` (100% local, sem rede) que rejeita preventivamente qualquer texto fora do allow-list `{"en"}` antes do modelo ser invocado. Textos curtos demais também são rejeitados (`< 20 chars`, `error_code=text_too_short_for_language_check`) porque `langid` é instável nesse regime.

---

## 4. Como o artefato é entregue

Cada treino grava em `models/YYYYMMDDTHHMMSSZ-<12hex>/` (versão **imutável**):

```
models/<versão>/
├── model.joblib       # pipeline treinado
├── classes.json       # ordem das classes
├── metadata.json      # manifesto validado (schema_version=1)
└── summary.json       # saída completa do treino
```

`metadata.json` registra:
- `model_version`, `task_type`, `language`, `classes`, `label_mapping`
- `metrics` (globais + per-classe), `selection` (candidatos, métrica, folds)
- `dependency_versions` (python, numpy, scipy, scikit-learn, joblib)
- `git_commit`, `git_dirty`, `created_at`
- `fingerprints` (SHA-256 do dataset preparado, splits e config)
- `checksum_sha256` do `model.joblib`

Antes da API aceitar um artefato, **3 camadas de validação** rodam:
1. Schema do manifesto (regex de SHA, ranges, tipos).
2. Checksum SHA-256 do `model.joblib` (via `hmac.compare_digest`).
3. `pipeline.classes_ == metadata.classes` antes e depois da desserialização.

Falha em qualquer camada → API aborta startup (`RuntimeError`). Não existe "modelo carregado pela metade".

---

## 5. A API de desenvolvimento

`src/triage_ml/dev_api/` expõe 5 endpoints (todos consomem o modelo real, não stub):

| Endpoint | O que faz |
|---|---|
| `GET /health` | diz se o modelo está carregado + versão atual |
| `GET /model-info` | devolve o `metadata.json` validado — pra inspecionar o que está em inferência sem tocar o filesystem |
| `GET /models` | lista versões imutáveis disponíveis em `models/` (read-only) |
| `POST /reload` | troca o holder em runtime após re-validar manifesto + checksum (erros preservam o holder anterior) |
| `POST /predict` | classifica um texto; aplica a política de idioma antes do pipeline |

Toda resposta traz `X-Request-ID` (gerado internamente, cliente não controla) e `Server-Timing: detect;dur=<ms>, predict;dur=<ms>`. Esses dois sinais viram `Histogram` no Prometheus na Fase 2.

### 5.1 Política de idioma (3 camadas)

| Camada | Quando falha | `error_code` |
|---|---|---|
| Comprimento mínimo (`< 20` chars) | texto muito curto | `text_too_short_for_language_check` |
| Confiança mínima (`min_language_score`) | detector incerto | `indeterminate_language` |
| Allow-list `{"en"}` | idioma fora | `unsupported_language` |

O corpo do erro carrega `detected_language` e `detected_language_score` (pra debug) mas **nunca** o `text` original. Mesma garantia vale para logs — nenhum texto clínico vaza.

---

## 6. Testes e qualidade

- **102 testes verdes** em `uv run pytest` (pipeline, artefato, treino, API, idioma, helpers do dashboard).
- **Lint limpo**: `ruff check .` + `ruff format --check .`.
- Evidência versionável (sem texto clínico) em `reports/evidence/api-dev.json`.

Cobertura por arquivo (ver `Etapa_2_Modelo_baseline_e_serialização.md` §6 para a tabela atualizada).

---

## 7. O dashboard de desenvolvimento

`front/app_dev.py` é um **painel Streamlit** opcional para validar a API localmente (sem `curl`). Não substitui Prometheus/Grafana (latência/taxa de erro/volume continuam lá).

**Sidebar**:
- **🔌 Conexão** — URL da API + botão "Atualizar health".
- **🔁 Trocar modelo** — lista versões via `GET /models`, escolhe no `<selectbox>` e dispara `POST /reload`. Estado em memória (some ao reiniciar o Streamlit).
- **🧠 Modelo** — consome `GET /model-info` e mostra, em 5 expanders:
  identidade (`model_version`, `task_type`, `language`) · treinamento (`n_train`/`n_test`, `random_state`, `git_commit`, `created_at`, `dependency_versions`) · seleção (candidatos logreg × linear_svc com mean ± std) · métricas globais + tabela per-classe · mapeamento `label ↔ label_name`.

**Abas**: **🩺 Health** · **🎯 Predição** · **🌐 Política de idioma** (4 cenários canônicos com validação automática do `error_code`).

---

## 8. O que falta (Fase 2)

- **Otimização ONNX** do pipeline (mesma inferência, latência menor) — critério: Δ macro-F1 ≤ 1pp e ≥ 20% de redução no p95.
- **Stack Prometheus + Grafana** via Compose, comparando baseline vs otimizado.
- **API oficial** com Docker e auth (trabalho do Romário, Etapa 3 — herda este contrato).
- **DAG Airflow** de retreino (trabalho do Denis, Etapa 7).

Detalhamento em [`docs/plans/PLAN-text-classifier.md`](../plans/PLAN-text-classifier.md).

---

## 9. TL;DR para os colegas

- Pipeline **TF-IDF + LinearSVC** num recorte de **5.000 amostras em inglês** (split 80/20, seed 42).
- LinearSVC escolhido por **CV 5-fold** só no treino — sem contaminar o teste.
- **Macro-F1 = 0.7296** no teste (accuracy 0.7460).
- **Artefato imutável** com manifesto validado e checksum SHA-256.
- **API HTTP** com `/health`, `/model-info`, `/models`, `/reload`, `/predict`; rejeita texto fora do inglês antes do modelo (`langid` local).
- **Sem camada de tradução** (LGPD + latência + custo).
- 80 testes verdes + ruff limpo.
- Detalhes completos: [`Etapa_2_Modelo_baseline_e_serialização.md`](./Etapa_2_Modelo_baseline_e_serialização.md) · [`docs/papers/README.md`](../papers/README.md) · [`docs/plans/PLAN-text-classifier.md`](../plans/PLAN-text-classifier.md).