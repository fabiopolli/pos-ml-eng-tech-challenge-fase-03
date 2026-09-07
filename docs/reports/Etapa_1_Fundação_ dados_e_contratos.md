# Relatório de implementação — Fase 1 (Dataset e EDA — Denis)

| Campo | Valor |
|---|---|
| Integrante | Denis Melo |
| Etapa do checklist | Etapa 1 — Fundação, dados e contratos (`docs/CHECKLIST.md` reordenado em 2026-08-23) |
| Período desta entrega | Foundation commit `05384d4` ("Add EDA") para `prepare.py`/`01_eda.ipynb`; revisões cruzadas em `d5973dd`, `b512d74`, `410b981`, `edf339c` (2026-09-07) |
| Status atual (2026-09-07) | ✅ Etapa 1 concluída no repositório, contrato de dados consumido pela Etapa 2, ADR 0001 e gate de Airflow/artefato fechados na revisão cruzada de Bill |

Este relatório registra o que está implementado e versionado em `main` para a Etapa 1 do checklist, mesmo que parte do trabalho tenha sido entregue em sessões anteriores à atual. O objetivo é padronizar a documentação por integrante, no mesmo formato de [`Etapa_2_Modelo_baseline_e_serialização.md`](./Etapa_2_Modelo_baseline_e_serialização.md), e servir como referência única para a banca. Foi atualizado em 2026-09-07 após a revisão cruzada de Bill.

## 1. Resumo executivo

- **Dataset escolhido**: *Medical Abstracts TC Corpus* (Schopf et al., NLPIR 2022), publicado em CC BY-SA 3.0. Fonte canônica: [repositório dos autores](https://github.com/sebischair/Medical-Abstracts-TC-Corpus). Decisão registrada como ADR em [`docs/adr/0001-escolha-recorte-dataset.md`](../adr/0001-escolha-recorte-dataset.md). O CSV não está na árvore atual; o histórico não foi reescrito.
- **Pipeline de preparação**: `triage_ml.data.prepare.prepare_dataset` aplica canonicização, validação de tipos (`medical_abstract` string e `condition_label` inteiro finito), validação da faixa `target ∈ {1..5}` (`VALID_TARGETS = frozenset(range(1, 6))`), exclusão de textos com targets conflitantes, deduplicação NFKC + casefold, amostragem estratificada e ordenação determinística. Toda execução devolve um `PreparationReport` com contagens verificáveis.
- **Recorte reproduzível**: 11.550 linhas de entrada → 0 missing/empty → 4.061 linhas removidas por conflito de label em 1.956 textos únicos → 0 duplicatas no pós-conflito → **7.489 elegíveis** → amostragem estratificada **5.000** com seed 42 → split estratificado 80/20 → **4.000 treino / 1.000 teste** sem leakage (asserção por chave textual normalizada).
- **Schema canônico**: `text` (string, normalizado) e `target` (inteiro em `{1, 2, 3, 4, 5}`, conforme declarado em [`.agents/contracts/README.md`](../../.agents/contracts/README.md) e [`docs/dataset.md`](../dataset.md)). Mapeamento para nomes clínicos vive no `metadata.json` do modelo (a partir da Etapa 2), e não em runtime da API.
- **EDA**: notebook `notebooks/01_eda.ipynb` (20 células, 9 figuras em `reports/figures/`) — distribuições, comprimento dos textos, palavras mais frequentes por classe, TF-IDF médio por classe e type-token ratio (TTR). Desde a revisão de 2026-09-07, o notebook **não imprime abstracts em HTML** (a expressão `df.head(3)` foi substituída por `print` agregado e todos os `outputs` foram zerados).
- **Privacidade operacional**: textos brutos não são impressos e foram removidos da árvore atual; somente agregados e figuras devem entrar em novos commits.

## 2. Decisão de dataset

A escolha entre Medical Abstracts TC Corpus e MIMIC-III Open Access está formalizada em [`docs/adr/0001-escolha-recorte-dataset.md`](../adr/0001-escolha-recorte-dataset.md) (ADR 0001) e reproduzida em [`docs/dataset.md`](../dataset.md):

| Critério | Medical Abstracts TC | MIMIC-III Open Access |
|---|---|---|
| Acesso | Download público direto | Acesso controlado e treinamento obrigatório |
| Unidade | Abstract científico | Registro clínico desidentificado |
| Target pronto | Cinco categorias (`target ∈ {1..5}`) | Exige definição e derivação de labels |
| Adequação ao prazo | Alta | Baixa (governança e preparação) |
| Risco de privacidade | Menor | Maior, apesar da desidentificação |

O Medical Abstracts TC foi escolhido porque oferece **labels públicos, schema simples e reprodutibilidade** compatível com o prazo. MIMIC-III permanece inadequado para esta etapa sem aprovação humana explícita para acesso, definição de labels e mudança de contrato.

### 2.1 Licença e proveniência

- **CC BY-SA 3.0** preservada pelo projeto; o CSV foi removido da árvore atual. O histórico
  não foi reescrito nesta alteração.
- Atribuição registrada em [`docs/dataset.md`](../dataset.md) e referenciada no `README.md`.
- O paper original é referenciado pelo [DOI oficial](https://doi.org/10.1145/3582768.3582795); [`docs/papers/README.md`](../papers/README.md) explica autoria e registra a remoção da árvore atual.

### 2.2 Diferença entre labels do enunciado e do dataset

O enunciado do Tech Challenge sugere triagem por urgência (normal/atenção/urgente). Os professores autorizaram — registrado no ADR 0001 — que o classificador devolva as 5 condições clínicas do corpus. O modelo **não emite decisão clínica**; apenas classifica o abstract em uma das 5 categorias. Esta diferença está registrada no `PLAN-text-classifier.md` (seção 2) e na Etapa 2 do `CHECKLIST.md`.

## 3. Pipeline de preparação (`src/triage_ml/data/prepare.py`)

### 3.1 Funções expostas

| Função | Entrada | Saída | Garantias |
|---|---|---|---|
| `prepare_dataset(raw_df, sample_size=5_000, random_state=42)` | DataFrame no schema original (`medical_abstract`, `condition_label`) | `(prepared_df, PreparationReport)` | Erro se `sample_size` ∉ [2.000, 5.000]; `medical_abstract` deve ser string; `condition_label` deve ser inteiro finito em `{1..5}`; textos com múltiplos targets são excluídos; saída é única, ordenada, sem leakage posterior |
| `split_dataset(prepared_df, test_size=0.2, random_state=42)` | DataFrame canônico (`text`, `target`) | `(train_df, test_df)` | Erro se houver duplicatas em `text` (chave NFKC + casefold) ou se colunas não baterem com `CANONICAL_COLUMNS`; asserção `set(train.text) ∩ set(test.text) == ∅` |

`PreparationReport` é um `@dataclass(frozen=True)` com `input_rows`, `missing_or_empty_rows`, `conflicting_texts`, `conflicting_rows`, `duplicate_rows`, `eligible_rows`, `output_rows`.

### 3.2 Passos executados

1. `_canonicalize` valida a presença das colunas originais e a tipagem: `medical_abstract` precisa ser string ou null; `condition_label` precisa ser inteiro finito (`math.isfinite` e `is_integer`) ou null. Qualquer outra forma é rejeitada com `ValueError`.
2. Filtra linhas sem texto ou sem target (`missing_or_empty_rows`).
3. **Validação de faixa**: rejeita qualquer label fora de `{1..5}` (`unknown_targets`) com `ValueError`, alinhando a tipagem ao contrato declarado em `.agents/contracts/README.md`.
4. Identifica textos associados a **mais de um target** (`conflicting_texts`) usando a chave textual normalizada (`NFKC` + `casefold`) e exclui todas as linhas correspondentes (`conflicting_rows`). A exclusão é obrigatória: escolher um label automaticamente para esses casos "não seria defensável" (comentário em `prepare.py`).
5. Deduplica por chave normalizada, mantendo a primeira ocorrência (`duplicate_rows`).
6. Se o total elegível excede `sample_size`, aplica `_stratified_sample` (amostragem por quotas inteiras determinísticas, em vez de `train_test_split` direto) para o recorte alvo, com `random_state=42`.
7. Ordena por `(target, text)` com `kind="stable"` para reprodutibilidade bit-a-bit.

### 3.3 Números reais desta versão (executados em 2026-08-23 e reproduzidos na revisão de 2026-09-07)

```
PreparationReport(input_rows=11550, missing_or_empty_rows=0,
                  conflicting_texts=1956, conflicting_rows=4061,
                  duplicate_rows=0, eligible_rows=7489, output_rows=5000)
```

**Distribuição por classe no recorte (5.000 amostras, ratio máximo/mínimo 3,33×):**

| target | Categoria clínica | Amostras | % |
|---|---|---|---|
| 1 | Neoplasms | 1262 | 25,24 |
| 2 | Digestive system diseases | 455 | 9,10 |
| 3 | Nervous system diseases | 618 | 12,36 |
| 4 | Cardiovascular diseases | 1151 | 23,02 |
| 5 | General pathological conditions | 1514 | 30,28 |

**Split 80/20 estratificado, seed 42:**

| Split | n | distribuição por classe |
|---|---|---|
| Treino | 4000 | 1010 / 364 / 494 / 921 / 1211 |
| Teste | 1000 | 252 / 91 / 124 / 230 / 303 |

`split_dataset` aplica `set(train.text) & set(test.text) == ∅` (chave NFKC + casefold) como asserção. O notebook 01 confirma a mesma distribuição.

### 3.4 Por que excluir textos com targets conflitantes

O dataset original tem **1.956 abstracts** que aparecem sob labels diferentes em linhas distintas (4.061 ocorrências). Em um classificador supervisionado isso seria **leakage** direto do treino para o teste (o mesmo texto teria dois targets verdadeiros) e, no mínimo, instabilidade do gradiente. Como não há metadado explicando a divergência, a única opção defensável é remover essas linhas. O `PreparationReport` mantém a contagem registrada para auditoria.

### 3.5 Por que amostragem estratificada por quotas

A distribuição original tem **ratio 3,33×** entre a classe mais frequente (5) e a menos frequente (2). A amostragem estratificada garante proporções equivalentes entre treino e teste e evita que alguma classe fique sub-representada no conjunto menor. O `PLAN-text-classifier.md` recomenda o uso de `class_weight="balanced"` no classificador — escolha que já está refletida em `configs/training.yaml` e no `metadata.json` do artefato vigente (`models/20260905T171611Z-f2cb6f23f9cd/` produzido pela Etapa 7, em substituição ao artefato legado `models/20260823T135811Z-bed2194376bc/` que não passa mais na validação por schema_version).

## 4. EDA (`notebooks/01_eda.ipynb`)

O notebook tem 20 células (9 de código + 11 de markdown) organizadas em 10 seções e gera 9 figuras em `reports/figures/`. Após a revisão cruzada de 2026-09-07 (commit `d5973dd`), a célula de inspeção inicial substituiu o `df.head(3)` por um `print` agregado e **todos os `outputs` persistidos foram zerados** — uma nova execução não regera abstracts no HTML do notebook.

### 4.1 Estrutura do notebook

| Seção | Célula | Conteúdo |
|---|---|---|
| 1. Imports | 2 | `pandas`, `numpy`, `matplotlib`, `seaborn`, `triage_ml.data.prepare.prepare_dataset` |
| 2. Carregamento e inspeção inicial | 4 | Carrega o CSV, imprime `PreparationReport`, valida schema e **não imprime abstracts** |
| 3. Distribuição de classes | 6 | Gráfico de barras + ratio max/min + alerta de desbalanceamento |
| 4. Análise estatística dos textos | 8 | `char_count`, `word_count`, `sentence_count`, `avg_word_length` |
| 5. Comprimento por label | 10 | Boxplots 2×2 (chars/words/sentenças/palavras médias) por label |
| 6. Palavras mais frequentes por label | 12 | Top-15 palavras por classe (5 painéis) |
| 7. Termos discriminativos (TF-IDF médio por classe) | 14 | Top-15 termos por classe via TF-IDF médio — sinais clínicos distintos |
| 8. Diversidade de vocabulário (TTR) | 16 | Type-Token Ratio por classe (riqueza lexical) |
| 9. Validação do recorte | 18 | Confirma `output_rows=5000`, **sem impressão de abstracts** |
| 10. Resumo e insights para modelagem | 19 | Lista decisões para Bill: TF-IDF + LR/SVC, `class_weight="balanced"`, evitar Random Forest |

### 4.2 Figuras geradas

| Arquivo | Conteúdo |
|---|---|
| `reports/figures/01_label_distribution.png` | Distribuição absoluta por label com percentuais e linha do ratio 3,33× |
| `reports/figures/02_text_stats_by_label.png` | Boxplots de chars/words/sentenças/comprimento médio de palavra por label |
| `reports/figures/03_top_words_per_label.png` | Top-15 palavras mais frequentes por classe |
| `reports/figures/04_tfidf_per_label.png` | Top-15 termos discriminativos por classe (TF-IDF médio) |
| `reports/figures/05_lexical_richness.png` | Type-Token Ratio por label (riqueza lexical) |
| `05_*.png` (demais figuras) | Tabelas agregadas e validações |

> A Etapa 2 (Bill) gera suas próprias figuras (`08_*.png`), evitando duplicação.

### 4.3 Estatísticas agregadas (extraídas do output da célula 8)

| Métrica | Média | Std | P50 (mediana) | P95 |
|---|---|---|---|---|
| Caracteres por abstract | 1.237 | 506 | ~1.220 | ~2.150 |
| Palavras por abstract | 181 | n/d | n/d | n/d |
| Sentenças por abstract | 10,8 | n/d | n/d | n/d |
| Comprimento médio da palavra | 5,9 | n/d | n/d | n/d |

Os textos são longos e médios (~181 palavras). Isso confirma a recomendação de `ngram_range=(1,2)` no TF-IDF (bigramas carregam contexto sem inflar a dimensionalidade) e de um classificador linear — embeddings densos por parágrafo seriam uma escolha pior para latência.

## 5. Contrato de dados consumido pela Etapa 2

A Etapa 2 (Bill) consome o contrato canônico da Etapa 1 sem nenhum workaround. A revisão de 2026-09-07 confirmou que o **enforcement conjunto** evita acoplamento oculto:

- `prepare_dataset(raw, sample_size=5_000, random_state=42)` é chamado em `triage_ml.models.train.run_training` para gerar o `DataFrame` canônico. A função rejeita qualquer label fora de `{1..5}`, garantindo que o dataset canônico esteja em conformidade antes de o pipeline da Etapa 2 tocar nele.
- Os **fingerprints** SHA-256 do CSV bruto, do dataset preparado e dos splits de treino/teste são registrados no `metadata.json` (`fingerprints.raw_csv_sha256`, `prepared_dataset_sha256`, `train_split_sha256`, `test_split_sha256`).
- `split_dataset` garante `text` ∩ `text` = ∅ entre treino e teste. A asserção é executada a cada treino e quebra o pipeline se for violada.
- O label mapping vive em `configs/training.yaml` e é copiado para `metadata.json`. `run_training` enforça `set(label_mapping) == {prepared classes}`, impedindo drift silencioso entre o contrato de dados e a configuração de inferência.
- A faixa `target ∈ {1..5}` está declarada em [`.agents/contracts/README.md`](../../.agents/contracts/README.md) (contrato de dados) e em [`docs/dataset.md`](../dataset.md) (schema e labels). A API de desenvolvimento da Etapa 2 lê o mapping do manifesto e **não** mais do CSV em runtime.

## 6. Privacidade operacional

Decisões aplicadas em todo o pipeline de dados (reforçadas pela revisão cruzada de Bill em 2026-09-07):

- `data/*.csv` listado em `.gitignore` (Fábio, Etapa 4 mantém o gate). Apenas `data/processed/` e `data/raw/` aparecem no diretório local para preparar o ambiente.
- Notebook 01 tem **regra explícita** de nunca imprimir `medical_abstract` em outputs versionados. Antes da revisão cruzada, a célula de inspeção terminava em `df.head(3)` e os `outputs` persistidos continham uma tabela HTML com 3 abstracts. O commit `d5973dd` removeu a expressão nua e zerou todos os `outputs`; uma nova execução não regera o leak.
- `metadata.json` registra apenas **contagens, agregados e fingerprints** — nenhum texto bruto.
- Logs do treino (`models/.../summary.json`) seguem o mesmo princípio.
- A Etapa 6 (observabilidade) também reforça essa política: o teste automatizado "varrer labels Prometheus, payloads de erro e formato de logs" valida que `text` nunca aparece nessas superfícies.

## 7. Testes e verificações

| Verificação | Comando | Resultado |
|---|---|---|
| Preparação completa em CSV real | `uv run python -c "from triage_ml.data.prepare import prepare_dataset, split_dataset; ..."` | `PreparationReport(input_rows=11550, missing_or_empty_rows=0, conflicting_texts=1956, conflicting_rows=4061, duplicate_rows=0, eligible_rows=7489, output_rows=5000)`; split 4000/1000 sem leakage |
| Notebook 01 executa de ponta a ponta | `uv run jupyter execute notebooks/01_eda.ipynb` | 20 células executadas, 9 figuras regeradas, **0 abstracts impressos** após `d5973dd` |
| Integração com a Etapa 2 | `uv run triage-ml-train` | Treino concluído, `metadata.fingerprints.prepared_dataset_sha256` consistente |
| Suíte da Etapa 1 | `uv run pytest tests/test_data_preparation.py -q` | 16 passed |
| Lint | `uv run ruff check src/triage_ml/data/ tests/test_data_preparation.py .agents/contracts/` | All checks passed |

> `tests/test_data_preparation.py` cobre contrato canônico, conflitos, duplicatas,
> equivalência Unicode/case-insensitive, parâmetros inválidos, estratificação e leakage.

## 8. Pendências e trabalho futuro

| Item | Status | Observação |
|---|---|---|
| Testes de unidade para `prepare_dataset` e `split_dataset` | ✅ Concluído | Cobertura em `tests/test_data_preparation.py`, incluindo inputs e parâmetros inválidos. 16 testes verdes em 2026-09-07. |
| Versão "bruta" do CSV com checksum registrado | Pendente | Hoje o `raw_csv_sha256` é capturado na hora do treino, mas o script de download ainda não está versionado. |
| Integração com a DAG do Airflow (Etapa 7) | ✅ Concluído | DAG `triage_ml_retraining` consome `prepare_dataset` via `triage_ml.data.prepare` com caminhos configuráveis (`os.getenv("TRIAGE_RAW_CSV" | "TRIAGE_MODELS_DIR" | "TRIAGE_REPORTS_DIR" | "TRIAGE_TRAINING_CONFIG")`) e `max_active_runs=1`. Evidência de `reused=true` na segunda execução registrada na Etapa 7 do `CHECKLIST.md`. O gate da Etapa 1 (linha 73 do checklist) foi formalmente fechado na revisão cruzada de 2026-09-07. |
| Política de retreino com novos dados | Pendente | Será definida na Etapa 7 com gate humano do Denis (em paralelo à publicação do CSV "bruto" versionado). |
| ADR da escolha de dataset | ✅ Concluído | `docs/adr/0001-escolha-recorte-dataset.md` criado em `b512d74` sintetizando esta decisão no formato ADR padrão. |
| `target ∈ {1..5}` declarado no contrato | ✅ Concluído | `.agents/contracts/README.md` e `docs/dataset.md` declaram a faixa explicitamente (`410b981`). |
| Notebook 01 sem abstracts nos outputs | ✅ Concluído | `df.head(3)` removido e `outputs` zerados em `d5973dd`. |

## 9. Como reproduzir a entrega

```bash
# 1. Baixar o CSV bruto da fonte canônica
# https://github.com/sebischair/Medical-Abstracts-TC-Corpus
# Salvar em data/medical_tc_train.csv (ignorado pelo Git)

# 2. Validar o pipeline de preparação
uv run python -c "
from triage_ml.data.prepare import prepare_dataset, split_dataset
import pandas as pd
raw = pd.read_csv('data/medical_tc_train.csv')
prep, rep = prepare_dataset(raw, sample_size=5_000, random_state=42)
print(rep)
tr, te = split_dataset(prep, random_state=42)
print('train', len(tr), '| test', len(te))
"

# 3. Rodar a suíte da Etapa 1
uv run pytest tests/test_data_preparation.py -q

# 4. Regenerar o notebook e as figuras (sem abstracts nos outputs)
uv run jupyter execute notebooks/01_eda.ipynb
```

## 10. Mapa dos artefatos da Etapa 1

```
docs/
├── CHECKLIST.md                                        (marcador oficial da Etapa 1)
├── dataset.md                                          (decisão do dataset, licença, schema com target ∈ {1..5})
├── papers/README.md                                    (paper de referência)
├── adr/
│   └── 0001-escolha-recorte-dataset.md                 (ADR da escolha do dataset)
└── reports/Etapa_1_Fundação_ dados_e_contratos.md    (este documento)

notebooks/
└── 01_eda.ipynb                                        (EDA, 20 células, 9 figuras, 0 abstracts nos outputs)

src/triage_ml/data/
├── __init__.py                                         (re-exporta PreparationReport, prepare_dataset, split_dataset)
└── prepare.py                                          (prepare_dataset + split_dataset + _text_key + _canonicalize)

src/triage_ml/models/train.py                           (consumido pela Etapa 2; chama prepare_dataset e enforça label_mapping)

.airflow/dags/
└── triage_retraining.py                                (consumido pela Etapa 7; chama prepare_dataset e valida PreparationReport)

.agents/contracts/README.md                             (contrato de dados declara text: string, target: int ∈ {1..5})

tests/
└── test_data_preparation.py                            (16 testes: contrato canônico, conflitos, duplicatas, equivalência Unicode/case, parâmetros inválidos, estratificação e leakage)

reports/figures/
├── 01_label_distribution.png
├── 02_text_stats_by_label.png
├── 03_top_words_per_label.png
├── 04_tfidf_per_label.png
└── 05_lexical_richness.png (e outras figuras agregadas)
```

## 11. Conclusão

A Etapa 1 entrega uma fundação de dados reprodutível e auditável: o `prepare.py` documenta cada exclusão, o notebook 01 produz figuras que justificam as decisões de modelagem da Etapa 2 (e, após `d5973dd`, não vaza abstracts), e o contrato canônico `text`/`target` é consumido sem mudanças pelos scripts de treino da Etapa 2 e pela DAG da Etapa 7. A escolha do Medical Abstracts TC Corpus foi registrada como ADR 0001, com critérios explícitos e licença preservada. A faixa `target ∈ {1..5}` foi declarada tanto no contrato (`410b981`) quanto em `docs/dataset.md`, eliminando o acoplamento silencioso que existia entre a validação rígida em `prepare.py` e a documentação. A pendência restante (script de download do CSV bruto versionado) é pequena e não bloqueia nenhuma etapa posterior.

## 12. Histórico de revisões deste relatório

| Data | Autor | Mudança |
|---|---|---|
| 2026-08-23 | Denis | Versão inicial descrevendo Etapa 1 entregue em sessões anteriores. |
| 2026-09-07 | Bill (revisão cruzada) | Atualização após o ciclo de revisão: cabeçalho, ADR 0001 referenciado, faixa `target ∈ {1..5}` declarada, gate de Airflow/artefato fechado, notebook sem abstracts, mapa de artefatos incluindo `docs/adr/` e DAG da Etapa 7, pendências reclassificadas, validações atualizadas com `pytest`/`ruff`. |
