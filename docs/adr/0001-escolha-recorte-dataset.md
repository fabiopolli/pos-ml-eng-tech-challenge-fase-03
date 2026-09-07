# ADR 0001 — Escolha do dataset e do recorte de 5.000 amostras

- **Status**: aceito
- **Data**: 2026-08-23 (sintetizado em 2026-09-07 a partir da Etapa 1 já concluída)
- **Decisores**: Denis Melo (proponente, persona de dados), com revisão cruzada de Bill em 2026-09-07.

## Contexto

O Tech Challenge Fase 3 exige um classificador de texto aplicado à triagem médica.
A Etapa 1 do `docs/CHECKLIST.md` precisava fixar:

1. Dataset público, com licença compatível e origem auditável.
2. Schema canônico (`text`, `target`) com tipagem simples.
3. Recorte entre 2.000 e 5.000 amostras (mínimo oficial de 2.000).
4. Splits reprodutíveis e sem leakage, prontos para a Etapa 2 (modelagem) e
   Etapa 7 (retreino orquestrado).

A proveniência, a licença e o recorte deviam ser registrados em
[`docs/dataset.md`](../dataset.md) antes de qualquer modelagem, para que Bill
(Etapa 2) e Romário (Etapa 3) pudessem consumir o contrato sem ambiguidade.

## Opções consideradas

### Opção A — Medical Abstracts TC Corpus (Schopf et al., NLPIR 2022)

- Distribuído sob **CC BY-SA 3.0** no repositório oficial dos autores.
- **5 classes** clínicas já rotuladas (Neoplasms, Digestive, Nervous,
  Cardiovascular, General pathological conditions).
- Schema simples: `medical_abstract` + `condition_label`.
- Pronto para uso após download direto; sem aprovação humana extra.
- Risco de privacidade baixo: literatura científica pública, não prontuário.

### Opção B — MIMIC-III Open Access

- Acesso controlado, exige curso CITI e aprovação de uso.
- Registros clínicos desidentificados (PHI residual ainda possível).
- Labels de triagem **não vêm prontos** — exigiria derivar classes a partir
  de metadados ICD, aumentando escopo e risco.
- Adequado ao prazo atual: **baixa**.

### Opção C — outro dataset público (PubMed 200k RCT, HealthTap, etc.)

- Avaliado informalmente: rótulos ruidosos, schema inconsistente ou licença
  restritiva à aplicação comercial. Sem ganho defensável frente à opção A.

## Decisão

Adotar a **Opção A — Medical Abstracts TC Corpus** com:

- Origem canônica: <https://github.com/sebischair/Medical-Abstracts-TC-Corpus>.
- DOI do paper: <https://doi.org/10.1145/3582768.3582795>.
- Atribuição e licença preservadas em [`docs/dataset.md`](../dataset.md) e em
  [`docs/papers/README.md`](../papers/README.md).
- **Recorte de 5.000 amostras**, estratificado por `condition_label`, com
  `random_state=42` fixado em
  [`triage_ml.data.prepare.prepare_dataset`](../../src/triage_ml/data/prepare.py).
- Split treino/teste 80/20 estratificado, com asserção de ausência de leakage
  pela chave textual normalizada (`NFKC` + `casefold`).
- CSV removido da árvore atual do Git (ver `.gitignore` e histórico não
  reescrito).

### Por que excluir textos com targets conflitantes

1.956 abstracts aparecem sob labels diferentes em linhas distintas
(4.061 ocorrências). Em aprendizado supervisionado isso seria leakage
treino↔teste direto e instabilidade do gradiente. Sem metadado explicando a
divergência, a única opção defensável é remover essas linhas
(`conflicting_texts`/`conflicting_rows` no `PreparationReport`).

### Por que a restrição numérica `target ∈ {1..5}`

O dataset canônico tem exatamente 5 categorias e o pipeline impõe essa
restrição em [`prepare.py:12`](../../src/triage_ml/data/prepare.py). A
documentação textual em [`docs/dataset.md`](../dataset.md) lista as 5
categorias mas, a partir desta ADR, o contrato em
[`.agents/contracts/README.md`](../../.agents/contracts/README.md) passa a
declarar explicitamente `target ∈ {1..5}` para evitar acoplamento oculto.

## Consequências

### Positivas

- Reprodutibilidade: `prepare_dataset(raw, sample_size=5_000, random_state=42)`
  devolve o mesmo `DataFrame` canônico independente de máquina.
- Auditoria: `PreparationReport` (`input_rows`, `conflicting_texts`,
  `eligible_rows`, `output_rows`) é persistido em `summary.json` da Etapa 2.
- Compatibilidade com a Etapa 2 (TF-IDF + LinearSVC) e com a Etapa 7
  (DAG `triage_ml_retraining` lê o CSV da mesma fonte).

### Negativas e mitigação

- **Dataset em inglês**: o classificador e a API `/predict` precisam rejeitar
  textos fora do allow-list `{"en"}`. Decidido e implementado na Etapa 2
  (config `configs/api.yaml`, detector `langid`).
- **5 classes não significam gravidade**: o relatório da Etapa 1 e o
  `PLAN-text-classifier.md` declaram explicitamente que o modelo **não emite
  decisão clínica**. Portal médico/paciente (Etapa 3) reforça isso.
- **CSV fora do Git**: a operação exige um passo de download manual
  (`git restore --source=dagshub/main --worktree -- data/medical_tc_train.csv`)
  documentado em [`docs/dataset.md`](../dataset.md).

## Referências

- [`docs/dataset.md`](../dataset.md) — origem, licença, schema e recorte.
- [`docs/papers/README.md`](../papers/README.md) — paper de referência.
- [`src/triage_ml/data/prepare.py`](../../src/triage_ml/data/prepare.py) —
  implementação da preparação canônica.
- [`tests/test_data_preparation.py`](../../tests/test_data_preparation.py) —
  cobertura de conflito, duplicatas, equivalência Unicode/case, parâmetros
  inválidos e leakage.
- [`docs/reports/Etapa_1_Fundação_ dados_e_contratos.md`](../reports/Etapa_1_Fundação_%20dados_e_contratos.md) —
  relatório de implementação da Etapa 1.
