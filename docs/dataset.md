# Dataset — Medical Abstracts TC Corpus

## Decisão

O projeto seleciona o **Medical Abstracts Text Classification Dataset**, publicado por
Tim Schopf, Daniel Braun e Florian Matthes. A fonte canônica é o
[repositório dos autores](https://github.com/sebischair/Medical-Abstracts-TC-Corpus),
associado ao artigo *Evaluating Unsupervised Text Classification: Zero-Shot and
Similarity-Based Approaches* (NLPIR 2022, publicado em 2023), DOI
[10.1145/3582768.3582795](https://doi.org/10.1145/3582768.3582795).

A cópia do Kaggle usada durante a exploração não é a referência de proveniência. O
arquivo original de treino possui 11.550 linhas; o repositório do projeto não redistribui
esse CSV.

## Licença e uso

A fonte canônica distribui o corpus sob **Creative Commons Attribution-ShareAlike 3.0
Unported (CC BY-SA 3.0)**. Uso e redistribuição exigem atribuição, indicação de alterações
e compartilhamento de adaptações sob licença compatível. O projeto mantém a referência à
licença original e não inclui os textos no Git.

Embora o corpus contenha abstracts médicos, ele é um conjunto de literatura científica,
não um prontuário clínico como o MIMIC-III. Ainda assim, textos não devem ser enviados a
logs, métricas ou artefatos de depuração.

## Comparação considerada

| Critério | Medical Abstracts TC | MIMIC-III Open Access |
|---|---|---|
| Acesso | Download público direto | Acesso controlado e treinamento obrigatório |
| Unidade | Abstract científico | Registro clínico desidentificado |
| Target pronto | Cinco categorias | Exige definição e derivação de labels |
| Adequação ao prazo | Alta | Baixa, devido à governança e preparação |
| Risco de privacidade | Menor | Maior, apesar da desidentificação |

O Medical Abstracts TC foi escolhido porque oferece labels públicos, schema simples e
reprodutibilidade compatível com o prazo. O MIMIC-III permanece inadequado para esta
etapa sem aprovação humana para acesso, definição de labels e mudança do contrato.

## Schema e labels

O processamento converte o schema original para o contrato canônico:

| Original | Canônico | Tipo |
|---|---|---|
| `medical_abstract` | `text` | string |
| `condition_label` | `target` | inteiro |

Os labels são categorias clínicas, não níveis ordenados de gravidade:

1. Neoplasms
2. Digestive system diseases
3. Nervous system diseases
4. Cardiovascular diseases
5. General pathological conditions

## Recorte reproduzível

`triage_ml.data.prepare.prepare_dataset` aplica, nesta ordem:

1. normalização de espaços e remoção de linhas vazias;
2. exclusão de todo texto associado a mais de um target;
3. deduplicação por texto normalizado;
4. amostragem estratificada de 5.000 registros com seed 42;
5. ordenação determinística e schema `text`, `target`.

Textos com labels conflitantes são excluídos porque escolher um label automaticamente
não seria defensável. O relatório retornado pela função registra todas as exclusões.

O split usa seed 42, estratificação por target e exige unicidade de texto. Uma asserção
confirma que nenhum texto exato aparece simultaneamente em treino e teste.

## Obtenção local

Baixe `medical_tc_train.csv` da fonte canônica e salve-o localmente em `data/`. O arquivo
é ignorado pelo Git. A partir da raiz do projeto, a preparação pode ser executada assim:

```python
import pandas as pd

from triage_ml.data.prepare import prepare_dataset

raw = pd.read_csv("data/medical_tc_train.csv")
prepared, report = prepare_dataset(raw, sample_size=5_000, random_state=42)
print(report)
```

Dados processados também permanecem fora do Git. Apenas código, documentação, testes e
figuras agregadas podem ser versionados.
