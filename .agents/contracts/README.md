# Contratos de integração

Estes contratos iniciais permitem trabalho paralelo. Alterações incompatíveis exigem aprovação dos produtores e consumidores e atualização deste documento.

## Dados

- entrada tabular pública, com origem e licença documentadas;
- 2.000 a 5.000 registros no recorte do projeto;
- schema canônico processado: `text: string`, `target: string|int`;
- splits reproduzíveis e sem vazamento;
- dados brutos/processados fora do Git.

## Modelo

- recebe uma coleção de textos no idioma declarado;
- expõe predição e, quando disponível, score/probabilidade;
- artefato acompanha versão, classes, preprocessing, métricas, seed e dependências;
- baseline e otimizado devem manter contrato comparável.

## API

Contrato inicial proposto, sujeito a validação por Romário:

- `GET /health`: estado do serviço;
- `GET /metrics`: métricas Prometheus;
- `POST /predict`: recebe `{"text": "..."}` e devolve classe, score opcional e versão do modelo;
- erros de validação não retornam dados internos nem o texto clínico em logs.

## Observabilidade

- total de requisições por rota/status;
- histograma de latência/tempo de resposta;
- total ou taxa derivável de erros;
- labels de baixa cardinalidade; nunca usar o conteúdo do laudo como label.

## Artefatos e Airflow

- DAG: ingestão → validação → treinamento → avaliação → persistência;
- tarefas idempotentes quando possível;
- caminho/registro do artefato configurável, não hardcoded para uma máquina;
- falhas deixam evidência acionável sem expor dados sensíveis.

