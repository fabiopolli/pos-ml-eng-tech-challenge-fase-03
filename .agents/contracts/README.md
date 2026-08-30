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
- `metadata.json` é a fonte canônica de `schema_version`, versão, tarefa, idioma, classes e nomes, configuração, seleção, métricas, dependências, commit/estado Git, fingerprints e checksum;
- versões seguem `YYYYMMDDTHHMMSSZ-<input_hash>` e nunca são sobrescritas;
- o loader aceita apenas artefatos locais confiáveis, valida manifesto e checksum antes do `joblib.load` e confirma `metadata.classes == model.classes_` depois da carga.

## API

Contrato inicial proposto, sujeito a validação por Romário:

- `GET /health`: estado do serviço;
- `GET /metrics`: métricas Prometheus;
- `POST /predict`: recebe `{"text": "..."}` e devolve classe, score opcional e versão do modelo;
- erros de validação não retornam dados internos nem o texto clínico em logs.
- `MODEL_PATH` aponta para o `model.joblib`; nomes de classes são lidos do manifesto, sem CSV de runtime;
- `GET /model-info` expõe o manifesto validado; `GET /models` lista somente versões íntegras do mesmo registry; `POST /reload` troca o modelo após nova validação sem publicar estado parcial;
- respostas de predição expõem `X-Request-ID` e `Server-Timing: detect;dur=<ms>, predict;dur=<ms>`;
- a checagem local usa `LanguageIdentifier(norm_probs=True)` e a allow-list deve coincidir com o idioma declarado no manifesto.

- **RBAC Estático e Proteção Clínica:** Validado por chave de API (header `X-API-Key`). Papel `patient` está restrito via `HTTP 403` a visualizar outputs da rota de predição (`POST /predict`), impedindo exposição sem revisão médica. Logs e respostas rejeitadas são estritamente sanitizadas contra vazamento de `text`.
- `POST /reload` restrito exclusivamente para o papel do sistema interserviços (`service`).
- As rotas protegidas aplicam limite independente por IP e por fingerprint SHA-256 da chave de API; a chave em texto puro nunca é usada como identificador, nem aparece em logs ou respostas.

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
