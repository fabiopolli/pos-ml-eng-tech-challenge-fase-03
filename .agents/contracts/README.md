# Contratos de integração

Estes contratos iniciais permitem trabalho paralelo. Alterações incompatíveis exigem aprovação dos produtores e consumidores e atualização deste documento.

## Dados

- entrada tabular pública, com origem e licença documentadas;
- 2.000 ou mais registros por corte (`prepare_dataset` aceita `sample_size >= 2_000`; o teto superior foi removido via ADR 0003 para permitir o recorte completo `14k` da Fase 2);
- schema canônico processado: `text: string`, `target: int ∈ {1, 2, 3, 4, 5}` mapeando para as cinco categorias clínicas do Medical Abstracts TC Corpus (Neoplasms, Digestive system diseases, Nervous system diseases, Cardiovascular diseases, General pathological conditions). Labels fora dessa faixa são rejeitados em `triage_ml.data.prepare` (`VALID_TARGETS = frozenset(range(1, 6))`);
- splits reproduzíveis e sem vazamento;
- dados brutos/processados fora do Git.

## Modelo

- recebe uma coleção de textos no idioma declarado;
- expõe predição e, quando disponível, score/probabilidade;
- artefato acompanha versão, classes, preprocessing, métricas, seed e dependências;
- **baseline e otimizado mantêm contrato comparável**: variante sklearn carrega um `model.joblib` `Pipeline("tfidf" -> "clf")`; variante ONNX carrega um `model.onnx` exportado pelo mesmo `joblib` via `skl2onnx.convert_sklearn` (opset 17). O adapter (`triage_ml.optimization.onnx_adapter`) implementa `predict`/`predict_proba` ou cai para `decision_function` quando o classificador é `LinearSVC` (não há superfície probabilística calibrada).
- `metadata.json` é a fonte canônica de `schema_version`, versão, tarefa, idioma, classes e nomes, configuração, seleção, métricas, dependências, commit/estado Git, fingerprints, checksum e `available_variants: ["sklearn", "onnx"]?`;
- versões seguem `YYYYMMDDTHHMMSSZ-<input_hash>` e nunca são sobrescritas;
- o loader aceita apenas artefatos locais confiáveis, valida manifesto e checksum antes do `joblib.load` e confirma `metadata.classes == model.classes_` depois da carga;
- `optimization_fingerprint` registra `opset`, `classifier`, e o `fingerprint_hash` usado pela DAG de otimização para reaproveitar artefatos ONNX já materializados.

## API

Contrato inicial proposto, sujeito a validação por Romário:

- `GET /health`: estado do serviço;
- `GET /metrics`: métricas Prometheus (texto puro `CONTENT_TYPE_LATEST`), aberto nesta fase — revisitar RBAC na Etapa 8;
- `POST /predict`: recebe `{"text": "..."}` e devolve classe, score opcional e versão do modelo. Variante ativa controlada por `TRIAGE_ML_MODEL_VARIANT={sklearn,onnx}` (default `sklearn`); etiqueta não muda o schema de resposta;
- erros de validação não retornam dados internos nem o texto clínico em logs.
- `MODEL_PATH` aponta para o `model.joblib`; nomes de classes são lidos do manifesto, sem CSV de runtime;
- `GET /model-info` expõe o manifesto validado; `GET /models` lista somente versões íntegras do mesmo registry; `POST /reload` troca o modelo após nova validação sem publicar estado parcial;
- respostas de predição expõem `X-Request-ID` e `Server-Timing: detect;dur=<ms>, predict;dur=<ms>`;
- a checagem local usa `LanguageIdentifier(norm_probs=True)` e a allow-list deve coincidir com o idioma declarado no manifesto.

- **RBAC Estático e Proteção Clínica:** Validado por chave de API (header `X-API-Key`). Papel `patient` está restrito via `HTTP 403` a visualizar outputs da rota de predição (`POST /predict`), impedindo exposição sem revisão médica. Logs e respostas rejeitadas são estritamente sanitizadas contra vazamento de `text`.
- `POST /reload` restrito exclusivamente para o papel do sistema interserviços (`service`).
- As rotas protegidas aplicam limite independente por IP e por fingerprint SHA-256 da chave de API; a chave em texto puro nunca é usada como identificador, nem aparece em logs ou respostas.

## Observabilidade

- **Stack**: `prometheus_client` com `REQUESTS_TOTAL{route,method,status,model_variant}`, `REQUEST_LATENCY_SECONDS{route,method,model_variant}` (histograma buckets `0.005..2.5` s) e `PREDICTION_ERRORS_TOTAL{route,error_code,model_variant}`. Registry **privado** para não vazar das bibliotecas globais. Implementado em `triage_ml.observability.metrics`; degrade gracioso quando `[observability]` extra está ausente.
- **Cardinalidade**: a única lista permitida de labels é `route/method/status/model_variant/error_code`; a label `le` é reservada pelos buckets do histograma. `text`, `label_name`, `request_id`, `request_body` jamais são usados como labels.
- **Pipeline de scrape**: overlay `infra/docker-compose.yml` sobe `api-sklearn:8000/metrics` e `api-onnx:8000/metrics`; Prometheus scrape em 5 s; Grafana provisionado a partir de `monitoring/grafana/provisioning` e dashboard `monitoring/grafana/dashboards/triage_ml.json`.
- **Privacidade**: `text` é classificado e descartado. Não persiste, não vaza em log, não vira label, não aparece em payload de erro. Verificado por `tests/test_observability_privacy.py` em três superfícies (métricas, respostas, logs).
- Total de requisições por rota/status;
- histograma de latência/tempo de resposta;
- total ou taxa derivável de erros;
- labels de baixa cardinalidade; nunca usar o conteúdo do laudo como label.

## Artefatos e Airflow

- **DAG base** `triage_ml_retraining`: ingestão → validação → treinamento → persistência (Etapa 7);
- **DAG Fase 2** `triage_ml_retraining_optimization`: itera sobre `dataset_sizing: [5000, 10000, 14000]` (`configs/training.yaml`, override `TRIAGE_DATASET_SLICES`). Para cada slice exporta `model.onnx`, mede `benchmark_for_version`, valida bundle e publica `reports/benchmarks/dataset_sizing.json` consolidado por `compare_slices`. Gateada por `TRIAGE_OPTIMIZATION_ENABLED=false` para preservar o comportamento da Etapa 7;
- tarefas idempotentes quando possível: `find_reusable_artifact` reusa versões por `(dataset_sha256, config_file_sha256)` e `train_with_sample_size` reusa por `(...sample_size)`;
- caminho/registro do artefato configurável, não hardcoded para uma máquina (`TRIAGE_MODELS_DIR`, `MODEL_PATH` no contêiner);
- falhas deixam evidência acionável sem expor dados sensíveis (logs JSON com `format_exc_info` mas sem `text`).
