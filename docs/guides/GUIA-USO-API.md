# Guia de uso da API (oficial e de desenvolvimento)

Este guia cobre a operação completa das duas APIs HTTP do projeto:

- **API oficial** (`src/triage_ml/api/`) — exposta por `docker compose up api-prod` em `http://localhost:8000`, com autenticação HMAC-SHA-256 e rate limit.
- **API de desenvolvimento** (`src/triage_ml/dev_api/`) — exposta por `uv run uvicorn triage_ml.dev_api.app:app` em `http://127.0.0.1:8000`, sem autenticação (uso local restrito).

As duas APIs compartilham o mesmo contrato de inferência (`POST /predict`), o mesmo contrato de erro e os mesmos endpoints administrativos de inspeção. Diferem apenas na superfície de segurança.

## Contrato compartilhado

### Métodos e URLs

| Método | URL | API oficial | API dev | Restrição |
|---|---|---|---|---|
| `GET` | `/health` | ✅ | ✅ | público |
| `GET` | `/model-info` | ✅ | ✅ | role `service` ou `doctor` (oficial) |
| `GET` | `/models` | ✅ | ✅ | role `service` ou `doctor` (oficial) |
| `POST` | `/reload` | ✅ | ✅ | role `service` (oficial); localhost-only (dev) |
| `POST` | `/predict` | ✅ | ✅ | role `doctor` (oficial) |
| `GET` | `/metrics` | ✅ (Fase 2) | — | exposto em texto-puro `prometheus_client` |

### Headers comuns

Toda resposta carrega:

- `X-Request-ID`: UUID4 gerado internamente; se o cliente enviar `X-Request-ID`, a API honra o valor (caso válido), permitindo correlação ponta-a-ponta.
- `Server-Timing`: tempos em milissegundos, formato `name;dur=<ms>, ...` — nomes possíveis: `detect` (checagem de idioma), `predict` (inferência) e `total` (tempo total na API).
- `Cache-Control: no-store` — vaza zero cache para dados clínicos.

### Política de privacidade

- O campo `text` **nunca** é persistido, copiado para log, copiado para label de métrica ou devolvido em payload de erro.
- O payload de erro carrega apenas `request_id`, `error_code`, `message`, `detected_language` e `detected_language_score` (este par só quando a checagem de idioma foi o motivo da rejeição).
- As labels Prometheus permitidas são `route`, `method`, `status`, `model_variant` e `error_code`. `text`, `label_name` e `request_id` jamais viram label.

## Preparação

### Artefato do modelo

Antes de subir qualquer API, garanta que existe pelo menos uma versão válida em `models/`:

```bash
ls models/   # deve listar ao menos um diretório YYYYMMDDTHHMMSSZ-<12hex>
ls models/20260101T000000Z-0123456789ab/
# model.joblib  classes.json  metadata.json  summary.json
```

Para criar um artefato novo do zero, veja [GUIA-TREINAMENTO.md](./GUIA-TREINAMENTO.md).

### API oficial via Docker Compose

```bash
# 1. .env (substitua os placeholders — chaves com 32+ caracteres)
cat > .env <<'EOF'
API_MODEL_PATH=/models/20260101T000000Z-0123456789ab/model.joblib
TRIAGE_ML_API_KEY_SERVICE=svc-000000000000000000000000000000
TRIAGE_ML_API_KEY_DOCTOR=doc-000000000000000000000000000000
TRIAGE_ML_API_KEY_PATIENT=pat-000000000000000000000000000000
TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME=medico-demo
TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD=uma-senha-local
TRIAGE_ML_DASHBOARD_PATIENT_USERNAME=paciente-demo
TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD=outra-senha-local
EOF

# 2. Subir apenas api-prod (portal é opcional para este guia)
docker compose up --build -d --wait api-prod

# 3. Conferir saúde
curl -s http://localhost:8000/health | jq
# Esperado: { "status": "ok", "model_version": "...", "model_loaded": true, "model_variant": "sklearn" }

# 4. Encerrar
docker compose down
```

> **Variante ONNX (Fase 2)**: para comparar latência sklearn vs ONNX, use `infra/docker-compose.yml`, que sobe instâncias separadas com `TRIAGE_ML_MODEL_VARIANT=sklearn` e `onnx`. A variante é configuração de startup; `/reload` troca apenas a versão e valida a variante ativa.

### API de desenvolvimento via uvicorn

```bash
# 1. (Opcional) fixar versão específica; sem MODEL_PATH, escolhe a mais recente em models/
export MODEL_PATH=models/20260101T000000Z-0123456789ab/model.joblib
uv run uvicorn triage_ml.dev_api.app:app --host 127.0.0.1 --port 8000

# 2. Conferir saúde em outro terminal
curl -s http://127.0.0.1:8000/health | jq

# 3. Encerrar (Ctrl-C no terminal da API)
```

A API dev **não tem autenticação** e o `POST /reload` afeta apenas o processo local; nunca exponha esta API em rede.

## Endpoints — uso detalhado

### `GET /health`

Sondagem rápida para Kubernetes, Docker healthcheck ou smoke tests.

```bash
curl -s http://localhost:8000/health
```

Resposta (200):

```json
{
  "status": "ok",
  "model_version": "20260101T000000Z-0123456789ab",
  "model_loaded": true
}
```

Status possíveis: `ok` (tudo carregado), `degraded` (aplicação sobe mas o holder está vazio — API oficial retorna 503). Em caso de modelo ausente ou incompatível, a aplicação **não sobe** (`RuntimeError` no startup); logs contêm a causa.

### `GET /model-info`

Devolve o manifesto completo do artefato em uso. Útil para auditoria e para confirmar a versão sem tocar o filesystem.

```bash
# API oficial — exige role service ou doctor
curl -s -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" http://localhost:8000/model-info | jq

# API dev — sem auth
curl -s http://127.0.0.1:8000/model-info | jq
```

Resposta (200) — campos principais:

```json
{
  "model_version": "20260101T000000Z-0123456789ab",
  "model_name": "tfidf_linear_svc",
  "task_type": "text_classification",
  "language": "en",
  "classes": [1, 2, 3, 4, 5],
  "label_mapping": {
    "1": "neoplasms",
    "2": "digestive system diseases",
    ...
  },
  "random_state": 42,
  "n_train": 4000,
  "n_test": 1000,
  "metrics": {
    "accuracy": 0.7460,
    "balanced_accuracy": 0.7221,
    "macro_f1": 0.7296,
    "weighted_f1": 0.7438
  },
  "preprocessing": { "tfidf": { "ngram_range": [1, 2], "min_df": 2 } },
  "selection": { "selected_classifier": "linear_svc", ... },
  "dependency_versions": { "scikit-learn": "1.6.0", "numpy": "1.26.4", ... },
  "git_commit": "abc1234...",
  "git_dirty": false,
  "created_at": "2026-01-01T00:00:00Z"
}
```

Quando o holder está vazio: 503 com `error_code=model_not_ready`.

### `GET /models`

Lista todas as versões íntegras no registry (`models/`), em ordem new-first, junto com a versão atualmente em uso.

```bash
curl -s -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" http://localhost:8000/models | jq
```

Resposta (200):

```json
{
  "current_model_version": "20260101T000000Z-0123456789ab",
  "models": [
    { "model_version": "20260101T000000Z-0123456789ab", "complete": true, "created_at": "..." },
    { "model_version": "20251231T235959Z-fedcba987654", "complete": true, "created_at": "..." }
  ]
}
```

Diretórios incompletos (sem `model.joblib` ou `metadata.json`) e symlinks são omitidos.

### `POST /reload`

Troca a versão em inferência sem reiniciar a aplicação. Atômico: o holder anterior permanece ativo até a validação completa do novo; só então a referência é trocada sob lock.

```bash
# API oficial — só role service
curl -s -X POST http://localhost:8000/reload \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" \
  -d '{"model_version": "20251231T235959Z-fedcba987654"}' | jq
```

```bash
# API dev — sem auth
curl -s -X POST http://127.0.0.1:8000/reload \
  -H "Content-Type: application/json" \
  -d '{"model_version": "20251231T235959Z-fedcba987654"}' | jq
```

A variante permanece a definida por `TRIAGE_ML_MODEL_VARIANT` no startup. Em uma instância ONNX, o reload só publica a nova versão depois de validar e inicializar também o `model.onnx` correspondente.

Resposta (200):

```json
{
  "model_version": "20251231T235959Z-fedcba987654",
  "model_loaded": true
}
```

Erros possíveis:

- `404 model_not_found` — versão não listada por `GET /models`.
- `500 model_incompatible` — manifesto/checksum/estrutura inválidos. Holder anterior permanece em uso.

### `POST /predict`

Endpoint principal de inferência. Recebe um texto livre e devolve a classe prevista, o nome legível e o score.

```bash
# API oficial — só role doctor
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $TRIAGE_ML_API_KEY_DOCTOR" \
  -d '{
    "text": "Patient presents with progressive dyspnea and lower extremity edema over the past two weeks."
  }' | jq
```

```bash
# API dev — sem auth
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient presents with progressive dyspnea and lower extremity edema over the past two weeks."
  }' | jq
```

Resposta (200):

```json
{
  "label": 4,
  "label_name": "cardiovascular diseases",
  "score": 1.23,
  "model_version": "20260101T000000Z-0123456789ab",
  "model_variant": "sklearn",
  "latency_ms": 28.4,
  "request_id": "f4a1b2c3-...",
  "warnings": []
}
```

`score` é `null` quando o classificador selecionado é `LinearSVC` (não expõe `predict_proba`); use `LogisticRegression` se a probabilidade calibrada for requisito.

Headers de resposta (sempre presentes em sucesso):

```
X-Request-ID: f4a1b2c3-...
Server-Timing: detect;dur=0.7, predict;dur=28.4, total;dur=29.1
Cache-Control: no-store
```

#### Política de idioma

Antes de chamar o pipeline, a API aplica uma política local em três camadas (configurada em `configs/api.yaml`):

| Camada | Configuração | Erro quando falha |
|---|---|---|
| Comprimento mínimo | `api.min_text_chars_for_language_check` (default `20`) | `text_too_short_for_language_check` |
| Confiança mínima | `api.min_language_score` (default `0.0`) | `indeterminate_language` |
| Allow-list de idiomas | `api.supported_languages` (default `["en"]`) | `unsupported_language` |

O detector é `langid`, roda 100% local. O valor retornado está em `[0, 1]`, mas **não é uma confiança calibrada** — qualquer limiar positivo deve ser validado em entradas representativas.

Resposta 422 com idioma fora do allow-list:

```json
{
  "request_id": "...",
  "error_code": "unsupported_language",
  "message": "Idioma detectado não suportado",
  "detected_language": "pt",
  "detected_language_score": 0.94
}
```

O campo `text` nunca aparece no payload de erro.

#### Lista completa de `error_code`

| `error_code` | HTTP | Quando ocorre |
|---|---|---|
| `validation_failed` | 422 | payload vazio, campo `text` ausente ou tipo errado |
| `text_too_short_for_language_check` | 422 | texto abaixo do `min_text_chars_for_language_check` |
| `indeterminate_language` | 422 | score abaixo do `min_language_score` |
| `unsupported_language` | 422 | idioma fora do `supported_languages` |
| `model_not_ready` | 503 | holder vazio (recarregue via `/reload`) |
| `model_not_found` | 404 | versão inexistente em `GET /models` |
| `model_incompatible` | 500 | manifesto/checksum inválido no `/reload` |
| `auth_missing` | 401 | `X-API-Key` ausente na API oficial |
| `auth_invalid` | 403 | chave inválida |
| `rate_limited` | 429 | excesso de requisições por IP + fingerprint |
| `request_failed` | 500 | exceção inesperada; request_id permite correlacionar com logs |

### `GET /metrics` (Fase 2 — Etapa 6)

Texto-puro no formato `prometheus_client`. Sem autenticação por padrão (revisitar na Etapa 8). Veja [GUIA-PROMETHEUS-GRAFANA.md](./GUIA-PROMETHEUS-GRAFANA.md) para interpretação e exemplos de PromQL.

```bash
curl -s http://localhost:8000/metrics
```

## RBAC e rate limit (API oficial)

| Role | `GET /health` | `GET /model-info` | `GET /models` | `POST /reload` | `POST /predict` |
|---|---|---|---|---|---|
| sem `X-API-Key` | ✅ | 401 | 401 | 401 | 401 |
| `service` | ✅ | ✅ | ✅ | ✅ | 403 |
| `doctor` | ✅ | ✅ | ✅ | 403 | ✅ |
| `patient` | ✅ | 403 | 403 | 403 | 403 |

O `X-API-Key` é o próprio valor da chave (`svc-...`, `doc-...` ou `pat-...`); o fingerprint HMAC-SHA-256 é derivado internamente a partir de um sal aleatório de 32 bytes gerado por processo.

Rate limit: contagem por IP + fingerprint. Headers de resposta quando ativo:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1234567890
```

Resposta 429 (body de erro padrão, nunca com `text`):

```json
{
  "request_id": "...",
  "error_code": "rate_limited",
  "message": "Limite de requisições excedido"
}
```

## Troubleshooting

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| API não sobe, log mostra `RuntimeError: no valid model version found` | Nenhum artefato válido em `models/` | Rodar `uv run triage-ml-train` (veja [GUIA-TREINAMENTO.md](./GUIA-TREINAMENTO.md)) |
| API sobe com `status: degraded`, holder vazio | `MODEL_PATH` aponta para versão incompatível | Conferir `GET /models` (lista versões íntegras) e `POST /reload` para a correta |
| `POST /predict` retorna 503 `model_not_ready` | Holder vazio no momento | `POST /reload {"model_version": "<válida>"}` ou reiniciar a API |
| `POST /predict` retorna 422 `unsupported_language` | Texto fora do allow-list | Revisar `configs/api.yaml::api.supported_languages` (não recomendo afrouxar sem revisão clínica) |
| `POST /predict` retorna 422 `text_too_short_for_language_check` | Texto abaixo do mínimo | Aumentar `min_text_chars_for_language_check` ou enviar mais contexto |
| `POST /reload` retorna 500 `model_incompatible` | Checksum ou versão de sklearn inconsistente | Recompilar a versão do artefato ou atualizar a imagem Docker |
| `401 auth_missing` mesmo com chave certa | Header `X-API-Key` ausente | Conferir nome do header (case-insensitive mas **não** `Authorization: Bearer`) |
| `429 rate_limited` em testes | IP + fingerprint acima do limite | Esperar `X-RateLimit-Reset` ou subir `--workers 1` em ambiente de teste |

## Boas práticas

- **Nunca** enviar dados reais de pacientes — os fronts e a API aceitam apenas texto clínico sintético para demonstração.
- **Nunca** copiar `text` em scripts de log — o canário `PRIVACY-CANARY-CARDIOVASCULAR-RESPIRATORY 2025 with severe stenosis and arrhythmia` é procurado em métricas, body de erro e logs por `tests/test_observability_privacy.py` para garantir que continua fora dessas superfícies.
- **Sempre** correlacionar erros via `X-Request-ID` ou `request_id` no payload — facilita o trace nos logs estruturados (`structlog` JSON).
- **Sempre** conferir `Server-Timing` em produção para detectar regressões de latência (pipeline de CI futuro pode alertar quando `predict;dur` crescer mais de X%).
