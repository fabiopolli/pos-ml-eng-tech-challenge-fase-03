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

- `X-Request-ID`: identificador curto gerado internamente (`uuid4().hex[:12]`); se o cliente enviar `X-Request-ID`, a API honora o valor (caso válido), permitindo correlação ponta-a-ponta.
- `Server-Timing`: tempos em milissegundos, formato `name;dur=<ms>, ...` — nomes possíveis: `detect` (checagem de idioma), `predict` (inferência) e `total` (tempo total na API).
- `Cache-Control: no-store` — garante que proxies e browsers nunca armazenam o payload clínico em cache compartilhado.
- Os headers `X-RateLimit-*` documentados em algumas bibliotecas `slowapi` **não são emitidos** por esta API: a contagem é interna e o 429 simplesmente carrega o `ErrorOut` padrão. Monitore latência via `/metrics`.

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
# 1. Gerar .env a partir de um único comando (detecta o modelo mais recente em
#    models/, gera secrets aleatórios via secrets.token_urlsafe(32))
uv run python scripts/bootstrap_observability_overlay.py

# 2. Subir apenas api-prod (portal é opcional para este guia)
docker compose up --build -d --wait api-prod

# 3. Conferir saúde
curl -s http://localhost:8000/health | jq
# Esperado: { "status": "ok", "model_version": "...", "model_loaded": true, "model_variant": "sklearn" }

# 4. Encerrar
docker compose down
```

> **Variante ONNX (Fase 2)**: para comparar latência sklearn vs ONNX, suba também `api-onnx` via `infra/docker-compose.yml` e flip a variável `TRIAGE_ML_MODEL_VARIANT` no `api-prod` ou passe `?variant=onnx` no `POST /reload`. Detalhes completos em [GUIA-PROMETHEUS-GRAFANA.md](./GUIA-PROMETHEUS-GRAFANA.md).
>
> **Armadilha frequente: `cat > .env <<EOF ... EOF` no shell.** Em alguns shells interativos o heredoc termina na primeira `EOF` solta, e o `docker compose` reclama `required variable X is missing a value`. Use sempre o helper [`scripts/bootstrap_observability_overlay.py`](../../scripts/bootstrap_observability_overlay.py).
>
> **Segunda armadilha: `docker compose` procura o `.env` no diretório do compose file.** O helper já cuida disso criando `infra/.env -> ../.env` automaticamente.

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
  "model_loaded": true,
  "model_variant": "sklearn"
}
```

`model_variant` reflete o valor de `TRIAGE_ML_MODEL_VARIANT` no startup (`sklearn` por padrão; `onnx` quando a imagem é montada com `TRIAGE_ML_MODEL_VARIANT=onnx`).

Status possíveis: `ok` (tudo carregado). Em caso de modelo ausente ou incompatível, a aplicação **não sobe** (`RuntimeError` no startup); logs contêm a causa. O valor `degraded` mencionado em versões antigas do plano não é mais emitido — o `/health` sempre retorna 503 quando o holder está vazio, com o `error_code` `model_not_ready`.

### `GET /model-info`

Devolve o manifesto completo do artefato em uso. Útil para auditoria e para confirmar a versão sem tocar o filesystem.

```bash
# API oficial — exige role service ou doctor
curl -s -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" http://localhost:8000/model-info | jq

# API dev — sem auth
curl -s http://127.0.0.1:8000/model-info | jq
```

Resposta (200) — campos principais (recortado do artefato real `20260909T234701Z-f2cb6f23f9cd`):

```json
{
  "model_version": "20260909T234701Z-f2cb6f23f9cd",
  "model_name": "triage_ml_tfidf_linear_svc",
  "task_type": "multiclass_text_classification",
  "language": "en",
  "classes": [1, 2, 3, 4, 5],
  "label_mapping": {
    "1": "neoplasms",
    "2": "digestive system diseases",
    "3": "nervous system diseases",
    "4": "cardiovascular diseases",
    "5": "general pathological conditions"
  },
  "random_state": 42,
  "n_train": 4000,
  "n_test": 1000,
  "metrics": {
    "accuracy": 0.752,
    "balanced_accuracy": 0.7280636354713258,
    "macro_f1": 0.7334849780556068,
    "weighted_f1": 0.7493632777168174,
    "per_class": {
      "1": {"precision": 0.837, "recall": 0.897, "f1": 0.866, "support": 252},
      "2": {"precision": 0.734, "recall": 0.637, "f1": 0.682, "support": 91},
      "3": {"precision": 0.652, "recall": 0.605, "f1": 0.628, "support": 124},
      "4": {"precision": 0.816, "recall": 0.848, "f1": 0.832, "support": 230},
      "5": {"precision": 0.667, "recall": 0.653, "f1": 0.660, "support": 303}
    }
  },
  "preprocessing": {
    "vectorizer": "tfidf",
    "tfidf": {
      "ngram_range": [1, 2],
      "min_df": 2,
      "max_df": 0.95,
      "sublinear_tf": true,
      "lowercase": true,
      "token_pattern": "(?u)\\b\\w+\\b"
    },
    "classifier": "linear_svc",
    "classifier_params": {"class_weight": "balanced", "C": 1.0, "random_state": 42}
  },
  "selection": {
    "metric": "macro_f1",
    "folds": 5,
    "candidates": {
      "logreg":    {"fold_macro_f1": [...], "mean_macro_f1": 0.7195, "std_macro_f1": 0.0040},
      "linear_svc": {"fold_macro_f1": [...], "mean_macro_f1": 0.7276, "std_macro_f1": 0.0058}
    },
    "best_classifier": "linear_svc",
    "selected_classifier": "linear_svc",
    "selection_policy": "highest_mean_macro_f1",
    "test_set_used_for_selection": false
  },
  "dependency_versions": {
    "python": "3.12.13", "numpy": "2.4.2", "scipy": "1.17.0",
    "scikit_learn": "1.8.0", "joblib": "1.5.3"
  },
  "git_commit": "abc1234...",
  "git_dirty": false,
  "created_at": "2026-09-09T23:47:01Z"
}
```

`model_name` é gerado pelo trainer como `triage_ml_<vectorizer>_<classifier>` (`triage_ml_tfidf_linear_svc` no baseline). `selection.candidates.*.fold_macro_f1` traz a lista completa das 5 dobras quando você precisar auditar a seleção.

Quando o holder está vazio: 503 com `error_code=model_not_ready`.

### `GET /models`

Lista todas as versões íntegras no registry (`models/`), em ordem new-first, junto com a versão atualmente em uso.

```bash
curl -s -H "X-API-Key: $TRIAGE_ML_API_KEY_SERVICE" http://localhost:8000/models | jq
```

Resposta (200):

```json
{
  "current": "20260909T234701Z-f2cb6f23f9cd",
  "versions": [
    "20260909T234701Z-f2cb6f23f9cd",
    "20260909T233930Z-d4ed3ea8d2ed",
    "20260909T233840Z-f2cb6f23f9cd",
    "20260909T230049Z-f2cb6f23f9cd",
    "20260909T215027Z-f2cb6f23f9cd",
    "20260823T135811Z-bed2194376bc",
    "20260823T134214Z-bed2194376bc"
  ]
}
```

`current` é a versão que o holder carrega neste momento (a mesma que `GET /health` retorna em `model_version`); `versions` é a lista de IDs ordenados lexicograficamente (timestamp ISO-8601 básico, então a ordem é equivalente a "mais novo primeiro"). Diretórios incompletos (sem `model.joblib` ou `metadata.json`) e symlinks são omitidos.

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
- `422 validation_failed` — body sem `model_version`, ou tipo errado, ou string vazia após `strip_whitespace`.
- O `model_incompatible` documentado em rascunhos anteriores foi absorvido pelo `model_not_found`: a checagem é feita no momento da carga e qualquer divergência de manifesto/checksum/estrutura já é barrada antes do 404.

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
  "model_version": "20260909T234701Z-f2cb6f23f9cd",
  "latency_ms": 28.4,
  "request_id": "443e7fab8912",
  "warnings": []
}
```

`score` é `null` quando o classificador selecionado é `LinearSVC` (não expõe `predict_proba`); use `LogisticRegression` se a probabilidade calibrada for requisito. **Não há campo `model_variant` no payload** — a variante ativa fica em `/health` e em `GET /models.current`; o cabeçalho `X-Request-ID` permite correlacionar a predição com o variant que aparece nas métricas Prometheus (`triage_ml_request_latency_seconds{model_variant="..."}`).

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
  "message": "Request could not be processed.",
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
| `model_not_found` | 404 | versão inexistente em `GET /models` (ou `model.joblib`/checksum inválido no `/reload`) |
| `unauthorized` | 401 | `X-API-Key` ausente ou diferente de qualquer chave conhecida |
| `forbidden` | 403 | chave válida, mas a role não tem permissão para o recurso (ex.: `service` chamando `/predict`, `patient` chamando `/model-info`) |
| `clinician_review_required` | 403 | especificamente em `POST /predict` com role `patient` — dashboards distinguem isso de "sem permissão genérica" |
| `request_failed` | 500 | exceção inesperada; request_id permite correlacionar com logs |
| rota inexistente | 404 | `{"error_code": "request_failed"}` — qualquer `HTTPException` cujo `detail` não esteja no allow-list cai nesse fallback |

### `GET /metrics` (Fase 2 — Etapa 6)

Texto-puro no formato `prometheus_client`. Sem autenticação por padrão (revisitar na Etapa 8). Veja [GUIA-PROMETHEUS-GRAFANA.md](./GUIA-PROMETHEUS-GRAFANA.md) para interpretação e exemplos de PromQL.

```bash
curl -s http://localhost:8000/metrics
```

## RBAC e rate limit (API oficial)

| Role | `GET /health` | `GET /model-info` | `GET /models` | `POST /reload` | `POST /predict` |
|---|---|---|---|---|---|
| sem `X-API-Key` | ✅ | 401 `unauthorized` | 401 `unauthorized` | 401 `unauthorized` | 401 `unauthorized` |
| `service` | ✅ | ✅ | ✅ | ✅ | 403 `forbidden` |
| `doctor` | ✅ | ✅ | ✅ | 403 `forbidden` | ✅ |
| `patient` | ✅ | 403 `forbidden` | 403 `forbidden` | 403 `forbidden` | 403 `clinician_review_required` |

O `X-API-Key` é o próprio valor da chave (`svc-...`, `doc-...` ou `pat-...`); o fingerprint HMAC-SHA-256 é derivado internamente a partir de um sal aleatório de 32 bytes gerado por processo.

`POST /predict` é intencionalmente restrito à role `doctor`: a arquitetura separa **inspeção** (qualquer role autenticada pode ler `model-info` e `models`) de **inferência clínica** (apenas `doctor` vê scores crus; `patient` recebe `clinician_review_required` para acionar fluxos de mediação humana). `service` pode chamar `/reload` mas não `/predict` — para automação de inferência, prefira um cliente dedicado com chave `doctor`.

Rate limit: contagem por IP + fingerprint (chave de API entra no segundo componente via `api_key_limiter`). O limite por endpoint é configurável via `TRIAGE_ML_RATELIMIT_DEFAULT` e `TRIAGE_ML_RATELIMIT_PREDICT` (sintaxe `slowapi`, ex.: `60/minute`). Quando excedido, a API devolve 429 com o `ErrorOut` padrão (`error_code=request_failed`, mensagem genérica) — não há header `X-RateLimit-*` e o body nunca carrega o `text`.

```json
{
  "request_id": "...",
  "error_code": "request_failed",
  "message": "Request could not be processed."
}
```

## Troubleshooting

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| API não sobe, log mostra `RuntimeError: no valid model version found` | Nenhum artefato válido em `models/` | Rodar `uv run triage-ml-train` (veja [GUIA-TREINAMENTO.md](./GUIA-TREINAMENTO.md)) |
| API oficial morre com `PermissionError: '/models/<ver>/metadata.json'` no startup | Artefato gravado com `umask 077` (uid do host não é world-readable) | `chmod -R a+rX models/*/` no host — o container `api-prod` roda como uid 10001 e exige leitura global |
| `/health` retorna 503 `model_not_ready` | Holder vazio (modelo incompatível ou `MODEL_PATH` errado) | Conferir `GET /models` (lista versões íntegras) e `POST /reload` para a correta |
| `POST /predict` retorna 422 `unsupported_language` | Texto fora do allow-list | Revisar `configs/api.yaml::api.supported_languages` (não recomendo afrouxar sem revisão clínica) |
| `POST /predict` retorna 422 `text_too_short_for_language_check` | Texto abaixo do mínimo | Aumentar `min_text_chars_for_language_check` ou enviar mais contexto |
| `POST /predict` retorna 403 `clinician_review_required` | Role `patient` em `/predict` | Esperado — redirecione o fluxo para um profissional; veja nota sobre `patient` em "RBAC e rate limit" |
| `POST /reload` retorna 404 `model_not_found` | Versão inexistente em `models/` ou `model.joblib`/checksum inválido | Conferir `ls models/<ver>` e re-exportar o ONNX se aplicável |
| `401 unauthorized` mesmo com chave certa | Header `X-API-Key` ausente | Conferir nome do header (case-insensitive mas **não** `Authorization: Bearer`) |
| `429` em testes de carga | IP + fingerprint acima do limite configurado em `TRIAGE_ML_RATELIMIT_*` | Esperar a janela (1 minuto) ou subir os limites para a suíte; lembre que `api_key_limiter` também conta |
| `/metrics` retorna 200 mas vazio | Container rodado a partir do target `runtime` (sem extras `optimization`/`observability`) | Use o overlay `infra/docker-compose.yml` (`api-sklearn`/`api-onnx`) ou rebuild com `--target runtime-observability` |
| ONNX retorna `onnxruntime.capi...Fail: Can not digest tokenexp: invalid perl operator: (?u)` | `model.onnx` exportado de um treino cujo `token_pattern` é `(?u)\b\w+\b` (Python re2 não aceita o modificador inline) | Re-exportar via `uv run python -c "from pathlib import Path; from triage_ml.orchestration.airflow_pipeline import export_onnx_for_version; export_onnx_for_version(Path('models/<ver>'))"` — o helper aplica `swap_token_pattern` automaticamente a partir do commit corrigido |

## Boas práticas

- **Nunca** enviar dados reais de pacientes — os fronts e a API aceitam apenas texto clínico sintético para demonstração.
- **Nunca** copiar `text` em scripts de log — o canário `PRIVACY-CANARY-CARDIOVASCULAR-RESPIRATORY 2025 with severe stenosis and arrhythmia` é procurado em métricas, body de erro e logs por `tests/test_observability_privacy.py` para garantir que continua fora dessas superfícies.
- **Sempre** correlacionar erros via `X-Request-ID` ou `request_id` no payload — facilita o trace nos logs estruturados (`structlog` JSON).
- **Sempre** conferir `Server-Timing` em produção para detectar regressões de latência (pipeline de CI futuro pode alertar quando `predict;dur` crescer mais de X%).
