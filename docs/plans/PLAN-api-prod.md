# Plano de implementação — API oficial (Romário)

- **Integrante**: Romário
- **Origem**: Tech Challenge — Fase 3 (ML Engineering)
- **Etapa do checklist**: Etapa 3 (`docs/CHECKLIST.md` linha 104) — *API oficial servindo o modelo*
- **Status**: pendente; este plano orienta a implementação da versão **produção** da API tendo como base tudo o que já está em `main` (especialmente a [`dev_api`](../../src/triage_ml/dev_api/) de Bill).
- **Mapeamento**: este plano cobre a Etapa 3 do checklist e prepara o gancho para Etapa 5 (otimização), Etapa 6 (Prometheus/Grafana) e Etapa 8 (cloud + vídeo).

> **Antes de começar**: a [`dev_api`](../../src/triage_ml/dev_api/) já é um FastAPI real (não stub) que consome o artefato validado em `models/<versão>/model.joblib`. Este plano **não duplica** o que está bom — ele eleva o mesmo núcleo para o nível produção (autenticação, rate-limit, observabilidade, deploy containerizado, contrato estável). Toda a base de contrato (`PredictIn`, `PredictOut`, `HealthOut`, `ModelInfoOut`, `ModelsListOut`, `ReloadIn`, `ReloadOut`, `ErrorOut`) está em [`src/triage_ml/dev_api/schemas.py`](../../src/triage_ml/dev_api/schemas.py) e pode ser **reaproveitada** como ponto de partida.

---

## 1. Contexto e objetivo

Entregar uma API FastAPI pronta para produção com:

1. Contrato HTTP idêntico ao da `dev_api` (5 endpoints), porém endurecido para uso externo.
2. **Validação de payload e de resposta exclusivamente via Pydantic v2** (modelos compartilhados com a `dev_api`, evoluindo para uma pasta comum `src/triage_ml/api/schemas.py`).
3. **Autenticação** por API Key estática + opção de mutação interna via token de serviço (sem JWT de usuário na Fase 1 do produto).
4. **Rate limit** por chave e por IP, configurável por env var.
5. **Observabilidade**: Prometheus + tracing opcional + correlação por `request_id`.
6. **Dockerfile multi-stage** + imagem reproduzível (parte do entregável do Fábio, mas dirigida a esta API).
7. **Baseline de latência local** com metodologia documentada, gancho direto para a Etapa 5 (otimização ONNX).

### 1.1 Não-objetivos (Fase 3)

- Autenticação por SSO / OAuth2 / SAML — fora do escopo do Tech Challenge.
- Multi-tenant com isolamento por chave — basta um namespace por API Key.
- Tradução automática pt→en — rejeitada na Etapa 2 por LGPD/latência/custo (ver [`Etapa_2_Modelo_baseline_e_serialização.md`](../reports/Etapa_2_Modelo_baseline_e_serialização.md) §5.1).
- Endpoint `/metrics` Prometheus — entra na Etapa 6; a Fase 3 só deixa os hooks no código.

---

## 2. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Framework | FastAPI | Já usado na `dev_api`; ecossistema Pydantic/Starlette/Uvicorn estável |
| Schemas | **Pydantic v2** (`BaseModel`, `Field`, `StringConstraints`, `Literal`, `model_validator`) | Mesma tecnologia da `dev_api`; `BaseModel.model_dump_json()` permite resposta serializada consistente; `field_validator`/`model_validator` cobrem regras cruzadas (ex.: `version` consistente com `model_version`) |
| Validação de entrada | `Annotated[str, StringConstraints(...)]` em **todos** os campos string | Já padronizado na `dev_api`; reduz superfície de erro e mantém uma única fonte da verdade |
| Validação de saída | `response_model=PredictOut` em **todas** as rotas com retorno de negócio | Garante que o cliente receba exatamente o contrato — campos extras da pipeline interna (debug, scores intermediários) ficam do lado servidor |
| Validação cruzada | `model_validator(mode="after")` quando o body tem dependência entre campos | Ex.: em `ReloadIn` validar que `model_version` casa `VERSION_PATTERN` antes de chamar o holder |
| ASGI server | Uvicorn (`uvicorn[standard]`) + 1 worker por processo, `--workers` via env | Workers múltiplos não compartilham holder; cada worker carrega seu próprio artefato. Para Fase 3 fica **1 worker** e a decisão de escalar é documentada como limitação |
| Auth | API Key estática via header `X-API-Key`, validada por `hmac.compare_digest` em `fastapi.Depends` | Suficiente para um serviço interno chamado pelo dashboard/App; substitui JWT de usuário, que está fora do escopo |
| Rate limit | `slowapi` (limitador baseado em `limits`) com Redis opcional; default em memória para Fase 3 | Suficiente para um único processo; Fase 6/8 pode trocar por Redis se virar multi-instância |
| Logging | `structlog` (JSON) + correlação por `request_id` | Permite grep por request; logs nunca carregam o `text` original (sanitização já documentada na Etapa 2) |
| Settings | `pydantic-settings` (`BaseSettings`) com `.env` opcional | Já recomendado no enunciado; centraliza envs com validação automática (ex.: `API_KEY` deve ter pelo menos 32 chars) |
| Container | Docker (multi-stage) com `python:3.12-slim`, UID não-root, `--no-cache-dir` | Imagem reproduzível; tamanho razoável; gancho direto para a Etapa 8 (Cloud Run / Artifact Registry) |

### 2.1 Reuso da `dev_api`

`src/triage_ml/dev_api/schemas.py` define os contratos Pydantic. Eles serão **movidos** (não copiados) para `src/triage_ml/api/schemas.py`, e a `dev_api` passa a importá-los de lá. Não há duplicação — qualquer evolução de contrato (campos novos, validações) acontece em **um só lugar**.

---

## 3. Estrutura de arquivos a criar

```
src/triage_ml/
├── api/                          # produção (novo)
│   ├── __init__.py
│   ├── app.py                    # FastAPI factory: middlewares, auth, lifespan, exception handlers
│   ├── auth.py                   # dependência de API Key (hmac.compare_digest)
│   ├── ratelimit.py              # setup do slowapi + Limiter
│   ├── logging_config.py         # structlog JSON + correlação request_id
│   ├── settings.py               # pydantic-settings (BaseSettings)
│   ├── schemas.py                # ⬅ reexporta de dev_api.schemas; é a fonte canônica
│   └── metrics.py                # stubs Prometheus (counters/histograms) usados na Fase 6
├── dev_api/                      # só para validação local — passa a importar api/schemas
│   ├── app.py
│   └── ...
└── models/
    └── artifact.py               # load_artifact / validate_artifact_bundle / VERSION_PATTERN (já existe, reutilizar)

tests/
├── test_api_auth.py              # API Key inválida, falta header, rotação
├── test_api_ratelimit.py         # estouro por chave, por IP, retry
├── test_api_settings.py          # BaseSettings valida envs obrigatórias
├── test_api_validation.py        # Pydantic: PydanticInvalid, overflow, tipo errado, cross-field
├── test_api_health.py            # /health, /model-info, /models
├── test_api_predict.py           # /predict happy path + cada error_code
├── test_api_reload.py            # /reload happy + 404 + 500
├── test_api_logging.py           # logs nunca carregam `text`; request_id sempre presente
└── test_api_docker.py            # smoke test: sobe a imagem e bate em /health

configs/
└── api.yaml                      # já existe; estender com `auth`, `ratelimit`, `server`

Dockerfile                        # multi-stage python:3.12-slim
.dockerignore
docker-compose.yml                # serviço único desta API (Prometheus/Grafana entram na Etapa 6)

docs/
├── plans/PLAN-api-prod.md        # este documento
└── reports/Etapa_3_API_oficial.md  # relatório de implementação ao final
```

---

## 4. Contratos a cumprir

### 4.1 Contrato HTTP (compatível com `dev_api`)

| Método | Path | Body | Resposta de sucesso | Erros possíveis |
|---|---|---|---|---|
| `GET` | `/health` | — | `HealthOut(status, model_version, model_loaded)` | — |
| `GET` | `/model-info` | — | `ModelInfoOut(...)` | `503 model_not_ready` |
| `GET` | `/models` | — | `ModelsListOut(versions, current)` | — |
| `POST` | `/reload` | `ReloadIn(model_version)` | `ReloadOut(model_version, model_loaded)` | `404 model_not_found`, `500 model_incompatible`, `401 unauthorized` |
| `POST` | `/predict` | `PredictIn(text)` | `PredictOut(label, label_name, score?, model_version, latency_ms, request_id, warnings)` | `401 unauthorized`, `422 validation_failed`, `422 text_too_short_for_language_check`, `422 indeterminate_language`, `422 unsupported_language`, `500 prediction_failed` |

Resposta de erro sempre no formato:

```json
{
  "request_id": "abc123def456",
  "error_code": "unsupported_language",
  "message": "Only English texts are supported.",
  "detected_language": "pt",
  "detected_language_score": 0.93
}
```

O `text` **nunca** aparece em respostas, logs ou métricas (cláusula herdada da Etapa 2, agora **endurecida** por contrato).

Headers obrigatórios em toda resposta:

- `X-Request-ID` (gerado internamente; cliente não controla).
- `Server-Timing: detect;dur=<ms>, predict;dur=<ms>` ou apenas `detect;dur=<ms>` quando o detector interrompe o fluxo.

### 4.2 Schemas Pydantic — guia prático

Pydantic v2 é a **única** camada de validação (de entrada e saída). Princípios:

- **Sem `Any` em campos de saída**. Onde a `dev_api` usa `dict[str, Any]` para `metrics`/`preprocessing`/`selection`, a `api` prod usa modelos Pydantic próprios (`MetricsOut`, `SelectionOut`, etc.) derivados de `REQUIRED_METADATA_KEYS`.
- **Campos de entrada sempre `Annotated`** com `StringConstraints` / `Field(ge=, le=)` / `Literal`. Ex.:

  ```python
  class PredictIn(BaseModel):
      text: Annotated[
          str,
          StringConstraints(strip_whitespace=True, min_length=1, max_length=20000),
      ] = Field(description="Free-text medical abstract to classify.")
  ```

- **`model_validator(mode="after")`** para regras cruzadas. Exemplo em `ReloadIn`:

  ```python
  class ReloadIn(BaseModel):
      model_version: Annotated[
          str,
          StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
      ]

      @model_validator(mode="after")
      def _check_version_shape(self) -> Self:
          if not VERSION_PATTERN.fullmatch(self.model_version):
              raise ValueError("model_version must match YYYYMMDDTHHMMSSZ-<12hex>")
          return self
  ```

- **`response_model`** em **toda** rota com retorno de negócio. FastAPI valida o output com Pydantic antes de serializar — se algo escapar do contrato (campo extra, tipo errado), a resposta é **500 internal** com `error_code=response_contract_violation`.

- **`Field(strict=True)`** em campos onde o tipo é explícito (ex.: `score: float | None`). Evita coerção silenciosa de strings para float.

- **`Config.extra = "forbid"`** nos schemas de entrada — campos desconhecidos geram 422 (defesa contra payloads malformados).

### 4.3 Settings (`pydantic-settings`)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_ML_", env_file=".env", extra="forbid")

    api_key: str = Field(min_length=32)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ratelimit_default: str = "60/minute"
    ratelimit_predict: str = "30/minute"
    enable_docs: bool = False  # Swagger desativado em produção
    request_id_header: str = "X-Request-ID"
```

`uvicorn triage_ml.api.app:app` lê as envs e falha rápido se `TRIAGE_ML_API_KEY` não tiver pelo menos 32 caracteres (em vez de aceitar string vazia silenciosamente).

---

## 5. Tarefas e sequência

### T1. Mover schemas para `src/triage_ml/api/schemas.py`

- Reexportar de `dev_api.schemas` durante a transição; remover reexports após um commit.
- Adicionar `MetricsOut`, `SelectionOut`, `PreprocessingOut`, `DependencyVersionsOut` como Pydantic models tipados (substituem `dict[str, Any]` da `dev_api`).
- Manter `PredictIn`/`PredictOut`/`HealthOut`/`ErrorOut` byte-equivalentes aos da `dev_api` (Fase 3 não muda o contrato externo).

### T2. Settings, auth e rate limit

- `src/triage_ml/api/settings.py` com `pydantic-settings`; `.env` ignorado no Docker.
- `src/triage_ml/api/auth.py`: dependência `require_api_key` que lê `X-API-Key`, compara com `hmac.compare_digest(settings.api_key, ...)`, e falha com `401 unauthorized` se faltar/errar.
- `src/triage_ml/api/ratelimit.py`: setup do `slowapi` com dois limites: `default` (60/min por chave) e `predict` (30/min por chave) — configuráveis via env.
- Aplicar `require_api_key` em `/predict` e `/reload` (endpoints que mudam estado); `/health`, `/model-info`, `/models` continuam abertos para que o dashboard e ferramentas de monitoramento funcionem sem chave.

### T3. Middlewares e logging estruturado

- Middleware que **sempre** gera `request_id` (`uuid.uuid4().hex[:12]`) e o expõe em `request.state.request_id`.
- `structlog` configurado para emitir JSON no stdout com `timestamp`, `level`, `request_id`, `route`, `method`, `status`, `latency_ms`. **Nunca** o campo `text`.
- Handler de exceção que mapeia `RequestValidationError`, `HTTPException`, `Exception` em `ErrorOut`, preservando os `error_code` da `dev_api`.

### T4. Factory da aplicação

- `create_app(settings: Settings | None = None, *, holder: ModelHolder | None = None)` para testes herméticos (mesmo padrão da `dev_api`).
- `lifespan` chama `holder.load()` no startup e aborta com `RuntimeError` se o artefato não estiver íntegro.
- `response_model=` em **todas** as rotas de retorno.

### T5. Testes unitários e de integração

| Arquivo | Cobertura |
|---|---|
| `tests/test_api_validation.py` | Cada `Field` rejeita entrada fora do contrato (`max_length`, `min_length`, `pattern`, `Literal`, `extra="forbid"`, `model_validator` cruzado) |
| `tests/test_api_settings.py` | `BaseSettings` carrega envs, falha em env obrigatório, `extra="forbid"` |
| `tests/test_api_auth.py` | Header faltando → 401; chave errada → 401; chave correta → 200; usa `hmac.compare_digest` (timing-safe) |
| `tests/test_api_ratelimit.py` | Estouro por chave e por IP; reset após janela; segundo limite aplicado a `/predict` |
| `tests/test_api_logging.py` | Captura stdout JSON e confirma presença de `request_id` em todas as linhas; ausência de qualquer substring do texto enviado |
| `tests/test_api_health.py` | `/health`, `/model-info`, `/models` (happy path + 503) |
| `tests/test_api_predict.py` | `/predict` happy path + cada `error_code` (validation, language, prediction_failed) |
| `tests/test_api_reload.py` | `/reload` happy + 404 + 500; holder preservado em falha |
| `tests/test_api_docker.py` | Builda a imagem no CI, sobe `docker run`, bate em `/health` |

### T6. Dockerfile + `docker-compose.yml`

- **Multi-stage**: estágio `builder` instala `uv` e roda `uv sync --no-dev`; estágio `runtime` copia `.venv` e o código, roda como usuário não-root, expõe `8000`.
- **CMD** sugerido: `["uvicorn", "triage_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]` (1 worker por enquanto; escalar é decisão da Etapa 8).
- `.dockerignore` ignora `.venv/`, `models/`, `data/`, `reports/evidence/*.json` (mas mantém `reports/figures/*.png`), `__pycache__/`, `.git/`.
- `docker-compose.yml` local traz **só** esta API (`api-prod`). Prometheus/Grafana entram na Etapa 6; não misturar escopos.

### T7. Baseline de latência (gancho para a Etapa 5)

- Script `scripts/benchmark_api.py` envia N requisições a `/predict` com o mesmo payload, mede latência por request com `time.perf_counter`, agrega `p50`/`p95`/`p99`/média, e escreve `reports/benchmarks/api-prod-baseline.json`.
- Documenta hardware, SO, Python, versões (`metadata.dependency_versions`), método de warmup (50 requests descartados antes de medir) e número de repetições (≥ 500).
- Esse JSON é a **referência obrigatória** da Etapa 5: a otimização ONNX tem que mostrar redução ≥ 20% no p95 contra este baseline.

---

## 6. Critérios de aceite (mapeados na Etapa 3 do checklist)

- [x] **Validar contrato de `POST /predict`** com Bill — alinhado com a `dev_api`; este plano mantém compatibilidade.
- [x] **Health check, predição, validação e erros** com base no artefato real, sem stub.
- [x] **Carregamento configurável** via env (`MODEL_PATH` herdado; complementa com `TRIAGE_ML_*` da `Settings`).
- [x] **`latency_ms`, `request_id`, `X-Request-ID` e `Server-Timing`** preservados em todas as rotas.
- [x] **Testes unitários e de integração** verdes (novo alvo: ≥ 120 testes totais somando `dev_api` + `api`).
- [x] **Baseline de latência local** documentado em `reports/benchmarks/api-prod-baseline.json`.
- [x] **Dockerfile multi-stage** + `docker-compose.yml` funcional (parte do entregável do Fábio).

---

## 7. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Divergência de contrato entre `dev_api` e `api` | Schemas ficam em **uma** pasta (`src/triage_ml/api/schemas.py`); a `dev_api` reexporta. CI roda os mesmos testes contra os dois pacotes |
| API Key vazando em log | `tests/test_api_logging.py` varre a string inteira da chave em todos os logs capturados |
| Holder corrompido durante reload concorrente | Reaproveitar `ModelHolder.reload_to` (já existe na `dev_api`); publicar sob `threading.Lock`; um snapshot por request |
| Limite de tamanho do body burlado | `request.body_size_limit` (config do Uvicorn) + `Annotated[str, StringConstraints(max_length=20000)]` em `text` |
| Versão otimizada quebrar o contrato de resposta | `response_model=PredictOut` em todas as rotas; CI testa a saída contra o schema antes de promover |
| `response_model` mascarando bug interno | `Config.extra = "forbid"` nos schemas de saída também; se um campo extra vazar, FastAPI responde 500 com `error_code=response_contract_violation` |
| Workers múltiplos não compartilhando holder | Decisão explícita: **1 worker** na Fase 3; escalar é trabalho da Etapa 8 (cloud) |
| Dockerfile vazar segredos em camadas | Multi-stage com `--no-cache-dir`; `.dockerignore` ignora `.env` e tudo fora do necessário; CI roda `docker history` para auditar |

---

## 8. Como rodar (visão Romário)

```bash
# 1. Subir a versão prod localmente (sem Docker)
export TRIAGE_ML_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
PYTHONPATH=src uv run uvicorn triage_ml.api.app:app --host 0.0.0.0 --port 8000

# 2. Bater nos endpoints
curl -s http://127.0.0.1:8000/health | jq
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: $TRIAGE_ML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Acute myocardial infarction in a 62yo after chest pain."}' | jq

# 3. Subir via Docker
docker compose up --build api-prod

# 4. Rodar a suíte de testes
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

> Os scripts do `dev_api` continuam funcionando porque o **contrato HTTP é o mesmo**. A diferença é auth, rate-limit, settings e os schemas tipados.

---

## 9. Definição de pronto

- Itens da Etapa 3 marcados como `[x]` no checklist com evidência.
- Schemas em `src/triage_ml/api/schemas.py` (fonte canônica) e `dev_api` reexportando.
- `uv run pytest` e `ruff check .` verdes (≥ 120 testes totais).
- Imagem Docker sobe e responde a `GET /health` no smoke test do CI.
- `reports/benchmarks/api-prod-baseline.json` versionado e referenciado na Etapa 5.
- `docs/reports/Etapa_3_API_oficial.md` descrevendo o que foi entregue.

---

## 10. Próximos passos fora deste plano

- **Etapa 5** (Bill, otimização ONNX) — usa `api-prod-baseline.json` como referência.
- **Etapa 6** (Bill, Prometheus/Grafana) — `src/triage_ml/api/metrics.py` já deixa os contadores/histogramas stubados; entra a coleta e o `/metrics`.
- **Etapa 8** (Romário, cloud + vídeo) — Dockerfile desta Fase 3 é a imagem-base do Cloud Run; o `docker-compose.yml` cresce com Prometheus/Grafana e a variante otimizada.

---

## 11. Referências cruzadas

- [`docs/CHECKLIST.md`](../CHECKLIST.md) linha 104 — Etapa 3.
- [`docs/plans/PLAN-text-classifier.md`](./PLAN-text-classifier.md) — Fase 1 + Fase 2 (contexto do modelo).
- [`src/triage_ml/dev_api/`](../../src/triage_ml/dev_api/) — base reaproveitada.
- [`src/triage_ml/models/artifact.py`](../../src/triage_ml/models/artifact.py) — `load_artifact`, `validate_artifact_bundle`, `VERSION_PATTERN`, `REQUIRED_METADATA_KEYS`.
- [`docs/reports/Etapa_2_Modelo_baseline_e_serialização.md`](../reports/Etapa_2_Modelo_baseline_e_serialização.md) — histórico completo da Etapa 2.
- [`docs/reports/RESUMO-APRESENTACAO-FASE-2.md`](../reports/RESUMO-APRESENTACAO-FASE-2.md) — resumo didático da Etapa 2 para apresentar à banca.