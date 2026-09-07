# Relatório de implementação — Etapa 3 (API oficial)

| Campo | Valor |
|---|---|
| Integrante | Romário |
| Etapa do checklist | Etapa 3 — API FastAPI oficial (`docs/CHECKLIST.md`) |
| Período desta entrega | 2026-08-30 a 2026-08-31 (implementação inicial) e 2026-09-07 (revisão cruzada de Bill) |
| Última revisão | 2026-09-07 — endurecimento de RBAC, observabilidade, telemetria de erro, auth em /model-info e /models, fingerprint HMAC e dashboard |
| Status | ✅ API, RBAC, benchmark, portal e empacotamento Docker concluídos localmente |

Este relatório documenta a promoção do contrato da API de desenvolvimento de Bill para uma API oficial. A implementação continua usando o artefato real de ML, mas adiciona autenticação por papel, limite de requisições, rastreabilidade, erros/logs sanitizados, baseline de latência e interface de demonstração clínica.

## 1. Resumo executivo

- A API oficial está em `src/triage_ml/api/` e sobe com Uvicorn na porta 8000.
- Ela reutiliza `ModelHolder`, validação de artefato, detector local de idioma e política da `dev_api`; o fluxo manual carrega `models/<versão>/model.joblib` real, sem resposta fixa ou stub.
- Mantém `GET /health`, `GET /model-info`, `GET /models`, `POST /predict` e `POST /reload`, com schemas Pydantic canônicos em `src/triage_ml/api/schemas.py`.
- RBAC estático mapeia `doctor`, `patient` e `service`: médico prediz, paciente recebe `403` sem classificação clínica e service recarrega modelo.
- `/predict` e `/reload` têm rate limit por IP e fingerprint HMAC-SHA-256 da API key (com sal aleatório de 32 bytes por processo).
- O middleware gera `request_id`, expõe `X-Request-ID` e `Server-Timing: total;dur=<ms>, detect;dur=<ms>, predict;dur=<ms>` (o `total;dur=` é sempre emitido; `detect;` e `predict;` aparecem quando os estágios foram executados), e envia logs JSON sem texto clínico ou chave de API.
- O baseline HTTP registra média `22,18 ms`, p95 `31,88 ms` e p99 `32,80 ms`.
- `front/app_prod.py` demonstra o RBAC com telas distintas de médico e paciente e nunca renderiza o body bruto de erro da API — exibe apenas o `request_id` para correlação.
- `/model-info` e `/models` (que retornam hyperparameters, métricas, label mapping, git commit e versões) **exigem** `X-API-Key` válida com role `service` ou `doctor`; `/health` segue público conforme `docs/plans/PLAN-api-prod.md`.
- A validação local terminou com **152 testes verdes** da suíte completa (`uv run pytest tests/`), dos quais 130 já estavam descritos no relatório original de 2026-08-31 e 22 são novos das revisões cruzadas de 2026-09-07 das Etapas 2 e 3. Lint e formatação verdes.

## 2. Escopo e alinhamento com o plano

Itens concluídos da Etapa 3:

- [x] Contrato de `POST /predict` alinhado à API de desenvolvimento;
- [x] Health, predição, validação e erros sobre o artefato real;
- [x] Carregamento configurável via `MODEL_PATH`;
- [x] `latency_ms`, `request_id`, `X-Request-ID` e `Server-Timing`;
- [x] Testes unitários e de integração;
- [x] Baseline de latência local com metodologia reproduzível.

O portal Streamlit por papel foi uma extensão acordada pelo time para demonstrar RBAC no vídeo STAR. O login local é demonstrativo, não um provedor de identidade para cloud. Docker/Compose/CI de imagem são de Fábio; ONNX e Prometheus/Grafana são de Bill; Airflow é de Denis.

## 3. Arquitetura e reuso

```text
modelo versionado (model.joblib + metadata.json)
                 │
                 ▼
      ModelHolder e validação do artefato
                 │
                 ▼
API oficial FastAPI ── RBAC, rate limit, logs, timing ──► Postman/cliente HTTP
                 │
                 └──────────────────────────────────────► portal Streamlit
                                                            médico / paciente
```

| Aspecto | `triage_ml.dev_api` (Bill) | `triage_ml.api` (Romário) |
|---|---|---|
| Modelo e artefato | Real e validado | Real e validado; mesmo núcleo |
| Contrato | Health, metadata, versões, reload e predict | Compatível com a API de desenvolvimento |
| Idioma | Detector `langid` local | Reutilizado |
| Segurança | Ferramenta de desenvolvimento | API key, RBAC e rate limit |
| Dashboard | `front/app_dev.py` | `front/app_prod.py` |

O dashboard de Bill continua funcionando como QA funcional para contrato, artefato, idioma, metadata, versões e reload. Ele não valida RBAC porque chama a API de desenvolvimento, não a API oficial.

### 3.1 Endpoints e permissões

| Método | Rota | Papel necessário | Uso |
|---|---|---|---|
| `GET` | `/health` | Público | Estado do processo e do modelo. |
| `GET` | `/model-info` | `service` ou `doctor` | Manifesto validado do artefato. Restrito a partir da revisão cruzada de 2026-09-07 porque expunha hyperparameters, métricas, label mapping, git commit e versões de dependência — superfície de reconhecimento útil a atacantes antes de qualquer exploração. |
| `GET` | `/models` | `service` ou `doctor` | Versões íntegras no registry. Mesma justificativa de `/model-info`. |
| `POST` | `/predict` | `doctor` | Predição sobre texto clínico em inglês. |
| `POST` | `/reload` | `service` | Troca para versão de modelo válida. |

O papel `patient` não pode predizer: recebe `403 clinician_review_required` sem `label`, `score` ou texto no body de erro. O RBAC é centralizado nas classes `RequireRole` (genérica) e `RequirePredictRole` (com código específico para `patient` em `triage_ml.api.auth`); o handler de `/predict` não tem mais checagens inline.

## 4. Implementação técnica

### 4.1 Configuração e artefato real

`src/triage_ml/api/app.py` usa `MODEL_PATH` quando definido e cria um `ModelHolder` com esse arquivo. O holder reutilizado valida manifesto, checksum e classes antes de desserializar; assim, a API não depende de CSV em runtime nem devolve dados pré-definidos.

`src/triage_ml/api/settings.py` define `TRIAGE_ML_API_KEY_SERVICE`, `TRIAGE_ML_API_KEY_DOCTOR` e `TRIAGE_ML_API_KEY_PATIENT`. Cada chave exige mínimo de 32 caracteres, e `extra="forbid"` rejeita configuração desconhecida.

### 4.2 Contrato, validação e erros

`src/triage_ml/api/schemas.py` é a fonte canônica de schemas. `PredictIn` rejeita texto ausente, vazio, tipo inválido, campo extra e texto maior que 20.000 caracteres. `PredictOut` expõe somente os campos do contrato:

```json
{
  "label": 1,
  "label_name": "neoplasms",
  "score": null,
  "model_version": "...",
  "latency_ms": 5.4,
  "request_id": "...",
  "warnings": []
}
```

`score: null` é válido para o LinearSVC atual. Exceções HTTP, de validação e internas são convertidas para `ErrorOut`, sem repetir texto clínico ou detalhes internos.

### 4.3 RBAC

`src/triage_ml/api/auth.py` lê `X-API-Key`, compara a chave com `hmac.compare_digest` e determina o papel no servidor:

| Papel | Pode fazer | Não pode fazer |
|---|---|---|
| `doctor` | `POST /predict` | `POST /reload` |
| `patient` | Endpoints públicos | Predizer ou receber classificação clínica |
| `service` | `POST /reload` | `POST /predict` |

O papel service não é “QA”; ele representa automação/operador interno. A decisão de segurança está em `docs/adr/0002-rbac-estatico-api-producao.md`.

### 4.4 Rate limit, privacidade e logs

`src/triage_ml/api/ratelimit.py` cria um limitador por IP e outro por fingerprint da chave. A partir da revisão cruzada de 2026-09-07 o fingerprint usa `HMAC-SHA-256` com um sal aleatório de 32 bytes gerado em `os.urandom` no momento de import do módulo — assim, mesmo um atacante rotacionando o header `X-API-Key` em requisições consecutivas cai em buckets imprevisíveis (rainbow tables pré-computadas ficam inúteis, e para correlacionar buckets seria necessário exfiltrar o sal do processo). O fingerprint carrega prefixo `k:` e 64 hex chars; é hash de mão única, não criptografia reversível. Os padrões são `30/minute` para `/predict` e `60/minute` para `/reload`, configuráveis por ambiente. Exceder o limite retorna `429 Too Many Requests`.

O middleware em `src/triage_ml/api/app.py` gera `request_id` por request, mede a resposta e emite log JSON no stdout via `structlog` (configurado em `logging_config.py`). A configuração de logging inclui os processadores `StackInfoRenderer` e `format_exc_info` antes do `JSONRenderer` para que chamadas a `logger.exception("internal_error")` carreguem o traceback na linha de log, e o setup é idempotente via flag `_logging_configured`. O log contém rota, método, status, latência e tipo de erro, mas não body clínico, API key ou fingerprint. Em cloud, stdout pode ser coletado pelo serviço de logs da plataforma.

O handler de `StarletteHTTPException` filtra `exc.detail` por uma allow-list compartilhada com a `dev_api` (`ALLOWED_ERROR_CODES` em `triage_ml.dev_api.app`, agora também incluindo `unauthorized`, `forbidden` e `clinician_review_required`); qualquer string fora da lista colapsa em `request_failed` para impedir que mensagens internas de exceção vazem no body. A mensagem canônica para erro de validação de payload é `Request body is invalid.` em ambas as APIs, alinhando o contrato entre dev e prod.

| Elemento | Onde aparece | Finalidade |
|---|---|---|
| `latency_ms` | Body de `PredictOut` | Tempo de inferência do pipeline apenas (`pipeline.predict`/`predict_proba`). Detecção de idioma tem slot próprio em `Server-Timing`. Arredondado para 3 casas decimais para coincidir com o header. |
| `request_id` | Body de sucesso/erro (ou `null` se o middleware não rodou) | Correlação de uma chamada. |
| `X-Request-ID` | Header HTTP | Correlação no nível HTTP. |
| `Server-Timing` | Header HTTP | Sempre `total;dur=<ms>`; acresce `detect;dur=<ms>` e/ou `predict;dur=<ms>` quando os estágios rodaram. Padroniza observabilidade entre `/health`, `/model-info`, `/models`, `/predict` e `/reload`. |

## 5. Portal Streamlit de demonstração

`front/app_prod.py` é separado de `front/app_dev.py` e lê a configuração abaixo somente do ambiente:

| Variável | Uso |
|---|---|
| `TRIAGE_ML_PROD_API_URL` | URL da API oficial. |
| `TRIAGE_ML_API_KEY_DOCTOR` | Chave usada server-side pelo Streamlit para `/predict`. |
| `TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME` / `PASSWORD` | Login demonstrativo da área médica. |
| `TRIAGE_ML_DASHBOARD_PATIENT_USERNAME` / `PASSWORD` | Login demonstrativo da área do paciente. |

Médico recebe formulário de predição e resposta real. Paciente não recebe chave médica, não possui formulário e não chama `/predict`. A chave não é exibida na página, URL ou `session_state`. Mesmo se a interface fosse contornada, a API continua aplicando RBAC.

A partir da revisão cruzada de 2026-09-07 o portal **não renderiza mais o body bruto de erro** da API: quando `/predict` retorna não-200, o Streamlit exibe apenas o `request_id` e uma mensagem genérica pedindo contato com a equipe técnica. Isso desacopla o dashboard da sanitização do backend: uma regressão futura em qualquer handler que exponha stack trace, caminho de arquivo ou mensagem interna não vazaria no navegador do médico.

Em cloud, a evolução correta é IdP/OIDC, tokens assinados, claims de papel e cofre de segredos; o login local não deve ser apresentado como autenticação final de produção.

## 6. Benchmark de latência

`scripts/benchmark_api.py` é executado separadamente de Uvicorn para medir a experiência real de um cliente HTTP. A API não roda benchmark automaticamente.

Evidência: `reports/benchmarks/api-prod-baseline.json`.

| Parâmetro | Valor registrado |
|---|---:|
| Cliente | `requests` via HTTP localhost |
| Warmup | 50 requisições descartadas |
| Requisições medidas | 500 |
| Modelo | `20260824T202622Z-f2cb6f23f9cd` |
| Média | `22,18 ms` |
| p50 | `28,89 ms` |
| p95 | `31,88 ms` |
| p99 | `32,80 ms` |

O benchmark foi gerado com `TRIAGE_ML_RATELIMIT_PREDICT=1000/minute` definido no terminal da API antes do startup, evitando bloqueio das 500 chamadas. Rodar o script novamente sobrescreve o JSON; use-o somente para nova evidência.

## 7. Testes e validação

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Resultado de referência local: **152 passed**, 3 deselected (testes e2e que requerem Docker daemon em execução). Warnings conhecidos de `joblib`/NumPy e `TestClient` não são falhas enquanto a suíte termina verde. A revisão cruzada de 2026-09-07 adicionou 4 testes novos nesta Etapa (`test_model_info_requires_authentication`, `test_models_requires_authentication`, `test_rate_limit_identifier_is_stable_for_same_key`, `test_rate_limit_identifier_changes_when_key_changes`) e atualizou `test_rate_limit_identifier_never_returns_api_key` para refletir o fingerprint HMAC.

| Arquivo | Funções `test_*` | Foco |
|---|---:|---|
| `tests/test_api_prod_rbac.py` | 4 | Paciente bloqueado com `clinician_review_required`, ausência de chave e permissões de doctor/service. |
| `tests/test_api_prod_security.py` | 15 | Settings, fingerprint HMAC, estabilidade/variância da fingerprint, rate limit, sanitização, health e metadata. |
| `tests/test_api_container.py` | 3 | Smoke do Dockerfile e da imagem construída em CI. |
| `tests/test_prod_dashboard_helpers.py` | 7 | Login, papéis, chave server-side e chamadas do portal. |
| Demais testes do projeto | 123 | Dados, modelo, artefato, dev API, idioma e dashboard de Bill. |
| **Total coletado** | **152** | Suíte completa (`uv run pytest tests/ -q`). |

Mocks/doubles aparecem apenas nos testes para isolar componentes. Uvicorn e Streamlit em execução manual usam HTTP e artefato reais.

## 8. Como reproduzir a entrega

### 8.1 API oficial e Postman

No primeiro terminal:

```powershell
$env:TRIAGE_ML_API_KEY_SERVICE = "srv-000000000000000000000000000000"
$env:TRIAGE_ML_API_KEY_DOCTOR = "doc-000000000000000000000000000000"
$env:TRIAGE_ML_API_KEY_PATIENT = "pat-000000000000000000000000000000"

$triageModel = Get-ChildItem .\models -Directory |
  Sort-Object Name -Descending |
  Select-Object -First 1
$env:MODEL_PATH = Join-Path $triageModel.FullName "model.joblib"
Test-Path $env:MODEL_PATH

uv run uvicorn triage_ml.api.app:app --host 127.0.0.1 --port 8000
```

No Postman, configure `base_url=http://127.0.0.1:8000` e três variáveis de chave. Teste `/health`, `/model-info`, `/models`, `/predict` com doctor, `/predict` com patient e `/reload` com service. O paciente deve receber 403 sanitizado.

### 8.2 Portal de produção

No segundo terminal:

```powershell
$env:TRIAGE_ML_PROD_API_URL = "http://127.0.0.1:8000"
$env:TRIAGE_ML_API_KEY_DOCTOR = "doc-000000000000000000000000000000"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME = "medico-demo"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD = "senha-medico-demo"
$env:TRIAGE_ML_DASHBOARD_PATIENT_USERNAME = "paciente-demo"
$env:TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD = "senha-paciente-demo"

uv run streamlit run front/app_prod.py
```

Abra `http://localhost:8501`. Faça login como paciente para demonstrar o bloqueio e como médico para executar uma predição real com texto sintético em inglês.

### 8.3 QA com dashboard de Bill

Use portas diferentes para evitar conflito:

```powershell
uv run uvicorn triage_ml.dev_api.app:app --host 127.0.0.1 --port 8001
uv run streamlit run front/app_dev.py --server.port 8502
```

No dashboard de Bill, informe `http://127.0.0.1:8001`. Valide health, manifesto, predição, idioma, versões e reload. Com mesma versão de artefato e mesma entrada, a classe retornada deve ser compatível com a API oficial.

## 9. Mapa de artefatos

```text
src/triage_ml/api/
├── app.py               # endpoints, lifecycle, middleware, handlers de erro, RBAC centralizado
├── auth.py              # API key, RequireRole, RequirePredictRole (RBAC centralizado)
├── settings.py          # settings de ambiente e limites
├── ratelimit.py         # IP e fingerprint HMAC-SHA-256 com sal por processo
├── logging_config.py    # structlog JSON no stdout com format_exc_info e setup idempotente
└── schemas.py           # contrato Pydantic canônico

src/triage_ml/dev_api/
└── app.py               # ALLOWED_ERROR_CODES + LANGUAGE_ERROR_CODES compartilhados com a prod_api

front/app_prod.py                              # portal médico/paciente (não renderiza body bruto de erro)
front/app_dev.py                               # dashboard de Bill preservado
scripts/benchmark_api.py                       # carga e percentis
reports/benchmarks/api-prod-baseline.json      # baseline versionado
tests/test_api_prod_rbac.py                    # autorização por papel
tests/test_api_prod_security.py                # segurança e privacidade (inclui fingerprint HMAC)
tests/test_api_container.py                    # smoke do Dockerfile
tests/test_prod_dashboard_helpers.py           # portal de produção
docs/adr/0002-rbac-estatico-api-producao.md    # decisão de RBAC
```

## 10. Riscos conhecidos e próximos passos

| Tema | Status | Próximo passo/responsável |
|---|---|---|
| Docker, Compose e imagem | Concluído localmente | Evidência em `Etapa_4_CI_CD_Docker.md`; CI remoto aguarda PR verde. |
| Rate limit distribuído | Limitação conhecida | Redis/armazenamento compartilhado em múltiplas instâncias. |
| Login do portal | Demonstração local | IdP, tokens assinados e cofre de segredos em cloud. |
| ONNX | Pendente | Bill compara baseline com variante otimizada. |
| Prometheus/Grafana | Pendente | Bill instrumenta e provisiona observabilidade. |
| Arquitetura cloud | Pendente | Romário registra decisão da Etapa 8. |

## 11. Conclusão

A Etapa 3 entrega uma API oficial funcional sobre o artefato real já treinado, sem duplicar a lógica de ML. A promoção adiciona configuração por ambiente, RBAC, proteção clínica do paciente, rate limit, rastreabilidade, logs sanitizados, testes, benchmark e portal visual. A base está pronta para as próximas etapas de containerização, observabilidade, otimização e cloud.

A revisão cruzada de Bill em 2026-09-07 endureceu a Etapa 3 com cuidado redobrado por se tratar de código de produção com RBAC. Treze pontos foram endereçados sem alterar comportamento observável pelos consumidores externos: (1) o handler genérico agora chama `logger.exception` e o structlog carrega `StackInfoRenderer` + `format_exc_info`; (2) `StarletteHTTPException` filtra `detail` por uma allow-list compartilhada (`ALLOWED_ERROR_CODES` na `dev_api`); (3) a mensagem de `validation_failed` foi padronizada para `Request body is invalid.` em ambas as APIs; (4) o fallback de `request_id` mudou de `"unknown"` para `None`; (5) `latency_ms` no `PredictOut` cobre apenas a inferência do pipeline; (6) `setup_logging` é idempotente; (7) o RBAC de `/predict` foi centralizado em `RequirePredictRole`; (8) `/reload` loga `model_version` no erro; (9) `latency_ms` arredondado para 3 casas; (10) `/model-info` e `/models` agora exigem `RequireRole(["service","doctor"])`; (11) o fingerprint do rate limit usa HMAC-SHA-256 com sal de 32 bytes por processo; (12) `front/app_prod.py` exibe apenas `request_id` em caso de erro; (13) `RequireRole.allowed_roles` é `frozenset`. A suíte passou de 130 para 152 testes verdes com 4 testes novos nesta Etapa.

## 12. Histórico de revisões deste relatório

| Data | Autor | Mudança |
|---|---|---|
| 2026-08-31 | Romário (versão inicial) | Entrega da API FastAPI oficial, RBAC, portal Streamlit, benchmark e Docker local; suíte de 130 testes. |
| 2026-09-07 | Bill (revisão cruzada) | Atualização do estado pós-revisão cruzada: cabeçalho aponta para a revisão de 2026-09-07 e lista os 13 hardenings por nome; §1 cita o fingerprint HMAC, o `Server-Timing` padronizado, o portal sem body bruto de erro, a autenticação de `/model-info` e `/models` e o total de 152 testes; §3.1 torna a tabela de endpoints consistente com o novo RBAC; §4.4 substitui SHA-256 por HMAC-SHA-256 com sal, detalha o structlog e adiciona o parágrafo sobre `ALLOWED_ERROR_CODES` e a mensagem canônica de `validation_failed`; §5 menciona a substituição do `json.dumps(response.body)` no portal; §7 atualiza para 152 testes com tabela de contagens reais; §9 expande o mapa com `RequirePredictRole`, fingerprint HMAC, `format_exc_info`, idempotência do setup de logging, `test_api_container.py` e os códigos compartilhados em `dev_api/app.py`; §11 inclui parágrafo dedicado à revisão cruzada. |
