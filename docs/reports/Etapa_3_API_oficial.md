# Relatório de implementação — Etapa 3 (API oficial)

| Campo | Valor |
|---|---|
| Integrante | Romário |
| Etapa do checklist | Etapa 3 — API FastAPI oficial (`docs/CHECKLIST.md`) |
| Período desta entrega | 2026-08-30 a 2026-08-31 |
| Última revisão | 2026-08-31 — relatório técnico e evidências de validação |
| Status | ✅ API, RBAC, benchmark, portal e empacotamento Docker concluídos localmente |

Este relatório documenta a promoção do contrato da API de desenvolvimento de Bill para uma API oficial. A implementação continua usando o artefato real de ML, mas adiciona autenticação por papel, limite de requisições, rastreabilidade, erros/logs sanitizados, baseline de latência e interface de demonstração clínica.

## 1. Resumo executivo

- A API oficial está em `src/triage_ml/api/` e sobe com Uvicorn na porta 8000.
- Ela reutiliza `ModelHolder`, validação de artefato, detector local de idioma e política da `dev_api`; o fluxo manual carrega `models/<versão>/model.joblib` real, sem resposta fixa ou stub.
- Mantém `GET /health`, `GET /model-info`, `GET /models`, `POST /predict` e `POST /reload`, com schemas Pydantic canônicos em `src/triage_ml/api/schemas.py`.
- RBAC estático mapeia `doctor`, `patient` e `service`: médico prediz, paciente recebe `403` sem classificação clínica e service recarrega modelo.
- `/predict` e `/reload` têm rate limit por IP e fingerprint SHA-256 da API key.
- O middleware gera `request_id`, expõe `X-Request-ID` e `Server-Timing`, e envia logs JSON sem texto clínico ou chave de API.
- O baseline HTTP registra média `22,18 ms`, p95 `31,88 ms` e p99 `32,80 ms`.
- `front/app_prod.py` demonstra o RBAC com telas distintas de médico e paciente.
- A validação local terminou com **130 testes verdes**, lint e formatação verdes.

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
| `GET` | `/model-info` | Público | Manifesto validado do artefato. |
| `GET` | `/models` | Público | Versões íntegras no registry. |
| `POST` | `/predict` | `doctor` | Predição sobre texto clínico em inglês. |
| `POST` | `/reload` | `service` | Troca para versão de modelo válida. |

O papel `patient` não pode predizer: recebe `403 clinician_review_required` sem `label`, `score` ou texto no body de erro.

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

`src/triage_ml/api/ratelimit.py` cria um limitador por IP e outro por fingerprint SHA-256 da chave. Fingerprint é hash de mão única, não criptografia reversível. Os padrões são `30/minute` para `/predict` e `60/minute` para `/reload`, configuráveis por ambiente. Exceder o limite retorna `429 Too Many Requests`.

O middleware em `src/triage_ml/api/app.py` gera `request_id`, mede a resposta e emite log JSON no stdout. O log contém rota, método, status, latência e tipo de erro, mas não body clínico ou API key. Em cloud, stdout pode ser coletado pelo serviço de logs da plataforma.

| Elemento | Onde aparece | Finalidade |
|---|---|---|
| `latency_ms` | Body de `PredictOut` | Tempo interno da predição. |
| `request_id` | Body de sucesso/erro | Correlação de uma chamada. |
| `X-Request-ID` | Header HTTP | Correlação no nível HTTP. |
| `Server-Timing` | Header HTTP | Tempo de idioma e predição, quando aplicável. |

## 5. Portal Streamlit de demonstração

`front/app_prod.py` é separado de `front/app_dev.py` e lê a configuração abaixo somente do ambiente:

| Variável | Uso |
|---|---|
| `TRIAGE_ML_PROD_API_URL` | URL da API oficial. |
| `TRIAGE_ML_API_KEY_DOCTOR` | Chave usada server-side pelo Streamlit para `/predict`. |
| `TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME` / `PASSWORD` | Login demonstrativo da área médica. |
| `TRIAGE_ML_DASHBOARD_PATIENT_USERNAME` / `PASSWORD` | Login demonstrativo da área do paciente. |

Médico recebe formulário de predição e resposta real. Paciente não recebe chave médica, não possui formulário e não chama `/predict`. A chave não é exibida na página, URL ou `session_state`. Mesmo se a interface fosse contornada, a API continua aplicando RBAC.

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

Resultado de referência local: **130 passed**. Warnings conhecidos de `joblib`/NumPy e `TestClient` não são falhas enquanto a suíte termina verde.

| Arquivo | Quantidade | Foco |
|---|---:|---|
| `tests/test_api_prod_rbac.py` | 4 | Paciente bloqueado, ausência de chave e permissões de doctor/service. |
| `tests/test_api_prod_security.py` | 14 | Settings, fingerprint, rate limit, sanitização, health e metadata. |
| `tests/test_prod_dashboard_helpers.py` | 10 | Login, papéis, chave server-side e chamadas do portal. |
| Demais testes do projeto | 102 | Dados, modelo, artefato, dev API, idioma e dashboard de Bill. |
| **Total** | **130** | Suíte completa. |

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
├── app.py               # endpoints, lifecycle, middleware e erros
├── auth.py              # API key, papel e RequireRole
├── settings.py          # settings de ambiente e limites
├── ratelimit.py         # IP e fingerprint de chave
├── logging_config.py    # structlog JSON no stdout
└── schemas.py           # contrato Pydantic canônico

front/app_prod.py                              # portal médico/paciente
front/app_dev.py                               # dashboard de Bill preservado
scripts/benchmark_api.py                       # carga e percentis
reports/benchmarks/api-prod-baseline.json      # baseline versionado
tests/test_api_prod_rbac.py                    # autorização por papel
tests/test_api_prod_security.py                # segurança e privacidade
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
