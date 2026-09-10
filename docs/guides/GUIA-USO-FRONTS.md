# Guia de uso dos fronts

Este guia permite reproduzir a demonstração dos dois dashboards Streamlit. Use
somente textos clínicos sintéticos: os fronts não são prontuários e não devem
receber dados reais de pacientes.

## Qual front usar

| Front | Público | Objetivo |
|---|---|---|
| `front/app_prod.py` | banca, médico e paciente demonstrativos | apresentar login por papel, RBAC e inferência protegida |
| `front/app_dev.py` | desenvolvedor ou avaliador técnico | inspecionar saúde, modelo, versões, reload e política de idioma |

Os fronts não são redundantes. O portal de produção demonstra a experiência por
papel; o dashboard de desenvolvimento expõe controles técnicos que não devem ser
oferecidos ao paciente.

## Pré-requisitos de configuração

Antes de subir qualquer front, exporte (no shell) ou preencha (no `.env`)
todas as variáveis abaixo. Os nomes seguem `TRIAGE_ML_*` para evitar colisão
com variáveis de outros projetos.

### Para o portal (`app_prod.py`)

| Variável | Obrigatória? | Origem |
|---|---|---|
| `TRIAGE_ML_PROD_API_URL` | sim | endpoint da API oficial (`http://127.0.0.1:8000` local; `http://api-prod:8000` no compose) |
| `TRIAGE_ML_API_KEY_DOCTOR` | sim | mesma chave configurada em `TRIAGE_ML_API_KEY_DOCTOR` da API oficial (não é a senha do médico) |
| `TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME` | sim | usuário local do portal para login médico |
| `TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD` | sim | senha local do portal para login médico |
| `TRIAGE_ML_DASHBOARD_PATIENT_USERNAME` | sim | usuário local do portal para login paciente |
| `TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD` | sim | senha local do portal para login paciente |
| `TRIAGE_ML_E2E_MODE` | opcional | defina como `true` somente em testes E2E para permitir URLs `127.0.0.1`/`localhost`. Em produção real, mantenha desligada para bloquear loopback. |

### Para o dashboard técnico (`app_dev.py`)

| Variável | Obrigatória? | Origem |
|---|---|---|
| `TRIAGE_ML_DEV_API_URL` | sim | endpoint da API (oficial ou dev) |
| `TRIAGE_ML_DEV_API_KEY_DOCTOR` | sim | chave de médico da API alvo |
| `TRIAGE_ML_DEV_API_KEY_SERVICE` | sim | chave de **service** da API alvo — necessária para `GET /model-info` e `GET /models`, que exigem papel `service` ou `doctor`. Sem essa chave, o dashboard fica 401 nesses endpoints. |

> **Importante:** mesmo contra a API oficial, o dashboard técnico envia as
> chaves via header `X-API-Key`. A dev API (`src/triage_ml/dev_api/app.py`)
> ignora o header; a oficial (`src/triage_ml/api/app.py`) exige.

## Subindo via Docker Compose

O `docker-compose.yml` declara `api-prod`, `portal-prod`, `api-dev` e
`dashboard-dev`. Os dois últimos estão no profile `dev` — use `--profile dev`
sempre que subir `dashboard-dev` ou `api-dev`:

```powershell
docker compose --profile dev --env-file .env up -d --build --wait api-prod portal-prod dashboard-dev
```

Pontos de atenção:

- `dashboard-dev` aponta para `TRIAGE_ML_DEV_API_URL` (que tem default
  `http://api-dev:8000` e pode ser sobrescrito no `.env`). Se quiser que o
  dashboard converse com `api-prod`, ajuste a env no `.env` (por exemplo,
  `TRIAGE_ML_DEV_API_URL=http://api-prod:8000`).
- Para evitar conflito de portas com outros projetos, sobrescreva `API_PORT`,
  `PORTAL_PORT` e `DEV_DASHBOARD_PORT` no `.env` (defaults: 8000, 8501 e
  8502, respectivamente).
- `--wait` bloqueia até que o `healthcheck` de `api-prod` retorne
  `status=ok`, evitando race conditions em que o portal tenta login antes
  da API estar pronta.

Acesse:

- Portal por papel: `http://localhost:8501` (ou a porta definida em
  `PORTAL_PORT`).
- Dashboard técnico: `http://localhost:8502` (ou `DEV_DASHBOARD_PORT`).
- Swagger da API: `http://localhost:8000/docs` (ou `API_PORT`).

Para encerrar: `docker compose --profile dev down`.

## Subindo via processos locais

Suba a API oficial em `http://127.0.0.1:8000` com o modelo e as chaves
descritos no README principal. Em outro terminal, configure o portal:

```powershell
$env:TRIAGE_ML_PROD_API_URL = "http://127.0.0.1:8000"
$env:TRIAGE_ML_API_KEY_DOCTOR = "<mesma-chave-da-api-em-TRIAGE_ML_API_KEY_DOCTOR>"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME = "medico-demo"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD = "<senha-local>"
$env:TRIAGE_ML_DASHBOARD_PATIENT_USERNAME = "paciente-demo"
$env:TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD = "<outra-senha-local>"
uv run streamlit run front/app_prod.py --server.port 8501
```

Em um terceiro terminal, suba o painel técnico:

```powershell
$env:TRIAGE_ML_DEV_API_URL = "http://127.0.0.1:8000"
$env:TRIAGE_ML_DEV_API_KEY_DOCTOR = "<chave-doctor-da-api>"
$env:TRIAGE_ML_DEV_API_KEY_SERVICE = "<chave-service-da-api>"
uv run streamlit run front/app_dev.py --server.port 8502
```

- Portal por papel: `http://localhost:8501`
- Dashboard técnico: `http://localhost:8502`

## Endpoints consumidos pelos fronts

A tabela abaixo lista exatamente o que cada front chama, com os contratos
correspondentes. Os endpoints vivem na API oficial
(`src/triage_ml/api/app.py`) e na dev API
(`src/triage_ml/dev_api/app.py`); ambos compartilham contrato.

| Método | Path | Quem chama | RBAC |
|---|---|---|---|
| `GET` | `/health` | ambos | público (sem `X-API-Key`) — retorna 200 `{status: "ok", model_loaded: true}` ou 503 `{status: "degraded", model_loaded: false}` enquanto o modelo não está pronto |
| `GET` | `/model-info` | `app_dev.py` | `service` ou `doctor` (chave `service` é enviada pelo dashboard técnico) |
| `GET` | `/models` | `app_dev.py` | `service` ou `doctor` |
| `POST` | `/predict` | `app_prod.py` (papel médico) e `app_dev.py` (aba Predição) | `doctor` para uso em produção; `dev_api` aceita qualquer payload sem chave |
| `POST` | `/reload` | `app_dev.py` (aba Trocar modelo) | `service` — body `{"model_version": "<versão>"}`; 200/404 (`model_not_found`)/422 (`validation_failed`) |

> **Atenção ao campo do body de `/reload`:** o schema aceita apenas
> `model_version` (não `version`). Versões inexistentes retornam 404
> `model_not_found`; payloads sem o campo retornam 422 `validation_failed`.

## Casos de uso do portal por papel

### Paciente

1. Entre com a credencial de paciente.
2. Percorra as abas **Entenda o processo**, **Revisão médica** e **Próximos passos**.
3. Confirme que não existe formulário de predição nem resultado do modelo.
4. Mostre que a área informa a necessidade de revisão profissional.
5. Use **Sair** e confirme o retorno ao login.

O que este cenário comprova: negação por padrão, separação de papéis e ausência
de diagnóstico, classe ou score na experiência do paciente.

### Médico

1. Entre com a credencial médica.
2. Confirme que a API está saudável e que o modelo foi carregado. Se `/health`
   responder 503 com `status="degraded"`, o painel mostra uma mensagem
   informativa — aguarde alguns segundos e tente de novo.
3. Envie este caso sintético em inglês:

```text
A 62-year-old patient presents with persistent chest pain, shortness of breath,
fatigue, and a history of hypertension. Clinical evaluation is recommended.
```

4. Mostre categoria, score (margem do LinearSVC, **não** probabilidade),
   versão do modelo, latência e request ID.
5. Destaque o aviso de que a saída apoia a triagem e não constitui diagnóstico.
6. Finalize com **Sair**.

O que este cenário comprova: somente o processo Streamlit autenticado como
médico usa a chave server-side para chamar `POST /predict`.

> **Sobre o `score` mostrado no portal:** a API retorna
> `decision_function` do LinearSVC, que é uma margem assinada (pode ser
> negativa). O portal exibe o valor como `+0.1234`/`-0.1950`. **Não é**
> uma probabilidade calibrada e **não** deve ser apresentada como
> percentual ao usuário final.

## Casos de uso do dashboard técnico

1. Em **Health**, confirme `status`, versão e carregamento do modelo. Lembre
   que 503 + `status="degraded"` significa modelo ainda carregando
   (boot pós-deploy ou reload em andamento), não falha permanente.
2. Em **Predição**, reutilize o texto sintético e inspecione a resposta
   completa, incluindo `detected_language`, `score` (margem do LinearSVC)
   e cabeçalhos `Server-Timing`/`X-Request-ID` para rastreabilidade.
3. Em **Política de idioma**, execute os presets de texto curto, português e
   inglês válido; confira o status HTTP e o `error_code` esperado
   (`text_too_short_for_language_check`, `unsupported_language` ou 200).
4. Na sidebar **Modelo**, apresente manifesto, métricas e mapeamento das
   classes consumindo `/model-info`.
5. Em **Trocar modelo**, apenas no ambiente local, liste versões
   (`GET /models`) e demonstre o reload de um artefato íntegro via
   `POST /reload` com body `{"model_version": "<versão>"}`.

O que este cenário comprova: contrato HTTP, validações, rastreabilidade do
artefato e comportamento técnico reproduzível. Ele não representa uma interface
destinada ao paciente.

## Observabilidade (Grafana + Prometheus)

Com os containers de overlay rodando (`docker compose --profile observability
up -d --wait`), há um dashboard Grafana provisionado automaticamente em
`http://localhost:3000`:

- Credenciais padrão: `admin` / `grafana-test-password-12345`
  (configuradas via `GF_SECURITY_ADMIN_*` no `.env.overlay`).
- Dashboard: **Triage ML — Optimization & Observability** com painéis para
  requests por rota/status, latência p95, taxa de erro de predição e
  comparação baseline vs otimizado.
- Fonte de dados: Prometheus em `http://localhost:9090` raspando
  `/metrics` das APIs `api-sklearn` e `api-onnx`.

## Roteiro curto para o vídeo

1. API saudável e modelo carregado.
2. Login do paciente, proteção clínica e logout.
3. Login do médico, predição sintética e metadados da resposta.
4. Dashboard técnico: health, modelo e política de idioma.
5. Encerrar explicando que RBAC e os dois fronts são extensões demonstrativas do
   projeto, enquanto a inferência, o modelo e o pipeline são os entregáveis centrais.

Nunca mostre senhas, chaves, `.env`, tokens do DagsHub ou dados clínicos reais na
gravação.
