# Etapa 4 — CI/CD, Docker e testes

Responsável: Fábio Polli.

## Escopo entregue

- imagens multi-stage da API oficial e dos dois fronts Streamlit em Python 3.12;
- dependências reproduzidas a partir de `uv.lock`;
- versões de NumPy, SciPy, scikit-learn e joblib alinhadas ao manifesto do modelo;
- processo Uvicorn com um worker e usuário sem privilégios (`uid=10001`);
- healthcheck HTTP sem dependência de `curl`;
- modelo montado em `/models` somente para leitura;
- filesystem somente leitura, `/tmp` limitado, capabilities removidas e
  `no-new-privileges` no Compose;
- chaves obrigatórias fornecidas apenas em runtime;
- job de CI para construir a imagem e importar a aplicação ASGI;
- testes automatizados das proteções estruturais do Dockerfile e Compose.
- Compose com `api-prod`, `portal-prod` e `dashboard-dev` (em `profiles: [dev]`), todos
  com healthcheck, usuário não-root e filesystem somente leitura;
- dashboard técnico configurável por ambiente, isolado em perfil `dev` para não
  interferir no stack mínimo de produção e consumir chaves dedicadas
  (`TRIAGE_ML_DEV_API_KEY_*`) em vez das chaves de produção;
- entrypoint `airflow/entrypoint.sh` que valida `TRIAGE_REQUIRE_AUTH=true` antes de
  iniciar o `airflow standalone`, evitando execuções silenciosas sem credenciais
  DagsHub quando o repositório é privado.

Prometheus e Grafana não fazem parte desta fatia. Eles serão integrados pelo responsável
pela observabilidade, estendendo o Compose existente.

## Validação local

Executada em 2026-09-05 com o artefato
`20260905T171611Z-f2cb6f23f9cd`, produzido pelo fluxo real do Airflow.

| Verificação | Resultado |
|---|---|
| Build multi-stage | sucesso |
| Tamanho das imagens | API 288.764.136 bytes; cada front 288.805.684 bytes |
| Healthcheck Docker | API, portal e dashboard `healthy` |
| Usuário do processo | `uid=10001` |
| `GET /health` | `status=ok`, modelo carregado |
| `GET /model-info` | mesma versão do `/health` |
| `POST /predict` como patient | `403 clinician_review_required` |
| `POST /predict` como doctor | sucesso com modelo real |
| `X-Request-ID` e `Server-Timing` | presentes |
| Portal por papel | HTTP 200 em `localhost:8501` |
| Dashboard técnico | HTTP 200 em `localhost:8502` |
| Ruff | aprovado |
| Pytest | 152 aprovados (140 iniciais + 12 da revisão cruzada 2026-09-07) |

O teste desconsiderado exige privilégio de criação de symlink no Windows e não representa
falha observada no código. A suíte completa permanece habilitada no Linux do CI.

## Configuração

O arquivo `.env.example` documenta as variáveis. No Compose, `API_MODEL_PATH` deve apontar
para `/models/<versao>/model.joblib`, enquanto o diretório local `./models` é montado em
`/models:ro`.

As três chaves da API de produção precisam ter pelo menos 32 caracteres:

- `TRIAGE_ML_API_KEY_SERVICE`;
- `TRIAGE_ML_API_KEY_DOCTOR`;
- `TRIAGE_ML_API_KEY_PATIENT`.

O dashboard técnico (`dashboard-dev`, em `profiles: [dev]`) usa chaves dedicadas para
não tocar nas chaves de produção:

- `TRIAGE_ML_DEV_API_URL` (default `http://api-dev:8000`);
- `TRIAGE_ML_DEV_API_KEY_DOCTOR`;
- `TRIAGE_ML_DEV_API_KEY_SERVICE`.

Nenhuma chave possui valor padrão. O contêiner da API não recebe as credenciais do
DagsHub nem outras variáveis presentes no `.env` compartilhado. No Airflow, a flag
`TRIAGE_REQUIRE_AUTH` (default `false`) habilita a checagem de
`DAGSHUB_USERNAME`/`DAGSHUB_USER_TOKEN` no entrypoint antes de subir o serviço.

## Comandos reproduzíveis

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest
docker compose config --quiet
# Stack mínimo de produção: api-prod + portal-prod
docker compose build api-prod portal-prod
docker compose up -d --wait api-prod portal-prod
docker compose ps
docker compose down
# Dashboard técnico isolado em perfil "dev" (não sobe junto do stack mínimo)
docker compose --profile dev build dashboard-dev
docker compose --profile dev up -d --wait dashboard-dev
docker compose --profile dev down
# Airflow (compose overlay) — exige TRIAGE_REQUIRE_AUTH=true quando o
# repositório de dados for privado.
docker compose -f docker-compose.airflow.yml build airflow
docker compose -f docker-compose.airflow.yml up -d --wait airflow
docker compose -f docker-compose.airflow.yml down
```

## Validação remota

O PR #5 foi integrado à `main` após o workflow `CI` nº 39 concluir com `success`.
O run executou qualidade, testes Playwright e build/auditoria das imagens de API,
portal e dashboard:

`https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-03/actions/runs/33995734506`

## Revisão cruzada 2026-09-07

Após o aceite oficial, Bill executou revisão estática cruzada com cuidado redobrado
(alto risco por empacotamento de produção, segredos e contratos CI) e cross-validação
com dois sub-agentes. Os pontos consolidados foram aplicados no commit `0e21dae`
direto à `main`:

- `dashboard-dev` desacoplado de `api-prod` (perfil `dev` dedicado, chaves
  `TRIAGE_ML_DEV_API_KEY_*`), evitando deploys cruzados acidentais;
- `fake_api.py` usa `hmac.compare_digest` em vez de comparação não constant-time;
- job `front-e2e` em `ci.yml` ganhou `set -euo pipefail`, captura explícita dos
  PIDs de `uvicorn`/`streamlit`, função `cleanup` e `trap cleanup EXIT` para
  encerrar processos em background mesmo em falha de teste;
- loop de readiness corrigido (`seq 1 30` com a checagem de "última tentativa"
  depois do `sleep 1`);
- smoke-test do job `container` valida `app.openapi()['info']['title']` e a presença
  de `/predict` em vez de comparar string literal;
- `Dockerfile` builder copia `.python-version` para reproducibilidade cruzada;
- Airflow ganhou `entrypoint.sh` registrado como `ENTRYPOINT` e flag
  `TRIAGE_REQUIRE_AUTH` (default `false`) no `docker-compose.airflow.yml` para
  falhar cedo quando faltarem credenciais DagsHub;
- testes estruturais (`test_api_container.py`, `test_repository_structure.py`)
  ajustados para refletir o novo isolamento do dashboard técnico e exigir a
  presença dos novos arquivos `airflow/Dockerfile` e `airflow/entrypoint.sh`;
- `src/triage_ml/api/app.py` reformado pelo `ruff format` (sem mudança de
  comportamento).

A suíte completa (152 testes, 3 e2e desativados por padrão) permanece verde após
as mudanças; `ruff check` e `ruff format --check` aprovados.
