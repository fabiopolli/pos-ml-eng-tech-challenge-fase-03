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

As três chaves precisam ter pelo menos 32 caracteres:

- `TRIAGE_ML_API_KEY_SERVICE`;
- `TRIAGE_ML_API_KEY_DOCTOR`;
- `TRIAGE_ML_API_KEY_PATIENT`.

Nenhuma chave possui valor padrão. O contêiner não recebe as credenciais do DagsHub nem
outras variáveis presentes no `.env` compartilhado.

## Comandos reproduzíveis

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest
docker compose config --quiet
docker compose build api-prod portal-prod dashboard-dev
docker compose up -d --wait api-prod portal-prod dashboard-dev
docker compose ps
docker compose down
```

## Validação remota

O PR #5 foi integrado à `main` após o workflow `CI` nº 39 concluir com `success`.
O run executou qualidade, testes Playwright e build/auditoria das imagens de API,
portal e dashboard:

`https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-03/actions/runs/33995734506`

## Revisão cruzada 2026-09-07

Após o aceite oficial, Bill executou revisão estática cruzada com cuidado redobrado
(alto risco por empacotamento de produção, segredos e contratos CI) e cross-validação
com dois sub-agentes. Os pontos consolidados foram aplicados em commit direto à `main`:

- `dashboard-dev` desacoplado de `api-prod` (perfil `dev` dedicado, chaves
  `TRIAGE_ML_DEV_API_KEY_*`), evitando deploys cruzados acidentais;
- `fake_api.py` usa `hmac.compare_digest` em vez de comparação não constant-time;
- job `front-e2e` em `ci.yml` ganhou `trap cleanup EXIT` e `set -euo pipefail`,
  encerrando `uvicorn`/`streamlit` mesmo em falha de teste;
- loop de readiness corrigido (`seq 1 30` + ordem correta do `sleep`);
- smoke-test do job `container` valida `app.openapi()['info']['title']` e a presença
  de `/predict` em vez de comparar string literal;
- `Dockerfile` builder copia `.python-version` para reproducibilidade cruzada;
- Airflow ganhou `entrypoint.sh` registrado como `ENTRYPOINT` e flag
  `TRIAGE_REQUIRE_AUTH` no compose para falhar cedo quando faltarem credenciais
  DagsHub;
- testes estruturais (`test_api_container.py`, `test_repository_structure.py`)
  ajustados para refletir o novo isolamento do dashboard técnico e incluir os novos
  arquivos `airflow/Dockerfile`/`airflow/entrypoint.sh`.

A suíte completa (152 testes) permanece verde após as mudanças.
