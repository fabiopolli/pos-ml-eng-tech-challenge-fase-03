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
- Compose com `api-prod`, `portal-prod` e `dashboard-dev`, todos com healthcheck,
  usuário não-root e filesystem somente leitura;
- dashboard técnico configurável por ambiente para acessar a API e enviar as chaves
  de médico/serviço somente no processo servidor.

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
| Pytest | 140 aprovados; 1 teste de symlink desconsiderado no Windows |

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

O resultado remoto do novo job `container` deve ser registrado no checklist somente depois
que o workflow do pull request terminar verde.
