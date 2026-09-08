# Relatório de implementação — Etapa 7 (Orquestração de retreino)

| Campo | Valor |
|---|---|
| Integrante | Denis |
| Etapa do checklist | Etapa 7 — Orquestração de retreino (`docs/CHECKLIST.md`) |
| Período desta entrega | 2026-09-04 a 2026-09-05 (implementação inicial) e 2026-09-07 (revisão cruzada de Bill) |
| Última revisão | 2026-09-07 — endurecimento de credenciais, atomicidade de publicação, validação alinhada ao `configs/training.yaml`, defesa contra symlink e cobertura de testes |
| Status | ✅ DAG `triage_ml_retraining` implantada, contêiner Airflow 3.1.7 standalone funcional, execução real validada contra a `main` do DagsHub e suíte de testes verde |

Este relatório documenta a entrega do pipeline de retreino orquestrado com Apache Airflow. A DAG `triage_ml_retraining` é disparada manualmente ou pela Etapa 8 (cloud) e realiza ingestão isolada do dataset versionado no DagsHub, validação do contrato de dados, treino/avaliação via `triage_ml.models.train.run_training`, persistência do artefato no caminho versionado do projeto e validação final do manifesto.

## 1. Resumo executivo

- DAG `triage_ml_retraining` declarada em `airflow/dags/triage_retraining.py` com `schedule=None`, `catchup=False`, `max_active_runs=1`, retries com backoff e quatro tasks encadeadas (`ingest` → `validate` → `train` → `verify`).
- `src/triage_ml/orchestration/airflow_pipeline.py` centraliza blocos testáveis: `file_sha256`, `ingest_from_git`, `validate_dataset_file`, `find_reusable_artifact`, `train_evaluate_persist` e `validate_training_output`.
- Ingestão publica o CSV por **substituição atômica** (`shutil.copyfile` para `.<name>.tmp` + `os.replace`) em um diretório temporário de clone, com `GIT_ASKPASS` descartável, `GIT_TERMINAL_PROMPT=0` + `GIT_ASKPASS_REQUIRE=force` e env mínimo (sem herdar segredos do worker).
- Validação reaproveita `triage_ml.data.prepare.prepare_dataset`, lê `sample_size`/`random_state` de `configs/training.yaml` (override explícito honrado), rejeita datasets > 200 MiB e devolve apenas metadados não sensíveis (`dataset_sha256`, contagens, `classes`), sem que `text`/`medical_abstract` trafegue por XCom ou logs.
- Persistência reutiliza o `run_training` canônico (nenhuma duplicação de lógica) e grava `airflow_run.json` com `_atomic_write_json` (tempfile + `os.replace`), recusando symlinks ancestrais via `_ensure_no_symlink_ancestor`.
- Idempotência estrutural: para o mesmo par (`dataset_sha256`, `config_file_sha256`), uma versão íntegra anterior é reutilizada por `find_reusable_artifact`, sem novo treino e com `reused=true` no retorno.
- Credenciais DagsHub nunca trafegam em URL, comando ou variável compartilhada: `_redact_credentials` mascara tanto `scheme://user:token@` quanto `DAGSHUB_USERNAME=...`/`DAGSHUB_USER_TOKEN=...` que vazem para stderr/stdout do `git`.
- Contêiner `airflow/Dockerfile` baseado em `apache/airflow:3.1.7-python3.12`, `git` instalado, `PYTHONPATH=/opt/triage-ml/src`, entrypoint `airflow/entrypoint.sh` validando `TRIAGE_REQUIRE_AUTH=true` + `DAGSHUB_USERNAME` + `DAGSHUB_USER_TOKEN` antes de subir `airflow standalone`.
- Compose overlay `docker-compose.airflow.yml` com healthcheck HTTP `/api/v2/monitor/health`, volumes nomeados para `airflow-state`/logs e bind-mounts somente leitura para `configs` e `dags`.
- Execução real registrada em 2026-09-05 via `airflow dags test triage_ml_retraining 2026-09-05` (DagsHub commit `069dc330e8f5c478a82c893cc224d63734781f6f`), com 11.550 linhas lidas, 7.489 elegíveis, 5.000 preparadas e artefato `20260905T171611Z-f2cb6f23f9cd` (accuracy 0,7520, balanced_accuracy 0,7281, macro_f1 0,7335). Segunda execução confirmada com `reused=true`.
- Suíte completa final: **173 testes verdes**, dos quais 13 são específicos da Etapa 7 (10 em `test_airflow_pipeline.py` + 3 de smoke/estrutura ligados a `docker-compose.airflow.yml` cobertos indiretamente).

## 2. Escopo e alinhamento com o plano

Itens concluídos da Etapa 7:

- [x] Consumir `triage_ml.models.train.run_training` (ou equivalente) em vez de duplicar lógica.
- [x] Implementar ingestão/leitura do CSV.
- [x] Implementar validação/preparação reaproveitando `triage_ml.data.prepare`.
- [x] Implementar treino e avaliação.
- [x] Persistir artefato e metadados no caminho configurável do contrato.
- [x] Garantir configuração portátil e tarefas idempotentes quando possível.
- [x] Testar/importar a DAG sem erros e registrar evidência de execução.
- [x] Suportar retreino disparando a partir da Etapa 8 (cloud) ou manualmente.

O aceite oficial (15%) — DAG funcional realizando ingestão e treino, com modelo salvo no caminho versionado — foi comprovado em 2026-09-05 contra a `main` do DagsHub.

## 3. Arquitetura e reuso

```text
                       ┌─────────────────────────────────────┐
                       │  DagsHub (data repo, branch main)   │
                       └────────────────┬────────────────────┘
                                        │ git clone (depth=1, askpass)
                                        ▼
   ┌───────────────────────┐   ┌─────────────────────────────┐
   │  airflow/Dockerfile   │──▶│ airflow/entrypoint.sh       │
   │  airflow/dags/*.py    │   │  (pre-flight TRIAGE_REQUIRE)│
   └────────────┬──────────┘   └────────────┬────────────────┘
                ▼                          ▼
      ┌──────────────────────────────────────────────┐
      │ DAG triage_ml_retraining (schedule=None)     │
      │  ingest → validate → train → verify          │
      └────────────┬─────────────────────────────────┘
                   ▼
   ┌─────────────────────────────────────────────────────┐
   │ src/triage_ml/orchestration/airflow_pipeline.py     │
   │  ingest_from_git / validate_dataset_file /          │
   │  find_reusable_artifact / train_evaluate_persist /  │
   │  validate_training_output                           │
   └────────────┬────────────────────────────────────────┘
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ triage_ml.data.prepare.prepare_dataset  (reuso)     │
   │ triage_ml.models.train.run_training     (reuso)     │
   │ triage_ml.models.artifact.validate_artifact_bundle  │
   └────────────┬────────────────────────────────────────┘
                ▼
     models/<YYYYMMDDTHHMMSSZ-12hex>/model.joblib
     models/<...>/airflow_run.json + classes.json + metadata.json
```

| Componente | Responsabilidade | Reuso |
|---|---|---|
| `airflow/dags/triage_retraining.py` | Definição da DAG, leitura de env, encadeamento de tasks e validação de credenciais. | Helper próprio `_require_env`/`_require_auth_credentials`; funções de `airflow_pipeline`. |
| `src/triage_ml/orchestration/airflow_pipeline.py` | Blocos testáveis: SHA-256 streaming, redacted git execution, validação alinhada ao YAML, persistência atômica. | `prepare_dataset`, `run_training`, `validate_artifact_bundle`, `VERSION_PATTERN`. |
| `airflow/Dockerfile` | Imagem `apache/airflow:3.1.7-python3.12` com `git`, `pyproject.toml` e `src/` da aplicação. | Reuso das dependências já declaradas em `pyproject.toml`. |
| `airflow/entrypoint.sh` | Falha cedo quando `TRIAGE_REQUIRE_AUTH=true` e faltam `DAGSHUB_USERNAME`/`DAGSHUB_USER_TOKEN`. | Bash mínimo, sem dependências extras. |
| `docker-compose.airflow.yml` | Sobe `airflow standalone` com healthcheck, volumes nomeados e bind-mounts read-only. | `docker-compose.yml` existente; perfis não compartilhados. |

## 4. Contrato da DAG

| Campo | Valor |
|---|---|
| `dag_id` | `triage_ml_retraining` |
| `schedule` | `None` (disparo manual via UI/API ou pela Etapa 8) |
| `catchup` | `False` |
| `max_active_runs` | `1` |
| `retries` | `2` (delay `2 min`) |
| Tasks | `ingest` (10 min) → `validate` (10 min) → `train` (1 h) → `verify` (5 min) |
| Variáveis lidas uma única vez no escopo do `@dag` | `DATA_REPOSITORY_URL` (obrigatória), `DATA_REPOSITORY_BRANCH` (`main`), `DATASET_RELATIVE_PATH` (`data/medical_tc_train.csv`), `TRIAGE_RAW_CSV`, `TRIAGE_TRAINING_CONFIG`, `TRIAGE_MODELS_DIR`, `TRIAGE_REPORTS_DIR`, `TRIAGE_REQUIRE_AUTH` (`false`), `DAGSHUB_USERNAME`, `DAGSHUB_USER_TOKEN` |
| Idempotência | Reuso por hash de dataset + hash de configuração via `find_reusable_artifact` |
| Saídas | `models/<versão>/airflow_run.json` com `{dataset_sha256, config_file_sha256, source_commit}` |

## 5. Endurecimento aplicado na revisão cruzada 2026-09-07

Após a implementação inicial, Bill executou revisão estática da Etapa 7 com cuidado redobrado (alto risco por execução de subprocessos com credenciais, contrato da DAG e publicação atômica) e cross-validação com dois sub-agentes. Os pontos consolidados foram aplicados no mesmo ciclo de revisão cruzada documentado no `docs/CHECKLIST.md`:

1. **Vazamento de segredos via `os.environ.copy()`** — `clone_environment = os.environ.copy()` na ingestão herdava todo o ambiente do worker Airflow (incluindo `DAGSHUB_USER_TOKEN` e demais segredos) para o subprocesso `git`, expondo-os em `/proc/<pid>/environ` e em qualquer filho. Substituído por `_git_environment(askpass, username, token)` que monta um env mínimo com apenas `PATH`, `LC_ALL`, `GIT_TERMINAL_PROMPT=0` e `GIT_ASKPASS_REQUIRE=force`, mais `GIT_ASKPASS`, `DAGSHUB_USERNAME` e `DAGSHUB_USER_TOKEN` quando há credenciais. O dict é limpo em memória (`""` em vez de remover) ao final, preservando a chave para testes que inspecionam o env recebido pelo subprocesso.
2. **`subprocess.run(..., capture_output=True)` engolia stderr do `git`** — erros de branch inexistente, 401 e falha TLS chegavam apenas como `CalledProcessError`. Empacotado em `_run_git` que sanitiza stderr/stdout via `_redact_credentials` (regex `(://)([^/\s:@]+):([^@\s/]+)@`) e re-raise como `RuntimeError` com mensagem útil.
3. **`GIT_TERMINAL_PROMPT=0` insuficiente** — combinado com `GIT_ASKPASS_REQUIRE=force`, que torna o helper obrigatório quando o `git` pedir credenciais.
4. **`validate_dataset_file` hardcodava `sample_size=5_000` e `random_state=42`** — divergia de `configs/training.yaml` consumido por `run_training`. Agora aceita `config_path` opcional e delega a `_load_preparation_settings`, que lê `sample_size`/`random_state` do YAML quando não fornecidos, com overrides explícitos honrados.
5. **`validate_dataset_file` retornava apenas 6 chaves** — descartava `missing_or_empty_rows`, `conflicting_texts`, `conflicting_rows` e `duplicate_rows` do `PreparationReport`. Agora retorna o conjunto completo para visibilidade operacional.
6. **CSV carregado sem limite** — adicionado `MAX_DATASET_BYTES = 200 MiB` antes de `pd.read_csv`, evitando OOM em datasets acidentais.
7. **`find_reusable_artifact` capturava exceções incompletas** — tupla ampliada para `(OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError)`, deixando a DAG falhar alto apenas em erros estruturais verdadeiros.
8. **`find_reusable_artifact` aceitava `models/.trash/...`** — filtros adicionados: `re.fullmatch(VERSION_PATTERN, manifest_path.parent.name)` e limite `MAX_MANIFEST_BYTES = 1 MiB` por manifesto.
9. **`airflow_run.json` sem atomicidade** — `Path.write_text` substituído por `_atomic_write_json` (tempfile + `os.replace`) em `train_evaluate_persist`.
10. **`ingest_from_git` aceitava symlinks ancestrais** — `_ensure_no_symlink_ancestor` recusa a publicação com erro explícito ("refusing to operate through symlink").
11. **`os.environ["DATA_REPOSITORY_URL"]` e `os.getenv(...) or None` mascaravam ausência** — substituídos por `_require_env` (mensagem amigável) e `_require_auth_credentials` que espelham a checagem do `entrypoint.sh` quando `TRIAGE_REQUIRE_AUTH=true`, eliminando bypass quando a DAG roda fora do compose.
12. **Leitura múltipla de variáveis no escopo de cada task** — leitura única de `TRIAGE_*` no escopo do `@dag` factory, garantindo avaliação única na construção da DAG.

Testes adicionados/atualizados: `test_git_subprocess_errors_redact_credentials_in_stderr`, `test_ingestion_refuses_destination_through_symlink`, `test_validate_dataset_file_aligns_with_training_config`, `test_redact_credentials_strips_dagshub_env_var_leaks`. Lint limpo e suíte completa permanece verde.

## 6. Configuração

O arquivo `.env.example` documenta as variáveis. Para a Etapa 7:

```dotenv
DATA_REPOSITORY_URL=https://dagshub.com/deniscelclaro/pos-ml-eng-tech-challenge-fase-03.git
DATA_REPOSITORY_BRANCH=main
DATASET_RELATIVE_PATH=data/medical_tc_train.csv
DAGSHUB_USERNAME=seu-usuario
DAGSHUB_USER_TOKEN=seu-token-de-leitura
TRIAGE_REQUIRE_AUTH=false
```

- `TRIAGE_REQUIRE_AUTH=true` exige `DAGSHUB_USERNAME` e `DAGSHUB_USER_TOKEN` preenchidos tanto no `entrypoint.sh` quanto em `_require_auth_credentials` no escopo da DAG.
- `TRIAGE_RAW_CSV`, `TRIAGE_MODELS_DIR`, `TRIAGE_REPORTS_DIR` e `TRIAGE_TRAINING_CONFIG` têm defaults internos e só precisam ser sobrescritos em ambientes customizados.

## 7. Comandos reproduzíveis

```bash
# Validação estática e testes da DAG
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/test_airflow_pipeline.py -v
uv run pytest                            # suíte completa (173 testes verdes)

# Subir o Airflow standalone e validar a DAG importada
docker compose -f docker-compose.airflow.yml build airflow
docker compose -f docker-compose.airflow.yml up -d --wait airflow
docker compose -f docker-compose.airflow.yml exec airflow airflow dags list
docker compose -f docker-compose.airflow.yml exec airflow airflow dags test \
  triage_ml_retraining 2026-09-05
docker compose -f docker-compose.airflow.yml logs --tail=200 airflow
docker compose -f docker-compose.airflow.yml down
```

## 8. Evidência de execução 2026-09-05

`airflow dags test triage_ml_retraining 2026-09-05`, executado com sucesso via Docker contra a `main` do DagsHub no commit `069dc330e8f5c478a82c893cc224d63734781f6f`.

| Etapa | Resultado |
|---|---|
| `ingest` | clone efêmero em `/tmp/triage-airflow-ingest-XXXX/source/`, CSV publicado em `/opt/triage-ml/data/medical_tc_train.csv` via `os.replace`, `dataset_sha256` calculado em streaming. |
| `validate` | 11.550 linhas lidas, 7.489 elegíveis, 5.000 preparadas, classes `[1, 2, 3, 4, 5]`. |
| `train` | artefato `20260905T171611Z-f2cb6f23f9cd` persistido em `models/20260905T171611Z-f2cb6f23f9cd/`. |
| Métricas | `accuracy=0.7520`, `balanced_accuracy=0.7281`, `macro_f1=0.7335`. |
| `verify` | manifesto validado; `checksum_sha256` consistente com `metadata.json`. |
| Segunda execução | `reused=true`, sem novo treino, confirmando idempotência para o mesmo par `(dataset_sha256, config_file_sha256)`. |

## 9. Validação final pós-revisão cruzada

| Verificação | Resultado |
|---|---|
| `uv run ruff format --check .` | aprovado |
| `uv run ruff check .` | aprovado |
| `uv run pytest tests/` | 173 aprovados (160 anteriores + 13 novos do ciclo 2 e dos ajustes da Etapa 7) |
| `docker compose -f docker-compose.airflow.yml config` | válido |
| DAG importada sem erros | `airflow dags list` inclui `triage_ml_retraining`; `airflow dags list-import-errors` vazio |
| Credenciais em logs | `s3cr3t-token` ausente em stderr sanitizado (assertions em `test_git_subprocess_errors_redact_credentials_in_stderr` e `test_redact_credentials_strips_dagshub_env_var_leaks`) |

A Etapa 7 está concluída e pode ser acionada manualmente pela UI do Airflow ou disparada pela Etapa 8 (cloud) sem mudanças adicionais.
