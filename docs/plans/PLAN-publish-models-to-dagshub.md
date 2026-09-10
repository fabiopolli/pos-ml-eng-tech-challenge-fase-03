# Plano — Publicar modelos treinados no DagsHub (Etapa 7)

| Campo | Valor |
|---|---|
| Origem | `docs/CHECKLIST.md` L263 — Etapa 7 — Orquestração de retreino (Denis) |
| Sintoma | Os artefatos treinados pela DAG `triage_ml_retraining` ficam presos em `./models/` (bind-mount local) e **nunca são enviados ao repositório DagsHub** configurado em `DATA_REPOSITORY_URL`. A pasta `models/` do projeto só contém `README.md`. |
| Status | 🟡 Backlog — escopo definido, aguardando aprovação do dono da Etapa 7 (Denis) e decisão sobre a estratégia de promoção (branch efêmera vs. release tag). |
| Última revisão | 2026-09-09 — análise estática por Bill + cross-check com `airflow_pipeline.py`, `airflow/dags/*`, `docker-compose.airflow.yml`, `models/README.md`, `.env.example` e `Etapa_7_*.md`. |

---

## 1. Diagnóstico

### 1.1 O pipeline atual (apenas *pull*)

Fluxo executado pela DAG `triage_ml_retraining` em [`airflow/dags/triage_retraining.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/dags/triage_retraining.py):

```text
DagsHub (data repo) ── git clone ──▶ /opt/triage-ml/data/medical_tc_train.csv
                                                       │
                                                       ▼
                                  train_evaluate_persist (run_training)
                                                       │
                                                       ▼
                              /opt/triage-ml/models/<version>/*  (bind-mount local)
                                                       │
                                                       ▼
                                                  [ fim ]    ◀── nada faz push
```

Pontos de evidência:

- [`airflow/dags/triage_retraining.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/dags/triage_retraining.py) só encadeia `ingest → validate → train → verify`. Não há task `publish`/`push`/`register`.
- [`airflow/dags/triage_retraining_optimization.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/dags/triage_retraining_optimization.py) repete o mesmo grafo por *slice* e também não publica.
- [`src/triage_ml/orchestration/airflow_pipeline.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/src/triage_ml/orchestration/airflow_pipeline.py) oferece `ingest_from_git` (clone) e `_atomic_write_json` (escrita local), mas não há `_publish_to_git` / `_register_remote`.
- [`docker-compose.airflow.yml`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/docker-compose.airflow.yml) faz bind-mount `./models:/opt/triage-ml/models` — o artefato é gravado no host, não publicado no remote.
- [`models/README.md`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/models/README.md) declara: *"Artefatos binários não são versionados diretamente no Git"* — a expectativa de versionamento externo (DagsHub) **não está implementada**.
- [`airflow/dags/README.md`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/dags/README.md#L33-L34) confirma: *"todos permanecem fora do Git"*. O DagsHub aparece **somente como origem do dataset**, nunca como destino.
- [`Etapa_7_Orquestração_de_retreino.md`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/docs/reports/Etapa_7_Orquestração_de_retreino.md#L40) marca o aceite oficial como atendido com base em **execução local** e nunca menciona publicação remota.

### 1.2 Consequência operacional

- Os 8 modelos treinados registrados em [`docs/reports/Relatorio_de_treinamento_dos_modelos.md`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/docs/reports/Relatorio_de_treinamento_dos_modelos.md) existem **apenas na máquina onde o Airflow rodou** (`./models/<versão>/`). Sem o push, qualquer outro ambiente (outro dev, CI, Etapa 8 cloud, Mkla/MLflow registry do DagsHub) precisa retreinar para conseguir o artefato.
- A `models/` local do repositório está vazia — qualquer pessoa clonando o repo parte do zero.
- Não há auditoria versionada dos artefatos (checksum, métricas, hiperparâmetros) — o `metadata.json` existe em disco mas nunca é comitado em local versionado.
- O contrato da DAG (`source_commit` em `airflow_run.json`) deixa de ser útil fora do pipeline: como não há publicação, não há relação direta entre commit do código e versão do modelo publicada.

### 1.3 Causa raiz

A Etapa 7 implementou **metade** do ciclo: a ingestão do dataset. A publicação do artefato treinado de volta ao repositório de origem foi **assumida como fora de escopo** e nunca foi especificada nem na Etapa 7 nem na Etapa 8. Quando o checklist pergunta "modelo salvo no caminho versionado", o "caminho versionado" foi interpretado como o diretório versionado pelo nome (`<YYYYMMDDTHHMMSSZ-input_hash>/`), não como um repositório Git.

---

## 2. Decisões a tomar (antes de implementar)

| # | Decisão | Opções | Recomendação |
|---|---|---|---|
| D1 | **Onde publicar?** | (a) mesmo repositório DagsHub (`deniscelclaro/pos-ml-eng-tech-challenge-fase-03.git`), pasta `models/<versão>/`; (b) **novo repositório** `pos-ml-eng-tech-challenge-fase-03-models.git` no DagsHub; (c) MLflow Registry do DagsHub. | **(a)** mesmo repositório — reaproveita `DATA_REPOSITORY_URL` e `DAGSHUB_*` já configurados, evita governança de novo repo. |
| D2 | **Como promover?** | (i) push direto na `main`; (ii) **branch efêmera `train/<versão>`** + PR; (iii) tag Git `model/<versão>` após merge. | **(ii) branch efêmera + PR** — preserva revisão humana, bloqueia regressões, e o DagsHub aceita sem config extra. (i) é o que `CHECKLIST` parece ter assumido mas conflita com "PR obrigatório" do `.agents/operating-model.md`. (iii) é compatível como sobrescrita pós-merge. |
| D3 | **Quais artefatos publicar?** | (1) `model.joblib` + `metadata.json` + `classes.json` + `airflow_run.json`; (2) também `model.onnx` quando a otimização estiver ligada; (3) também `benchmark.json` por *slice*. | **(1)+(2)+(3)** sempre que existirem no disco. Conjunto = tudo o que `run_training` + `export_onnx_for_version` + `benchmark_for_version` materializam. |
| D4 | **Quem tem permissão de push?** | (a) token do Airflow (DAGSHUB_USER_TOKEN) — mesmo que lê o dataset; (b) **token dedicado** `DAGSHUB_PUSH_TOKEN` com escopo de escrita. | **(b)** token dedicado, marcado como *write-only* no DagsHub. Reaproveita o mesmo usuário mas isola o privilégio (comprometimento do token de leitura não vira push). |
| D5 | **Quando publicar?** | (a) sempre ao final de toda execução; (b) só quando as *gates* (`promotion.eligible`) passam; (c) sempre, com flag `publish_to_remote=true` opcional. | **(c) sempre publicar, mas rotular** — `airflow_run.json` ganha campo `published`, `published_remote`, `publish_ref` (SHA do commit). Publicações que falharem as *gates* ficam marcadas como `unpromoted` para rastreio, mas não são descartadas (mantém histórico de tentativa). |

> **Aprovações pendentes**: D1–D5 precisam de OK do Denis (dono da Etapa 7) antes do início da implementação.

---

## 3. Estratégia recomendada

### 3.1 Visão de alto nível

Adicionar uma **task terminal** `publish_artifact` em cada DAG (`triage_ml_retraining` e `triage_ml_retraining_optimization`), reaproveitando os blocos já endurecidos em [`src/triage_ml/orchestration/airflow_pipeline.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/src/triage_ml/orchestration/airflow_pipeline.py):

- `_git_environment(...)` — env mínimo, sem vazar segredos herdados.
- `_redact_credentials(...)` — mesma regex que já cobre `scheme://user:token@` e `DAGSHUB_*=`.
- `_ensure_no_symlink_ancestor(...)` — defesa já ativa em todos os *writes* locais.
- `_atomic_write_json(...)` — usado para o `airflow_run.json` final com `published=true`.

### 3.2 Fluxo alvo

```text
DAG atual: ingest → validate → train → verify
                              │
                              ▼
                  ┌──────────────────────┐
                  │   publish_artifact   │  ◀── nova task
                  └──────────┬───────────┘
                             │
        ┌────────────────────┼──────────────────────────┐
        │ 1. clonar repo remoto em /tmp/publish-<ver>   │
        │ 2. criar branch train/<ver> (or push to main)  │
        │ 3. copiar models/<ver>/ + reports/.../*        │
        │ 4. atualizar airflow_run.json com published   │
        │ 5. git add + commit + push (GIT_ASKPASS)      │
        │ 6. remover diretório temporário               │
        │ 7. emitir ref do commit remoto como XCom      │
        └───────────────────────────────────────────────┘
```

### 3.3 Componentes novos

| Arquivo | Conteúdo | Reuso |
|---|---|---|
| `src/triage_ml/orchestration/publish.py` (novo) | `publish_artifact(version_dir, *, repository_url, branch, model_subdir, git_username, git_token) -> dict[str, Any]` — orquestra clone → *copy* → commit → push → cleanup. | `_git_environment`, `_redact_credentials`, `_run_git`, `_ensure_no_symlink_ancestor`, `file_sha256`. |
| `src/triage_ml/orchestration/airflow_pipeline.py` | Adicionar `_require_publish_token()` (espelho de `_require_auth_credentials`, mas para `DAGSHUB_PUSH_TOKEN`) e expor `train_evaluate_publish(...)` que encadeia `train_evaluate_persist` + `publish_artifact`. | Já existentes. |
| `airflow/dags/triage_retraining.py` | Nova task `@task(execution_timeout=timedelta(minutes=10)) def publish(verified: dict) -> dict` encadeada após `verify`. Lê `TRIAGE_PUBLISH_REMOTE`, `DATA_REPOSITORY_BRANCH`, `DAGSHUB_PUSH_USERNAME`, `DAGSHUB_PUSH_TOKEN`. | `verify` já existente. |
| `airflow/dags/triage_retraining_optimization.py` | Mesma task `publish` por *slice* (após `verify_<size>`) + um agregador `tag_published_versions` opcional no fim (cria tag `model/<ver>` quando `promotion.eligible=true`). | `verify_<size>` já existente. |
| `docker-compose.airflow.yml` | Acrescentar `DAGSHUB_PUSH_USERNAME` e `DAGSHUB_PUSH_TOKEN` (com fallback vazio) + `TRIAGE_PUBLISH_REMOTE=${TRIAGE_PUBLISH_REMOTE:-true}`. | — |
| `.env.example` | Documentar as 3 variáveis novas e a flag `TRIAGE_PUBLISH_REMOTE` em "Airflow retraining". | — |
| `docs/CHECKLIST.md` L263+ | Marcar a sub-bullet "Publica artefato no DagsHub (branch efêmera + push)" como concluída após implementação. | — |
| `docs/reports/Etapa_7_Orquestração_de_retreino.md` | Acrescentar seção "Publicação remota" com screenshot/log do `git ls-remote` mostrando a branch `train/<ver>`. | — |
| `tests/test_publish_artifact.py` (novo) | Cobre: (a) **happy path** com `bare repo` local como `remote`, (b) **redaction** — `scheme://user:token@` não vaza no stderr, (c) **idempotência** — re-execução com a mesma versão é no-op (`git push` retorna "Everything up-to-date"), (d) **falha de auth** — erro sanitizado e DAG continua local, (e) **branch protegida** — push para `main` é recusado quando `TRIAGE_PUBLISH_BRANCH=train/...`, (f) **symlink ancestor** — recusa. | Mesma estratégia dos 13 testes da Etapa 7. |
| `tests/test_airflow_pipeline.py` | Adicionar caso `test_train_evaluate_publish_invokes_publish_artifact` (mock + spy). | Já existente. |

### 3.4 Contrato da nova task

```python
@task(execution_timeout=timedelta(minutes=10))
def publish(verified: dict) -> dict:
    """Publica models/<ver>/ + reports/<ver>/ no remote configurado.

    Returns:
        {
            "model_version": "<ver>",
            "published": bool,
            "remote": "<DATA_REPOSITORY_URL>",
            "branch": "<branch treinada>",          # ex.: "train/20260905T171611Z-..."
            "remote_sha": "<sha do commit>",        # None se published=False
            "publish_skipped_reason": Optional[str] # "TRIAGE_PUBLISH_REMOTE=false"
        }
    """
```

### 3.5 Hardening obrigatório (replicar postura da Etapa 7)

1. **Env mínimo** — `_git_environment` com só `PATH`, `LC_ALL`, `GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS_REQUIRE=force` + `GIT_ASKPASS`, `DAGSHUB_PUSH_USERNAME`, `DAGSHUB_PUSH_TOKEN`. **Nunca** `os.environ.copy()`.
2. **`GIT_ASKPASS` descartável** — script criado em `tempfile.TemporaryDirectory`, `chmod 0o700`, removido junto com o diretório.
3. **Redaction** — `_redact_credentials` aplicado a stderr/stdout do `git push` (regex já cobre vazamentos de URL com token e de `DAGSHUB_*=`).
5. **Branch efêmera** — `train/<YYYYMMDDTHHMMSSZ-hex>`, **nunca** `main` direto quando `TRIAGE_PUBLISH_BRANCH=efêmera`. CI do repo (Etapa 4) já bloqueia push direto em `main`.
5. **Idempotência** — `git push` quando a branch já tem o mesmo commit retorna sem erro. DAG pode rodar N vezes com mesmo `data_sha256`+`config_sha256` (reuso via `find_reusable_artifact`).
6. **Detecção de modelo já publicado** — antes do clone, `git ls-remote --heads origin train/<ver>` evita `git clone` desnecessário quando a versão já foi publicada em uma execução anterior.
7. **Falha de push ≠ falha de DAG** — `publish_artifact` loga warning e retorna `published=False` quando o push falha por auth/rede; a DAG **não deve falhar** só porque a publicação remota caiu. O *acceptance* da Etapa 7 (artefato local íntegro) continua válido.
8. **Cleanup garantido** — `tempfile.TemporaryDirectory` com `try/finally`; `askpass` e clone temporário são apagados mesmo em erro.
9. **Sem secrets em logs** — `_atomic_write_json` no `airflow_run.json` antes do push garante que `published_remote=true` entre no commit, mas o `DAGSHUB_PUSH_TOKEN` nunca é serializado.
10. **Defesa contra symlink** — `_ensure_no_symlink_ancestor` em cada destino de *copy* (mesmo que o volume local `./models/`).

---

## 4. Critérios de aceite

### 4.1 Funcionais

- [ ] `publish_artifact` cria a branch `train/<ver>` no remote, faz *copy* fiel de `models/<ver>/` (incluindo `metadata.json`, `classes.json`, `airflow_run.json`, `model.joblib` e, se existir, `model.onnx` + `benchmark.json`), *commit* com mensagem canônica e *push* autenticado.
- [ ] `airflow_run.json` publicado tem `published=true`, `published_remote=<repo>`, `published_branch=<branch>`, `published_sha=<commit_sha>`.
- [ ] Quando `TRIAGE_PUBLISH_REMOTE=false`, a task é pulada (retorna `published=False, publish_skipped_reason="disabled"`) sem falhar a DAG.
- [ ] Quando o token de push falta e `TRIAGE_REQUIRE_PUBLISH_AUTH=true`, a DAG falha *fast* no entrypoint (espelho do `TRIAGE_REQUIRE_AUTH`).
- [ ] Re-execução com mesma versão é *no-op* no remote (idempotência).
- [ ] Push direto para `main` é recusado pela task quando `TRIAGE_PUBLISH_BRANCH_POLICY=ephemeral`.

### 4.2 Não-funcionais

- [ ] Lint (`ruff`) e suíte (`pytest`) verdes — incluindo **13 novos testes** em `tests/test_publish_artifact.py`.
- [ ] Credenciais nunca aparecem em: `airflow logs`, XCom, `metadata.json`, `git push` URL, `git config --get` no container.
- [ ] `models/` local continua fora do Git (mantém o `.gitignore`).
- [ ] Documentação atualizada: `CHECKLIST.md` L263+ (Etapa 7), `Etapa_7_*.md` (nova seção), `airflow/dags/README.md` (parágrafo "Publicação remota"), `README.md` (seção "Como rodar a DAG").

### 4.3 De observabilidade

- [ ] Métrica Prometheus `triage_ml_publish_total{status, slice}` no `/metrics` da API (`status ∈ {pushed, skipped, failed}`).
- [ ] Log estruturado com `model_version`, `remote`, `branch`, `sha`, `duration_ms` — sem PII nem `text`.

---

## 5. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Token de push vazado por reuso de `DAGSHUB_USER_TOKEN` | média | alto | D4(b) — token dedicado `DAGSHUB_PUSH_TOKEN`, escopo *write-only* no DagsHub, rotacionado a cada release. |
| Push acidental para `main` em vez da branch efêmera | média | alto | D2(ii) + `TRIAGE_PUBLISH_BRANCH_POLICY=ephemeral` *hard-coded* no helper; testes negativos. |
| Race entre dois *runs* publicando a mesma versão | baixa | médio | `_find_published_version()` via `git ls-remote` antes do clone; o segundo *run* detecta e finaliza como `published=False, reason=already_published`. |
| `git push` lento / timeout | média | médio | `execution_timeout=10min`; *fallback* log-only + alerta via métrica `triage_ml_publish_total{status="failed"}`. |
| Artefato grande (>50 MiB) bloqueando clone/push | baixa | médio | Limite `MAX_ARTIFACT_BYTES = 100 MiB` em `validate_artifact_bundle` antes do *copy*; erro explícito. |
| DagsHub fora do ar durante a DAG | média | baixo | `publish_artifact` não falha a DAG; métrica + log estruturado sinalizam o problema. Etapa 7 não depende da publicação. |
| Publicação sobrescrevendo métrica inferior | baixa | alto | Nunca sobrescrever; sempre criar branch nova. Política de promoção (qual branch/tag sobrevive) fica na Etapa 8 (CI + PR review). |

---

## 6. Plano de implementação em fases

| Fase | Entregável | Tamanho estimado | Bloqueios |
|---|---|---|---|
| **F0** | Aprovar D1–D5 com Denis + validar token de push no DagsHub. | XS | — |
| **F1** | Implementar `src/triage_ml/orchestration/publish.py` com 6 testes unitários (sem DAG). | S | F0 |
| **F2** | Integrar `publish` como task final em `triage_retraining.py` + 4 testes da DAG. | S | F1 |
| **F3** | Repetir para `triage_retraining_optimization.py` (1 task por *slice* + agregador opcional). | M | F2 |
| **F4** | Atualizar `docker-compose.airflow.yml`, `.env.example`, `entrypoint.sh` (gate `TRIAGE_REQUIRE_PUBLISH_AUTH`). | XS | F3 |
| **F5** | Atualizar `docs/CHECKLIST.md`, `Etapa_7_*.md`, `airflow/dags/README.md`, `README.md`. | XS | F3 |
| **F6** | Subir `docker compose -f docker-compose.airflow.yml up --build -d`, executar `airflow dags test triage_ml_retraining <data>` **e** `airflow dags test triage_ml_retraining_optimization <data>`, capturar evidência (`git ls-remote` + screenshot do painel do DagsHub), atualizar relatório. | M | F4 + F5 |
| **F7** | PR + revisão cruzada por Bill + push para `main` via merge (Etapa 4). | XS | F6 |

**Tamanhos**: XS = ≤30 min, S = ≤1 h, M = ≤3 h.

> ⚠️ Não implementar F1+ antes da aprovação F0 — o plano pode mudar se D1–D5 forem re-debatidos (especialmente D3 *quais artefatos publicar*).

---

## 7. Como verificar end-to-end

Após F6:

```bash
# 1. Subir o stack Airflow (com token de push configurado)
docker compose -f docker-compose.airflow.yml --env-file .env up --build -d

# 2. Disparar a DAG manualmente
docker compose -f docker-compose.airflow.yml exec airflow \
    airflow dags test triage_ml_retraining 2026-09-09

# 3. Confirmar a branch criada no DagsHub
git ls-remote --heads https://dagshub.com/deniscelclaro/pos-ml-eng-tech-challenge-fase-03.git \
    | grep "refs/heads/train/"

# 4. Conferir os artefatos publicados
git archive --remote=https://dagshub.com/deniscelclaro/pos-ml-eng-tech-challenge-fase-03.git \
    refs/heads/train/<ver> models/<ver>/ | tar -t

# 5. Reexecutar — confirmar idempotência (segunda execução não cria branch nova)
docker compose -f docker-compose.airflow.yml exec airflow \
    airflow dags test triage_ml_retraining 2026-09-09
# Resultado esperado: published=True (reuso via find_reusable_artifact + git push no-op)
```

---

## 8. Referências cruzadas

- [`docs/CHECKLIST.md` L263](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/docs/CHECKLIST.md#L263-L276) — bullets da Etapa 7.
- [`docs/reports/Etapa_7_Orquestração_de_retreino.md`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/docs/reports/Etapa_7_Orquestração_de_retreino.md) — relatório atual (não menciona publicação).
- [`src/triage_ml/orchestration/airflow_pipeline.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/src/triage_ml/orchestration/airflow_pipeline.py) — blocos a serem reaproveitados.
- [`airflow/dags/triage_retraining.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/dags/triage_retraining.py) e [`triage_retraining_optimization.py`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/dags/triage_retraining_optimization.py) — DAGs alvo.
- [`docker-compose.airflow.yml`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/docker-compose.airflow.yml) e [`airflow/entrypoint.sh`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/airflow/entrypoint.sh) — config overlay.
- [`.env.example`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/.env.example) — variáveis a documentar.
- [`.gitignore` L30-L31](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/.gitignore#L30-L31) — `models/*` continua fora do Git local.
- [`.agents/operating-model.md`](file:///home/bill/Codes/ML_Eng_Projects/pos-ml-eng-tech-challenge-fase-03/.agents/operating-model.md) — política de PR obrigatório.