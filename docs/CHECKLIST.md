# Checklist do Tech Challenge — Fase 3

Fonte canônica do progresso. Legenda: `[ ]` pendente, `[~]` em andamento/parcial, `[x]` concluído. Um item só fica concluído quando seu critério de aceite possui evidência verificável.

Última atualização: 2026-09-12 — proposta de arquitetura GCP registrada no ADR 0004; aguarda revisão arquitetural antes do provisionamento.

## Visão geral e responsáveis

- [x] Repositório e arquitetura inicial — Fábio
- [x] EDA e escolha do dataset — Denis
- [x] Classificador de texto (baseline) — Bill
- [x] API FastAPI — Romário
- [x] CI/CD, Docker e testes — Fábio
- [x] DAG Airflow — Denis
- [ ] Otimização de latência e observabilidade — Bill
- [~] Arquitetura em nuvem — Romário (proposta registrada; revisão pendente)
- [~] Documentação detalhada — Fábio
- [ ] Vídeo STAR — Romário

> Mudança 2026-08-23: o item "Classificador de texto" agora descreve apenas o baseline. A otimização do modelo aparece dentro de "Otimização de latência e observabilidade", alinhada à Etapa 5 do novo plano. Mantemos dois itens no checklist por refletir a divisão de pesos da banca.
>
> Atualização 2026-08-23 (revisão de Fase 1): a Etapa 2 foi endurecida com `metadata.schema_version`, versionamento imutável `YYYYMMDDTHHMMSSZ-<input_hash>`, validação de manifesto + checksum antes da desserialização e seleção entre LogisticRegression/LinearSVC por validação cruzada estratificada de 5 folds somente no treino. LinearSVC foi o vencedor (`0.7335` vs `0.7319` mean macro-F1). Artefatos em `models/v1/` legados não passam mais na validação — cada novo treino cria uma versão imutável.
>
> Atualização 2026-08-23 (idioma da API): a `/predict` ganhou checagem de idioma com `langid` rodando localmente. Textos com menos de 20 caracteres são rejeitados como `text_too_short_for_language_check`; detecções com confiança baixa ou idioma fora do allow-list `{"en"}` são rejeitadas como `indeterminate_language` ou `unsupported_language`. A política vive em `configs/api.yaml`; o body de erro carrega `detected_language` e `detected_language_score`, mas nunca o `text`.
>
> Atualização 2026-08-23 (dashboard de desenvolvimento): `front/app_dev.py` é um dashboard Streamlit opcional para exercitar `/health`, `/model-info` e `/predict` manualmente. Fala HTTP contra qualquer URL configurada (local, container ou cloud). Tem 3 abas (Health, Predição, Política de idioma) com validação automática do `error_code` nos cenários canônicos, e uma seção fixa na sidebar chamada **🧠 Modelo** que consome `GET /model-info` e mostra identidade do artefato carregado, métricas de treino (n_train/n_test/random_state/git/created_at), seleção do classificador (logreg × linear_svc com mean ± std do cross-validation) e métricas globais/per-classe (accuracy, balanced_accuracy, macro_f1, weighted_f1, precision/recall/F1/support). Não substitui Prometheus/Grafana para produção — é ferramenta de validação do desenvolvedor.
>
> Atualização 2026-08-23 (`GET /model-info`): a API de desenvolvimento passou a expor `GET /model-info` retornando o manifesto validado do artefato (`metadata.json` validado por `validate_metadata`). O contrato é descrito por `ModelInfoOut` em `src/triage_ml/dev_api/schemas.py` e inclui model_version/model_name/task_type/language/classes/label_mapping/random_state/n_train/n_test/metrics/preprocessing/selection/dependency_versions/git_commit/git_dirty/created_at. Quando o artefato não está carregado, retorna `503 model_not_ready`.
>
> Atualização 2026-08-23 (`GET /models` + `POST /reload`): o model picker do dashboard agora é end-to-end. `GET /models` lista as versões imutáveis disponíveis em `models/` (newest-first) + a atualmente em uso; `POST /reload {"model_version": "..."}` troca o holder da API após re-validar manifesto + checksum (`ReloadIn`/`ReloadOut` em `src/triage_ml/dev_api/schemas.py`, `ModelHolder.reload_to` em `src/triage_ml/dev_api/app.py`). Erros: `404 model_not_found` para versão inexistente e `500 model_incompatible` para falha de validação; o holder anterior permanece em uso. `ModelHolder.reload_to` é unitário (não toca o filesystem do chamador) e o dashboard guarda o estado só em memória — não há persistência entre sessões Streamlit.
>
> Atualização 2026-08-23 (limpeza da sidebar): a seção "📚 Atalhos" e as constantes `DOC_PLAN`/`DOC_CHECKLIST`/`DOC_REPORT_FASE_1` foram removidas do `front/app_dev.py`. O acesso ao Plan/Checklist/Relatório continua via Git/GitHub, e o teste `test_documentation_shortcuts_point_to_existing_files` foi excluído.
>
> Atualização 2026-08-23 (robustez): o score de idioma agora vem de `LanguageIdentifier(norm_probs=True)`, sem exponenciar o score bruto não normalizado. A configuração falha no startup quando inválida ou incompatível com o idioma do modelo. O treino publica via staging + rename atômico; `/models` omite artefatos incompletos/symlinks; reload e predição usam snapshot consistente do holder.
>
> Atualização 2026-08-23 (auditoria integral): o CSV bruto e o PDF de terceiros foram
> removidos do estado atual do Git, com referências externas preservadas. Preparação e
> split agora rejeitam coerções ambíguas e equivalências textuais com leakage; o loader
> valida o bundle completo e a versão de scikit-learn antes de desserializar. O projeto
> passou a gerar wheel/sdist e a CI verifica lockfile, formatação, lint, testes e build.

## Requisitos transversais

- [x] Estrutura versionada separando dados, código, modelos, Airflow, observabilidade, infraestrutura, testes e documentos.
- [x] Workflow de branches, commits, PRs, revisão e gates documentado.
- [x] Dados e modelos grandes excluídos do Git.
- [ ] Histórico de commits semântico e organizado durante todo o projeto.
- [ ] Instruções finais de execução reproduzidas em máquina limpa.
- [x] Licença/origem do dataset documentada.
- [ ] Nenhum segredo ou dado clínico sensível versionado ou emitido em logs.
- [ ] Ativar proteção da `main` após o primeiro CI verde: PR, checks, uma aprovação e bloqueio de force-push.

> Mudança 2026-08-23: o requisito "nenhum segredo/dado clínico em logs" ganhou dono implícito (Bill, na Etapa 6 — observabilidade) e critério verificável (teste automatizado que varre labels Prometheus, payloads de erro e formato de logs).

> Atualização 2026-09-07 (revisão cruzada da Etapa 1): Bill executou revisão estática da Etapa 1 e encontrou quatro pontos. (1) `notebooks/01_eda.ipynb` tinha um `df.head(3)` na célula de inspeção cujos `outputs` persistidos embutiam 3 abstracts clínicos no HTML do notebook — substituído por `print` agregado e `outputs` zerados. (2) `docs/adr/0001-escolha-recorte-dataset.md` foi criado sintetizando o que já estava em `docs/dataset.md` e no relatório da Etapa 1. (3) O checkbox "Contrato de Airflow/artefato" da linha 73 foi marcado como concluído porque o gate já foi satisfeito pela Etapa 7 (DAG configurável e idempotente). (4) A faixa `target ∈ {1..5}` foi explicitada em `.agents/contracts/README.md` (contrato de dados) e em `docs/dataset.md` (schema e labels), alinhando a documentação ao que `prepare.py` já enforçava. Lint e os 16 testes de `tests/test_data_preparation.py` continuam verdes.

## Etapa 1 — Fundação, dados e contratos

### Dataset e EDA — Denis

- [x] Comparar Medical Abstracts TC Corpus e MIMIC-III Open Access.
- [x] Confirmar licença, proveniência, schema e condições de uso.
- [x] Selecionar recorte entre 2.000 e 5.000 amostras (mínimo oficial: 2.000).
- [x] Documentar distribuição, duplicatas, ausências, comprimento dos textos e balanceamento.
- [x] Definir `text`, `target`, labels e estratégia de split sem leakage.
- [x] Registrar decisão e evidências; manter dados fora do Git.

Aceite: notebook/relatório reprodutível, dataset escolhido e contrato de dados aprovado.

### Contratos compartilhados — todos (gate humano)

- [x] Contrato de dados definido em `.agents/contracts/README.md`.
- [x] Contrato de modelo (versão, classes, métricas, preprocessing) e serialização segundo contrato.
- [x] Contrato de API inicial proposto; sujeito à validação de Romário antes de promover.
- [x] Contrato de Airflow/artefato (caminho configurável, idempotência) revisado por Denis antes da Etapa 7. Coberto pela seção "Artefatos e Airflow" de `.agents/contracts/README.md` e validado na Etapa 7 (DAG `triage_ml_retraining` com `TRIAGE_RAW_CSV`/`TRIAGE_MODELS_DIR`/`TRIAGE_REPORTS_DIR`/`TRIAGE_TRAINING_CONFIG` por `os.getenv`, `max_active_runs=1` e evidência de `reused=true` na segunda execução registrada na Etapa 7).

Aceite: contratos em `.agents/contracts/README.md` estáveis antes do início da Etapa 2.

### Arquitetura em nuvem — Romário (ADR pode começar em paralelo)

- [x] Definir direção: real-time para inferência e batch para treino/re-treino.
- [x] Comparar opções e validar a proposta GCP com Cloud Run, Artifact Registry e Cloud Storage.
- [x] Definir Cloud Composer/Airflow como orquestrador e Cloud Run Job como executor batch de treino.
- [x] Avaliar segurança, privacidade, disponibilidade, escala e custos.
- [x] Registrar [ADR 0004](adr/0004-arquitetura-cloud-gcp.md) e sintetizar a proposta no [README](../README.md).

**Evidência (2026-09-12):** o ADR 0004 propõe Cloud Run para portal/API,
Artifact Registry para imagens por digest, Cloud Storage para bundles imutáveis,
Secret Manager e identidades de serviço com privilégio mínimo, e Cloud Composer
mais Cloud Run Job para retreino batch. A decisão não declara deploy existente;
ela requer revisão arquitetural de Fábio e dos consumidores Denis/Bill antes de
provisionamento ou de marcar a arquitetura como concluída.

Aceite oficial (parte da Etapa 8): decisão arquitetural textual clara e coerente com batch versus real-time.

## Etapa 2 — Modelo baseline e serialização (Bill)

### Modelagem baseline — Bill

- [x] Criar baseline leve TF-IDF; comparar LogisticRegression e LinearSVC somente no treino por macro-F1 estratificado e selecionar LinearSVC antes do teste.
- [x] Seeds, preprocessing, fingerprints e versões fixas (seed 42, versões em `metadata.json`, SHA-256 de input/dataset/splits/config/modelo).
- [x] Métricas por classe e agregadas, com figuras em `reports/figures/08_confusion_matrix_linear_svc.png` e `08_top_features_linear_svc.png`.
- [x] Modelo e manifesto canônico serializados em diretório imutável, com schema/classes/mapping/checksum validados antes da desserialização.
- [x] API de desenvolvimento (`/health` + `/model-info` + `/models` + `/reload` + `/predict`) em `src/triage_ml/dev_api/`, consumindo o artefato real treinado, com erros sanitizados, `latency_ms`, `request_id` e headers. `/reload` permite trocar o holder em runtime re-validando manifesto + checksum.
- [x] Checagem de idioma na `/predict` via `langid` (allow-list `{"en"}`, rejeitando texto curto, score baixo e idioma não suportado, com `detected_language`/`detected_language_score` no body de erro).
- [x] Dashboard de desenvolvimento `front/app_dev.py` (Streamlit) para exercitar `/health`, `/model-info`, `/models`, `/reload` e `/predict` manualmente, com cenários canônicos da política de idioma e model picker na sidebar.

**Evidência (recorte preparado 5.000, split 80/20):** `n_train=4000`, `n_test=1000`, `accuracy=0.7460`, `balanced_accuracy=0.7221`, `macro_f1=0.7296`, `weighted_f1=0.7438`. Seleção: LinearSVC `0.7335` contra LR `0.7319` em macro-F1 CV. Resumo local em `models/20260823T135811Z-bed2194376bc/summary.json`; evidência versionável em `reports/evidence/api-dev.json` (inclui `models`, `reload_success` e `reload_not_found` do model picker).

Aceite parcial (soma com Etapa 5 para fechar 20% do item oficial): modelo NLP funcional, otimização bem-sucedida e melhoria demonstrada. A otimização em si entra na Etapa 5.

> Atualização 2026-09-07 (revisão cruzada da Etapa 2): Bill executou revisão estática da Etapa 2 e encontrou cinco pontos. (1) `front/app_dev.py` lia `metrics.get("overall")` no sidebar — chave que nunca existiu no manifesto; substituído por leitura direta das métricas no nível raiz de `metrics`. (2) `load_artifact` validava apenas `expected_params ⊆ actual_params`, sem detectar o caso simétrico (parâmetro declarado no manifesto mas ausente do `joblib`); adicionado erro `ArtifactCompatibilityError("missing declared parameter")` e teste novo. (3) `/predict` levantava `HTTPException(detail="language_config_incompatible")` mas o handler de Starlette caía no fallback `"request_failed"`, perdendo telemetria de drift de configuração; código adicionado ao `allowed_codes` e teste novo. (4) `validate_metadata` aceitava `std_macro_f1 > 1` (assimetria com `mean_macro_f1` que tem limite ≤1); adicionado limite superior simétrico e teste novo. (5) `triage_ml.models.__init__` só exportava `build_pipeline`, exigindo import paths longos; re-exportados `ArtifactCompatibilityError`, `ArtifactIntegrityError`, `ArtifactPaths`, `build_metadata`, `load_artifact`, `validate_artifact_bundle`, `validate_metadata`, `build_classifier`, `VALID_CLASSIFIERS`, `DEFAULT_TFIDF`, `DEFAULT_LOGREG`, `DEFAULT_LINEAR_SVC`. Lint limpo e 88 testes verdes (84 anteriores + 4 novos).

## Etapa 3 — API oficial servindo o modelo (Romário)

### API FastAPI — Romário

- [X] Validar contrato de `POST /predict` (já alinhado com a API de desenvolvimento de Bill).
- [X] Implementar health check, predição, validação e erros com base no artefato real, sem stub.
- [X] Carregar artefato do modelo de forma configurável (env `MODEL_PATH`).
- [X] Manter `latency_ms`, `request_id`, `X-Request-ID` e `Server-Timing` herdados da Etapa 2.
- [X] Adicionar testes unitários e de integração.
- [X] Medir baseline de latência local com metodologia documentada (gancho para a Etapa 5).
- [x] Empacotar o serviço em Docker (parte do entregável do Fábio, mas dirigido a esta API).
- [x] Demonstrar login por papel e jornada do paciente sem expor diagnóstico, classe ou score.

**Evidência (2026-08-30):** a API oficial aplica RBAC estático com negação padrão,
limites independentes por IP e fingerprint de chave, e não emite `text` clínico nem
chaves de API em respostas ou logs. O portal Streamlit `front/app_prod.py` demonstra
login médico/paciente, mantém a chave médica no processo servidor e nunca chama
`/predict` na sessão de paciente. A suíte local possui 130 testes, incluindo cenários
positivos/negativos de papéis, rate limit, settings estritos, sanitização e helpers
do portal. O baseline HTTP permanece em `reports/benchmarks/api-prod-baseline.json`.
A imagem da API foi validada em 2026-09-05 com o artefato real produzido pelo Airflow:
healthcheck saudável, execução como `uid=10001`, modelo carregado, bloqueio patient `403`,
predição doctor e headers `X-Request-ID`/`Server-Timing` confirmados.

> Atualização 2026-09-07 (revisão cruzada da Etapa 3): Bill executou revisão estática da Etapa 3 com cuidado redobrado (alto risco por código de produção com RBAC) e encontrou treze pontos. (1) `general_handler` engolia o traceback porque o `JSONRenderer` do structlog não tinha `format_exc_info`; adicionados `StackInfoRenderer` e `format_exc_info` em `logging_config.py` e trocado `logger.error` por `logger.exception`. (2) `StarletteHTTPException` aceitava qualquer string como `error_code`, poluindo telemetria e expondo detalhes internos; criado `_resolve_error_code` que filtra contra o `ALLOWED_ERROR_CODES` extraído como constante compartilhada em `dev_api/app.py` (adicionados `unauthorized`, `forbidden`, `clinician_review_required`). (3) Mensagem de `validation_failed` divergia entre Etapas 2 e 3 (`"Invalid payload."` vs `"Request body is invalid."`); padronizada para `"Request body is invalid."` em ambos. (4) Fallback de `request_id` usava `"unknown"` em vez de `None`; helper `_request_id_for` agora retorna `None`, alinhando a semântica de correlação perdida nas duas APIs. (5) `latency_ms` no `PredictOut` incluía detecção de idioma — escopo renomeado para `predict_latency_ms` e o body agora mede só a inferência; `Server-Timing` sempre emite `total;dur=` mais `detect;dur=` e `predict;dur=` quando disponíveis. (6) `setup_logging` reconfigurava structlog em cada `create_app`; idempotente via `_logging_configured`. (7) `/predict` tinha RBAC inline (`if role == "patient" / role != "doctor"`) enquanto `/reload` usava `RequireRole`; extraída `RequirePredictRole` em `auth.py` que centraliza a lógica. (8) `/reload` perdia `model_version` no log de erro; agora loga `model_version=payload.model_version`. (9) Server-Timing truncava para 3 casas enquanto `latency_ms` no body ficava float cru; `latency_ms` agora `round(x, 3)`. (10) `/model-info` e `/models` eram públicos e expunham hyperparameters, métricas, label mapping e git commit; agora restritos a `RequireRole(["service","doctor"])`. (11) Fingerprint de chave de API usava SHA-256 simples, permitindo bypass do rate limit por rotação de header; substituído por `HMAC-SHA-256` com sal aleatório de 32 bytes gerado em import-time. (12) `front/app_prod.py` renderizava `json.dumps(response.body)` no Streamlit em caso de erro, acoplando o dashboard à sanitização da API; agora exibe apenas o `request_id` para correlação. (13) `RequireRole.allowed_roles` era lista mutável; convertido para `frozenset` no `__init__`. Lint limpo e 152 testes verdes (130 anteriores + 4 da Etapa 2 ainda ativos + 4 novos desta rodada: `test_model_info_requires_authentication`, `test_models_requires_authentication`, `test_rate_limit_identifier_is_stable_for_same_key`, `test_rate_limit_identifier_changes_when_key_changes`).

Aceite oficial (parte da Etapa 8): API funcional, baseline de tempo de resposta documentado.

## Etapa 4 — CI/CD e Docker (Fábio)

### CI/CD, Docker e testes — Fábio

- [x] Criar CI com lockfile, formatação, lint, pytest e build do pacote em push/PR para `main`.
- [x] Ampliar testes conforme API, modelo, front e DAG forem integrados.
- [x] Testar o portal no Chromium com Playwright e preservar evidências de falha no CI.
- [x] Criar Dockerfile funcional para inferência (imagem da API oficial).
- [x] Empacotar portal por papel e dashboard técnico em targets Docker reproduzíveis.
- [x] Adicionar build da imagem ao CI.
- [x] Documentar execução local e no CI.
- [x] Confirmar primeiro workflow verde no GitHub.

Aceite oficial (15%): GitHub Actions executando ao menos lint e testes básicos, build da imagem verde.

**Evidência local do front 2026-09-05:** três testes Playwright aprovados no
Chromium: credenciais inválidas permanecem no login; paciente percorre revisão e
orientação, encerra a sessão e produz zero chamadas a `/predict`; médico autentica,
visualiza o aviso de apoio não diagnóstico e executa uma predição server-side. O job
`front-e2e` repete os cenários no GitHub Actions e retém logs, screenshots e traces de
falha no artefato `front-e2e-evidence` por 14 dias.

**Evidência local 2026-09-05:** build multi-stage concluído; imagem de aproximadamente
289 MB; Compose saudável com modelo real montado em modo somente leitura; Ruff aprovado e
140 testes aprovados (1 teste de symlink desconsiderado por privilégio do Windows). Detalhes
e comandos em `docs/reports/Etapa_4_CI_CD_Docker.md`.

**Evidência da stack 2026-09-05:** `api-prod`, `portal-prod` e `dashboard-dev`
construídos e iniciados pelo Compose com healthchecks saudáveis, UID `10001:10001` e
filesystem somente leitura. A API carregou o artefato real do Airflow; portal e dashboard
responderam HTTP 200 nas portas 8501 e 8502. O smoke test médico retornou a classe
cardiovascular usando a chave fornecida somente em runtime.

**Evidência remota 2026-09-05:** PR #5 integrado à `main`; workflow `CI` nº 39
concluído com `success`, incluindo os jobs de qualidade, Playwright e build/auditoria das
três imagens Docker.

> Atualização 2026-09-07 (revisão cruzada da Etapa 4): Bill executou revisão estática
> cruzada da Etapa 4 com cuidado redobrado (alto risco por empacotamento de produção,
> segredos e contratos CI). Cross-validação com dois sub-agentes confirmou as correções
> aplicadas: (1) `dashboard-dev` no `docker-compose.yml` ainda apontava para
> `http://api-prod:8000` e reusava `TRIAGE_ML_API_KEY_DOCTOR` de produção, acoplando o
> dev ao prod; movido para `profiles: [dev]`, removido `depends_on: api-prod` e trocadas
> as variáveis para `TRIAGE_ML_DEV_API_URL` e `TRIAGE_ML_DEV_API_KEY_DOCTOR`
> dedicadas, garantindo que `docker compose up` rode apenas o stack mínimo. (2)
> `tests/e2e/fake_api.py` comparava `x_api_key != "doc-e2e-key"` (não constant-time);
> substituído por `hmac.compare_digest` contra `EXPECTED_API_KEY` (`bytes`). (3) Job
> `front-e2e` em `ci.yml` não tinha `trap` para encerrar `uvicorn`/`streamlit`
> abertos em background; adicionado `cleanup` + `trap cleanup EXIT` mais
> `set -euo pipefail`, evitando processos-zumbi e encerramentos silenciosos. (4) Loop
> de readiness `for attempt in {1..30}` + `if [ "$attempt" -eq 30 ]` antes de `sleep 1`
> fazia a 30ª iteração falhar sem tentar; trocado por `seq 1 30` e o `if` movido para
> depois do `sleep`, mantendo o intervalo máximo de 30s com retentativas. (5) Smoke-test
> do job `container` em `ci.yml` usava `assert app.title == 'Triage ML - Prod API'`
> (acoplamento por string); trocado por `app.openapi()['info']['title']` mais
> checagem de `'/predict' in spec['paths']`, eliminando string-coupling e validando o
> contrato OpenAPI. (6) `Dockerfile` builder não copiava `.python-version`, deixando a
> versão de Python implícita pela imagem base; adicionado ao `COPY` inicial para
> reproducibilidade cruzada. (7) `airflow/Dockerfile` não validava credenciais DagsHub
> quando o repositório era privado; criado `airflow/entrypoint.sh` com `set -euo pipefail`
> que falha rápido se `TRIAGE_REQUIRE_AUTH=true` e `DAGSHUB_USERNAME`/`DAGSHUB_USER_TOKEN`
> estiverem vazios, registrado como `ENTRYPOINT`. (8) `docker-compose.airflow.yml` não
> expunha a flag de exigência de auth; adicionada `TRIAGE_REQUIRE_AUTH` ao bloco
> `environment` com default `false`. (9) `tests/test_api_container.py` ainda exigia
> `depends_on: api-prod` para `dashboard-dev`; reescrito o teste para refletir o
> desacoplamento e exigir `profiles == ["dev"]` mais chave de API dedicada. (10)
> `tests/test_repository_structure.py` ignorava os novos arquivos do Airflow; incluídos
> `docker-compose.airflow.yml`, `airflow/Dockerfile` e `airflow/entrypoint.sh` na lista
> de arquivos obrigatórios. Lint limpo, 152 testes verdes (149 anteriores + 3 ajustados
> desta rodada: `test_api_container.py::test_front_containers_are_isolated_and_depend_on_healthy_api`,
> `test_repository_structure.py::test_required_files_exist`, sem novos testes —
> mudanças estruturais).

## Etapa 5 — Otimização do modelo (Bill)

### Otimização do classificador — Bill

- [x] Aplicar ao menos uma técnica vista em aula: ONNX (export via `skl2onnx.convert_sklearn`, opset 17).
- [x] Comparar baseline e otimizado nas mesmas entradas/condições (mesmo split, mesma função de inferência do contrato).
- [x] Demonstrar melhoria de latência sem degradação inaceitável de qualidade (Δ macro-F1 ≤ 1 pp no split de teste).
- [x] Persistir `model.onnx` (ou equivalente) e `benchmark.json` ao lado do `model.joblib`.
- [x] Expor a versão otimizada na API oficial atrás de uma flag (ex.: `TRIAGE_ML_MODEL_VARIANT=onnx|sklearn`) para a Etapa 6 medir os dois lados.

Aceite parcial (junto com Etapa 6 fecha o 20% oficial): otimização bem-sucedida e melhoria demonstrada.

**Implementação 2026-09-07/08 (Fase 2):**

- `src/triage_ml/optimization/` introduz `optimize.py` (export ONNX atômico e validado), `onnx_adapter.py` (lazy `onnxruntime.InferenceSession` thread-safe), `registry.py` (variantes + checksums), `dataloader.py` (cortes `dataset_sizing` 5K/6K/7K) e `benchmark.py` (p50/p95/p99, macro-F1 no split real, throughput e hardware fingerprint).
- DAG nova `triage_ml_retraining_optimization` (`airflow/dags/triage_retraining_optimization.py`) — `schedule=None`, `catchup=False`, `max_active_runs=1`, gated por `TRIAGE_OPTIMIZATION_ENABLED=false` (default) para preservar o stack da Etapa 7; itera sobre `dataset_sizing: [5000, 6000, 7000]` em `configs/training.yaml` (override `TRIAGE_DATASET_SLICES`), rejeita cortes maiores que a população elegível e publica `reports/benchmarks/dataset_sizing.json` após os gates de promoção.
- Helpers novos em `airflow_pipeline.py`: `train_with_sample_size`, `_find_reusable_for_size`, `export_onnx_for_version`, `benchmark_for_version`, `build_optimization_manifest` — todos idempotentes por `(dataset_sha256, config_file_sha256, sample_size)` e usam `run_training` canônico (sem duplicar lógica de preparação ou treino).
- `configs/training.yaml` usa `dataset_sizing: [5000, 6000, 7000]`, compatível com as 7.489 linhas elegíveis documentadas (espelhado em `src/triage_ml/training.yaml`).
- `pyproject.toml` registra o grupo opcional `[optimization]` (`onnx`, `onnxruntime`, `skl2onnx`).
- ADR 0003 (`docs/adr/0003-flexibilizar-sample-size.md`) documenta a remoção do teto `sample_size <= 5_000` em `prepare_dataset` (limite inferior `>= 2_000` mantido como invariante de CV-folds-por-classe).
- 27 novos testes adicionados em `tests/test_optimization_*.py`, `tests/test_model_optimization.py`, `tests/test_airflow_optimization.py`, `tests/test_train_with_sample_size.py` — total **229 testes verdes**, 6 pulados por dependência opcional.

## Etapa 6 — Observabilidade e stack Prometheus/Grafana (Bill)

### Instrumentação e stack — Bill

- [x] Expor métricas com `prometheus_client` no middleware da API oficial.
- [x] Medir total de requisições por rota/status.
- [x] Medir latência/tempo de resposta (reaproveitando `Server-Timing` da Etapa 2).
- [x] Medir total/taxa de erros.
- [x] Evitar labels de alta cardinalidade e conteúdo clínico. Teste automatizado varre labels aceitos.
- [x] Configurar Compose com API, Prometheus e Grafana.
- [x] Provisionar dashboard reprodutível em JSON com pelo menos quatro painéis: requisições, latência p95, erros e comparação baseline vs otimizado.
- [x] Salvar print e JSON do dashboard em `reports/figures/`.

Aceite oficial (junto com Etapa 5 fecha 20%): stack completa no Compose e dashboard exibindo as métricas propostas, incluindo o comparativo baseline vs otimizado.

### Privacidade e segurança operacional — Bill

- [x] Garantir que `text` nunca aparece em logs, payloads de erro ou labels de métrica (teste de fumaça).
- [x] Documentar a política de não retenção do `text` após a resposta.

**Implementação 2026-09-08 (Fase 2):**

- `src/triage_ml/observability/metrics.py` define `REQUESTS_TOTAL`, `REQUEST_LATENCY_SECONDS`, `PREDICTION_ERRORS_TOTAL` em registry **privado** (não vaza do global), buckets `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5]` e degrade gracioso quando o `[observability]` extra está ausente.
- `src/triage_ml/observability/middleware.py` — ASGI middleware route-aware que mapeia `path → /predict|/reload|/health|/model-info|/models|/metrics|/" via templates, mantendo a label `route` de cardinalidade baixa.
- API oficial (`src/triage_ml/api/app.py`): `add_middleware(PrometheusMiddleware)`, novo endpoint `GET /metrics` (sem RBAC nesta fase; revisitar na Etapa 8), `TRIAGE_ML_MODEL_VARIANT={sklearn,onnx}` resolvido em `app.state.model_variant` no `lifespan`/`startup`. `predict` agora consulta o adapter ONNX quando a variante ativa é `onnx` (LinearSVC cai para `decision_function`, conforme ADR 0003).
- HTTP exceptions gravam `request.state.error_code` (allow-list público) para a métrica `prediction_errors_total{route, error_code, model_variant}`.
- `infra/docker-compose.yml` overlay roda `api-sklearn:8000`, `api-onnx:8000`, `prometheus:9090`, `grafana:3000` em rede privada, com `read_only: true`, `cap_drop: ALL`, `no-new-privileges` (espelha o compose de produção).
- `monitoring/prometheus/prometheus.yml` scrape em `api-sklearn:8000/metrics` e `api-onnx:8000/metrics` (5s); datasource + provider YAML em `monitoring/grafana/provisioning/`; dashboard JSON canônico `monitoring/grafana/dashboards/triage_ml.json` com 4 painéis (`Requests by route/status`, `Latency p95` com `model_variant`, `Prediction error rate`, `Baseline vs optimized` table).
- `Dockerfile` ganhou `target=runtime-observability` (instala `[observability,optimization]` em cima do `runtime-base`); produção continua usando `target=runtime` sem extras.
- `scripts/generate_observability_traffic.py` — gerador de carga sintética benigna, materializa 32 amostras com tópicos clínicos rotacionados, sem persistência nem payloads sensíveis.
- 13 novos testes em `tests/test_observability_metrics.py` e `tests/test_observability_privacy.py` — verifica `ALLOWED_LABELS`, cardinalidade do payload Prometheus (com `le` reservado para buckets de histograma), ausência do fixture de texto `"PRIVACY-CANARY-CARDIOVASCULAR..."` em logs/resposta/métricas, e `render_metrics()`.
- Política de privacidade publicada em [`README.md`](../README.md) e em [`.agents/contracts/README.md`](../.agents/contracts/README.md): texto é classificado e descartado; nunca persistido, logado, copiado para label ou retornado em erro. Labels Prometheus permitidas: `route`, `method`, `status`, `model_variant`, `error_code`.

## Etapa 7 — Orquestração de retreino (Denis)

### DAG Airflow — Denis

- [x] Consumir `triage_ml.models.train.run_training` (ou equivalente) em vez de duplicar lógica.
- [x] Implementar ingestão/leitura do CSV.
- [x] Implementar validação/preparação reaproveitando `triage_ml.data.prepare`.
- [x] Implementar treino e avaliação.
- [x] Persistir artefato e metadados no caminho configurável do contrato.
- [x] Garantir configuração portátil e tarefas idempotentes quando possível.
- [x] Testar/importar a DAG sem erros e registrar evidência de execução.
- [x] Suportar retreino disparando a partir da Etapa 8 (cloud) ou manualmente.

Aceite oficial (15%): DAG funcional realizando ingestão e treino, com modelo salvo no caminho versionado.

**Evidência 2026-09-05:** `airflow dags test triage_ml_retraining 2026-09-05`
executado com sucesso via Docker contra a `main` do DagsHub no commit
`069dc330e8f5c478a82c893cc224d63734781f6f`. A ingestão leu 11.550 registros,
validou 7.489 elegíveis e preparou 5.000; o artefato
`20260905T171611Z-f2cb6f23f9cd` foi persistido e validado com `accuracy=0.7520`,
`balanced_accuracy=0.7281` e `macro_f1=0.7335`. Uma segunda execução terminou com
sucesso e `reused=true`, confirmando a idempotência para os mesmos dataset e configuração.

> Atualização 2026-09-07 (revisão cruzada da Etapa 7): Bill executou revisão estática
> cruzada da Etapa 7 com cuidado redobrado (alto risco por execução de subprocessos
> com credenciais, contrato de DAG e publicação atômica) e cross-validação com dois
> sub-agentes. Os pontos consolidados foram aplicados: (1) `clone_environment =
> os.environ.copy()` em `airflow_pipeline.py:61` herdava todo o ambiente do worker
> Airflow — incluindo segredos — para o subprocess do `git`, deixando o token do
> DagsHub em `/proc/<pid>/environ` e em qualquer subprocesso filho; substituído por
> `_git_environment()` que monta um env mínimo com apenas `PATH`, `LC_ALL`,
> `GIT_TERMINAL_PROMPT=0` e `GIT_ASKPASS_REQUIRE=force`, mais `GIT_ASKPASS`,
> `DAGSHUB_USERNAME` e `DAGSHUB_USER_TOKEN` quando há credenciais. Após o clone, o
> dict é limpo em memória (`""` em vez de remover, preservando a chave para testes
> que inspecionam o env recebido pelo subprocesso). (2) `subprocess.run(..., check=True,
> capture_output=True)` engolia o stderr do `git`; empacotado em `_run_git` que
> sanitiza credenciais via `_redact_credentials` (regex
> `(://)([^/\s:@]+):([^@\s/]+)@`) e re-raise como `RuntimeError` com mensagem útil
> para branch inexistente, 401 ou falha TLS. (3) `GIT_TERMINAL_PROMPT=0` era
> insuficiente — combinado com `GIT_ASKPASS_REQUIRE=force` que torna o helper
> obrigatório. (4) `validate_dataset_file` hardcodava `sample_size=5_000` e
> `random_state=42`, divergindo do `configs/training.yaml` consumido por
> `run_training`; agora aceita `config_path` opcional e delega a `_load_preparation_settings`
> que lê `sample_size`/`random_state` do YAML quando não fornecidos, com overrides
> explícitos honrados. (5) `validate_dataset_file` retornava apenas 6 chaves e
> descartava `missing_or_empty_rows`, `conflicting_texts`, `conflicting_rows` e
> `duplicate_rows` do `PreparationReport`; agora retorna o conjunto completo para
> visibilidade operacional. (6) `validate_dataset_file` carregava o CSV inteiro
> sem limite; adicionada proteção contra OOM com `MAX_DATASET_BYTES = 200 MiB`
> antes de `pd.read_csv`. (7) `find_reusable_artifact` capturava exceções
> incompletas; tupla ampliada para `(OSError, ValueError, RuntimeError, KeyError,
> TypeError, AttributeError)`. (8) `find_reusable_artifact` aceitava qualquer
> subdiretório com `airflow_run.json` (inclusive `models/.trash/...`) sem checar o
> `VERSION_PATTERN`; agora filtra `re.fullmatch(VERSION_PATTERN, manifest_path.parent.name)`
> e pula manifestos maiores que `MAX_MANIFEST_BYTES = 1 MiB`. (9)
> `train_evaluate_persist` escrevia `airflow_run.json` com `Path.write_text`, sem
> atomicidade; substituído por `_atomic_write_json` (tempfile + `os.replace`). (10)
> `ingest_from_git` aceitava destino através de symlink na hierarquia, criando
> potencial escape via `mkdir(parents=True)`; `_ensure_no_symlink_ancestor` recusa
> a publicação com erro explícito. (11) `airflow/dags/triage_retraining.py` usava
> `os.environ["DATA_REPOSITORY_URL"]` (KeyError sem mensagem) e `os.getenv(...)
> or None` mascarando credenciais vazias; substituído por `_require_env` (mensagem
> amigável) e `_require_auth_credentials` que valida `TRIAGE_REQUIRE_AUTH=true`
> exige `DAGSHUB_USERNAME`/`DAGSHUB_USER_TOKEN` preenchidos — espelhando a checagem
> do `entrypoint.sh` dentro do DAG Python, eliminando bypass quando o DAG roda fora
> do compose. (12) DAG movido para fora do escopo das tasks: leitura única de
> `TRIAGE_*` no escopo do `@dag` factory, garantindo avaliação única na construção
> da DAG. Testes adicionados: `test_git_subprocess_errors_redact_credentials_in_stderr`,
> `test_ingestion_refuses_destination_through_symlink`, `test_validate_dataset_file_aligns_with_training_config`.
> Lint limpo, 155 testes verdes (152 anteriores + 3 novos desta rodada).

## Etapa 8 — Cloud, vídeo e documentação final

### Documentação — Fábio

- [~] Manter README e checklist como documentos vivos.
- [ ] Documentar setup, execução, testes, API, Airflow, Compose e troubleshooting.
- [ ] Consolidar arquitetura em nuvem após ADR de Romário.
- [ ] Documentar metodologia e resultados do benchmark baseline vs otimizado (entrega da Etapa 5).
- [ ] Revisar links, comandos e afirmações contra o sistema final.

Aceite parcial (15% oficial, junto com cloud ADR e vídeo): arquitetura em nuvem explicada e instruções claras de execução.

### Vídeo STAR — Romário

- [ ] Situation: problema clínico e importância da triagem rápida.
- [ ] Task: requisitos de latência, CI/CD e monitoramento.
- [ ] Action: arquitetura, otimização e observabilidade.
- [ ] Result: pipeline funcionando, latência e lições aprendidas.
- [ ] Demonstrar os componentes essenciais e manter duração de até cinco minutos.
- [ ] Inserir link final no README.

Aceite oficial (15%): demonstração técnica clara, impacto explicado e duração respeitada.

## Tradução

- [x] Não inserir tradução online no caminho crítico da fundação (decisão por LGPD/latência).
- [x] Confirmar que o produto final recebe apenas inglês (modelo treinado em abstracts em inglês).
- [x] Checagem de idioma local via `langid` na `/predict`, rejeitando texto fora do allow-list `{"en"}` — sem tradução automática, sem chamada externa.
- [ ] Se necessário no futuro, avaliar tradução offline, versionada e mensurada e revisar impactos em qualidade, privacidade, custo e latência.

## Regra de atualização

Todo agente confere este arquivo em cada tarefa. Só o edita quando status, escopo, responsável, critério de aceite ou evidência mudar. Todo PR declara “checklist atualizado” ou “checklist conferido, sem alteração necessária”.
