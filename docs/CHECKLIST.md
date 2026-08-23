# Checklist do Tech Challenge — Fase 3

Fonte canônica do progresso. Legenda: `[ ]` pendente, `[~]` em andamento/parcial, `[x]` concluído. Um item só fica concluído quando seu critério de aceite possui evidência verificável.

Última atualização: 2026-08-23 — revisão de robustez da Fase 1 (dados, artefatos,
registry, packaging e CI).

## Visão geral e responsáveis

- [x] Repositório e arquitetura inicial — Fábio
- [x] EDA e escolha do dataset — Denis
- [x] Classificador de texto (baseline) — Bill
- [ ] API FastAPI — Romário
- [~] CI/CD, Docker e testes — Fábio
- [ ] DAG Airflow — Denis
- [ ] Otimização de latência e observabilidade — Bill
- [ ] Arquitetura em nuvem — Romário
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
- [ ] Contrato de Airflow/artefato (caminho configurável, idempotência) revisado por Denis antes da Etapa 7.

Aceite: contratos em `.agents/contracts/README.md` estáveis antes do início da Etapa 2.

### Arquitetura em nuvem — Romário (ADR pode começar em paralelo)

- [~] Direção inicial: real-time para inferência e batch para treino/re-treino.
- [ ] Comparar opções e validar/refutar GCP Cloud Run, Artifact Registry e Cloud Storage.
- [ ] Definir execução/orquestração do Airflow na proposta.
- [ ] Avaliar segurança, privacidade, disponibilidade, escala e custos.
- [ ] Registrar ADR e sintetizar decisão no README.

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

## Etapa 3 — API oficial servindo o modelo (Romário)

### API FastAPI — Romário

- [ ] Validar contrato de `POST /predict` (já alinhado com a API de desenvolvimento de Bill).
- [ ] Implementar health check, predição, validação e erros com base no artefato real, sem stub.
- [ ] Carregar artefato do modelo de forma configurável (env `MODEL_PATH`).
- [ ] Manter `latency_ms`, `request_id`, `X-Request-ID` e `Server-Timing` herdados da Etapa 2.
- [ ] Adicionar testes unitários e de integração.
- [ ] Medir baseline de latência local com metodologia documentada (gancho para a Etapa 5).
- [ ] Empacotar o serviço em Docker (parte do entregável do Fábio, mas dirigido a esta API).

Aceite oficial (parte da Etapa 8): API funcional, baseline de tempo de resposta documentado.

## Etapa 4 — CI/CD e Docker (Fábio)

### CI/CD, Docker e testes — Fábio

- [x] Criar CI com lockfile, formatação, lint, pytest e build do pacote em push/PR para `main`.
- [ ] Ampliar testes conforme API, modelo e DAG forem integrados.
- [ ] Criar Dockerfile funcional para inferência (imagem da API oficial).
- [ ] Adicionar build da imagem ao CI.
- [ ] Documentar execução local e no CI.
- [ ] Confirmar primeiro workflow verde no GitHub.

Aceite oficial (15%): GitHub Actions executando ao menos lint e testes básicos, build da imagem verde.

## Etapa 5 — Otimização do modelo (Bill)

### Otimização do classificador — Bill

- [ ] Aplicar ao menos uma técnica vista em aula: ONNX, quantização ou pruning.
- [ ] Comparar baseline e otimizado nas mesmas entradas/condições (mesmo split, mesma função de inferência do contrato).
- [ ] Demonstrar melhoria de latência sem degradação inaceitável de qualidade (Δ macro-F1 ≤ 1 pp no split de teste).
- [ ] Persistir `model.onnx` (ou equivalente) e `benchmark.json` ao lado do `model.joblib`.
- [ ] Expor a versão otimizada na API oficial atrás de uma flag (ex.: `MODEL_VARIANT=onnx|sklearn`) para a Etapa 6 medir os dois lados.

Aceite parcial (junto com Etapa 6 fecha o 20% oficial): otimização bem-sucedida e melhoria demonstrada.

## Etapa 6 — Observabilidade e stack Prometheus/Grafana (Bill)

### Instrumentação e stack — Bill

- [ ] Expor métricas com `prometheus_client` no middleware da API oficial.
- [ ] Medir total de requisições por rota/status.
- [ ] Medir latência/tempo de resposta (reaproveitando `Server-Timing` da Etapa 2).
- [ ] Medir total/taxa de erros.
- [ ] Evitar labels de alta cardinalidade e conteúdo clínico. Teste automatizado varre labels aceitos.
- [ ] Configurar Compose com API, Prometheus e Grafana.
- [ ] Provisionar dashboard reprodutível em JSON com pelo menos quatro painéis: requisições, latência p95, erros e comparação baseline vs otimizado.
- [ ] Salvar print e JSON do dashboard em `reports/figures/`.

Aceite oficial (junto com Etapa 5 fecha 20%): stack completa no Compose e dashboard exibindo as métricas propostas, incluindo o comparativo baseline vs otimizado.

### Privacidade e segurança operacional — Bill

- [ ] Garantir que `text` nunca aparece em logs, payloads de erro ou labels de métrica (teste de fumaça).
- [ ] Documentar a política de não retenção do `text` após a resposta.

## Etapa 7 — Orquestração de retreino (Denis)

### DAG Airflow — Denis

- [ ] Consumir `triage_ml.models.train.run_training` (ou equivalente) em vez de duplicar lógica.
- [ ] Implementar ingestão/leitura do CSV.
- [ ] Implementar validação/preparação reaproveitando `triage_ml.data.prepare`.
- [ ] Implementar treino e avaliação.
- [ ] Persistir artefato e metadados no caminho configurável do contrato.
- [ ] Garantir configuração portátil e tarefas idempotentes quando possível.
- [ ] Testar/importar a DAG sem erros e registrar evidência de execução.
- [ ] Suportar retreino disparando a partir da Etapa 8 (cloud) ou manualmente.

Aceite oficial (15%): DAG funcional realizando ingestão e treino, com modelo salvo no caminho versionado.

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
