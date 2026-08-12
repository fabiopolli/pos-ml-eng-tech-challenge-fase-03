# Checklist do Tech Challenge — Fase 3

Fonte canônica do progresso. Legenda: `[ ]` pendente, `[~]` em andamento/parcial, `[x]` concluído. Um item só fica concluído quando seu critério de aceite possui evidência verificável.

Última atualização: 2026-08-12 — bootstrap arquitetural.

## Visão geral e responsáveis

- [x] Repositório e arquitetura inicial — Fábio
- [ ] EDA e escolha do dataset — Denis
- [ ] Classificador de texto — Bill
- [ ] API FastAPI — Romário
- [~] CI/CD, Docker e testes — Fábio
- [ ] DAG Airflow — Denis
- [ ] Otimização de latência e observabilidade — Bill
- [ ] Arquitetura em nuvem — Romário
- [~] Documentação detalhada — Fábio
- [ ] Vídeo STAR — Romário

## Requisitos transversais

- [x] Estrutura versionada separando dados, código, modelos, Airflow, observabilidade, infraestrutura, testes e documentos.
- [x] Workflow de branches, commits, PRs, revisão e gates documentado.
- [x] Dados e modelos grandes excluídos do Git.
- [ ] Histórico de commits semântico e organizado durante todo o projeto.
- [ ] Instruções finais de execução reproduzidas em máquina limpa.
- [ ] Licença/origem do dataset documentada.
- [ ] Nenhum segredo ou dado clínico sensível versionado ou emitido em logs.
- [ ] Ativar proteção da `main` após o primeiro CI verde: PR, checks, uma aprovação e bloqueio de force-push.

## Etapa 1 — Decisão arquitetural e API inicial

### Dataset e EDA — Denis

- [ ] Comparar Medical Abstracts TC Corpus e MIMIC-III Open Access.
- [ ] Confirmar licença, proveniência, schema e condições de uso.
- [ ] Selecionar recorte entre 2.000 e 5.000 amostras (mínimo oficial: 2.000).
- [ ] Documentar distribuição, duplicatas, ausências, comprimento dos textos e balanceamento.
- [ ] Definir `text`, `target`, labels e estratégia de split sem leakage.
- [ ] Registrar decisão e evidências; manter dados fora do Git.

Aceite: notebook/relatório reprodutível, dataset escolhido e contrato de dados aprovado.

### API FastAPI — Romário

- [ ] Validar contrato de `POST /predict`.
- [ ] Implementar health check, predição, validação e erros.
- [ ] Carregar artefato do modelo de forma configurável.
- [ ] Adicionar testes unitários e de integração.
- [ ] Empacotar o serviço em Docker.
- [ ] Medir baseline de latência local com metodologia documentada.

Aceite oficial: API funcional em Docker e baseline de tempo de resposta.

### Arquitetura em nuvem — Romário

- [~] Direção inicial: real-time para inferência e batch para treino/re-treino.
- [ ] Comparar opções e validar/refutar GCP Cloud Run, Artifact Registry e Cloud Storage.
- [ ] Definir execução/orquestração do Airflow na proposta.
- [ ] Avaliar segurança, privacidade, disponibilidade, escala e custos.
- [ ] Registrar ADR e sintetizar decisão no README.

Aceite oficial: decisão arquitetural textual clara e coerente com batch versus real-time.

## Etapa 2 — CI/CD e pipeline automatizado

### CI/CD, Docker e testes — Fábio

- [x] Criar CI inicial com lint e pytest em push/PR para `main`.
- [ ] Ampliar testes conforme API, modelo e DAG forem integrados.
- [ ] Criar Dockerfile funcional para inferência.
- [ ] Adicionar build da imagem ao CI.
- [ ] Documentar execução local e no CI.
- [ ] Confirmar primeiro workflow verde no GitHub.

Aceite oficial (15%): GitHub Actions executando ao menos lint e testes básicos.

### DAG Airflow — Denis

- [ ] Implementar ingestão/leitura do CSV.
- [ ] Implementar validação/preparação.
- [ ] Implementar treino e avaliação.
- [ ] Persistir artefato e metadados.
- [ ] Garantir configuração portátil e tarefas idempotentes quando possível.
- [ ] Testar/importar a DAG sem erros e registrar evidência de execução.

Aceite oficial (15%): DAG funcional realizando ingestão e treino, com modelo salvo.

## Etapa 3 — Monitoramento e observabilidade

### Instrumentação e stack — Bill

- [ ] Expor métricas com `prometheus_client`.
- [ ] Medir total de requisições por rota/status.
- [ ] Medir latência/tempo de resposta.
- [ ] Medir total/taxa de erros.
- [ ] Evitar labels de alta cardinalidade e conteúdo clínico.
- [ ] Configurar Compose com API, Prometheus e Grafana.
- [ ] Provisionar dashboard reprodutível em JSON.
- [ ] Criar pelo menos três painéis: requisições, latência e erros.
- [ ] Salvar print e JSON do dashboard.

Aceite oficial (20%): stack completa no Compose e dashboard exibindo as métricas propostas.

## Etapa 4 — Modelo, otimização e entrega

### Modelagem e otimização — Bill

- [ ] Criar baseline leve, por exemplo TF-IDF + classificador Scikit-Learn.
- [ ] Fixar seeds, preprocessing e versões relevantes.
- [ ] Reportar métricas adequadas por classe e agregadas.
- [ ] Serializar modelo e metadados segundo o contrato.
- [ ] Aplicar ao menos uma técnica vista em aula: ONNX, quantização ou pruning.
- [ ] Comparar baseline e otimizado nas mesmas entradas/condições.
- [ ] Demonstrar melhoria de latência sem degradação inaceitável de qualidade.

Aceite oficial (20%): modelo NLP funcional, otimização bem-sucedida e melhoria demonstrada.

### Tradução

- [x] Não inserir tradução online no caminho crítico da fundação.
- [ ] Após escolha do dataset, confirmar se o produto precisa receber português.
- [ ] Se necessário, avaliar tradução offline, versionada e mensurada.
- [ ] Documentar efeitos em qualidade, privacidade, custo e latência.

### Documentação — Fábio

- [~] Manter README e checklist como documentos vivos.
- [ ] Documentar setup, execução, testes, API, Airflow, Compose e troubleshooting.
- [ ] Consolidar arquitetura em nuvem após ADR de Romário.
- [ ] Documentar metodologia e resultados do benchmark.
- [ ] Revisar links, comandos e afirmações contra o sistema final.

Aceite oficial (15%): arquitetura em nuvem explicada e instruções claras de execução.

### Vídeo STAR — Romário

- [ ] Situation: problema clínico e importância da triagem rápida.
- [ ] Task: requisitos de latência, CI/CD e monitoramento.
- [ ] Action: arquitetura, otimização e observabilidade.
- [ ] Result: pipeline funcionando, latência e lições aprendidas.
- [ ] Demonstrar os componentes essenciais e manter duração de até cinco minutos.
- [ ] Inserir link final no README.

Aceite oficial (15%): demonstração técnica clara, impacto explicado e duração respeitada.

## Regra de atualização

Todo agente confere este arquivo em cada tarefa. Só o edita quando status, escopo, responsável, critério de aceite ou evidência mudar. Todo PR declara “checklist atualizado” ou “checklist conferido, sem alteração necessária”.

