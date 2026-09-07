# Relatório de implementação — Etapa 8 (Cloud, vídeo e documentação final)

| Campo | Valor |
|---|---|
| Integrantes | Fábio Polli (documentação), Romário (cloud + vídeo STAR) |
| Etapa do checklist | Etapa 8 — `docs/CHECKLIST.md` |
| Período desta entrega | 2026-08-30 a 2026-09-07 (consolidação); revisões cruzadas de Bill nas Etapas 2, 3, 4 e 7 entre 2026-09-07 |
| Última revisão | 2026-09-07 |
| Status | 🟡 Documentação contínua ativa; cloud ADR e vídeo STAR pendentes |

A Etapa 8 fecha a fase: consolida a documentação detalhada (README, CHECKLIST, reports
por etapa, ADRs e plans), publica a decisão textual sobre arquitetura em nuvem e
registra o vídeo no formato STAR. Este relatório descreve o que está pronto,
o que ainda falta e o que foi corrigido durante as revisões cruzadas.

## 1. Resumo executivo

- Documentação operacional completa em `README.md`, `docs/CHECKLIST.md`,
  `docs/WORKFLOW_AGENTICO.md`, `docs/dataset.md`, `docs/guides/GUIA-USO-FRONTS.md`
  e nos planos `docs/plans/PLAN-text-classifier.md` e
  `docs/plans/PLAN-api-prod.md`. ADRs publicados em `docs/adr/0001-escolha-recorte-dataset.md`
  e `docs/adr/0002-rbac-estatico-api-producao.md`.
- Quatro relatórios por etapa já consolidados em `docs/reports/`:
  - `Etapa_1_Fundação_ dados_e_contratos.md`;
  - `Etapa_2_Modelo_baseline_e_serialização.md`;
  - `Etapa_3_API_oficial.md`;
  - `Etapa_4_CI_CD_Docker.md` (atualizado em 2026-09-07 com a revisão cruzada da
    Etapa 4).
- `infra/README.md` registra a hipótese arquitetural GCP (Cloud Run, Artifact
  Registry, Cloud Storage) ainda **não validada**; a decisão textual e o ADR
  correspondente permanecem com Romário.
- Vídeo STAR de até cinco minutos pendente de gravação por Romário; cenários
  sugeridos e checkpoints em `docs/guides/GUIA-USO-FRONTS.md`.
- 155 testes verdes, lint e formatação limpos, workflow remoto `CI` no `main`
  verde desde o PR #5 (runs #39, #40 e subsequentes).

## 2. Estado por subgrupo do checklist

Itens da seção "Etapa 8 — Cloud, vídeo e documentação final" do
`docs/CHECKLIST.md`:

### Documentação — Fábio

- [x] **Manter README e checklist como documentos vivos.** README atualizado em
  2026-09-07 reflete o estado pós revisão cruzada das Etapas 4 e 7
  (dashboard-dev em `profiles: [dev]`, chaves dedicadas, Airflow com
  `TRIAGE_REQUIRE_AUTH`).
- [x] **Documentar setup, execução, testes, API, Airflow, Compose e troubleshooting.**
  O README cobre Docker Compose (produção e overlay do Airflow), execução
  local sem Docker, Playwright local, e cenários de troubleshooting (token
  DagsHub, permissões de `models/`, leitura de logs estruturados).
- [~] **Consolidar arquitetura em nuvem após ADR de Romário.** Pendente do ADR
  de Romário; enquanto isso, o `infra/README.md` registra a hipótese GCP.
- [ ] **Documentar metodologia e resultados do benchmark baseline vs otimizado.**
  Depende da Etapa 5 (otimização de modelo, owner Bill).
- [x] **Revisar links, comandos e afirmações contra o sistema final.** Revisão
  feita em 2026-09-07 contra `docker compose config --quiet` e a árvore atual
  do repositório.

### Vídeo STAR — Romário

- [ ] **Situation.** Problema clínico e importância da triagem rápida.
- [ ] **Task.** Requisitos de latência, CI/CD e monitoramento.
- [ ] **Action.** Arquitetura, otimização e observabilidade.
- [ ] **Result.** Pipeline funcionando, latência e lições aprendidas.

Roteiro proposto (resumo, não normativo):

1. **Situation** — triagem textual em hospital, gargalo humano, motivação do
   classificador automático; volume esperado e classes (cinco categorias
   clínicas do Medical Abstracts TC Corpus).
2. **Task** — requisitos não funcionais: latência alvo (≤ 50 ms p95 por
   inferência), RBAC com três papéis, auditabilidade via `X-Request-ID`,
   `Server-Timing` e logs JSON sem texto clínico.
3. **Action** — arquitetura end-to-end (Airflow → treino → artefato
   versionado → API FastAPI → fronts Streamlit), separação dev/prod, pipeline
   de retreino com `find_reusable_artifact`, gating por `TRIAGE_REQUIRE_AUTH`
   e empacotamento multi-stage com UID não-root.
4. **Result** — métricas de baseline (accuracy `0.7460`, balanced_accuracy
   `0.7221`, macro_f1 `0.7296` no split de teste), baseline HTTP (média
   `22,18 ms`, p95 `31,88 ms`, p99 `32,80 ms`), 155 testes verdes, workflow
   CI verde, DAG idempotente com `reused=true`.

## 3. Decisão textual sobre arquitetura em nuvem (pendente de ADR)

> **Status:** esta é a posição **proposta** pelo time e ainda depende do ADR
> formal de Romário. Nenhuma infraestrutura está provisionada.

A solução é desenhada para inferência **real-time**, mantendo o pipeline
batch apenas para ingestão, preparação e retreino (Airflow). A hipótese
arquitetural é **GCP**:

| Componente GCP | Substitui | Justificativa |
|---|---|---|
| Cloud Run | `api-prod` no Compose | Pay-per-use, escala a zero, suporte nativo a containers com UID não-root e healthcheck HTTP. Variáveis de ambiente injetadas via Secret Manager. |
| Artifact Registry | `triage-ml-api:ci` | Imagens multi-stage reproduzidas a partir do `Dockerfile`; sem alteração de contrato. |
| Cloud Storage (bucket versionado) | `./models/<versao>` montado em `/models:ro` | Hospeda `model.joblib`, `metadata.json`, `classes.json` e `airflow_run.json`; permite assinatura de URL de curta duração para a Cloud Run carregar o artefato sem expor o bucket. |
| Cloud Scheduler + Cloud Run Job | Airflow para gatilhos externos | Disparo opcional do retreino quando o dataset do DagsHub muda; o DAG continua executando dentro de um cluster Airflow gerenciado (Cloud Composer) ou em um cluster Kubernetes interno. |
| Cloud Logging + Cloud Monitoring | Logs JSON locais e Prometheus/Grafana locais | Estrutura JSON já compatível (`structlog` com `format_exc_info`); métricas do Prometheus client podem ser exportadas via OpenTelemetry. |
| Secret Manager | `.env` local | Armazena as três chaves da API (`TRIAGE_ML_API_KEY_*`), credenciais do portal Streamlit e, quando aplicável, `DAGSHUB_USER_TOKEN` apenas para o job de retreino. |

Premissas da proposta:

1. **Latência**: o baseline HTTP local ficou em `≈22 ms` de média; Cloud Run
   em região `southamerica-east1` deve preservar p95 < `50 ms` para textos
   clínicos curtos.
2. **Custos**: pay-per-use com `min-instances=0` é economicamente viável para
   baixo tráfego (triagem manual辅助). Se o tráfego for constante, configurar
   `min-instances=1` para eliminar cold-start.
3. **Segredos**: Secret Manager como única fonte; nada de chaves no `.env`
   versionado nem em variáveis de ambiente do Cloud Run definidas inline.
4. **Observabilidade**: Cloud Logging ingere os logs JSON sem mudança de
   formato; rótulos com cardinalidade alta (texto, score) permanecem
   excluídos por contrato (ver `docs/adr/0002-rbac-estatico-api-producao.md`).
5. **Reprocessamento**: cada nova versão em `gs://<bucket>/<model_version>/`
   preserva o diretório imutável; o `metadata.json` continua sendo a fonte
   de verdade para auditoria.

O ADR correspondente (`docs/adr/0003-arquitetura-cloud.md`) será aberto por
Romário após a Etapa 5 (otimização) para incorporar os números de latência do
modelo otimizado.

## 4. Consolidação de documentação

### 4.1 Estrutura final

```
docs/
├── CHECKLIST.md                  # checklist vivo (155 testes, 7 etapas concluídas)
├── WORKFLOW_AGENTICO.md          # protocolo de revisão entre agentes
├── dataset.md                    # decisão, licença e procedimento do dataset
├── adr/
│   ├── README.md
│   ├── 0001-escolha-recorte-dataset.md
│   └── 0002-rbac-estatico-api-producao.md
├── guides/
│   └── GUIA-USO-FRONTS.md        # cenários para o vídeo e validação manual
├── plans/
│   ├── PLAN-text-classifier.md   # plano da Etapa 2
│   └── PLAN-api-prod.md          # plano da Etapa 3
└── reports/
    ├── RESUMO-APRESENTACAO-FASE-2.md
    ├── Etapa_1_Fundação_ dados_e_contratos.md
    ├── Etapa_2_Modelo_baseline_e_serialização.md
    ├── Etapa_3_API_oficial.md
    ├── Etapa_4_CI_CD_Docker.md
    └── Etapa_8_Cloud_video_documentacao.md  # este relatório
```

### 4.2 Revisão de links e comandos (2026-09-07)

- `README.md` aponta para `docs/CHECKLIST.md`, `docs/dataset.md`, `front/README.md`,
  `docs/guides/GUIA-USO-FRONTS.md` e `airflow/dags/README.md` — todos
  existentes na árvore pós-revisão.
- Os comandos de `docker compose` do `README.md` continuam válidos para a
  stack mínima (`api-prod` + `portal-prod`); o `dashboard-dev` agora exige
  `--profile dev`. A nota foi incorporada na seção "Plataforma local em
  Docker".
- Os blocos de troubleshooting (permissões, token, reload) foram ajustados
  após a revisão cruzada da Etapa 7, onde `_require_auth_credentials` passou
  a falhar cedo quando `TRIAGE_REQUIRE_AUTH=true` sem credenciais.

### 4.3 Estado do CHECKLIST após revisões cruzadas de Bill (2026-09-07)

| Etapa | Owner | Status oficial | Atualização transversal |
|---|---|---|---|
| 1 | Denis | ✅ concluída | — |
| 2 | Bill | ✅ concluída | Drift detection em `artifact.py`, bounds de `std_macro_f1`. |
| 3 | Romário | ✅ concluída | RBAC centralizado, `RequirePredictRole`, `/model-info` protegido, fingerprint HMAC, `Server-Timing`. |
| 4 | Fábio | ✅ concluída | `dashboard-dev` em `profiles: [dev]`, `hmac.compare_digest`, `trap cleanup` no front-e2e, smoke-test sem string coupling. |
| 5 | Bill | ⏳ pendente | Otimização de latência + benchmark. |
| 6 | Bill | ⏳ pendente | Prometheus/Grafana provisionados pelo Compose. |
| 7 | Denis | ✅ concluída | DAG endurecida: env mínimo, `GIT_ASKPASS_REQUIRE=force`, validação simétrica com `entrypoint.sh`, escrita atômica de `airflow_run.json`. |
| 8 | Fábio / Romário | 🟡 parcial | Este relatório + ADR cloud pendente + vídeo pendente. |

## 5. Aceite oficial e próximos passos

O aceite oficial da Etapa 8 está condicionado a três entregas ainda abertas:

1. **ADR de arquitetura em nuvem** por Romário, referenciando este relatório
   e a Etapa 5 quando os números de latência do modelo otimizado estiverem
   disponíveis.
2. **Benchmark baseline vs otimizado** (Etapa 5, owner Bill), com
   `benchmark.json` versionado em `models/<versao>/` e refletido no
   `docs/reports/Etapa_2_Modelo_baseline_e_serialização.md` (atualização
   adicional) e em um futuro `docs/reports/Etapa_5_Otimizacao_modelo.md`.
3. **Vídeo STAR** gravado e anexado ao repositório ou link público, seguindo
   o roteiro da seção 2 deste relatório.

A documentação operacional, os relatórios por etapa, o checklist vivo e a
proposta textual de cloud estão prontos e versionados; a Etapa 8 será
marcada como concluída tão logo as três entregas acima sejam integradas.

## 6. Referências

- `docs/CHECKLIST.md` — checklist oficial da fase 3.
- `README.md` — quickstart, plataforma local e troubleshooting.
- `docs/WORKFLOW_AGENTICO.md` — protocolo de revisão entre agentes.
- `docs/adr/0001-escolha-recorte-dataset.md` — decisão do recorte.
- `docs/adr/0002-rbac-estatico-api-producao.md` — política de RBAC.
- `docs/reports/Etapa_4_CI_CD_Docker.md` — pós-revisão cruzada de Bill.
- `infra/README.md` — hipótese GCP (a ser validada por ADR).