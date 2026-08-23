# Plano de implementação — Classificador de Texto NLP (Bill)

- **Integrante**: Bill
- **Origem**: Tech Challenge — Fase 3 (ML Engineering)
- **Status**: plano, ainda não executado
- **Mapeamento no checklist**: cobre as Etapas 2, 5 e 6 do `docs/CHECKLIST.md` reordenado (2026-08-23). Está organizado em duas fases para deixar claro o que é trabalho **agora** e o que é trabalho **depois**.

## Estrutura por fases

| Fase | Etapas do checklist | Peso oficial | Conteúdo |
|---|---|---|---|
| **Fase 1 — Modelo baseline + API de smoke** | Etapa 2 | (soma com Fase 2 → 20%) | Treino, métricas, serialização, API de smoke local |
| **Fase 2 — Otimização + observabilidade** | Etapas 5 + 6 | 20% (Fase 2) + 20% (otimização+obs) | ONNX, benchmark, Prometheus/Grafana/Compose, dashboard comparativo |

A API oficial FastAPI (Etapa 3) é trabalho do Romário e não está no escopo deste plano; ela consome o artefato e o contrato de `metadata.json` definidos na Fase 1.

## Pré-requisitos do plano

- **Etapa 1 do checklist concluída**: dataset definido (`triage_ml.data.prepare.prepare_dataset`), contratos estáveis em `.agents/contracts/README.md`.
- **Decisão registrada do "Modelo (Bill)" no README**: justificar formalmente o não-uso de Random Forest (vide seção 2 abaixo).
- **Alinhamento com Romário**: contrato de `POST /predict` validado antes do merge da Fase 1 (Risco #1).

## Mudanças em relação à versão anterior do plano

A versão anterior deste arquivo cobria apenas a Fase 1 e tratava otimização/observabilidade como "próxima fase sem detalhe". Agora:

- **Fase 1 mantida** com as melhorias acumuladas (latência exposta, `request_id`, `Server-Timing`).
- **Fase 2 detalhada** com tarefas concretas (otimização, benchmark, Prometheus, Compose, dashboard).
- **Critérios de aceite remapeados** para os itens oficiais do checklist (Etapas 2, 5 e 6), com referência explícita aos números de linha.
- **Riscos revisitados** incluindo os da Fase 2 (dependência ONNX, custo de manter Compose local).
- **Sequência de commits estendida** para incluir as entregas da Fase 2.

---

# Fase 1 — Modelo baseline + API de smoke (Etapa 2 do checklist)

## F1. Contexto e objetivo

Entregar:

1. Classificador NLP leve (TF-IDF + classificador linear Scikit-Learn) treinado no recorte preparado pela fundação (5.000 amostras, seed 42).
2. Artefato serializado segundo o contrato (`models/<versão>/model.joblib` + `metadata.json` + `classes.json`).
3. Métricas por classe e agregadas, com figuras em `reports/figures/`.
4. API FastAPI de smoke local (`/health` + `/predict`) consumindo o artefato, com `latency_ms` e `request_id` já expostos para reuso na Fase 2.

A API oficial é do Romário (Etapa 3). Esta API de smoke é substituída ou estendida por ele; o contrato (esquemas Pydantic, headers) é ponto de alinhamento obrigatório antes de qualquer promoção.

## F1. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Vetorizador | `TfidfVectorizer` | Sugestão do enunciado; leve, determinístico, sem dependência externa |
| Classificador baseline (primário) | `LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")` | Linear, probabilístico, inferência barata, suporte nativo a `predict_proba` |
| Classificador baseline (secundário) | `LinearSVC(class_weight="balanced")` | SVM linear é estado da arte em texto esparso; mais rápido em inferência que LR; comparativo |
| Por que não Random Forest? | — | Exemplo do enunciado. Em TF-IDF, RF explode o custo de inferência (centenas de árvores) sem ganho consistente sobre modelos lineares em texto. Justificativa registrada no README seção Bill |
| Serialização | `joblib` para o pipeline scikit-learn | Padrão sklearn |
| API (smoke) | FastAPI + Uvicorn, em processo local sem Docker | Suficiente para teste manual; Docker e Compose ficam para Etapa 4 (Fábio) e Fase 2 |
| Seeds | 42 em todos os pontos estocásticos | Reprodutibilidade exigida pelo checklist |

Referência do paper (Schopf et al., NLPIR 2022) no Medical Corpus, F1 micro (unsupervised): LSA 31,6; SBERT MiniLM 46,5; DeBERTa zero-shot 57,3. Como o classificador supervisionado linear tende a superar todas essas marcas, esses números ficam só como contexto narrativo no relatório — não como meta.

## F1. Estrutura de arquivos a criar

```
src/triage_ml/
├── data/prepare.py            (existente, não modificar)
├── models/
│   ├── __init__.py
│   ├── pipeline.py            # fábrica do Pipeline TF-IDF + classificador
│   ├── train.py               # treino + métricas + serialização
│   └── artifact.py            # lê/grava metadata.json e classes.json
└── api/
    ├── __init__.py
    ├── app.py                 # FastAPI mínimo com /health e /predict
    └── schemas.py             # Pydantic de entrada/saída
tests/
├── test_model_pipeline.py
└── test_model_artifact.py
models/
└── .gitkeep
reports/
└── figures/
    ├── 08_confusion_matrix_lr.png
    └── 08_top_features_lr.png
configs/
└── training.yaml              # hiperparâmetros versionados
```

A pasta `monitoring/` não é tocada na Fase 1.

## F1. Contratos a cumprir

Pontos relevantes de `.agents/contracts/README.md`:

- **Dados**: já garantido por `prepare_dataset` (sem duplicatas, sem leakage, 5.000 amostras seed 42).
- **Modelo**: recebe lista de textos; devolve classe e score. Artefato acompanha `metadata.json` com `model_version`, `classes`, versões, seed, métricas, preprocessing.
- **API (proposta, sujeito à validação de Romário antes de promover)**:
  - `GET /health` → `HealthOut(status, model_version, model_loaded)`.
  - `POST /predict` → `PredictIn(text)` → `PredictOut(label, label_name, score, model_version, latency_ms, request_id, warnings)`.
  - Erros: `ErrorOut(request_id, error_code, message)` — nunca conteúdo clínico.
  - Headers: `X-Request-ID` ecoando o `request_id`; `Server-Timing: predict;dur=<latency_ms>`.
- **Observabilidade**: adiada como stack completa. Mas a Fase 1 já nasce expondo latência e `request_id` em toda resposta para reuso na Fase 2.

### O que **não** entra na Fase 1

- Endpoint `GET /metrics` Prometheus.
- Middleware `prometheus_fastapi_instrumentator`.
- Otimização ONNX, quantização ou pruning.
- Compose, Prometheus, Grafana, dashboard.
- Autenticação, rate limit, tracing distribuído.

## F1. Tarefas e sequência

Trabalho direto na `main` (Bill é o único colaborador ativo nesta sessão).

### F1.T1. Esqueleto e configuração
- Criar `src/triage_ml/models/{__init__.py, pipeline.py, artifact.py}`.
- Criar `configs/training.yaml` com hiperparâmetros versionados.
- Garantir `fastapi` e `uvicorn` no `pyproject.toml`.
- Smoke test: `python -c "from triage_ml.models.pipeline import build_pipeline; print(build_pipeline())"`.

### F1.T2. Pipeline e treino
- `pipeline.py`: `build_pipeline(classifier="logreg", **kwargs)` retorna `Pipeline([("tfidf", ...), ("clf", ...)])`.
- `train.py`: `run_training(raw_csv_path, out_dir, *, classifier="logreg", sample_size=5_000, test_size=0.2, random_state=42)`.
  - Carrega CSV bruto.
  - Aplica `prepare_dataset` e `split_dataset`.
  - Fita o pipeline.
  - Calcula accuracy, macro-F1, weighted-F1, classification report por classe, matriz de confusão.
  - Serializa em `models/<versão>/{model.joblib, classes.json, metadata.json}`.
  - Salva figuras em `reports/figures/`.
  - Retorna dicionário com métricas e caminhos.
- CLI: `python -m triage_ml.models.train --classifier logreg`.

### F1.T3. Tests do baseline
- `tests/test_model_pipeline.py`: cobre `build_pipeline`, shapes, presença de `predict_proba` no LR.
- `tests/test_model_artifact.py`: round-trip `model.joblib` + `metadata.json` + `classes.json`; assert `classes == model.classes_`.

### F1.T4. API de smoke
- `src/triage_ml/api/schemas.py`:
  - `PredictIn(text: constr(strip_whitespace=True, min_length=1, max_length=20000))`.
  - `PredictOut(label, label_name, score, model_version, latency_ms, request_id, warnings)`.
  - `HealthOut(status, model_version, model_loaded)`.
  - `ErrorOut(request_id, error_code, message)`.
- `src/triage_ml/api/app.py`:
  - `GET /health` → `HealthOut`.
  - `POST /predict` → `PredictIn` → `PredictOut`.
- Carregamento preguiçoso do modelo (`lifespan`), com `MODEL_PATH` configurável por env var.
- Mapeamento `condition_label → condition_name` lido de `data/medical_tc_labels.csv`.
- Middleware/dependência:
  - Gera `request_id` (`uuid.uuid4().hex[:12]`) em `request.state.request_id`.
  - Mede `latency_ms` com `time.perf_counter()` em torno do `pipeline.predict`/`predict_proba`.
  - Ecoa `request_id` em `X-Request-ID`.
  - Emite `Server-Timing: predict;dur=<latency_ms>`.
  - Captura exceções, retorna `ErrorOut` com `error_code` genérico; loga apenas `request_id` + classe prevista (nunca `text`).
- `uvicorn triage_ml.api.app:app --reload` deve subir e responder nos dois endpoints.

### F1.T5. Teste manual e evidências
- Subir a API local, enviar 5 abstracts (incluindo 1 da classe 1 e 1 da classe 5) via `curl`/`httpie` e salvar respostas em `reports/figures/api_smoke.log`.
- Para cada resposta, registrar `request_id`, `label`, `score`, `latency_ms`, headers `X-Request-ID` e `Server-Timing`. Confirmar que `latency_ms` varia entre chamadas.
- Validar `metadata.json` (chaves, classes, versões).
- Validar que `text` vazio retorna `422` Pydantic sem vazar conteúdo; `prediction_failed` retorna `ErrorOut` com `request_id`.
- Salvar `reports/figures/api_smoke.log`.

### F1.T6. Documentação
- Atualizar `docs/CHECKLIST.md`: Etapa 2 → `[~]` em progresso, depois `[x]` com evidência. Não tocar em itens de outros donos.
- Adicionar seção "Modelo (Bill)" no `README.md` resumindo abordagem, classes e como rodar treino + API local; **incluir a justificativa formal de não-uso de Random Forest**.
- Atualizar `.agents/contracts/README.md` se o formato de `metadata.json` divergir.

## F1. Critérios de aceite

Mapeados na Etapa 2 do `docs/CHECKLIST.md`:

- [x] Baseline TF-IDF + classificador Scikit-Learn (LR e/ou LinearSVC).
- [x] Seeds, preprocessing e versões fixas.
- [x] Métricas por classe e agregadas, com figuras em `reports/figures/`.
- [x] Modelo e metadados serializados segundo contrato.
- [x] API de smoke local (`/health` + `/predict`) consumindo o artefato, com `latency_ms`, `request_id` e headers.

## F1. Riscos específicos

| Risco | Mitigação |
|---|---|
| Divergência entre esta API e a API oficial de Romário (Etapa 3) | Marcar como "smoke/provisória" no README; alinhar contrato Pydantic com Romário **antes** de qualquer promoção |
| Modelo não serializa classes corretamente | `tests/test_model_artifact.py` assert `classes == model.classes_` |
| Conteúdo clínico em logs | Não logar `text` em nenhum caminho; revisão de PR |
| CI quebrando | Fixar versões em `pyproject.toml`/`uv.lock`; preferir libs já presentes |

## F1. Sequência de commits (apenas Fase 1)

1. `chore: scaffold triage_ml.models package and training config`
2. `feat(models): tf-idf + logistic regression pipeline with reproducible training`
3. `test(models): cover pipeline fit/predict and artifact round-trip`
4. `feat(api): minimal fastapi smoke app exposing /health and /predict`
5. `docs(models): update checklist and readme for the baseline classifier`

## F1. Definição de pronto da Fase 1

- Itens da Etapa 2 marcados com evidência no checklist.
- `uv run pytest` e `uv run ruff check .` verdes.
- API sobe com `uvicorn triage_ml.api.app:app` e responde `/health` e `/predict` com o artefato.
- `README.md` e `CHECKLIST.md` refletem o estado real.

---

# Fase 2 — Otimização + observabilidade (Etapas 5 e 6 do checklist)

Pré-requisito: Fase 1 concluída e API oficial do Romário (Etapa 3) em estado de servir o artefato. Também depende do CI/Docker (Etapa 4) para que a stack Compose seja reprodutível.

## F2. Contexto e objetivo

Entregar:

1. Variante otimizada do classificador (ONNX, quantização ou pruning) com benchmark baseline vs otimizado **nas mesmas entradas e split**.
2. Stack de observabilidade completa: `prometheus_client` na API, `prometheus.yml`, Grafana provisionado, Compose com API+Prometheus+Grafana, dashboard versionado com painel extra "baseline vs otimizado".
3. Garantia formal de que `text` nunca aparece em logs, payloads de erro ou labels de métrica.

## F2. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Técnica de otimização | Export **ONNX** via `skl2onnx.convert_sklearn` | Técnica vista em aula, com maturidade, sem alterar a API Python do classificador; quantização fica como experimento secundário documentado |
| Runtime ONNX | `onnxruntime` (CPU) | Já é o padrão; abstração do backend; suporta LR/SVM linear com `zipmap=False` |
| Alternativa avaliada | Quantização dinâmica ONNX | Documentada como experimento adicional; só promover se a primeira não cumprir o critério |
| Critério de aceitação | Δ macro-F1 ≤ 1 pp e redução ≥ 20% em p95 de latência no split de teste | Margem rígida contra regressão e piso razoável de ganho |
| Métricas Prometheus | `prometheus_client` + middleware FastAPI próprio (sem `prometheus_fastapi_instrumentator` por padrão) | Controle sobre labels e buckets do histograma; instrumentator pode entrar depois se Romário quiser |
| Labels aceitas | `route`, `method`, `status`, `model_variant` | Baixa cardinalidade; **nunca** `text`, `label_name`, `request_id` |
| Buckets do histograma | `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5]` (s) | Cobre o range esperado para inferência TF-IDF |
| Compose | `infra/docker-compose.yml` com serviços `api`, `prometheus`, `grafana` | Reproduz local; mesma base serve para a Etapa 8 (cloud) |
| Dashboard | JSON em `monitoring/grafana/dashboards/triage_ml.json` + print em `reports/figures/` | Reprocessamento via provisioning do Grafana |

## F2. Estrutura de arquivos a criar (incremento sobre Fase 1)

```
src/triage_ml/
├── models/
│   ├── optimize.py            # export ONNX + quantização opcional
│   └── benchmark.py           # latência p50/p95/p99 + macro-F1 comparativo
└── monitoring/
    ├── __init__.py
    ├── metrics.py             # Counter/Histogram Prometheus compartilhados
    └── middleware.py          # middleware FastAPI que mede e expõe
monitoring/
├── prometheus/
│   ├── prometheus.yml
│   └── alerts.yml             (opcional, versão inicial sem)
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasource.yml
    │   └── dashboards/dashboards.yml
    └── dashboards/
        └── triage_ml.json
infra/
└── docker-compose.yml         (somente os serviços de observabilidade aqui; API vai em outro compose de Fábio, alinhado)
tests/
├── test_model_optimization.py
└── test_monitoring_metrics.py
reports/
└── figures/
    ├── 09_latency_comparison.png
    └── 09_dashboard.png
```

## F2. Contratos a cumprir

Evolução dos contratos da Fase 1:

- **Modelo otimizado**:
  - Carrega via `onnxruntime.InferenceSession` com a mesma interface (`predict`/`predict_proba`) usada pelo pipeline sklearn.
  - Variação escolhida por env var `MODEL_VARIANT=sklearn|onnx`.
  - `metadata.json` ganha campos `model_variant`, `onnx_path`, `benchmark` (com p50/p95/p99 e macro-F1 de ambos).
- **API**:
  - Continua expondo `latency_ms`, `request_id`, `X-Request-ID`, `Server-Timing`.
  - Acrescenta `/metrics` (Prometheus text format) servido pelo `prometheus_client.generate_latest`.
  - `model_variant` aparece como **label** (não no body) para permitir comparativo via Prometheus.
- **Observabilidade**:
  - Métricas: `triage_ml_requests_total{route,method,status,model_variant}`, `triage_ml_request_latency_seconds{route,method,model_variant}`, `triage_ml_prediction_errors_total{route,error_code,model_variant}`.
  - **Sem** `text`, **sem** `label`, **sem** `request_id` em labels.
- **Privacidade** (subseção nova da Etapa 6 do checklist):
  - Teste automatizado varre `metadata.json`, fixtures de erro e logs gerados em teste; falha se encontrar string que case com `medical_abstract` ou outro texto de `data/`.

## F2. Tarefas e sequência

### F2.T1. Otimização ONNX
- `optimize.py`: `export_onnx(pipeline, out_path, *, opset=17)` via `skl2onnx.convert_sklearn`.
- Restrição: `LinearSVC` precisa de wrapper para `predict_proba` (Calibrado) ou então expor apenas `decision_function` e mapear para classe. Documentar.
- Salvar `models/<versão>/model.onnx`.

### F2.T2. Benchmark comparativo
- `benchmark.py`: `benchmark(predict_fn, texts, *, warmup=50, n=1000, model_variant)` retorna p50/p95/p99 (ms) e macro-F1.
- Carrega split de teste preparado pela Fase 1, roda em ambos `model.joblib` e `model.onnx`, escreve `models/<versão>/benchmark.json` e gera `reports/figures/09_latency_comparison.png`.

### F2.T3. Critério "sem degradação inaceitável"
- Regra implementada em teste: Δ macro-F1 ≤ 1 pp **e** redução ≥ 20% no p95 da latência, medidos no **mesmo split e nas mesmas entradas** (não em amostras aleatórias novas).
- Teste falha se o critério não for cumprido; reporta valores no output.

### F2.T4. Instrumentação Prometheus
- `monitoring/metrics.py`:
  - Define `REQUESTS_TOTAL`, `REQUEST_LATENCY_SECONDS`, `PREDICTION_ERRORS_TOTAL` com buckets e labels especificados.
  - Função `render_metrics()` retorna o `generate_latest()` do registry próprio (não usar o registry global para evitar vazamento de métricas de bibliotecas).
- `monitoring/middleware.py`:
  - Middleware FastAPI que mede latência total, contabiliza erros com `error_code` genérico, e popula `Server-Timing` (já existente) e o histograma.
- API oficial (Romário) ou o `app.py` da Fase 1 passa a usar esse middleware.

### F2.T5. Endpoint `/metrics`
- Adicionar rota `GET /metrics` retornando `Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)`.
- Garantir que o endpoint **não** está atrás de autenticação nesta fase (será revisitado na Etapa 8).

### F2.T6. Compose, Prometheus e Grafana
- `monitoring/prometheus/prometheus.yml`: scrape da API em `api:8000/metrics` a cada 5 s.
- `monitoring/grafana/provisioning/datasources/datasource.yml`: provisiona o Prometheus como datasource.
- `monitoring/grafana/provisioning/dashboards/dashboards.yml`: provisiona o dashboard `triage_ml.json`.
- `infra/docker-compose.yml`: serviços `api`, `prometheus`, `grafana` na mesma rede. Variáveis de compose documentam como alternar `MODEL_VARIANT=sklearn|onnx`.

### F2.T7. Dashboard
- Painéis mínimos:
  1. **Requisições por rota/status** (stat panel + gráfico de barras empilhadas).
  2. **Latência p95** (timeseries) com filtro `model_variant`.
  3. **Taxa de erros** (stat panel).
  4. **Comparativo baseline vs otimizado** (table panel lendo `triage_ml_request_latency_seconds_bucket` filtrado por `model_variant`).
- Print do dashboard em `reports/figures/09_dashboard.png`.
- JSON versionado em `monitoring/grafana/dashboards/triage_ml.json`.

### F2.T8. Teste de privacidade
- `tests/test_monitoring_metrics.py`: chama `/predict` com texto controlado, captura logs e métricas, assert que `text` **não** aparece em nenhum label e nem nos payloads de erro.
- Regex de varredura: padrões como `medical_abstract`, trechos do fixture, headers sensíveis.

### F2.T9. Documentação
- Atualizar `docs/CHECKLIST.md`: Etapa 5 e Etapa 6 marcadas com evidência (figuras, `benchmark.json`, JSON do dashboard, print).
- Adicionar seção "Otimização e observabilidade" no `README.md` com comandos `docker compose up`, URLs locais e como ler o dashboard.
- Atualizar `.agents/contracts/README.md` se a API ganhar `/metrics`.

## F2. Critérios de aceite

Mapeados nas Etapas 5 e 6 do `docs/CHECKLIST.md`:

**Etapa 5:**
- [ ] Otimização aplicada (ONNX).
- [ ] Baseline vs otimizado comparados nas mesmas entradas.
- [ ] Melhoria de latência demonstrada (≥ 20% no p95) sem degradação inaceitável (Δ macro-F1 ≤ 1 pp).
- [ ] `model.onnx` e `benchmark.json` persistidos.

**Etapa 6:**
- [ ] Métricas `prometheus_client` expostas em `/metrics`.
- [ ] Total de requisições por rota/status, latência e erros medidos.
- [ ] Sem labels de alta cardinalidade e sem conteúdo clínico (teste automatizado).
- [ ] Compose com API, Prometheus e Grafana.
- [ ] Dashboard JSON reprodutível com pelo menos 4 painéis (incluindo comparativo).
- [ ] Print e JSON do dashboard em `reports/figures/`.

## F2. Riscos específicos

| Risco | Mitigação |
|---|---|
| `skl2onnx` não suportar alguma combinação (ex.: `LinearSVC` puro) | Testar a conversão já em F2.T1; fallback documentado para LR puro |
| Latência ONNX não melhorar no ambiente local | Critério exige ≥ 20% de redução; se não cumprir, registrar como evidência e tentar quantização dinâmica antes de voltar atrás |
| Dashboard divergir entre versões do Grafana | Fixar versão da imagem no Compose; testar provisionamento no CI quando possível |
| `prometheus_client` registrar texto clínico acidentalmente | Teste de privacidade em F2.T8 + revisão de PR |
| Compose local pesado para o time | Documentar pré-requisitos; CI só faz lint+pytest, não sobe Compose |

## F2. Sequência de commits (Fase 2, em adição aos 5 da Fase 1)

6. `feat(models): onnx export and latency benchmark for the baseline classifier`
7. `test(models): enforce no-regression latency and macro-f1 in optimization`
8. `feat(monitoring): prometheus metrics middleware for /health and /predict`
9. `feat(api): expose /metrics and ship prometheus + grafana compose`
10. `feat(monitoring): provision grafana dashboard with baseline vs optimized panel`
11. `test(monitoring): privacy regression test for logs and metric labels`
12. `docs(models): document optimization, observability and dashboard usage`

## F2. Definição de pronto da Fase 2

- Etapas 5 e 6 marcadas como `[x]` no checklist, com evidência.
- `uv run pytest` e `uv run ruff check .` verdes.
- `docker compose up` sobe API, Prometheus e Grafana; dashboard visível com os 4 painéis.
- Critério de "sem degradação inaceitável" cumprido em CI (ou evidência registrada quando CI local não for suficiente).
- `README.md` e `CHECKLIST.md` refletem o estado real.

---

# Próximas fases (fora do escopo deste plano)

- **API oficial** (Etapa 3, Romário): substitui ou incorpora a API de smoke.
- **CI/CD e Docker da imagem da API** (Etapa 4, Fábio): cria o Dockerfile e o build no CI; este plano depende disso, mas não é dono.
- **DAG Airflow** (Etapa 7, Denis): consome `triage_ml.models.train.run_training` definido na Fase 1.
- **Cloud, ADR, vídeo STAR, documentação final** (Etapa 8, Romário + Fábio).
