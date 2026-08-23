# Plano de implementação — Classificador de Texto NLP (Bill)

- **Integrante**: Bill
- **Origem**: Tech Challenge — Fase 3 (ML Engineering)
- **Status**: Fase 1 entregue e revisada; Fase 2 pendente
- **Mapeamento no checklist**: cobre as Etapas 2, 5 e 6 do `docs/CHECKLIST.md` reordenado (2026-08-23). Está organizado em duas fases para deixar claro o que é trabalho **agora** e o que é trabalho **depois**.

## Estrutura por fases

| Fase | Etapas do checklist | Peso oficial | Conteúdo |
|---|---|---|---|
| **Fase 1 — Modelo baseline + API de smoke** | Etapa 2 | Parte do item modelo + otimização (20%) | Treino, métricas, serialização, API de smoke local |
| **Fase 2 — Otimização + observabilidade** | Etapas 5 + 6 | Etapa 5 completa modelo + otimização (20%); Etapa 6 cobre observabilidade (20%) | ONNX, benchmark, Prometheus/Grafana/Compose, dashboard comparativo |

A API oficial FastAPI (Etapa 3) é trabalho do Romário e não está no escopo deste plano; ela consome o artefato e o contrato de `metadata.json` definidos na Fase 1.

## Pré-requisitos do plano

- **Etapa 1 do checklist concluída**: dataset definido (`triage_ml.data.prepare.prepare_dataset`), contratos estáveis em `.agents/contracts/README.md`.
- **Gate semântico aprovado pelo time**: confirmar que o produto entregue classifica categorias clínicas, e não níveis de urgência, ou alterar dataset/labels antes de estabilizar o contrato do modelo.
- **Alinhamento com Romário**: contrato de `POST /predict` validado antes de concluir e promover a Fase 1.
- **Contrato de artefato validado por Denis**: caminho versionado, metadados e regeneração do split devem ser consumíveis pela futura DAG sem lógica duplicada.

## Mudanças em relação à versão anterior do plano

A versão anterior deste arquivo cobria apenas a Fase 1 e tratava otimização/observabilidade como "próxima fase sem detalhe". Agora:

- **Fase 1 mantida** com as melhorias acumuladas (latência exposta, `request_id`, `Server-Timing`).
- **Fase 2 detalhada** com tarefas concretas (otimização, benchmark, Prometheus, Compose, dashboard).
- **Critérios de aceite remapeados** para os itens oficiais do checklist (Etapas 2, 5 e 6), com referência explícita aos números de linha.
- **Riscos revisitados** incluindo os da Fase 2 (dependência ONNX, custo de manter Compose local).
- **Sequência de commits estendida** para incluir as entregas da Fase 2.
- **Revisão técnica de 2026-08-23**: solver multiclasses corrigido, seleção sem uso indevido do teste, contrato de artefato detalhado, testes automatizados da API, benchmark isolado do CI comum e Compose comparativo com duas variantes.

---

# Fase 1 — Modelo baseline + API de smoke (Etapa 2 do checklist)

## F1. Contexto e objetivo

Entregar:

1. Classificador NLP leve (TF-IDF + classificador linear Scikit-Learn) treinado no recorte preparado pela fundação (5.000 amostras, seed 42).
2. Artefato serializado segundo o contrato (`models/<versão>/model.joblib` + `metadata.json`), com classes e nomes no próprio manifesto.
3. Métricas por classe e agregadas, com figuras em `reports/figures/`.
4. API FastAPI de smoke local (`/health` + `/predict`) consumindo o artefato, com `latency_ms` e `request_id` já expostos para reuso na Fase 2.

A API oficial é do Romário (Etapa 3). Esta API de smoke é substituída ou estendida por ele; o contrato (esquemas Pydantic, headers) é ponto de alinhamento obrigatório antes de qualquer promoção.

## F1. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Vetorizador | `TfidfVectorizer` | Sugestão do enunciado; leve, determinístico, sem dependência externa |
| Candidatos do baseline | `LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")` e `LinearSVC(class_weight="balanced")` | Comparados por macro-F1 em validação estratificada somente no treino |
| Modelo selecionado | `LinearSVC` | Maior macro-F1 médio na validação (`0.7335` contra `0.7319`); o teste permaneceu isolado até a escolha |
| Por que não Random Forest? | — | Exemplo do enunciado. Em TF-IDF, RF explode o custo de inferência (centenas de árvores) sem ganho consistente sobre modelos lineares em texto. Justificativa registrada no README seção Bill |
| Serialização | `joblib` para o pipeline scikit-learn | Padrão sklearn |
| API (smoke) | FastAPI + Uvicorn, em processo local sem Docker | Suficiente para teste manual; Docker e Compose ficam para Etapa 4 (Fábio) e Fase 2 |
| Seeds | 42 em todos os pontos estocásticos | Reprodutibilidade exigida pelo checklist |

Referência do paper (Schopf et al., NLPIR 2022) no Medical Corpus, F1 micro (unsupervised): LSA 31,6; SBERT MiniLM 46,5; DeBERTa zero-shot 57,3. Esses números ficam apenas como contexto narrativo, pois usam metodologia diferente e não são meta nem evidência antecipada de superioridade do baseline supervisionado.

## F1. Estrutura de arquivos a criar

```
src/triage_ml/
├── data/prepare.py            (existente, não modificar)
├── models/
│   ├── __init__.py
│   ├── pipeline.py            # fábrica do Pipeline TF-IDF + classificador
│   ├── train.py               # treino + métricas + serialização
│   └── artifact.py            # lê/valida/grava model.joblib e metadata.json
└── api/
    ├── __init__.py
    ├── app.py                 # FastAPI mínimo com /health e /predict
    └── schemas.py             # Pydantic de entrada/saída
tests/
├── test_model_pipeline.py
├── test_model_artifact.py
└── test_api.py
models/
└── README.md                  (existente; artefatos permanecem fora do Git)
reports/
├── evidence/
│   └── api-smoke.json         # resposta sanitizada, sem os textos enviados
└── figures/
    ├── 08_confusion_matrix_linear_svc.png
    └── 08_top_features_linear_svc.png
configs/
└── training.yaml              # hiperparâmetros e label mapping versionados
```

A pasta `monitoring/` não é tocada na Fase 1.

## F1. Contratos a cumprir

Pontos relevantes de `.agents/contracts/README.md`:

- **Dados**: `prepare_dataset` garante amostra sem duplicatas e seed 42. O treino registra fingerprints do CSV preparado e dos índices de cada split para comprovar que a avaliação e o benchmark usam as mesmas entradas.
- **Seleção e avaliação**: Logistic Regression e LinearSVC são comparados somente no treino por validação estratificada. O classificador é fixado antes da avaliação final no teste; a escolha não usa métricas do test set.
- **Modelo**: recebe lista de textos; devolve classe e score quando disponível. `score` representa confiança do classificador e não deve ser descrito como probabilidade calibrada sem avaliação específica.
- **Artefato**: `metadata.json` é a fonte canônica para `schema_version`, `model_version`, `task_type`, idioma, classes e nomes, configuração do pipeline, seed, métricas, versões das dependências, commit Git, fingerprints de dataset/splits e checksum de `model.joblib`.
- **Versão**: `<versão>` segue `YYYYMMDDTHHMMSSZ-<input_hash_curto>`, com hash derivado do dataset preparado e da configuração, e nunca sobrescreve um diretório existente. O carregador aceita apenas artefatos locais/confiáveis e valida schema, classes e checksum antes de desserializar o `joblib`.
- **API (proposta, sujeito à validação de Romário antes de promover)**:
  - `GET /health` → `HealthOut(status, model_version, model_loaded)`.
  - `POST /predict` → `PredictIn(text)` → `PredictOut(label, label_name, score?, model_version, latency_ms, request_id, warnings)`, com score opcional para classificadores sem `predict_proba`.
  - Erros: `ErrorOut(request_id, error_code, message)` — nunca conteúdo clínico. Um handler próprio sanitiza o `422` do FastAPI/Pydantic, removendo o campo `input` antes da resposta.
  - Headers: `X-Request-ID` ecoando o `request_id`; `Server-Timing: predict;dur=<latency_ms>`.
- **Observabilidade**: adiada como stack completa. Mas a Fase 1 já nasce expondo latência e `request_id` em toda resposta para reuso na Fase 2.

### O que **não** entra na Fase 1

- Endpoint `GET /metrics` Prometheus.
- Middleware `prometheus_fastapi_instrumentator`.
- Otimização ONNX, quantização ou pruning.
- Compose, Prometheus, Grafana, dashboard.
- Autenticação, rate limit, tracing distribuído.

## F1. Tarefas e sequência

Por decisão explícita de Bill em 2026-08-23, o trabalho desta semana será feito diretamente na `main`, pois ele é o único colaborador ativo no projeto durante o período. Esta é uma exceção temporária e autorizada ao workflow padrão: antes de cada incremento, confirmar `main` sincronizada e worktree limpa; manter commits semânticos pequenos; nunca usar force-push nem incluir alterações alheias.

### F1.T1. Esqueleto e configuração
- Criar `src/triage_ml/models/{__init__.py, pipeline.py, artifact.py}`.
- Criar `configs/training.yaml` com hiperparâmetros e mapeamento `condition_label → condition_name` versionados; o treino copia esse mapeamento para `metadata.json`.
- Declarar diretamente `joblib`, `PyYAML`, `fastapi` e `uvicorn` no `pyproject.toml`; dependências de notebook permanecem no grupo de desenvolvimento.
- Smoke test: `uv run python -c "from triage_ml.models.pipeline import build_pipeline; print(build_pipeline())"`.

### F1.T2. Pipeline e treino
- `pipeline.py`: `build_pipeline(classifier="logreg", config=...)` retorna `Pipeline([("tfidf", ...), ("clf", ...)])` com parâmetros explícitos e validados.
- `train.py`: `run_training(raw_csv_path, out_dir, *, classifier=None, sample_size=5_000, test_size=0.2, random_state=42)`; `classifier=None` seleciona por CV e um valor explícito funciona como override auditável.
  - Carrega CSV bruto.
  - Aplica `prepare_dataset` e `split_dataset`.
  - Compara LR e LinearSVC por validação cruzada estratificada somente no conjunto de treino, usando macro-F1 como métrica primária.
  - Fixa a escolha, refaz o fit no treino completo e usa o test set somente para avaliação do artefato congelado, nunca para seleção ou ajuste.
  - Calcula accuracy, balanced accuracy, macro-F1, weighted-F1, classification report por classe e matriz de confusão.
  - Serializa em `models/<versão>/{model.joblib, metadata.json}` e grava fingerprints suficientes para regenerar e verificar os splits sem persistir textos no Git.
  - Salva figuras em `reports/figures/`.
  - Retorna dicionário com métricas e caminhos.
- CLI: `python -m triage_ml.models.train` seleciona por CV; `--classifier logreg|linear_svc` força um candidato e registra o override.

### F1.T3. Tests do baseline
- `tests/test_model_pipeline.py`: usa fixture sintética pequena e cobre fit multiclasses, shapes, configuração reproduzível e presença de `predict_proba` no LR.
- `tests/test_model_artifact.py`: round-trip `model.joblib` + `metadata.json`; valida schema, checksum, fingerprints, label mapping e `metadata.classes == model.classes_`.

### F1.T4. API de smoke
- `src/triage_ml/api/schemas.py`:
  - `PredictIn(text: constr(strip_whitespace=True, min_length=1, max_length=20000))`.
  - `PredictOut(label, label_name, score: float | None, model_version, latency_ms, request_id, warnings)`.
  - `HealthOut(status, model_version, model_loaded)`.
  - `ErrorOut(request_id, error_code, message)`.
- `src/triage_ml/api/app.py`:
  - `GET /health` → `HealthOut`.
  - `POST /predict` → `PredictIn` → `PredictOut`.
- App factory com injeção do carregador nos testes; carregamento no startup via `lifespan`, com falha rápida para artefato ausente/incompatível e `MODEL_PATH` configurável por env var.
- Mapeamento `condition_label → condition_name` lido do `metadata.json`, sem depender de CSV ignorado pelo Git em runtime.
- Handler de `RequestValidationError` remove valores de entrada do `422` e responde no formato `ErrorOut`.
- Middleware/dependência:
  - Gera `request_id` (`uuid.uuid4().hex[:12]`) em `request.state.request_id`.
  - Mede `latency_ms` com `time.perf_counter()` em torno do `pipeline.predict`/`predict_proba`.
  - Ecoa `request_id` em `X-Request-ID`.
  - Emite `Server-Timing: predict;dur=<latency_ms>`.
  - Captura exceções inesperadas, preserva erros HTTP conhecidos e retorna `ErrorOut` com `error_code` genérico; loga apenas `request_id`, rota, status e latência (nunca `text` nem corpo da requisição).
- `uvicorn triage_ml.api.app:app --reload` deve subir e responder nos dois endpoints.
- `tests/test_api.py` cobre `/health`, `/predict`, artefato inválido, texto vazio, sanitização de erro, `X-Request-ID`, `Server-Timing` e ausência do texto em logs/respostas de erro.

### F1.T5. Teste manual e evidências
- Subir a API local, enviar 5 abstracts (incluindo 1 da classe 1 e 1 da classe 5) via `curl`/`httpie` e salvar somente respostas sanitizadas em `reports/evidence/api-smoke.json`; os textos de entrada não são persistidos.
- Para cada resposta, registrar `request_id`, `label`, `score`, `latency_ms`, headers `X-Request-ID` e `Server-Timing`. Confirmar que `latency_ms` varia entre chamadas.
- Validar `metadata.json` (chaves, classes, versões).
- Validar que `text` vazio retorna `422` Pydantic sem vazar conteúdo; `prediction_failed` retorna `ErrorOut` com `request_id`.
- Confirmar que a evidência versionável não contém os abstracts nem campos `input` do Pydantic.

### F1.T6. Documentação
- Atualizar `docs/CHECKLIST.md`: Etapa 2 → `[~]` em progresso, depois `[x]` com evidência. Não tocar em itens de outros donos.
- Adicionar seção "Modelo (Bill)" no `README.md` resumindo tarefa real (categorias clínicas), abordagem, classes e como rodar treino + API local; incluir a justificativa formal de não-uso de Random Forest.
- Atualizar `.agents/contracts/README.md` se o formato de `metadata.json` divergir.

## F1. Critérios de aceite

Mapeados na Etapa 2 do `docs/CHECKLIST.md`:

- [x] Baseline TF-IDF + classificador Scikit-Learn selecionado sem usar o test set.
- [x] Seeds, preprocessing, fingerprints e versões fixas.
- [x] Métricas por classe e agregadas, com figuras em `reports/figures/`.
- [x] Modelo e metadados serializados segundo contrato e validados por checksum.
- [x] API de smoke local (`/health` + `/predict`) consumindo o artefato, com erros sanitizados, `latency_ms`, `request_id` e headers.

## F1. Riscos específicos

| Risco | Mitigação |
|---|---|
| Contrato fala em urgência, mas o dataset possui categorias clínicas | Gate humano antes de estabilizar labels, metadata e API; documentar a decisão no README/ADR aplicável |
| Divergência entre esta API e a API oficial de Romário (Etapa 3) | Marcar como "smoke/provisória" no README; alinhar contrato Pydantic com Romário **antes** de qualquer promoção |
| Seleção otimista pelo test set | Comparar modelos somente por validação estratificada no treino e fixar a escolha antes da avaliação final |
| Modelo não serializa classes corretamente | `tests/test_model_artifact.py` valida `metadata.classes == model.classes_`, schema e checksum |
| Conteúdo clínico em logs ou no `422` | Handler de validação sanitizado, testes automatizados e revisão das evidências antes de versionar |
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
- Treino reproduzível a partir de clone limpo, dado o CSV local versionado e o `data/medical_tc_labels.csv` (mapeamento `condition_label → condition_name`).
- `README.md` e `CHECKLIST.md` refletem o estado real.

---

# Fase 2 — Otimização + observabilidade (Etapas 5 e 6 do checklist)

Pré-requisito: Fase 1 concluída e API oficial do Romário (Etapa 3) em estado de servir o artefato. Também depende do CI/Docker (Etapa 4) para que a stack Compose seja reprodutível.

## F2. Contexto e objetivo

Entregar:

1. Variante otimizada do classificador (ONNX, quantização ou pruning) com benchmark baseline vs otimizado **nas mesmas entradas e split**.
2. Stack de observabilidade completa: `prometheus_client` na API, `prometheus.yml`, Grafana provisionado, Compose com as duas variantes da API + Prometheus + Grafana e dashboard versionado com painel extra "baseline vs otimizado".
3. Garantia formal de que `text` nunca aparece em logs, payloads de erro ou labels de métrica.

## F2. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Técnica de otimização | Export **ONNX** via `skl2onnx.convert_sklearn` | Técnica vista em aula; uma prova antecipada valida a conversão do pipeline completo, incluindo TF-IDF |
| Runtime ONNX | `onnxruntime` (CPU) | Backend acessado por adapter próprio com o mesmo contrato de predição do pipeline sklearn e `zipmap=False` |
| Alternativa avaliada | Quantização dinâmica ONNX | Experimento secundário, sem pressupor compatibilidade ou ganho; avaliado na validação antes de promover |
| Critério de aceitação | Δ macro-F1 ≤ 1 pp e redução ≥ 20% em p95 de latência no split de teste | Margem rígida contra regressão e piso razoável de ganho |
| Métricas Prometheus | `prometheus_client` + middleware FastAPI próprio (sem `prometheus_fastapi_instrumentator` por padrão) | Controle sobre labels e buckets do histograma; instrumentator pode entrar depois se Romário quiser |
| Labels aceitas | `route`, `method`, `status`, `model_variant` | Baixa cardinalidade; **nunca** `text`, `label_name`, `request_id` |
| Buckets do histograma | `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5]` (s) | Cobre o range esperado para inferência TF-IDF |
| Compose | `infra/docker-compose.yml` com `api-sklearn`, `api-onnx`, `prometheus` e `grafana` | Permite comparação simultânea das variantes e atende à stack local; alinhamento com o Dockerfile de Fábio é obrigatório |
| Dashboard | JSON em `monitoring/grafana/dashboards/triage_ml.json` + print em `reports/figures/` | Reprocessamento via provisioning do Grafana |

## F2. Estrutura de arquivos a criar (incremento sobre Fase 1)

```
src/triage_ml/
├── models/
│   ├── optimize.py            # export ONNX + quantização opcional
│   ├── onnx_adapter.py        # interface comum para InferenceSession
│   └── benchmark.py           # benchmark controlado + relatório comparativo
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
└── docker-compose.yml         # duas APIs + Prometheus + Grafana, usando Dockerfile alinhado com Fábio
scripts/
└── generate_observability_traffic.py  # popula métricas sem persistir os textos
tests/
├── test_model_optimization.py
└── test_monitoring_metrics.py
reports/
├── benchmarks/
│   └── benchmark.json         # evidência agregada e versionável, sem entradas
└── figures/
    ├── 09_latency_comparison.png
    └── 09_dashboard.png
```

## F2. Contratos a cumprir

Evolução dos contratos da Fase 1:

- **Modelo otimizado**:
  - Carrega via adapter sobre `onnxruntime.InferenceSession` com a mesma interface (`predict`/`predict_proba`) usada pelo pipeline sklearn.
  - Variação escolhida em runtime por env var `MODEL_VARIANT=sklearn|onnx`.
  - `metadata.json` declara `available_variants: ["sklearn", "onnx"]`, `checksum_sha256` do `model.onnx` e a versão fixa da variante ativa no treinamento. Os resultados detalhados de latência/qualidade ficam em `reports/benchmarks/benchmark.json`, evitando duas fontes divergentes.
- **API**:
  - Continua expondo `latency_ms`, `request_id`, `X-Request-ID`, `Server-Timing`.
  - Acrescenta `/metrics` (Prometheus text format) servido pelo `prometheus_client.generate_latest`.
  - `model_variant` aparece como **label** (não no body) para permitir comparativo via Prometheus.
- **Observabilidade**:
  - Métricas: `triage_ml_requests_total{route,method,status,model_variant}`, `triage_ml_request_latency_seconds{route,method,model_variant}`, `triage_ml_prediction_errors_total{route,error_code,model_variant}`.
  - **Sem** `text`, **sem** `label`, **sem** `request_id` em labels.
- **Privacidade** (subseção nova da Etapa 6 do checklist):
  - Teste automatizado varre respostas de erro, métricas e logs gerados em teste; falha se encontrar o **texto controlado da fixture** (string completa e trechos exclusivos), os campos `input`/`text` indevidos ou headers sensíveis. Asserção adicional: `metadata.json` é validado por schema e nenhum de seus campos contém texto do dataset. O teste **carrega apenas o fixture**, nunca o dataset bruto.

## F2. Tarefas e sequência

### F2.T1. Otimização ONNX
- Fazer primeiro um spike de conversão do pipeline LR completo, incluindo `TfidfVectorizer`, e registrar limitações antes de implementar a abstração definitiva.
- Declarar diretamente `skl2onnx`, `onnx` e `onnxruntime` em um grupo de dependências de otimização.
- `optimize.py`: `export_onnx(pipeline, out_path, *, opset=17)` via `skl2onnx.convert_sklearn`.
- `onnx_adapter.py`: normaliza entradas/saídas do `InferenceSession`, classes e scores sem alterar o contrato consumido pela API.
- Restrição: `LinearSVC` precisa de wrapper para `predict_proba` (Calibrado) ou então expor apenas `decision_function` e mapear para classe. Documentar.
- Salvar `models/<versão>/model.onnx`.

### F2.T2. Benchmark comparativo
- Decisões entre ONNX puro e alternativa quantizada são feitas em validação. Depois de fixada a variante, o test set é usado para o comparativo final nas mesmas entradas e ordem.
- `benchmark.py`: mede batch size 1 e retorna p50/p95/p99, média, throughput, tempo de carregamento, tamanho do artefato, macro-F1, concordância de classes e desvio máximo dos scores.
- Metodologia fixa warmup, número de repetições, threads sklearn/ONNX e fronteira de medição incluindo TF-IDF + inferência, mas excluindo carregamento. Hardware, SO, Python, dependências e parâmetros são registrados.
- Regenera o split a partir dos parâmetros da Fase 1, valida seus fingerprints, roda ambos `model.joblib` e `model.onnx` e escreve o resultado local em `models/<versão>/benchmark.json` e a evidência sem entradas em `reports/benchmarks/benchmark.json`.
- Gera `reports/figures/09_latency_comparison.png` a partir do JSON versionável.

### F2.T3. Critério "sem degradação inaceitável"
- Testes determinísticos do CI validam conversão, shape, classes, concordância de predições e tolerância de qualidade (Δ macro-F1 ≤ 1 pp), sem impor limite de tempo dependente de hardware.
- O benchmark controlado avalia redução ≥ 20% no p95, nas mesmas entradas e condições. O resultado é critério de promoção documentado, não teste unitário no runner compartilhado do GitHub Actions.
- Se ONNX puro não cumprir o piso na validação, avaliar quantização; somente a alternativa fixada antes do teste final é usada como evidência oficial.

### F2.T4. Instrumentação Prometheus
- Declarar `prometheus-client` como dependência direta da aplicação.
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
- `monitoring/prometheus/prometheus.yml`: scrape de `api-sklearn:8000/metrics` e `api-onnx:8000/metrics` a cada 5 s.
- `monitoring/grafana/provisioning/datasources/datasource.yml`: provisiona o Prometheus como datasource.
- `monitoring/grafana/provisioning/dashboards/dashboards.yml`: provisiona o dashboard `triage_ml.json`.
- `infra/docker-compose.yml`: serviços `api-sklearn`, `api-onnx`, `prometheus` e `grafana` na mesma rede, com variantes e portas documentadas. O Dockerfile da API é o mesmo nas duas instâncias.
- `scripts/generate_observability_traffic.py`: envia carga controlada às duas APIs para popular o dashboard sem gravar textos, respostas ou request IDs.

### F2.T7. Dashboard
- Painéis mínimos:
  1. **Requisições por rota/status** (stat panel + gráfico de barras empilhadas).
  2. **Latência p95** (timeseries) com filtro `model_variant`.
  3. **Taxa de erros** (stat panel).
  4. **Comparativo baseline vs otimizado** (table panel lendo `triage_ml_request_latency_seconds_bucket` filtrado por `model_variant`).
- Print do dashboard em `reports/figures/09_dashboard.png`.
- JSON canônico versionado em `monitoring/grafana/dashboards/triage_ml.json` e referenciado como evidência no checklist, sem cópia divergente em `reports/figures/`.

### F2.T8. Teste de privacidade
- `tests/test_monitoring_metrics.py`: chama `/predict` com texto controlado (fixture pequena), captura logs, respostas de erro e métricas, e confirma que **nem o texto completo nem trechos exclusivos da fixture** aparecem em nenhum desses canais. Verifica também ausência de campos `input`/`text` em respostas e de headers sensíveis.
- O teste carrega apenas a fixture local; o dataset bruto (`medical_tc_train.csv`, `medical_tc_test.csv`) **não** é aberto durante o teste, evitando leitura desnecessária do conteúdo clínico.
- O `metadata.json` é validado por schema; nenhum dos seus campos pode conter string que case com a fixture.

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
- [ ] `model.onnx` persistido localmente e `reports/benchmarks/benchmark.json` versionado como evidência agregada.

**Etapa 6:**
- [ ] Métricas `prometheus_client` expostas em `/metrics`.
- [ ] Total de requisições por rota/status, latência e erros medidos.
- [ ] Sem labels de alta cardinalidade e sem conteúdo clínico (teste automatizado).
- [ ] Compose com as variantes sklearn/ONNX da API, Prometheus e Grafana.
- [ ] Dashboard JSON reprodutível com pelo menos 4 painéis (incluindo comparativo).
- [ ] Print em `reports/figures/` e JSON canônico em `monitoring/grafana/dashboards/`.

## F2. Riscos específicos

| Risco | Mitigação |
|---|---|
| `skl2onnx` não converter o TF-IDF ou classificador escolhido | Fazer spike do pipeline completo no início de F2.T1; manter LR como caminho preferencial compatível e adapter isolado |
| Latência ONNX não melhorar no ambiente controlado | Avaliar quantização somente na validação; registrar metodologia e resultado sem transformar benchmark de hardware em teste unitário flakey |
| Dashboard divergir entre versões do Grafana | Fixar versão da imagem no Compose; testar provisionamento no CI quando possível |
| `prometheus_client` registrar texto clínico acidentalmente | Teste de privacidade em F2.T8 + revisão de PR |
| Compose local pesado para o time | Documentar pré-requisitos; CI só faz lint+pytest, não sobe Compose |
| Dashboard vazio ou sem comparação simultânea | Executar duas instâncias da API e fornecer gerador de tráfego controlado para ambas as variantes |

## F2. Sequência de commits (Fase 2, em adição aos 5 da Fase 1)

6. `feat(models): onnx export and latency benchmark for the baseline classifier`
7. `test(models): validate onnx prediction compatibility and macro-f1 tolerance`
8. `feat(monitoring): prometheus metrics middleware for /health and /predict`
9. `feat(api): expose /metrics and ship prometheus + grafana compose`
10. `feat(monitoring): provision grafana dashboard with baseline vs optimized panel`
11. `test(monitoring): privacy regression test for logs and metric labels`
12. `docs(models): document optimization, observability and dashboard usage`

## F2. Definição de pronto da Fase 2

- Etapas 5 e 6 marcadas como `[x]` no checklist, com evidência.
- `uv run pytest` e `uv run ruff check .` verdes.
- `docker compose up` sobe as duas variantes da API, Prometheus e Grafana; gerador de tráfego popula os 4 painéis.
- Testes determinísticos de compatibilidade passam no CI e o critério de latência/qualidade é demonstrado em benchmark controlado com ambiente registrado.
- `README.md` e `CHECKLIST.md` refletem o estado real.

---

# Próximas fases (fora do escopo deste plano)

- **API oficial** (Etapa 3, Romário): substitui ou incorpora a API de smoke.
- **CI/CD e Docker da imagem da API** (Etapa 4, Fábio): cria o Dockerfile e o build no CI; este plano depende disso, mas não é dono.
- **DAG Airflow** (Etapa 7, Denis): consome `triage_ml.models.train.run_training` definido na Fase 1.
- **Cloud, ADR, vídeo STAR, documentação final** (Etapa 8, Romário + Fábio).
