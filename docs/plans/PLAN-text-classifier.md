# Plano de implementação — Classificador de Texto NLP (Bill)

- **Integrante**: Bill
- **Origem**: Tech Challenge — Fase 3 (ML Engineering)
- **Status**: Fase 1 entregue e revisada; Fase 2 pendente
- **Mapeamento no checklist**: cobre as Etapas 2, 5 e 6 do `docs/CHECKLIST.md` reordenado (2026-08-23). Está organizado em duas fases para deixar claro o que é trabalho **agora** e o que é trabalho **depois**.

## Estrutura por fases

| Fase | Etapas do checklist | Peso oficial | Conteúdo |
|---|---|---|---|
| **Fase 1 — Modelo baseline + API de desenvolvimento** | Etapa 2 | Parte do item modelo + otimização (20%) | Treino, métricas, serialização, API de desenvolvimento local |
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
- **Revisão de 2026-08-23 (model picker)**: novos endpoints `GET /models` e `POST /reload` na API de desenvolvimento (`src/triage_ml/dev_api/app.py`) com `ModelHolder.reload_to` re-validando manifesto + checksum antes do swap. Seção "🔁 Trocar modelo" na sidebar do dashboard consome ambos. Removida a seção "📚 Atalhos" da sidebar e as constantes `DOC_*` correspondentes.
- **Revisão técnica de robustez (2026-08-23)**: corrigida a semântica do score de idioma com `LanguageIdentifier(norm_probs=True)`; configuração validada no startup; registry unificado e protegido contra symlinks/artefatos incompletos; snapshot consistente do holder sob concorrência; rotas bloqueantes delegadas ao thread pool; publicação do treino feita por staging + rename atômico; CV falha cedo para folds inválidos.

---

# Fase 1 — Modelo baseline + API de desenvolvimento (Etapa 2 do checklist)

## F1. Contexto e objetivo

Entregar:

1. Classificador NLP leve (TF-IDF + classificador linear Scikit-Learn) treinado no recorte preparado pela fundação (5.000 amostras, seed 42).
2. Artefato serializado segundo o contrato (`models/<versão>/model.joblib` + `metadata.json`), com classes e nomes no próprio manifesto.
3. Métricas por classe e agregadas, com figuras em `reports/figures/`.
4. API FastAPI de desenvolvimento (`/health` + `/model-info` + `/models` + `/reload` + `/predict`) em `src/triage_ml/dev_api/`, consumindo o artefato real treinado, com `latency_ms` e `request_id` já expostos para reuso na Fase 2. `/reload` permite trocar o holder em runtime após re-validar manifesto + checksum.

A API oficial é do Romário (Etapa 3). Esta API de desenvolvimento é substituída ou estendida por ele; o contrato (esquemas Pydantic, headers) é ponto de alinhamento obrigatório antes de qualquer promoção.

## F1. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Vetorizador | `TfidfVectorizer` | Sugestão do enunciado; leve, determinístico, sem dependência externa |
| Candidatos do baseline | `LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")` e `LinearSVC(class_weight="balanced")` | Comparados por macro-F1 em validação estratificada somente no treino |
| Modelo selecionado | `LinearSVC` | Maior macro-F1 médio na validação (`0.7335` contra `0.7319`); o teste permaneceu isolado até a escolha |
| Por que não Random Forest? | — | Exemplo do enunciado. Em TF-IDF, RF explode o custo de inferência (centenas de árvores) sem ganho consistente sobre modelos lineares em texto. Justificativa registrada no README seção Bill |
| Serialização | `joblib` para o pipeline scikit-learn | Padrão sklearn |
| API (desenvolvimento) | FastAPI + Uvicorn, em processo local sem Docker | Suficiente para validação local; Docker e Compose ficam para Etapa 4 (Fábio) e Fase 2 |
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
└── dev_api/
    ├── __init__.py
    ├── app.py                 # FastAPI mínimo com /health e /predict
    ├── schemas.py             # Pydantic de entrada/saída
    ├── language.py            # detector langid + política de idioma
    └── config.py              # ApiConfig (LRU cache) carregado de configs/api.yaml
tests/
├── test_model_pipeline.py
├── test_model_artifact.py
├── test_dev_api.py
├── test_dev_api_language.py
└── test_dev_dashboard_helpers.py  # helpers HTTP do dashboard (mocks requests)
models/
└── README.md                  (existente; artefatos permanecem fora do Git)
reports/
├── evidence/
│   └── api-dev.json           # resposta sanitizada, sem os textos enviados
└── figures/
    ├── 08_confusion_matrix_linear_svc.png
    └── 08_top_features_linear_svc.png
configs/
├── training.yaml              # hiperparâmetros e label mapping versionados
└── api.yaml                   # allow-list de idiomas + thresholds da política de idioma
front/
├── app_dev.py                 # dashboard Streamlit para exercitar /health, /model-info, /models, /reload e /predict
└── README.md                  # instruções de uso e escopo do dashboard
```

A pasta `monitoring/` não é tocada na Fase 1.

## F1. Contratos a cumprir

Pontos relevantes de `.agents/contracts/README.md`:

- **Dados**: `prepare_dataset` garante amostra sem duplicatas e seed 42. O treino registra fingerprints do CSV preparado e dos índices de cada split para comprovar que a avaliação e o benchmark usam as mesmas entradas.
- **Seleção e avaliação**: Logistic Regression e LinearSVC são comparados somente no treino por validação estratificada. O classificador é fixado antes da avaliação final no teste; a escolha não usa métricas do test set.
- **Modelo**: recebe lista de textos; devolve classe e score quando disponível. `score` representa confiança do classificador e não deve ser descrito como probabilidade calibrada sem avaliação específica.
- **Artefato**: `metadata.json` é a fonte canônica para `schema_version`, `model_version`, `task_type`, idioma, classes e nomes, configuração do pipeline, seed, métricas, versões das dependências, commit Git, fingerprints de dataset/splits e checksum de `model.joblib`.
- **Versão**: `<versão>` segue `YYYYMMDDTHHMMSSZ-<input_hash_curto>`, com hash derivado do dataset preparado e da configuração, e nunca sobrescreve um diretório existente. O treino publica por diretório de staging + rename atômico. O carregador aceita apenas artefatos locais/confiáveis e valida schema, classes, estrutura/parâmetros declarados e checksum antes/depois de desserializar o `joblib`.
- **API (proposta, sujeito à validação de Romário antes de promover)**:
  - `GET /health` → `HealthOut(status, model_version, model_loaded)`.
  - `GET /model-info` → `ModelInfoOut(...)` com o manifesto validado do artefato (503 `model_not_ready` se nada estiver carregado).
  - `GET /models` → `ModelsListOut(versions, current)` listando somente versões completas e íntegras no registry configurado (newest-first) + a atualmente em uso.
  - `POST /reload` → `ReloadIn(model_version)` → `ReloadOut(model_version, model_loaded)`. Troca o holder para a versão solicitada após re-validar manifesto + checksum; erros viram `404 model_not_found` ou `500 model_incompatible` (holder anterior permanece em uso).
  - `POST /predict` → `PredictIn(text)` → `PredictOut(label, label_name, score?, model_version, latency_ms, request_id, warnings)`, com score opcional para classificadores sem `predict_proba`.
  - **Política de idioma** (camada adicional, fora do contrato inicial da Fase 1 mas incorporada em 2026-08-23): `langid` local roda antes do pipeline. Configuração em `configs/api.yaml` (`supported_languages`, `min_text_chars_for_language_check`, `min_language_score`). Rejeições viram `ErrorOut(request_id, error_code="text_too_short_for_language_check"|"indeterminate_language"|"unsupported_language", message, detected_language?, detected_language_score?)` — o `text` nunca aparece na resposta nem em logs.
  - Erros: `ErrorOut(request_id, error_code, message, detected_language?, detected_language_score?)` — nunca conteúdo clínico. Um handler próprio sanitiza o `422` do FastAPI/Pydantic, removendo o campo `input` antes da resposta.
  - Headers: `X-Request-ID` ecoando o `request_id`; `Server-Timing: detect;dur=<ms>, predict;dur=<ms>` quando ambos rodam, ou apenas `detect;dur=<ms>` quando a checagem de idioma interrompe o fluxo.
- **Observabilidade**: adiada como stack completa. Mas a Fase 1 já nasce expondo latência e `request_id` em toda resposta para reuso na Fase 2.

### O que **não** entra na Fase 1

- Endpoint `GET /metrics` Prometheus.
- Middleware `prometheus_fastapi_instrumentator`.
- Otimização ONNX, quantização ou pruning.
- Compose, Prometheus, Grafana, dashboard.
- Autenticação, rate limit, tracing distribuído.
- Tradução automática pt-BR→en (rejeitada por LGPD/latência). A política de idioma com `langid` é puramente **checagem** local, sem tradução.

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

### F1.T4. API de desenvolvimento
- `src/triage_ml/dev_api/schemas.py`:
  - `PredictIn(text: constr(strip_whitespace=True, min_length=1, max_length=20000))`.
  - `PredictOut(label, label_name, score: float | None, model_version, latency_ms, request_id, warnings)`.
  - `HealthOut(status, model_version, model_loaded)`.
  - `ModelInfoOut(model_version, model_name, task_type, language, classes, label_mapping, random_state, n_train, n_test, metrics, preprocessing, selection, dependency_versions, git_commit, git_dirty, created_at)`.
  - `ModelsListOut(versions, current)`.
  - `ReloadIn(model_version)`, `ReloadOut(model_version, model_loaded)`.
  - `ErrorOut(request_id, error_code, message, detected_language?, detected_language_score?)`.
- `src/triage_ml/dev_api/app.py`:
  - `GET /health` → `HealthOut`.
  - `GET /model-info` → `ModelInfoOut` (503 `model_not_ready` se o artefato não estiver carregado). Permite que ferramentas externas (dashboard, smoke, monitoramento) inspecionem o manifesto sem tocar o filesystem.
  - `GET /models` → `ModelsListOut` (read-only, usa `_list_model_versions` que respeita `VERSION_DIR_PATTERN`).
  - `POST /reload` → `ReloadOut`. `ModelHolder.reload_to(version)` resolve `<repo>/models/<version>/model.joblib`, chama `load_artifact` para re-validar manifesto + checksum, e só então substitui os campos do holder — em caso de falha, o holder anterior permanece em uso.
  - `POST /predict` → `PredictIn` → `PredictOut` (com checagem de idioma antes do pipeline).
  - Camada de idioma: `detect_language(text, ...)` aplicado após validação de schema. Falhas viram `UnsupportedLanguageError`, mapeadas para `error_code` apropriado.
- App factory com injeção do carregador nos testes; carregamento no startup via `lifespan`, com falha rápida para artefato ausente/incompatível e `MODEL_PATH` configurável por env var.
- Mapeamento `condition_label → condition_name` lido do `metadata.json`, sem depender de CSV ignorado pelo Git em runtime.
- Handler de `RequestValidationError` remove valores de entrada do `422` e responde no formato `ErrorOut`.
- Middleware/dependência:
  - Gera `request_id` (`uuid.uuid4().hex[:12]`) em `request.state.request_id`.
  - Mede `latency_ms` com `time.perf_counter()` em torno do `pipeline.predict`/`predict_proba`. Mede `detect_latency_ms` ao redor de `detect_language`.
  - Ecoa `request_id` em `X-Request-ID`.
  - Emite `Server-Timing: detect;dur=<ms>, predict;dur=<ms>` (ou apenas `detect;dur=<ms>` se a checagem de idioma interrompeu o fluxo).
  - Captura exceções inesperadas, preserva erros HTTP conhecidos e retorna `ErrorOut` com `error_code` genérico; loga apenas `request_id`, rota, status e latência (nunca `text` nem corpo da requisição).
- `uvicorn triage_ml.dev_api.app:app --reload` deve subir e responder nos cinco endpoints.
- `tests/test_dev_api.py` cobre `/health`, `/predict`, `/model-info`, `/models`, `/reload`, artefato inválido, texto vazio, sanitização de erro, `X-Request-ID`, `Server-Timing` e ausência do texto em logs/respostas de erro.
- `tests/test_dev_api_language.py` cobre cada branch da política (`text_too_short_for_language_check`, `indeterminate_language`, `unsupported_language`), valida os campos `detected_language`/`detected_language_score` e confirma que o `text` nunca vaza.

### F1.T5. Teste manual e evidências
- Subir a API local, enviar 5 abstracts (incluindo 1 da classe 1 e 1 da classe 5) via `curl`/`httpie` e salvar somente respostas sanitizadas em `reports/evidence/api-dev.json`; os textos de entrada não são persistidos.
- Para cada resposta, registrar `request_id`, `label`, `score`, `latency_ms`, headers `X-Request-ID` e `Server-Timing`. Confirmar que `latency_ms` varia entre chamadas.
- **Cenários da política de idioma**: texto curto (`<20` chars, `error_code=text_too_short_for_language_check`), probabilidade baixa (mock do identificador normalizado com `("en", 0.1)`, `error_code=indeterminate_language`) e idioma fora do allow-list (mock `("pt", 0.9)`, `error_code=unsupported_language`). Cada cenário roda em um `TestClient` isolado com `_strict_lang_config` para forçar o limiar quando necessário.
- Validar `metadata.json` (chaves, classes, versões).
- Validar que `text` vazio retorna `422` Pydantic sem vazar conteúdo; `prediction_failed` retorna `ErrorOut` com `request_id`.
- Confirmar que a evidência versionável não contém os abstracts nem campos `input` do Pydantic, nem o `text` enviado.

### F1.T6. Documentação
- Atualizar `docs/CHECKLIST.md`: Etapa 2 → `[~]` em progresso, depois `[x]` com evidência. Não tocar em itens de outros donos.
- Adicionar seção "Modelo (Bill)" no `README.md` resumindo tarefa real (categorias clínicas), abordagem, classes e como rodar treino + API local; incluir a justificativa formal de não-uso de Random Forest.
- Atualizar `.agents/contracts/README.md` se o formato de `metadata.json` divergir.
- Documentar a checagem de idioma (`langid`): nova subseção no README + entrada no `docs/reports/Etapa_2_Modelo_baseline_e_serialização.md` + linha de evolução no CHECKLIST; atualizar o plano (este arquivo) com a camada de contrato e os arquivos novos.
- Documentar o dashboard de desenvolvimento (`front/app_dev.py`): subseção no README, README próprio em `front/README.md`, evolução no CHECKLIST; deixar claro que **não substitui Prometheus/Grafana**.

## F1. Critérios de aceite

Mapeados na Etapa 2 do `docs/CHECKLIST.md`:

- [x] Baseline TF-IDF + classificador Scikit-Learn selecionado sem usar o test set.
- [x] Seeds, preprocessing, fingerprints e versões fixas.
- [x] Métricas por classe e agregadas, com figuras em `reports/figures/`.
- [x] Modelo e metadados serializados segundo contrato e validados por checksum.
- [x] API de desenvolvimento local (`/health` + `/model-info` + `/models` + `/reload` + `/predict`) em `src/triage_ml/dev_api/`, consumindo o artefato real treinado, com erros sanitizados, `latency_ms`, `request_id` e headers.

## F1. Riscos específicos

| Risco | Mitigação |
|---|---|
| Contrato fala em urgência, mas o dataset possui categorias clínicas | Gate humano antes de estabilizar labels, metadata e API; documentar a decisão no README/ADR aplicável |
| Divergência entre esta API e a API oficial de Romário (Etapa 3) | Marcar como "dev/provisória" no README (`src/triage_ml/dev_api/`); alinhar contrato Pydantic com Romário **antes** de qualquer promoção |
| Seleção otimista pelo test set | Comparar modelos somente por validação estratificada no treino e fixar a escolha antes da avaliação final |
| Modelo não serializa classes corretamente | `tests/test_model_artifact.py` valida `metadata.classes == model.classes_`, schema e checksum |
| Conteúdo clínico em logs ou no `422` | Handler de validação sanitizado, testes automatizados e revisão das evidências antes de versionar |
| CI quebrando | Fixar versões em `pyproject.toml`/`uv.lock`; preferir libs já presentes |
| Receber texto em idioma fora do allow-list (pt-BR, por exemplo) | `langid` local com allow-list `{"en"}` configurável em `configs/api.yaml`; rejeição explícita com `error_code` apropriado e `text` nunca exposto na resposta nem em logs |
| `langid` instável em entradas muito curtas | Mínimo de 20 caracteres (`min_text_chars_for_language_check`) antes do detector; abaixo disso a API rejeita com `text_too_short_for_language_check` |
| Probabilidade normalizada do `langid` não ser confiança calibrada | Instância própria com `norm_probs=True`; `min_language_score` permanece opt-in e deve ser calibrado em entradas representativas |
| Treino interrompido deixar a versão mais recente incompleta | Gravação em staging e `rename` atômico somente após manifesto, checksum e figuras serem produzidos |
| Reload concorrer com predição e misturar pipeline/metadata | Holder publica sob lock e cada request captura um snapshot único; inferência e reload síncronos rodam no thread pool do FastAPI |

## F1. Sequência de commits (apenas Fase 1)

1. `chore: scaffold triage_ml.models package and training config`
2. `feat(models): tf-idf + logistic regression pipeline with reproducible training`
3. `test(models): cover pipeline fit/predict and artifact round-trip`
4. `feat(api): minimal fastapi dev app exposing /health and /predict`
5. `docs(models): update checklist and readme for the baseline classifier`
6. `feat(api): checagem de idioma na /predict via langid local` (entregue em 2026-08-23; adiciona `language.py`, `config.py`, `configs/api.yaml`, testes e evidência do dev API)

## F1. Definição de pronto da Fase 1

- Itens da Etapa 2 marcados com evidência no checklist.
- `uv run pytest`, `uv run ruff check .` e `uv run ruff format --check .` verdes (80 testes após a revisão de robustez de 2026-08-23).
- API sobe com `uvicorn triage_ml.dev_api.app:app` e responde `/health`, `/model-info`, `/models`, `/reload` e `/predict` com o artefato.
- Dashboard sobe com `streamlit run front/app_dev.py`, lista as versões válidas do registry, permite trocar o holder global da API via `POST /reload` e renderiza o manifesto carregado em `🧠 Modelo`.
- Treino reproduzível a partir de clone limpo após obter o CSV conforme `docs/dataset.md`; o mapeamento `condition_label → condition_name` versionado vem de `configs/training.yaml`.
- `README.md`, `front/README.md`, `CHECKLIST.md` e `docs/reports/Etapa_2_Modelo_baseline_e_serialização.md` refletem o estado real (model picker + remoção dos Atalhos da sidebar).

---

# Fase 2 — Otimização + observabilidade (Etapas 5 e 6 do checklist)

> **Revisão 2026-09-08 (Bill):** o plano da Fase 2 foi redesenhado a partir do estado real do projeto. As mudanças se justificam em três pontos que o usuário pediu para considerar: (a) o treino histórico foi feito com `sample_size=5_000` em `configs/training.yaml` mas o dataset bruto `data/medical_tc_train.csv` tem 14K linhas; (b) a **Etapa 7 (orquestração de retreino) já está concluída** e a DAG consome `triage_ml.models.train.run_training` — qualquer novo passo da Etapa 5 (cortes alternativos, export ONNX, quantização) precisa ser plugado nesse pipeline existente, sem duplicar lógica de preparação/treino; (c) a **API oficial (`src/triage_ml/api/app.py`, Romário)** é o consumidor real do artefato e é nela que `/metrics`, middleware Prometheus e flag `MODEL_VARIANT` devem operar, não na API de desenvolvimento. Estrutura modular `triage_ml.optimization`/`triage_ml.observability` é introduzida para essa separação.

## F2. Contexto e objetivo

Entregar:

1. **Cortes adicionais do dataset** — incluir treino com 5K (linha de base atual, em uso pela Etapa 7), com 14K (dataset bruto completo) e, opcionalmente, com um corte intermediário (ex.: 10K) para comparar generalização vs. custo. A matriz de comparação é registrada em `reports/benchmarks/dataset_sizing.json` (ou tabela Markdown equivalente) e cada corte vira uma versão de artefato (`models/<versão>/`) no padrão existente.
2. **Variante otimizada ONNX** — export via `skl2onnx.convert_sklearn` do pipeline canônico, salva em `models/<versão>/model.onnx`, com adapter que implementa o mesmo contrato (`predict`/`predict_proba` quando aplicável, `decision_function` quando não).
3. **Plug no pipeline Airflow (Etapa 7)** — `train_evaluate_persist` ou um novo passo orquestrado chama `run_training` + `optimize.export_onnx` + `benchmark.compare` em uma única DAG (`triage_ml_retraining_optimization`, `schedule=None`), reaproveitando `find_reusable_artifact` para idempotência por hash e publicando `available_variants` em `metadata.json`.
4. **Stack de observabilidade na API oficial** — `prometheus_client` no middleware FastAPI de `src/triage_ml/api/app.py`, endpoint `GET /metrics` sem autenticação nesta fase, label `model_variant={sklearn,onnx}` para comparativo.
5. **Compose + Prometheus + Grafana** — `infra/docker-compose.yml` (overlay separado do `docker-compose.yml` de produção) sobe `api-sklearn`, `api-onnx`, `prometheus` e `grafana` provisionado com dashboard JSON.
6. **Privacidade** — teste de fumaça que varre logs, métricas e respostas de erro de `/predict` para garantir que `text` jamais aparece. Política de não-retenção documentada.

## F2. Decisões de stack e justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Cortes do dataset | 5K (atual), 14K (bruto) e opcional 10K | Comparar generalização real sem inflar custo de validação mais que o necessário; cada corte gera versão de artefato reutilizável |
| Quem dispara os cortes | DAG `triage_ml_retraining_optimization` (Etapa 7 ampliada) | Idempotência via `find_reusable_artifact`; reaproveita `run_training` sem duplicar lógica |
| Técnica de otimização | Export **ONNX** via `skl2onnx.convert_sklearn` (opset 17) | Vista em aula; `skl2onnx` cobre `Pipeline([TfidfVectorizer, LogReg/LinearSVC])` |
| Runtime ONNX | `onnxruntime` (CPU) | Backend estável; adapter próprio com mesmo contrato de predição; `zipmap=False` |
| Restrição do LinearSVC | Sem `predict_proba` nativo | Adapter expõe `decision_function` mapeada para classe; score do `PredictOut` vira opcional e documentado (escala de margem, não probabilidade) |
| Quantização dinâmica | Avaliada como alternativa secundária na validação; só é promovida se ONNX puro não cumprir o piso de latência | Evita presumir compatibilidade antes de medir |
| Critério de aceitação de latência | Δ p95 ≥ 20% (baseline sklearn vs. ONNX) **e** Δ macro-F1 ≤ 1 pp no mesmo split | Margem rígida contra regressão; medição controlada, não flakey |
| Métricas Prometheus | `prometheus_client` + middleware FastAPI próprio | Controle sobre labels/buckets; sem `prometheus_fastapi_instrumentator` por padrão |
| Labels aceitas | `route`, `method`, `status`, `model_variant` | Baixa cardinalidade; **nunca** `text`, `label_name`, `request_id` |
| Buckets do histograma | `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5]` (s) | Cobre o range esperado para TF-IDF |
| Compose | `infra/docker-compose.yml` com `api-sklearn`, `api-onnx`, `prometheus` e `grafana` — overlay do `docker-compose.yml` de produção | Permite comparação simultânea das variantes sem misturar com o stack de produção |
| Dashboard | JSON em `monitoring/grafana/dashboards/triage_ml.json` + print em `reports/figures/09_dashboard.png` | Reprocessamento via provisioning do Grafana |
| Privacidade | Teste de fumaça em fixture local, sem abrir o dataset bruto | Mesmo padrão da Etapa 3; alinhado com LGPD |

## F2. Estrutura de arquivos a criar/modificar (incremento sobre Fase 1 + Etapa 7)

```
src/triage_ml/
├── models/
│   ├── optimize.py            # export ONNX + quantização opcional
│   ├── onnx_adapter.py        # interface comum para InferenceSession
│   └── benchmark.py           # benchmark controlado + relatório comparativo
├── optimization/              # nova fronteira dedicada
│   ├── __init__.py
│   ├── dataloader.py          # cortes 5K/6K/7K, delega para prepare_dataset
│   └── registry.py            # resolve variant ativo (sklearn/onnx) para uma versão
├── observability/             # nova fronteira dedicada
│   ├── __init__.py
│   ├── metrics.py             # Counter/Histogram Prometheus compartilhados
│   └── middleware.py          # middleware FastAPI que mede e expõe
└── api/                       # MODIFICAÇÕES para Fase 2
    └── app.py                 # ganha /metrics + middleware + MODEL_VARIANT

src/triage_ml/orchestration/
└── airflow_pipeline.py        # AMPLIADO: novas funções export_onnx_for_version,
                                # build_optimization_manifest, sem duplicar lógica

airflow/dags/
└── triage_retraining_optimization.py   # NOVA DAG: schedule=None, depende de
                                          # ingest/validate/train/verify + export_onnx
                                          # + benchmark + health-check ONNX

configs/
├── training.yaml              # MODIFICADO: lista `dataset_sizing` opcional
└── training.sizing.yaml       # NOVO (opcional): catalogar cortes 5K/6K/7K

monitoring/
├── prometheus/
│   ├── prometheus.yml
│   └── README.md              # (já existe)
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasource.yml
    │   └── dashboards/dashboards.yml
    └── dashboards/
        └── triage_ml.json

infra/
└── docker-compose.yml         # overlay: api-sklearn + api-onnx + prometheus + grafana

scripts/
├── benchmark_api.py           # já existe; ampliado para variant
└── generate_observability_traffic.py  # popula métricas sem persistir textos

tests/
├── test_model_optimization.py        # contrato ONNX, classes, macro-f1, Δ ≤ 1 pp
├── test_observability_metrics.py     # labels aceitas, ausência de text, /metrics
├── test_optimization_registry.py     # resolver variant ativo por versão
├── test_optimization_dataloader.py   # cortes 5K/6K/7K chamam prepare_dataset
└── test_observability_privacy.py      # smoke: text em logs/métricas/respostas

reports/
├── benchmarks/
│   ├── benchmark.json         # evidência agregada e versionável, sem entradas
│   └── dataset_sizing.json    # comparação 5K vs. 10K vs. 14K
└── figures/
    ├── 09_latency_comparison.png
    ├── 09_sizing_comparison.png
    └── 09_dashboard.png
```

## F2. Contratos a cumprir

Evolução dos contratos da Fase 1 e Etapa 7:

- **Modelo / artefato**:
  - `metadata.json` ganha `available_variants: ["sklearn", "onnx"]`, `onnx_checksum_sha256`, `onnx_opset` e `selected_variant` (igual ao `selected_classifier` da Etapa 7).
  - Se o classificador escolhido for `LinearSVC`, `score` em `PredictOut` permanece opcional; a métrica retornada é o `decision_function` da classe predita (margem, não probabilidade calibrada) — documentado em `metadata.preprocessing.classifier_score_kind = "decision_function" | "predict_proba"`.
  - Idempotência: `find_reusable_artifact` da Etapa 7 é ampliada com `optimization_fingerprint` (configuração de export ONNX) para reuso de variantes ONNX já materializadas.
- **API oficial (`src/triage_ml/api/app.py`)**:
  - Variante ativa escolhida por env `TRIAGE_ML_MODEL_VARIANT={sklearn,onnx}` (default `sklearn`), carregada no `lifespan`.
  - Mantém `latency_ms`, `request_id`, `X-Request-ID`, `Server-Timing`.
  - Nova rota `GET /metrics` (Prometheus text format) **sem** autenticação nesta fase; documentação cita que será revisitado na Etapa 8.
  - `model_variant` aparece como **label** Prometheus (não no body) — `PredictOut` segue igual, evitando quebrar o contrato da Etapa 3.
- **Observabilidade**:
  - Métricas: `triage_ml_requests_total{route,method,status,model_variant}`, `triage_ml_request_latency_seconds{route,method,model_variant}`, `triage_ml_prediction_errors_total{route,error_code,model_variant}`.
  - **Sem** `text`, **sem** `label`, **sem** `request_id` em labels (verificado pelo teste de privacidade).
- **Privacidade**:
  - Política documentada: `text` é lido do request, classificado e descartado; nunca persistido, nunca copiado para log, nunca copiado para label de métrica, nunca retornado em payload de erro.
  - Teste automatizado varre respostas de erro, métricas e logs gerados em teste; falha se encontrar o **texto controlado da fixture** (string completa e trechos exclusivos), os campos `input`/`text` indevidos ou headers sensíveis. `metadata.json` validado por schema: nenhum campo pode conter string que case com a fixture. O teste carrega apenas a fixture local; o dataset bruto **não** é aberto durante o teste.

## F2. Tarefas e sequência

### F2.T1. Cortes do dataset (5K / 10K / 14K) plugados no pipeline Airflow
- Adicionar `dataset_sizing` em `configs/training.yaml` como lista opcional de inteiros (defaults `[5_000, 14_000]`; 10K entra só se o ambiente permitir).
- `src/triage_ml/optimization/dataloader.py` expõe `iter_dataset_slices(config, *, base_csv)` que delega para `triage_ml.data.prepare.prepare_dataset` em cada tamanho (sem duplicar regras de exclusão/seed).
- DAG `triage_ml_retraining_optimization` itera sobre os cortes; cada corte vira uma versão de artefato (`models/<YYYYMMDDTHHMMSSZ-<12hex>/`) no padrão da Etapa 7.
- Cada corte roda o fluxo atual `validate → train → verify` da Etapa 7 (zero duplicação) e adiciona um passo `compare_slices` (F2.T7) que escreve `reports/benchmarks/dataset_sizing.json`.

### F2.T2. Otimização ONNX
- **Pré-condição**: spike em `notebooks/03_onnx_spike.ipynb` convertendo o pipeline atual (`TfidfVectorizer` + `LinearSVC` ou `LogisticRegression`) e medindo latência. Resultado do spike alimenta a decisão `LinearSVC` vs. `LogReg` no adapter.
- Adicionar grupo de dependências opcional `[optimization]` em `pyproject.toml` com `skl2onnx`, `onnx`, `onnxruntime` (somente quando o `pip install -e .[optimization]` for usado pelo Airflow/Compose da Fase 2).
- `optimize.py`: `export_onnx(pipeline, out_path, *, opset=17)` via `skl2onnx.convert_sklearn`; `optimization_fingerprint` calculado a partir do pipeline + `opset` + flag de quantização.
- `onnx_adapter.py`:
  - Implementa `predict(X)` igual ao pipeline sklearn (mesmo array de classes).
  - Implementa `predict_proba(X)` **somente** quando o `metadata.preprocessing.classifier == "logreg"`; senão expõe `decision_function` mapeada para `PredictOut.score`.
  - Carregamento preguiçoso: instancia `onnxruntime.InferenceSession` na primeira chamada para manter a inicialização barata.
- Salvar `models/<versão>/model.onnx` ao lado de `model.joblib`; atualizar `metadata.json` com `onnx_checksum_sha256`, `onnx_opset`, `available_variants`.

### F2.T3. Benchmark comparativo
- `benchmark.py`: mede batch size 1 e retorna p50/p95/p99, média, throughput, tempo de carregamento, tamanho do artefato, macro-F1, concordância de classes e desvio máximo dos scores.
- Metodologia fixa: warmup, número de repetições, threads sklearn/ONNX, fronteira de medição (inclui TF-IDF + inferência, exclui carregamento), hardware/SO/Python/dependências/parâmetros registrados.
- Regenera o split a partir dos parâmetros da Etapa 7, valida seus fingerprints, roda `model.joblib` e `model.onnx` no mesmo split e escreve `models/<versão>/benchmark.json` + `reports/benchmarks/benchmark.json` (evidência agregada, sem entradas).
- Gera `reports/figures/09_latency_comparison.png` a partir do JSON versionável.

### F2.T4. Critério "sem degradação inaceitável"
- Testes determinísticos do CI validam: conversão ONNX ocorre, shape bate, classes batem, concordância de predições, Δ macro-F1 ≤ 1 pp. **Sem** limite de tempo dependente de hardware.
- O benchmark controlado avalia Δ p95 ≥ 20%. Documentado como critério de promoção (não teste flakey).
- Se ONNX puro não cumprir o piso na validação, avaliar quantização dinâmica; somente a alternativa fixada antes do teste final é usada como evidência oficial.
- Custo do export e do carregamento ONNX é registrado e comparado ao caminho sklearn.

### F2.T5. Plug no pipeline Airflow (Etapa 7 ampliada)
- Nova DAG `triage_ml_retraining_optimization` em `airflow/dags/triage_retraining_optimization.py`:
  - `schedule=None`, `catchup=False`, `max_active_runs=1`, mesmo padrão de retries da DAG atual.
  - Tasks: `ingest` → `validate` → `train_slice` (parametrizada por `sample_size`) → `export_onnx` → `benchmark` → `verify` → `compare_slices`.
  - Reutiliza funções de `triage_ml.orchestration.airflow_pipeline` (idempotência por hash via `find_reusable_artifact`, manifest em `models/<versão>/airflow_run.json`).
  - Variável nova `TRIAGE_OPTIMIZATION_ENABLED` (`false` por padrão) para o Compose atual da Etapa 7 (`docker-compose.airflow.yml`) continuar funcionando sem código novo de otimização.
- `airflow_pipeline.py` ganha: `export_onnx_for_version(version_dir)`, `benchmark_for_version(version_dir)` e `build_optimization_manifest`. **Não** duplica `run_training`/`prepare_dataset`/`load_config`.

### F2.T6. Instrumentação Prometheus
- Adicionar `prometheus-client` em `pyproject.toml` (dependência principal da aplicação, mesmo grupo da FastAPI).
- `src/triage_ml/observability/metrics.py`:
  - Define `REQUESTS_TOTAL`, `REQUEST_LATENCY_SECONDS`, `PREDICTION_ERRORS_TOTAL` com buckets e labels especificados.
  - Função `render_metrics()` retorna `generate_latest()` do registry **próprio** (não global) para evitar vazamento de bibliotecas.
- `src/triage_ml/observability/middleware.py`:
  - Middleware FastAPI medindo latência total, contabilizando erros com `error_code` genérico, e populando `Server-Timing` (já existente) **e** o histograma.
- A **API oficial (`src/triage_ml/api/app.py`)** instala o middleware e expõe `/metrics`; a API de desenvolvimento e os fronts Streamlit **não** ganham Prometheus nesta fase (seria sombra).

### F2.T7. Comparativo de dataset sizing
- Após o loop da DAG, `compare_slices` agrega as métricas por `sample_size` e publica `reports/benchmarks/dataset_sizing.json` com `macro_f1`, `balanced_accuracy`, `latency_p50`, `latency_p95`, `latency_p99`, `artifact_size_kb`, e variante ONNX quando disponível.
- Gera `reports/figures/09_sizing_comparison.png` (gráfico de barras por métrica).
- Tabela vira evidência da Etapa 5 no checklist.

### F2.T8. Compose, Prometheus e Grafana
- `infra/docker-compose.yml` (novo, overlay) com `api-sklearn`, `api-onnx`, `prometheus` e `grafana`. `api-sklearn` e `api-onnx` usam o mesmo `Dockerfile` da Etapa 4 mas com env `TRIAGE_ML_MODEL_VARIANT={sklearn,onnx}` e portas distintas.
- `monitoring/prometheus/prometheus.yml`: scrape de `api-sklearn:8000/metrics` e `api-onnx:8000/metrics` a cada 5 s.
- `monitoring/grafana/provisioning/datasources/datasource.yml`: provisiona Prometheus.
- `monitoring/grafana/provisioning/dashboards/dashboards.yml`: provisiona `triage_ml.json`.
- `scripts/generate_observability_traffic.py`: envia carga controlada às duas APIs para popular o dashboard sem gravar textos, respostas ou request IDs.
- README em `infra/` lista pré-requisitos e o comando único `docker compose -f infra/docker-compose.yml up -d --wait`.

### F2.T9. Dashboard
- Painéis mínimos:
  1. **Requisições por rota/status** (stat panel + gráfico de barras empilhadas).
  2. **Latência p95** (timeseries) com filtro `model_variant`.
  3. **Taxa de erros** (stat panel) com filtro `error_code`.
  4. **Comparativo baseline vs otimizado** (table panel lendo `triage_ml_request_latency_seconds_bucket` filtrado por `model_variant`).
- Print do dashboard em `reports/figures/09_dashboard.png`.
- JSON canônico versionado em `monitoring/grafana/dashboards/triage_ml.json`; o print é apenas evidência visual.

### F2.T10. Teste de privacidade
- `tests/test_observability_privacy.py`: chama `/predict` com texto controlado (fixture pequena), captura logs, respostas de erro, métricas Prometheus e responses e confirma que **nem o texto completo nem trechos exclusivos da fixture** aparecem em nenhum desses canais. Verifica também ausência de campos `input`/`text` em respostas e de headers sensíveis.
- O teste carrega apenas a fixture local; o dataset bruto **não** é aberto.
- `metadata.json` é validado por schema; nenhum campo pode conter string que case com a fixture.
- Adicionalmente, `tests/test_observability_metrics.py` valida que `/metrics` retorna `text/plain` no formato Prometheus, lista exatamente as métricas declaradas e nega qualquer label fora da whitelist.

### F2.T11. Documentação
- Atualizar `docs/CHECKLIST.md`: Etapas 5 e 6 marcadas com evidência (`benchmark.json`, `dataset_sizing.json`, JSON do dashboard, prints).
- Adicionar seção "Otimização + observabilidade" no `README.md` com `docker compose -f infra/docker-compose.yml up -d --wait`, URLs locais e como ler o dashboard.
- Atualizar `.agents/contracts/README.md` com `available_variants`, `MODEL_VARIANT` e `/metrics`.
- Documentar política de não-retenção do `text` (Etapa 6) em `docs/papers/` ou `docs/adr/` (escolher quando uma das duas pastas ganhar uma entrada consistente).

## F2. Critérios de aceite

Mapeados nas Etapas 5 e 6 do `docs/CHECKLIST.md`:

**Etapa 5:**
- [ ] Otimização aplicada (ONNX) com `model.onnx` salvo e `metadata.onnx_checksum_sha256` registrado.
- [ ] Cortes 5K e 14K treinados, comparados no mesmo split, com `reports/benchmarks/dataset_sizing.json` versionado.
- [ ] Comparativo baseline vs otimizado **e** 5K vs 14K, com Δ p95 ≥ 20% e Δ macro-F1 ≤ 1 pp no mesmo split.
- [ ] `model.onnx` persistido localmente e `reports/benchmarks/benchmark.json` versionado como evidência agregada.
- [ ] Variante ativa exposta pela API oficial via env `TRIAGE_ML_MODEL_VARIANT`.
- [ ] Pipeline de retreino da Etapa 7 (`triage_ml_retraining` / `triage_ml_retraining_optimization`) cobre os novos passos sem duplicar lógica de `run_training`/`prepare_dataset`.

**Etapa 6:**
- [ ] Métricas `prometheus_client` expostas em `/metrics` da API oficial (`src/triage_ml/api/app.py`).
- [ ] Total de requisições por rota/status, latência e erros medidos, com label `model_variant`.
- [ ] Sem labels de alta cardinalidade e sem conteúdo clínico (teste automatizado).
- [ ] `infra/docker-compose.yml` com `api-sklearn`, `api-onnx`, `prometheus` e `grafana`.
- [ ] Dashboard JSON reprodutível com pelo menos 4 painéis (incluindo comparativo).
- [ ] Prints em `reports/figures/` e JSON canônico em `monitoring/grafana/dashboards/`.
- [ ] Política de não-retenção do `text` documentada e teste de privacidade verde.

## F2. Riscos específicos

| Risco | Mitigação |
|---|---|
| `skl2onnx` não converter o `TfidfVectorizer` ou classificador escolhido | Spike no `notebooks/03_onnx_spike.ipynb` no início de F2.T2; registrar limitações antes da abstração definitiva; adapter isolado permite trocar a estratégia |
| `LinearSVC` sem `predict_proba` nativo | Adapter expõe `decision_function` mapeada para classe; `PredictOut.score` permanece opcional e documentado como margem (não probabilidade) |
| Latência ONNX não melhorar no ambiente controlado | Avaliar quantização somente na validação; registrar metodologia e resultado sem transformar o benchmark de hardware em teste unitário flakey |
| Cortes 14K explodirem tempo/recursos da DAG | `dataset_sizing` em `configs/training.yaml` é editável; `TRIAGE_OPTIMIZATION_ENABLED=false` mantém o fluxo atual da Etapa 7; tamanhos podem ser ajustados sem mudar código |
| Divergência entre a API oficial e a de desenvolvimento | Otimização e Prometheus operam **na oficial**; a de desenvolvimento fica inalterada nesta fase |
| Dashboard divergir entre versões do Grafana | Fixar versão da imagem no Compose (`grafana:11.x`); testar provisioning no CI quando possível |
| `prometheus_client` registrar texto clínico acidentalmente | Teste de privacidade em F2.T10 + revisão de PR |
| Compose local pesado para o time | `infra/docker-compose.yml` é overlay separado do `docker-compose.yml` de produção; CI só faz lint+pytest, não sobe Compose |
| Dashboard vazio ou sem comparação simultânea | Duas instâncias da API no Compose + `generate_observability_traffic.py` com carga controlada |
| DAG quebrando o container da Etapa 7 | Novas tarefas atrás de `TRIAGE_OPTIMIZATION_ENABLED` (`false` por padrão); imagem da DAG só instala extras se a env estiver setada |

## F2. Sequência de commits (Fase 2, em adição aos 7+ das Fases 1 e 7)

8. `chore(deps): add optional [optimization] group with skl2onnx/onnxruntime`
9. `feat(models): onnx export and adapter for sklearn/onnx variants`
10. `test(models): cover onnx conversion, prediction agreement and macro-f1 tolerance`
11. `feat(optimization): dataloader for 5K/6K/7K slices delegating to prepare_dataset`
12. `feat(orchestration): extend airflow_pipeline with export_onnx and benchmark helpers`
13. `feat(airflow): add triage_ml_retraining_optimization DAG reusing the Etapa 7 helpers`
14. `feat(observability): prometheus metrics and middleware for the official API`
15. `feat(api): expose /metrics and load selected variant from TRIAGE_ML_MODEL_VARIANT`
16. `feat(infra): overlay compose with api-sklearn, api-onnx, prometheus and grafana`
17. `feat(monitoring): provision grafana dashboard with baseline-vs-optimized panel`
18. `test(observability): privacy regression test for logs, metrics and error payloads`
19. `docs(models): document optimization, observability and dashboard usage`

## F2. Definição de pronto da Fase 2

- Etapas 5 e 6 marcadas como `[x]` no `docs/CHECKLIST.md`, com evidência (`reports/benchmarks/benchmark.json`, `reports/benchmarks/dataset_sizing.json`, dashboard JSON e prints em `reports/figures/`).
- `uv run pytest` (marcador `-m 'not e2e'`, mesmo padrão atual) e `uv run ruff check .` verdes.
- `uv run pytest -m e2e` (Playwright) verdes para os fronts atualizados com `/metrics` e `MODEL_VARIANT` na sidebar.
- `docker compose -f docker-compose.airflow.yml up` continua funcionando com `TRIAGE_OPTIMIZATION_ENABLED=false`.
- `docker compose -f infra/docker-compose.yml up -d --wait` sobe as duas variantes da API, Prometheus e Grafana; `scripts/generate_observability_traffic.py` popula os 4 painéis.
- Testes determinísticos de compatibilidade passam no CI; critério de latência/qualidade é demonstrado em benchmark controlado com ambiente registrado (sem virar teste flakey).
- `README.md`, `.agents/contracts/README.md` e `docs/CHECKLIST.md` refletem o estado real (com `available_variants`, `MODEL_VARIANT`, `/metrics`, DAG de otimização e política de privacidade).


---

# Próximas fases (fora do escopo deste plano)

- **API oficial** (Etapa 3, Romário): substitui ou incorpora a API de desenvolvimento.
- **CI/CD e Docker da imagem da API** (Etapa 4, Fábio): cria o Dockerfile e o build no CI; este plano depende disso, mas não é dono.
- **DAG Airflow** (Etapa 7, Denis): consome `triage_ml.models.train.run_training` definido na Fase 1.
- **Cloud, ADR, vídeo STAR, documentação final** (Etapa 8, Romário + Fábio).
