# DAGs do Airflow

`triage_retraining.py` define a DAG manual `triage_ml_retraining`:

1. clona a branch configurada do DagsHub em um diretório temporário;
2. publica somente o CSV esperado em `data/`, por troca atômica;
3. valida o contrato de dados sem enviar textos por XCom ou logs;
4. chama `triage_ml.models.train.run_training` para preparar, treinar, avaliar e persistir;
5. valida manifesto e checksum do artefato produzido.

A execução é idempotente para a mesma combinação de bytes do dataset e do arquivo de
configuração: uma versão íntegra anterior é reutilizada. O diretório temporário de clone é
removido ao final da tarefa de ingestão.

## Execução local

Requer Docker Desktop com WSL2 no Windows. A partir da raiz do repositório:

```bash
docker compose -f docker-compose.airflow.yml up --build -d
docker compose -f docker-compose.airflow.yml exec airflow airflow dags list
docker compose -f docker-compose.airflow.yml exec airflow airflow dags test \
  triage_ml_retraining 2026-09-05
```

A interface fica em `http://localhost:8080`. Na primeira inicialização, as credenciais do
modo standalone aparecem nos logs:

```bash
docker compose -f docker-compose.airflow.yml logs airflow
```

O dataset fica em `data/medical_tc_train.csv` e os resultados em `models/<versão>/` e
`reports/figures/<versão>/`; todos permanecem fora do Git.

## Configuração

As variáveis documentadas em `.env.example` selecionam URL, branch e caminho do CSV no
repositório de dados. Para um repositório privado, não inclua credenciais na URL: injete o
segredo no ambiente do container por um mecanismo de secrets.

## Diagnóstico

```bash
docker compose -f docker-compose.airflow.yml ps
docker compose -f docker-compose.airflow.yml exec airflow airflow dags list-import-errors
docker compose -f docker-compose.airflow.yml logs --tail=200 airflow
```

