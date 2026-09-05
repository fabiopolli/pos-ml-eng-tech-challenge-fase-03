FROM apache/airflow:3.1.7-python3.12

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY --chown=airflow:root pyproject.toml README.md /opt/airflow/project/
COPY --chown=airflow:root src /opt/airflow/project/src
COPY --chown=airflow:root configs /opt/airflow/project/configs

RUN pip install --no-cache-dir /opt/airflow/project \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.7/constraints-3.12.txt
