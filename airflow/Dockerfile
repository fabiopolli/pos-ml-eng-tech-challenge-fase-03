FROM apache/airflow:3.1.7-python3.12

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY --chown=airflow:root pyproject.toml README.md /opt/triage-ml/
COPY --chown=airflow:root src /opt/triage-ml/src
COPY --chown=airflow:root configs /opt/triage-ml/configs

ENV PYTHONPATH=/opt/triage-ml/src

RUN pip install --no-cache-dir \
    "matplotlib>=3.7" \
    "seaborn>=0.12" \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.7/constraints-3.12.txt
