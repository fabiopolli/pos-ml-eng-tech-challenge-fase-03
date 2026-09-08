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
COPY --chown=airflow:root airflow/entrypoint.sh /opt/triage-ml/entrypoint.sh

ENV PYTHONPATH=/opt/triage-ml/src

RUN chmod +x /opt/triage-ml/entrypoint.sh \
    && pip install --no-cache-dir \
    "matplotlib>=3.7" \
    "seaborn>=0.12" \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.7/constraints-3.12.txt \
    && pip install --no-cache-dir \
    "onnx>=1.15,<2" \
    "onnxruntime>=1.17,<2" \
    "skl2onnx>=1.16,<2"

ENTRYPOINT ["/opt/triage-ml/entrypoint.sh"]
