FROM python:3.12.11-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv==0.11.23

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12.11-slim-bookworm AS runtime-base

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGE_ML_API_CONFIG=/app/configs/api.yaml

RUN groupadd --gid 10001 triage \
    && useradd --uid 10001 --gid triage --no-create-home --shell /usr/sbin/nologin triage

WORKDIR /app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 configs ./configs

USER 10001:10001


FROM runtime-base AS portal-runtime

COPY --chown=10001:10001 front ./front
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["streamlit", "run", "front/app_prod.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]


FROM runtime-base AS dev-dashboard-runtime

COPY --chown=10001:10001 front ./front
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["streamlit", "run", "front/app_dev.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]


FROM runtime-base AS runtime

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "triage_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
