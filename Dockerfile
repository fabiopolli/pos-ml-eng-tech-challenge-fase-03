FROM python:3.12.11-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv==0.11.23

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12.11-slim-bookworm AS runtime

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
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "triage_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
