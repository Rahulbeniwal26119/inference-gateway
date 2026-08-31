FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 gateway
WORKDIR /app
COPY --from=builder --chown=gateway:gateway /app/.venv /app/.venv
USER gateway
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
