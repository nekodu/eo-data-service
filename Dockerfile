FROM ghcr.io/astral-sh/uv:0.13.0 AS uv

FROM python:3.13-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev --extra ingest --no-editable

FROM python:3.13-slim-bookworm
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "eods.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--proxy-headers", "--no-access-log"]
