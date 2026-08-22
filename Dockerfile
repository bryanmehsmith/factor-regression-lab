FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# The interactive demo is now the static frontend/ (open frontend/index.html
# via any static file server), so this image serves only the small backend
# behind live data refresh and ticker lookups. See README.md's Usage section.
CMD ["python", "app/api.py", "--port=8000", "--host=0.0.0.0"]
