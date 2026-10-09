# SolTrade, headless. Paper trades unless the command drops --dry-run (see compose.yaml).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py ./
COPY sol_trade ./sol_trade
RUN uv sync --frozen --no-dev \
    && mkdir -p /app/data /app/logs /app/strategies /config \
    && chmod 0777 /app/data /app/logs /app/strategies /config

# Any UID works (compose sets PUID:PGID); the code and venv are read-only for it.
USER 1000:1000
ENV PATH="/app/.venv/bin:$PATH" \
    SOLTRADE_CONFIG_PATH=/config/config.json \
    SOLTRADE_DOTENV_PATH=/config/.env

HEALTHCHECK --interval=60s --timeout=30s --start-period=180s --retries=3 \
    CMD ["python", "-m", "sol_trade.cli", "status", "--healthcheck"]

ENTRYPOINT ["python", "main.py", "--headless"]
CMD ["--dry-run"]
