FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

LABEL io.modelcontextprotocol.server.name="io.github.stockmcp/stock-data-mcp"

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    TRANSPORT=http \
    PORT=80

WORKDIR /app

# 只复制依赖定义文件，最大化缓存命中
COPY pyproject.toml uv.lock README.md 

./RUN uv sync --frozen --no-dev --no-cache

# 再复制源码
COPY . .

RUN apt update && apt install -y --no-install-recommends netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

CMD ["uv", "run", "-m", "stock_data_mcp"]
HEALTHCHECK --interval=1m --start-period=30s CMD nc -zn 0.0.0.0 $PORT || exit 1

