# 基础镜像
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

LABEL io.modelcontextprotocol.server.name="io.github.stockmcp/stock-data-mcp"

ARG SETUPTOOLS_SCM_PRETEND_VERSION

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    TRANSPORT=http \
    PORT=80 \
    SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

WORKDIR /app

# 先复制依赖文件和 README，利用 Docker 缓存
COPY pyproject.toml uv.lock README.md ./

# 安装依赖（frozen 不安装 dev 依赖）
RUN uv sync --frozen --no-dev --no-cache

# 再复制完整源码
COPY . .

# 安装必要工具
RUN apt update && apt install -y --no-install-recommends netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 启动命令
CMD ["uv", "run", "-m", "stock_data_mcp"]

# 健康检查
HEALTHCHECK --interval=1m --start-period=30s CMD nc -zn 0.0.0.0 $PORT || exit 1
