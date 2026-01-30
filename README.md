# Stock Data MCP Server

[English](README_EN.md) | 中文

MCP 服务器，提供 A股/港股/美股、加密货币数据查询，支持多数据源自动故障转移。

## 安装

```bash
# uvx
uvx stock-data-mcp

# pip
pip install stock-data-mcp

# Docker
docker run -p 8808:80 ghcr.io/stockmcp/stock-data-mcp
```

## MCP 配置

```json
{
  "mcpServers": {
    "stock-data": {
      "command": "uvx",
      "args": ["stock-data-mcp"]
    }
  }
}
```

<details>
<summary>环境变量（可选）</summary>

| 变量 | 说明 |
|------|------|
| `TUSHARE_TOKEN` | Tushare Pro API Token |
| `OKX_BASE_URL` | OKX API 代理地址 |
| `BINANCE_BASE_URL` | Binance API 代理地址 |

</details>

## 工具列表

| 工具 | 说明 |
|------|------|
| `search` | 股票搜索 |
| `stock_info` | 股票信息 |
| `stock_prices` | 历史价格+技术指标 |
| `stock_realtime` | 实时行情 |
| `stock_chip` | 筹码分布 |
| `stock_news` | 个股新闻 |
| `stock_news_global` | 全球财经快讯 |
| `stock_indicators_a` | A股财务指标 |
| `stock_indicators_hk` | 港股财务指标 |
| `stock_indicators_us` | 美股财务指标 |
| `stock_zt_pool_em` | A股涨停股池 |
| `stock_lhb_ggtj_sina` | A股龙虎榜 |
| `stock_sector_fund_flow_rank` | 板块资金流 |
| `okx_prices` | 加密货币K线 |
| `binance_ai_report` | 加密货币AI分析 |
| `data_source_status` | 数据源状态 |

## License

MIT
