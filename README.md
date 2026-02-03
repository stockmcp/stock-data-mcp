# Stock Data MCP Server

[English](README_EN.md) | 中文

MCP 服务器，提供 A股/港股/美股、加密货币数据查询，支持多数据源自动故障转移。

## 安装

### uvx

- MCP 配置

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

### Docker
```
docker run -p 8808:8808 ghcr.io/stockmcp/stock-data-mcp
```

- MCP 配置

```json
{
  "mcpServers": {
    "stock-data": {
      "url": "http://0.0.0.0:8808/mcp" # Streamable HTTP
    }
  }
}
```

<details>
<summary>环境变量（可选）</summary>

| 变量 | 说明 |
|------|------|
| `TUSHARE_TOKEN` | Tushare Pro API Token（A股高优先数据源） |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API Key（美股数据增强，不配置则使用 yfinance 免费源） |
| `OKX_BASE_URL` | OKX API 代理地址 |
| `BINANCE_BASE_URL` | Binance API 代理地址 |

</details>

## Claude Code

```
claude mcp add stock-data \
    -e TUSHARE_TOKEN=your_token \
    -e ALPHA_VANTAGE_API_KEY=your_key \
    -e OKX_BASE_URL=https://okx.4url.cn\
    -e BINANCE_BASE_URL=https://bian.4url.cn \
    -- uvx stock-data-mcp
```

## 工具列表

| 工具 | 说明 |
|------|------|
| `search` | 股票搜索 |
| `stock_info` | 股票信息 |
| `stock_prices` | 历史价格+技术指标 |
| `stock_realtime` | 实时行情 |
| `stock_batch_realtime` | 批量实时行情 |
| `stock_chip` | 筹码分布 |
| `stock_fund_flow` | 个股资金流向 |
| `stock_period_stats` | 多周期统计 |
| `stock_sector_spot` | 个股所属板块 |
| `stock_board_cons` | 板块成分股 |
| `stock_news` | 个股新闻 |
| `stock_news_global` | 全球财经快讯 |
| `stock_indicators_a` | A股财务指标 |
| `stock_indicators_hk` | 港股财务指标 |
| `stock_indicators_us` | 美股财务指标 |
| `stock_zt_pool_em` | A股涨停股池 |
| `stock_zt_pool_strong_em` | A股强势股池 |
| `stock_lhb_ggtj_sina` | A股龙虎榜 |
| `stock_sector_fund_flow_rank` | 板块资金流 |
| `stock_overview_us` | 美股公司概览（Alpha Vantage / yfinance） |
| `stock_financials_us` | 美股财务报表（Alpha Vantage / yfinance） |
| `stock_news_us` | 美股新闻情绪（需 Alpha Vantage API key） |
| `stock_earnings_us` | 美股盈利数据（Alpha Vantage / yfinance） |
| `stock_insider_us` | 美股内部交易（Alpha Vantage / yfinance） |
| `stock_tech_indicators_us` | 美股技术指标（需 Alpha Vantage API key） |
| `okx_prices` | 加密货币K线 |
| `okx_loan_ratios` | 加密货币借贷比 |
| `okx_taker_volume` | 加密货币买卖量 |
| `binance_ai_report` | 加密货币AI分析 |
| `get_current_time` | 系统时间及交易日 |
| `trading_suggest` | 投资建议 |
| `data_source_status` | 数据源状态 |

## License

MIT
