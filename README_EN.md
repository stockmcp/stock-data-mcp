# Stock Data MCP Server

English | [中文](README.md)

MCP server for China A-shares, HK, US stocks and cryptocurrency data with multi-source failover.

## Installation

```bash
# uvx
uvx stock-data-mcp

# pip
pip install stock-data-mcp

# Docker
docker run -p 8808:80 ghcr.io/stockmcp/stock-data-mcp
```

## MCP Configuration

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
<summary>Environment Variables (Optional)</summary>

| Variable | Description |
|----------|-------------|
| `TUSHARE_TOKEN` | Tushare Pro API Token |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API Key (enhanced US data, falls back to yfinance if not set) |
| `OKX_BASE_URL` | OKX API proxy URL |
| `BINANCE_BASE_URL` | Binance API proxy URL |

</details>

## Tools

| Tool | Description |
|------|-------------|
| `search` | Stock search |
| `stock_info` | Stock information |
| `stock_prices` | Historical prices + indicators |
| `stock_realtime` | Real-time quotes |
| `stock_chip` | Chip distribution |
| `stock_news` | Stock news |
| `stock_news_global` | Global financial news |
| `stock_indicators_a` | A-share financials |
| `stock_indicators_hk` | HK stock financials |
| `stock_indicators_us` | US stock financials |
| `stock_zt_pool_em` | A-share limit-up pool |
| `stock_lhb_ggtj_sina` | A-share top traders |
| `stock_sector_fund_flow_rank` | Sector fund flow |
| `stock_overview_us` | US company overview (Alpha Vantage / yfinance) |
| `stock_financials_us` | US financial statements (Alpha Vantage / yfinance) |
| `stock_news_us` | US news sentiment (Alpha Vantage API key required) |
| `stock_earnings_us` | US earnings data (Alpha Vantage / yfinance) |
| `stock_insider_us` | US insider transactions (Alpha Vantage / yfinance) |
| `stock_tech_indicators_us` | US technical indicators (Alpha Vantage API key required) |
| `okx_prices` | Crypto K-lines |
| `binance_ai_report` | Crypto AI analysis |
| `data_source_status` | Data source status |

## License

MIT
