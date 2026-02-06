# Stock Data MCP Server

English | [中文](README.md)

MCP server for China A-shares, HK, US stocks and cryptocurrency data with multi-source failover.

## Installation

### uvx

#### MCP Configuration

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

#### MCP Configuration

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
<summary>Environment Variables (Optional)</summary>

| Variable | Description |
|----------|-------------|
| `TUSHARE_TOKEN` | Tushare Pro API Token (high-priority A-share data source) |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API Key (enhanced US data, falls back to yfinance if not set) |
| `OKX_BASE_URL` | OKX API proxy URL |
| `BINANCE_BASE_URL` | Binance API proxy URL |

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

## Tools

| Tool | Description |
|------|-------------|
| `search` | Stock search |
| `stock_info` | Stock information |
| `stock_prices` | Historical prices + indicators |
| `stock_realtime` | Real-time quotes |
| `stock_batch_realtime` | Batch real-time quotes |
| `stock_chip` | Chip distribution |
| `stock_fund_flow` | Individual stock fund flow |
| `stock_period_stats` | Multi-period statistics |
| `stock_sector_spot` | Stock sector membership |
| `stock_board_cons` | Sector constituents |
| `stock_news` | Stock news |
| `stock_news_global` | Global financial news |
| `stock_indicators_a` | A-share financials |
| `stock_indicators_hk` | HK stock financials |
| `stock_indicators_us` | US stock financials |
| `stock_zt_pool_em` | A-share limit-up pool |
| `stock_zt_pool_strong_em` | A-share strong stock pool |
| `stock_lhb_ggtj_sina` | A-share top traders |
| `stock_sector_fund_flow_rank` | Sector fund flow |
| `stock_overview_us` | US company overview (Alpha Vantage / yfinance) |
| `stock_financials_us` | US financial statements (Alpha Vantage / yfinance) |
| `stock_news_us` | US news sentiment (Alpha Vantage API key required) |
| `stock_earnings_us` | US earnings data (Alpha Vantage / yfinance) |
| `stock_insider_us` | US insider transactions (Alpha Vantage / yfinance) |
| `stock_tech_indicators_us` | US technical indicators (Alpha Vantage API key required) |
| `okx_prices` | Crypto K-lines |
| `okx_loan_ratios` | Crypto loan ratios |
| `okx_taker_volume` | Crypto taker volume |
| `binance_ai_report` | Crypto AI analysis |
| `get_current_time` | System time & trading day |
| `data_source_status` | Data source status |

## License

MIT
