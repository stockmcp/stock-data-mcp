# 港股/美股 stock_prices 故障转移方案

## 当前状态

| 市场 | 当前数据源 | 故障转移 |
|------|-----------|---------|
| A股 | DataFetcherManager | ✓ 多源 |
| 港股 | `ak.stock_hk_hist` | ✗ 无 |
| 美股 | `ak.stock_us_daily` | ✗ 无 |

## 设计方案

在 `stock_prices` 函数中增加两个带故障转移的辅助函数：

### 1. 港股故障转移链
```
ak.stock_hk_hist → yfinance (code.HK)
```

### 2. 美股故障转移链
```
ak.stock_us_daily → yfinance → Alpha Vantage (TIME_SERIES_DAILY)
```

## 实现步骤

1. 新增 `_fetch_hk_prices(symbol, start_date, limit)` 函数
   - 优先调用 `ak.stock_hk_hist`
   - 失败后用 yfinance（股票代码转为 `XXXX.HK` 格式）

2. 新增 `_fetch_us_prices(symbol, start_date, limit)` 函数
   - 优先调用 `ak.stock_us_daily`
   - 失败后用 yfinance
   - 再失败用 Alpha Vantage（如果配置了 API Key）

3. 修改 `stock_prices` 函数
   - 港股调用 `_fetch_hk_prices`
   - 美股调用 `_fetch_us_prices`

## 列名映射

所有数据源统一输出为中文列名：
- `日期`, `开盘`, `收盘`, `最高`, `最低`, `成交量`, `换手率`

## 关键代码位置

- `stock_prices`: `__init__.py:134-183`
- `YfinanceFetcher`: `data_provider/yfinance_fetcher.py`
- `AlphaVantageFetcher`: `data_provider/alphavantage_fetcher.py`
