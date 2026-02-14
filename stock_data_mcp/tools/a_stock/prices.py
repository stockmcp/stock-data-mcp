"""
A股价格与行情模块

包含历史价格、实时行情等工具
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from pydantic import Field

from ...core import (
    mcp,
    get_data_manager,
    format_source_name,
    field_symbol,
    field_market,
    ak_cache,
)
from ...data_provider import to_chinese_columns, validate_stock_type, market_to_stock_type, StockType
from ...indicators import add_technical_indicators, STOCK_PRICE_COLUMNS


# ==================== 辅助函数 ====================

def _fund_etf_hist_sina(symbol, market="sh", start_date="2025-01-01", period="daily"):
    """获取 ETF 历史数据"""
    dfs = ak.fund_etf_hist_sina(symbol=f"{market}{symbol}")
    if dfs is None or dfs.empty:
        return None
    dfs.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}, inplace=True)
    dfs["换手率"] = None
    dfs.index = pd.to_datetime(dfs["日期"], errors="coerce")
    return dfs.loc[start_date:]


# ==================== 历史价格 ====================

@mcp.tool(
    title="获取股票历史价格",
    description="根据股票代码和市场获取股票历史价格及技术指标, 不支持加密货币。支持多数据源自动故障转移。",
)
def stock_prices(
    symbol: str = field_symbol,
    market: str = field_market,
    period: str = Field("daily", description="周期，如: daily(日线), weekly(周线，不支持美股)"),
    limit: int = Field(30, description="返回数量(int)", strict=False),
):
    stock_type = market_to_stock_type(market)

    # 对于 A 股，优先使用多数据源管理器
    if stock_type == StockType.A_STOCK:
        try:
            manager = get_data_manager()
            df = manager.get_daily_data(symbol, days=limit + 62)
            if df is not None and not df.empty:
                source = format_source_name(df.attrs.get('source', ''))
                df = to_chinese_columns(df)
                if "换手率" not in df.columns:
                    df["换手率"] = None
                add_technical_indicators(df, df["收盘"], df["最低"], df["最高"], df.get("成交量"))
                available_cols = [c for c in STOCK_PRICE_COLUMNS if c in df.columns]
                all_lines = df.to_csv(columns=available_cols, index=False, float_format="%.2f").strip().split("\n")
                lines = [f"# {symbol} 历史价格", f"# 数据来源: {source}", f"# 市场: A股"]
                lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
                return "\n".join(lines)
        except Exception as e:
            pass  # 回退到原有逻辑

    # 计算起始日期
    if period == "weekly":
        delta = {"weeks": limit + 62}
    else:
        delta = {"days": limit + 62}
    start_date = (datetime.now() - timedelta(**delta)).strftime("%Y%m%d")

    # 港股/美股：使用统一的带故障转移函数
    if stock_type in (StockType.HK, StockType.US):
        from ..us_stock import _fetch_global_prices
        dfs = _fetch_global_prices(symbol, market, start_date, period)
        if dfs is not None and not dfs.empty:
            add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
            all_lines = dfs.to_csv(columns=STOCK_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
            source = dfs.attrs.get('source', 'unknown')
            market_label = "港股" if stock_type == StockType.HK else "美股"
            lines = [f"# {symbol} 历史价格", f"# 数据来源: {source}", f"# 市场: {market_label}"]
            lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
            return "\n".join(lines)
        return f"Not Found for {symbol}.{market}"

    # 其他市场（A股回退、ETF）
    markets = [
        ["sh", ak.stock_zh_a_hist, {}],
        ["sz", ak.stock_zh_a_hist, {}],
        ["sh", _fund_etf_hist_sina, {"market": "sh"}],
        ["sz", _fund_etf_hist_sina, {"market": "sz"}],
    ]
    for m in markets:
        if m[0] != market:
            continue
        kws = {"period": period, "start_date": start_date, **m[2]}
        dfs = ak_cache(m[1], symbol=symbol, ttl=3600, **kws)
        if dfs is None or dfs.empty:
            continue
        add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
        all_lines = dfs.to_csv(columns=STOCK_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
        lines = [f"# {symbol} 历史价格", f"# 数据来源: akshare", f"# 市场: A股/ETF"]
        lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
        return "\n".join(lines)
    return f"Not Found for {symbol}.{market}"


# ==================== 实时行情 ====================

@mcp.tool(
    title="获取股票实时行情",
    description="获取A股/港股实时行情数据，包括最新价、涨跌幅、成交量、换手率、市盈率等。支持多数据源自动故障转移。",
)
def stock_realtime(
    symbol: str = field_symbol,
    market: str = Field("sh", description="股票市场，仅支持: sh(上证), sz(深证), hk(港股)"),
):
    try:
        stock_type, validated_market = validate_stock_type(symbol, market)

        manager = get_data_manager()
        quote = manager.get_realtime_quote(symbol, stock_type=stock_type)
        if quote is None:
            return f"Not Found for {symbol}.{validated_market}"

        row = {
            "代码": quote.code,
            "名称": quote.name or "-",
            "最新价": quote.price,
            "涨跌幅": quote.change_pct,
            "涨跌额": quote.change_amount,
            "今开": quote.open_price,
            "最高": quote.high,
            "最低": quote.low,
            "昨收": quote.pre_close,
            "成交量": quote.volume,
            "成交额": quote.amount,
            "换手率": quote.turnover_rate,
            "量比": quote.volume_ratio,
            "振幅": quote.amplitude,
            "市盈率": quote.pe_ratio,
            "市净率": quote.pb_ratio,
            "总市值": quote.total_mv,
            "流通市值": quote.circ_mv,
        }
        df = pd.DataFrame([row])
        source = quote.source.value if quote.source else "-"
        lines = [f"# {symbol} 实时行情", f"# 数据来源: {source}"]
        lines.append(df.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 实时行情失败: {e}"


# ==================== 批量实时行情 ====================

@mcp.tool(
    title="批量获取实时行情",
    description="批量获取多只A股实时行情数据。支持多数据源自动故障转移。",
)
def stock_batch_realtime(
    symbols: str = Field(description="股票代码列表，用逗号分隔，如: 600519,000858,601318"),
    limit: int = Field(20, description="返回数量(int)", strict=False),
):
    try:
        codes = [s.strip() for s in symbols.split(",") if s.strip()]
        if not codes:
            return "请提供有效的股票代码"

        codes = codes[:limit]
        manager = get_data_manager()
        quotes = manager.prefetch_realtime_quotes(codes)

        if not quotes:
            return "未获取到任何行情数据"

        rows = []
        sources = set()
        for code, quote in quotes.items():
            rows.append({
                "代码": quote.code,
                "名称": quote.name or "-",
                "最新价": quote.price,
                "涨跌幅": quote.change_pct,
                "涨跌额": quote.change_amount,
                "今开": quote.open_price,
                "最高": quote.high,
                "最低": quote.low,
                "昨收": quote.pre_close,
                "成交量": quote.volume,
                "成交额": quote.amount,
                "换手率": quote.turnover_rate,
                "市盈率": quote.pe_ratio,
                "市净率": quote.pb_ratio,
            })
            if quote.source:
                sources.add(quote.source.value)

        df = pd.DataFrame(rows)
        source_str = ", ".join(sorted(sources)) if sources else "-"
        lines = [f"# 批量实时行情", f"# 数据来源: {source_str}"]
        lines.append(df.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        return f"批量获取实时行情失败: {e}"
