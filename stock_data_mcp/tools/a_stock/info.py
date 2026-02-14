"""
A股基本信息与搜索模块

包含股票搜索、基本信息、财务指标、交易时间等工具
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from pydantic import Field

from ...core import (
    mcp,
    field_symbol,
    field_market,
    ak_cache,
)
from ...data_provider import market_to_stock_type, StockType


# ==================== 辅助函数 ====================

def _search_us_stock_fast(symbol: str) -> pd.Series | None:
    """使用 yfinance 快速验证美股代码"""
    import yfinance as yf
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info and info.get("symbol") and info.get("shortName"):
            return pd.Series({
                "symbol": info.get("symbol", symbol),
                "name": info.get("shortName", ""),
                "cname": info.get("longName", info.get("shortName", "")),
                "market": "us",
            })
    except Exception:
        pass
    return None


def _ak_search(symbol=None, keyword=None, market=None):
    """通用股票搜索"""
    if market == "us" and (symbol or keyword):
        us_result = _search_us_stock_fast(symbol or keyword)
        if us_result is not None:
            return us_result

    markets = [
        ["sh", ak.stock_info_a_code_name, "code", "name"],
        ["sh", ak.stock_info_sh_name_code, "证券代码", "证券简称"],
        ["sz", ak.stock_info_sz_name_code, "A股代码", "A股简称"],
        ["hk", ak.stock_hk_spot, "代码", "中文名称"],
        ["hk", ak.stock_hk_spot_em, "代码", "名称"],
        ["us", ak.get_us_stock_name, "symbol", "cname"],
        ["us", ak.get_us_stock_name, "symbol", "name"],
        ["sh", ak.fund_etf_spot_ths, "基金代码", "基金名称"],
        ["sz", ak.fund_etf_spot_ths, "基金代码", "基金名称"],
        ["sh", ak.fund_info_index_em, "基金代码", "基金名称"],
        ["sz", ak.fund_info_index_em, "基金代码", "基金名称"],
        ["sh", ak.fund_etf_spot_em, "代码", "名称"],
        ["sz", ak.fund_etf_spot_em, "代码", "名称"],
    ]
    for m in markets:
        if market and market != m[0]:
            continue
        all = ak_cache(m[1], ttl=86400, ttl2=86400*7)
        if all is None or all.empty:
            continue
        for _, v in all.iterrows():
            code, name = str(v[m[2]]).upper(), str(v[m[3]]).upper()
            if symbol and symbol.upper() == code:
                return v
            if keyword and keyword.upper() in [code, name]:
                return v
        for _, v in all.iterrows() if keyword else []:
            name = str(v[m[3]])
            if len(keyword) >= 4 and keyword in name:
                return v
            if name.startswith(keyword):
                return v
    return None


# ==================== 搜索与基本信息 ====================

@mcp.tool(
    title="查找股票代码",
    description="根据股票名称、公司名称等关键词查找股票代码, 不支持加密货币。"
                "该工具比较耗时，当你知道股票代码或用户已指定股票代码时，建议直接通过股票代码使用其他工具",
)
def search(
    keyword: str = Field(description="搜索关键词，公司名称、股票名称、股票代码、证券简称"),
    market: str = field_market,
):
    info = _ak_search(None, keyword, market)
    if info is not None:
        lines = [f"# 搜索结果: {keyword}", f"# 数据来源: akshare", f"# 交易市场: {market}"]
        # 转为 CSV 格式：表头行 + 数据行
        if isinstance(info, pd.Series):
            lines.append(",".join(str(k) for k in info.index))
            lines.append(",".join(str(v) for v in info.values))
        else:
            lines.append(info.to_csv(index=False).strip())
        return "\n".join(lines)
    return f"Not Found for {keyword}"


@mcp.tool(
    title="获取股票信息",
    description="根据股票代码和市场获取股票基本信息, 不支持加密货币",
)
def stock_info(
    symbol: str = field_symbol,
    market: str = field_market,
):
    markets = [
        ["sh", ak.stock_individual_info_em],
        ["sz", ak.stock_individual_info_em],
        ["hk", ak.stock_hk_security_profile_em],
    ]
    for m in markets:
        if m[0] != market:
            continue
        all = ak_cache(m[1], symbol=symbol, ttl=43200)
        if all is None or all.empty:
            continue
        lines = [f"# {symbol} 基本信息", f"# 数据来源: akshare", f"# 市场: {market}"]
        lines.append(all.to_csv(index=False).strip())
        return "\n".join(lines)

    info = _ak_search(symbol, market)
    if info is not None:
        lines = [f"# {symbol} 基本信息", f"# 数据来源: akshare"]
        # 转为 CSV 格式：表头行 + 数据行
        if isinstance(info, pd.Series):
            lines.append(",".join(str(k) for k in info.index))
            lines.append(",".join(str(v) for v in info.values))
        else:
            lines.append(info.to_csv(index=False).strip())
        return "\n".join(lines)
    return f"Not Found for {symbol}.{market}"


# ==================== 财务指标 ====================

@mcp.tool(
    title="股票财务指标",
    description="获取股票财务报告关键指标，支持A股、港股、美股市场",
)
def stock_indicators(
    symbol: str = field_symbol,
    market: str = Field("sh", description="市场: 'sh'/'sz'(A股), 'hk'(港股), 'us'(美股)"),
):
    try:
        stock_type = market_to_stock_type(market)

        if stock_type == StockType.A_STOCK:
            dfs = ak_cache(ak.stock_financial_abstract_ths, symbol=symbol)
            if dfs is None or dfs.empty:
                return f"获取A股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标", f"# 数据来源: akshare", f"# 市场: A股"]
            lines.append("\n".join([keys[0], *keys[-15:]]))
            return "\n".join(lines)
        elif stock_type == StockType.HK:
            dfs = ak_cache(ak.stock_financial_hk_analysis_indicator_em, symbol=symbol, indicator="报告期")
            if dfs is None or dfs.empty:
                return f"获取港股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标", f"# 数据来源: akshare", f"# 市场: 港股"]
            lines.append("\n".join(keys[0:15]))
            return "\n".join(lines)
        elif stock_type == StockType.US:
            dfs = ak_cache(ak.stock_financial_us_analysis_indicator_em, symbol=symbol, indicator="单季报")
            if dfs is None or dfs.empty:
                return f"获取美股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标", f"# 数据来源: akshare", f"# 市场: 美股"]
            lines.append("\n".join(keys[0:15]))
            return "\n".join(lines)
        else:
            return f"不支持的市场类型: {market}"
    except Exception as exc:
        return f"获取财务指标失败: {exc}"


# ==================== 交易时间 ====================

@mcp.tool(
    title="获取当前时间及A股交易日信息",
    description="获取当前系统时间及A股交易日信息，建议在调用其他需要日期参数的工具前使用该工具",
)
def get_current_time():
    now = datetime.now()
    week = "日一二三四五六日"[now.isoweekday()]
    texts = [f"当前时间: {now.isoformat()}, 星期{week}"]
    dfs = ak_cache(ak.tool_trade_date_hist_sina, ttl=43200)
    if dfs is not None:
        start = now.date() - timedelta(days=5)
        ended = now.date() + timedelta(days=5)
        dates = [
            d.strftime("%Y-%m-%d")
            for d in dfs["trade_date"]
            if start <= d <= ended
        ]
        texts.append(f", 最近交易日有: {','.join(dates)}")
    return "".join(texts)
