import os
import time
import json
import random
import logging
import threading
import akshare as ak
import efinance as ef
import argparse
import requests
import pandas as pd
import numpy as np
from fastmcp import FastMCP
from pydantic import Field
from datetime import datetime, timedelta
from starlette.middleware.cors import CORSMiddleware
from .cache import CacheKey
from ._version import __version__
from .data_provider import (
    DataFetcherManager,
    to_chinese_columns,
    StockType,
    validate_stock_type,
)

_LOGGER = logging.getLogger(__name__)
# 日志级别可通过环境变量配置
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOGGER.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

mcp = FastMCP(name="stock-data-mcp", version=__version__)

# 全局数据获取管理器（支持多数据源自动故障转移）
_data_manager = None
_data_manager_lock = threading.Lock()

# 技术指标列定义（复用于股票和加密货币K线输出）
MA_COLUMNS = ["MA5", "MA10", "MA20", "MA30", "MA60"]
INDICATOR_COLUMNS = [
    "MACD", "DIF", "DEA",
    "KDJ.K", "KDJ.D", "KDJ.J",
    "RSI6", "RSI", "RSI24",
    "BOLL.U", "BOLL.M", "BOLL.L", "BOLL.W",  # 新增布林带宽度
    "OBV", "VMA5", "VMA10",  # 新增成交量均线
    "ATR",
    "ADX", "+DI", "-DI",  # 新增DMI方向指标
    "CCI", "WR", "VWAP",
]
STOCK_PRICE_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "换手率"] + MA_COLUMNS + INDICATOR_COLUMNS
CRYPTO_PRICE_COLUMNS = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额"] + MA_COLUMNS + INDICATOR_COLUMNS

def get_data_manager() -> DataFetcherManager:
    """获取全局数据管理器（延迟初始化，线程安全）"""
    global _data_manager
    if _data_manager is None:
        with _data_manager_lock:
            if _data_manager is None:
                _data_manager = DataFetcherManager()
    return _data_manager


# 数据源名称映射：将 Fetcher 类名转换为友好显示名称
_SOURCE_NAME_MAP = {
    "EfinanceFetcher": "efinance",
    "AkshareFetcher": "akshare",
    "TushareFetcher": "tushare",
    "BaostockFetcher": "baostock",
    "YfinanceFetcher": "yfinance",
    "AlphaVantage": "alphavantage",
    "AlphaVantageFetcher": "alphavantage",
}


def format_source_name(source: str) -> str:
    """格式化数据源名称为友好显示格式"""
    if not source:
        return "-"
    # 处理带后缀的情况，如 "AkshareFetcher_ratio"
    base_source = source.split("_")[0]
    friendly_name = _SOURCE_NAME_MAP.get(base_source, source)
    # 如果有后缀（如 _ratio），添加回去
    if "_" in source:
        suffix = source.split("_", 1)[1]
        friendly_name = f"{friendly_name} ({suffix})"
    return friendly_name

field_symbol = Field(description="股票代码")
field_market = Field("sh", description="股票市场，仅支持: sh(上证), sz(深证), hk(港股), us(美股), 不支持加密货币")

OKX_BASE_URL = os.getenv("OKX_BASE_URL") or "https://www.okx.com"
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL") or "https://www.binance.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10) AppleWebKit/537.36 Chrome/139"


def _http_get_with_retry(url, params=None, headers=None, max_retries=3, timeout=20):
    """带重试的 HTTP GET 请求"""
    return _http_request_with_retry("GET", url, params=params, headers=headers, max_retries=max_retries, timeout=timeout)


def _http_post_with_retry(url, json=None, headers=None, max_retries=3, timeout=20):
    """带重试的 HTTP POST 请求"""
    return _http_request_with_retry("POST", url, json=json, headers=headers, max_retries=max_retries, timeout=timeout)


def _http_request_with_retry(method, url, params=None, json=None, headers=None, max_retries=3, timeout=20):
    """带重试的 HTTP 请求"""
    if headers is None:
        headers = {"User-Agent": USER_AGENT}
    last_error = None
    for i in range(max_retries):
        try:
            res = requests.request(method, url, params=params, json=json, headers=headers, timeout=timeout)
            if res.status_code == 200:
                return res
        except Exception as e:
            last_error = e
            _LOGGER.warning(f"HTTP {method} 第{i+1}次失败 [{url}]: {e}")
            if i < max_retries - 1:
                time.sleep(1 * (i + 1))
    if last_error:
        raise last_error
    return None


@mcp.tool(
    title="查找股票代码",
    description="根据股票名称、公司名称等关键词查找股票代码, 不支持加密货币。"
                "该工具比较耗时，当你知道股票代码或用户已指定股票代码时，建议直接通过股票代码使用其他工具",
)
def search(
    keyword: str = Field(description="搜索关键词，公司名称、股票名称、股票代码、证券简称"),
    market: str = field_market,
):
    info = ak_search(None, keyword, market)
    if info is not None:
        lines = [f"# 搜索结果: {keyword}\n", f"数据来源: akshare\n", f"交易市场: {market}\n"]
        lines.append(info.to_string())
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
        lines = [f"# {symbol} 基本信息\n", f"数据来源: akshare\n", f"市场: {market}\n"]
        lines.append(all.to_string())
        return "\n".join(lines)

    info = ak_search(symbol, market)
    if info is not None:
        lines = [f"# {symbol} 基本信息\n", f"数据来源: akshare\n"]
        lines.append(info.to_string())
        return "\n".join(lines)
    return f"Not Found for {symbol}.{market}"


def _fetch_hk_prices(symbol: str, start_date: str, period: str = "daily") -> pd.DataFrame | None:
    """港股价格获取（带故障转移）: akshare → yfinance"""
    import yfinance as yf

    # 处理 Field 对象作为默认值的情况
    if hasattr(period, 'default'):
        period = period.default or "daily"

    # 1. 优先尝试 akshare
    try:
        dfs = ak_cache(ak.stock_hk_hist, symbol=symbol, period=period, start_date=start_date, ttl=3600)
        if dfs is not None and not dfs.empty:
            _LOGGER.debug(f"[港股] akshare 获取成功: {symbol}")
            dfs.attrs['source'] = 'akshare'
            return dfs
    except Exception as e:
        _LOGGER.warning(f"[港股] akshare 获取失败 {symbol}: {e}")

    # 2. 回退到 yfinance
    try:
        # 转换代码格式: 09988 → 9988.HK
        yf_symbol = f"{symbol.lstrip('0').zfill(4)}.HK"
        _LOGGER.debug(f"[港股] 尝试 yfinance: {yf_symbol}")

        # 转换日期格式
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}" if len(start_date) == 8 else start_date

        df = yf.download(yf_symbol, start=start_dt, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            df = df.reset_index()
            # 处理 MultiIndex 列名（yfinance 可能返回多层列名）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.rename(columns={
                "Date": "日期", "Open": "开盘", "Close": "收盘",
                "High": "最高", "Low": "最低", "Volume": "成交量"
            }, inplace=True)
            df["换手率"] = None
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
            _LOGGER.debug(f"[港股] yfinance 获取成功: {symbol}")
            df.attrs['source'] = 'yfinance'
            return df
    except Exception as e:
        _LOGGER.warning(f"[港股] yfinance 获取失败 {symbol}: {e}")

    return None


def _fetch_us_prices(symbol: str, start_date: str, period: str = "daily") -> pd.DataFrame | None:
    """美股价格获取（带故障转移）: akshare → yfinance → Alpha Vantage"""
    import yfinance as yf

    # 处理 Field 对象作为默认值的情况
    if hasattr(period, 'default'):
        period = period.default or "daily"

    # 1. 优先尝试 akshare
    try:
        dfs = ak_cache(stock_us_daily, symbol=symbol, start_date=start_date, period=period, ttl=3600)
        if dfs is not None and not dfs.empty:
            _LOGGER.debug(f"[美股] akshare 获取成功: {symbol}")
            dfs.attrs['source'] = 'akshare'
            return dfs
    except Exception as e:
        _LOGGER.warning(f"[美股] akshare 获取失败 {symbol}: {e}")

    # 2. 回退到 yfinance
    try:
        yf_symbol = symbol.upper()
        _LOGGER.debug(f"[美股] 尝试 yfinance: {yf_symbol}")

        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}" if len(start_date) == 8 else start_date

        df = yf.download(yf_symbol, start=start_dt, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            df = df.reset_index()
            # 处理 MultiIndex 列名（yfinance 可能返回多层列名）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.rename(columns={
                "Date": "日期", "Open": "开盘", "Close": "收盘",
                "High": "最高", "Low": "最低", "Volume": "成交量"
            }, inplace=True)
            df["换手率"] = None
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
            _LOGGER.debug(f"[美股] yfinance 获取成功: {symbol}")
            df.attrs['source'] = 'yfinance'
            return df
    except Exception as e:
        _LOGGER.warning(f"[美股] yfinance 获取失败 {symbol}: {e}")

    # 3. 回退到 Alpha Vantage（如果配置了 API Key）
    if ALPHA_VANTAGE_API_KEY:
        try:
            _LOGGER.debug(f"[美股] 尝试 Alpha Vantage: {symbol}")
            from .data_provider import AlphaVantageFetcher
            av = AlphaVantageFetcher()
            df = av._fetch_raw_data(symbol, start_date, datetime.now().strftime("%Y%m%d"))
            if df is not None and not df.empty:
                df = av._normalize_data(df, symbol)
                # 转换为中文列名
                df.rename(columns={
                    "date": "日期", "open": "开盘", "close": "收盘",
                    "high": "最高", "low": "最低", "volume": "成交量"
                }, inplace=True)
                df["换手率"] = None
                _LOGGER.debug(f"[美股] Alpha Vantage 获取成功: {symbol}")
                df.attrs['source'] = 'alphavantage'
                return df
        except Exception as e:
            _LOGGER.warning(f"[美股] Alpha Vantage 获取失败 {symbol}: {e}")

    return None


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
    # 对于 A 股，优先使用多数据源管理器
    if market in ("sh", "sz"):
        try:
            manager = get_data_manager()
            df = manager.get_daily_data(symbol, days=limit + 62)
            if df is not None and not df.empty:
                # 获取数据来源
                source = format_source_name(df.attrs.get('source', ''))
                # 转换为中文列名
                df = to_chinese_columns(df)
                # 添加换手率列（如果没有）
                if "换手率" not in df.columns:
                    df["换手率"] = None
                # 添加技术指标
                add_technical_indicators(df, df["收盘"], df["最低"], df["最高"], df.get("成交量"))
                available_cols = [c for c in STOCK_PRICE_COLUMNS if c in df.columns]
                all_lines = df.to_csv(columns=available_cols, index=False, float_format="%.2f").strip().split("\n")
                lines = [f"# {symbol} 历史价格\n", f"数据来源: {source}\n", f"市场: A股\n"]
                lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
                return "\n".join(lines)
        except Exception as e:
            _LOGGER.warning(f"多数据源获取失败，回退到原有逻辑: {e}")

    # 计算起始日期
    if period == "weekly":
        delta = {"weeks": limit + 62}
    else:
        delta = {"days": limit + 62}
    start_date = (datetime.now() - timedelta(**delta)).strftime("%Y%m%d")

    # 港股：使用带故障转移的函数
    if market == "hk":
        dfs = _fetch_hk_prices(symbol, start_date, period)
        if dfs is not None and not dfs.empty:
            add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
            all_lines = dfs.to_csv(columns=STOCK_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
            source = dfs.attrs.get('source', 'unknown')
            lines = [f"# {symbol} 历史价格\n", f"数据来源: {source}\n", f"市场: 港股\n"]
            lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
            return "\n".join(lines)
        return f"Not Found for {symbol}.{market}"

    # 美股：使用带故障转移的函数
    if market == "us":
        dfs = _fetch_us_prices(symbol, start_date, period)
        if dfs is not None and not dfs.empty:
            add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
            all_lines = dfs.to_csv(columns=STOCK_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
            source = dfs.attrs.get('source', 'unknown')
            lines = [f"# {symbol} 历史价格\n", f"数据来源: {source}\n", f"市场: 美股\n"]
            lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
            return "\n".join(lines)
        return f"Not Found for {symbol}.{market}"

    # 其他市场（A股回退、ETF）
    markets = [
        ["sh", ak.stock_zh_a_hist, {}],
        ["sz", ak.stock_zh_a_hist, {}],
        ["sh", fund_etf_hist_sina, {"market": "sh"}],
        ["sz", fund_etf_hist_sina, {"market": "sz"}],
    ]
    for m in markets:
        if m[0] != market:
            continue
        kws = {"period": period, "start_date": start_date, **m[2]}
        dfs = ak_cache(m[1], symbol=symbol, ttl=3600, **kws)
        if dfs is None or dfs.empty:
            continue
        add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
        all = dfs.to_csv(columns=STOCK_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
        lines = [f"# {symbol} 历史价格\n", f"数据来源: akshare\n", f"市场: A股/ETF\n"]
        lines.append("\n".join([all[0], *all[-limit:]]))
        return "\n".join(lines)
    return f"Not Found for {symbol}.{market}"


def stock_us_daily(symbol, start_date="2025-01-01", period="daily"):
    dfs = ak.stock_us_daily(symbol=symbol)
    if dfs is None or dfs.empty:
        return None
    dfs.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}, inplace=True)
    dfs["换手率"] = None
    dfs.index = pd.to_datetime(dfs["日期"], errors="coerce")
    return dfs.loc[start_date:]

def fund_etf_hist_sina(symbol, market="sh", start_date="2025-01-01", period="daily"):
    dfs = ak.fund_etf_hist_sina(symbol=f"{market}{symbol}")
    if dfs is None or dfs.empty:
        return None
    dfs.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}, inplace=True)
    dfs["换手率"] = None
    dfs.index = pd.to_datetime(dfs["日期"], errors="coerce")
    return dfs.loc[start_date:]


@mcp.tool(
    title="获取股票/加密货币相关新闻",
    description="根据股票代码或加密货币符号获取近期相关新闻",
)
def stock_news(
    symbol: str = Field(description="股票代码/加密货币符号"),
    limit: int = Field(15, description="返回数量(int)", strict=False),
):
    try:
        result = ak_cache(stock_news_em, symbol=symbol, ttl=3600)
        if result is None or (hasattr(result, 'empty') and result.empty):
            return f"未找到 {symbol} 相关新闻"
        news = list(dict.fromkeys([
            v["新闻内容"]
            for v in result.to_dict(orient="records")
            if isinstance(v, dict)
        ]))
        if news:
            lines = [f"# {symbol} 相关新闻\n", f"数据来源: 东方财经\n"]
            lines.extend(news[0:limit])
            return "\n".join(lines)
        return f"未找到 {symbol} 相关新闻"
    except Exception as e:
        _LOGGER.warning(f"获取新闻失败: {e}")
        return f"获取 {symbol} 新闻失败: {e}"

def stock_news_em(symbol, limit=20):
    cbk = "jQuery351013927587392975826_1763361926020"
    resp = requests.get(
        "http://search-api-web.eastmoney.com/search/jsonp",
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"https://so.eastmoney.com/news/s?keyword={symbol}",
        },
        params={
            "cb": cbk,
            "param": '{"uid":"",'
                     f'"keyword":"{symbol}",'
                     '"type":["cmsArticleWebOld"],"client":"web","clientType":"web","clientVersion":"curr",'
                     '"param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default","pageIndex":1,"pageSize":10,'
                     '"preTag":"<em>","postTag":"</em>"}}}',
        },
    )
    text = resp.text.replace(cbk, "").strip().strip("()")
    data = json.loads(text) or {}
    dfs = pd.DataFrame(data.get("result", {}).get("cmsArticleWebOld") or [])
    dfs.sort_values("date", ascending=False, inplace=True)
    dfs = dfs.head(limit)
    dfs["新闻内容"] = dfs["content"].str.replace(r"</?em>", "", regex=True)
    return dfs


@mcp.tool(
    title="股票财务指标",
    description="获取股票财务报告关键指标，支持A股、港股、美股市场",
)
def stock_indicators(
    symbol: str = field_symbol,
    market: str = Field("sh", description="市场: 'sh'/'sz'(A股), 'hk'(港股), 'us'(美股)"),
):
    """获取股票财务指标"""
    try:
        if market in ["sh", "sz"]:
            # A股
            dfs = ak_cache(ak.stock_financial_abstract_ths, symbol=symbol)
            if dfs is None or dfs.empty:
                return f"获取A股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标\n", f"数据来源: akshare\n", f"市场: A股\n"]
            lines.append("\n".join([keys[0], *keys[-15:]]))
            return "\n".join(lines)
        elif market == "hk":
            # 港股
            dfs = ak_cache(ak.stock_financial_hk_analysis_indicator_em, symbol=symbol, indicator="报告期")
            if dfs is None or dfs.empty:
                return f"获取港股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标\n", f"数据来源: akshare\n", f"市场: 港股\n"]
            lines.append("\n".join(keys[0:15]))
            return "\n".join(lines)
        elif market == "us":
            # 美股
            dfs = ak_cache(ak.stock_financial_us_analysis_indicator_em, symbol=symbol, indicator="单季报")
            if dfs is None or dfs.empty:
                return f"获取美股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标\n", f"数据来源: akshare\n", f"市场: 美股\n"]
            lines.append("\n".join(keys[0:15]))
            return "\n".join(lines)
        else:
            return f"不支持的市场类型: {market}"
    except Exception as exc:
        _LOGGER.warning(f"获取财务指标失败: {exc}")
        return f"获取财务指标失败: {exc}"


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

def recent_trade_date():
    now = datetime.now().date()
    dfs = ak_cache(ak.tool_trade_date_hist_sina, ttl=43200)
    if dfs is None:
        return now
    dfs.sort_values("trade_date", ascending=False, inplace=True)
    for d in dfs["trade_date"]:
        if d <= now:
            return d
    return now


@mcp.tool(
    title="A股涨停/强势股池",
    description="获取中国A股市场(上证、深证)的涨停股池或强势股池数据",
)
def stock_zt_pool(
    pool_type: str = Field("涨停", description="股池类型: '涨停'(涨停股池), '强势'(强势股池), '跌停'(跌停股池), '昨日涨停'(昨日涨停股今日表现)"),
    date: str = Field("", description="交易日日期(可选)，默认为最近的交易日，格式: 20251231"),
    limit: int = Field(50, description="返回数量(int,30-100)", strict=False),
):
    if not date:
        date = recent_trade_date().strftime("%Y%m%d")

    try:
        if pool_type == "强势":
            dfs = ak_cache(ak.stock_zt_pool_strong_em, date=date, ttl=1200)
            title = "强势股池"
        elif pool_type == "跌停":
            dfs = ak_cache(ak.stock_zt_pool_dtgc_em, date=date, ttl=1200)
            title = "跌停股池"
        elif pool_type == "昨日涨停":
            dfs = ak_cache(ak.stock_zt_pool_zbgc_em, date=date, ttl=1200)
            title = "昨日涨停股今日表现"
        else:
            dfs = ak_cache(ak.stock_zt_pool_em, date=date, ttl=1200)
            title = "涨停股池"

        if dfs is None or dfs.empty:
            return f"获取{title}数据失败"

        cnt = len(dfs)
        dfs.drop(columns=["序号", "流通市值", "总市值"], inplace=True, errors='ignore')
        if "成交额" in dfs.columns:
            dfs.sort_values("成交额", ascending=False, inplace=True)
        dfs = dfs.head(int(limit))
        lines = [f"# {title}\n", f"数据来源: akshare\n", f"共{cnt}只股票\n"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as exc:
        _LOGGER.warning(f"获取股池数据失败: {exc}")
        return f"获取股池数据失败: {exc}"


@mcp.tool(
    title="A股龙虎榜统计",
    description="获取中国A股市场(上证、深证)的龙虎榜个股上榜统计数据。支持多数据源。",
)
def stock_lhb_ggtj_sina(
    days: str = Field("5", description="统计最近天数，仅支持: [5/10/30/60]"),
    limit: int = Field(50, description="返回数量(int,30-100)", strict=False),
):
    try:
        manager = get_data_manager()
        dfs = manager.get_billboard(days)

        if dfs is None or dfs.empty:
            return "获取龙虎榜数据失败"

        source = format_source_name(dfs.attrs.get('source', ''))
        dfs = dfs.head(int(limit))
        lines = [f"# 龙虎榜统计\n", f"数据来源: {source}\n"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取龙虎榜失败: {e}")
        return f"获取龙虎榜数据失败: {e}"


@mcp.tool(
    title="A股板块资金流",
    description="获取中国A股市场(上证、深证)的行业资金流向数据",
)
def stock_sector_fund_flow_rank(
    days: str = Field("今日", description="天数，仅支持: {'今日','5日','10日'}，如果需要获取今日数据，请确保是交易日"),
    cate: str = Field("行业资金流", description="仅支持: {'行业资金流','概念资金流','地域资金流'}"),
):
    # 主数据源：东方财富板块资金流
    try:
        dfs = fetch_with_retry(
            ak.stock_sector_fund_flow_rank,
            max_retries=2,
            delay=2.0,
            initial_delay=0.5,
            indicator=days,
            sector_type=cate
        )
        if dfs is not None and not dfs.empty:
            if "今日涨跌幅" in dfs.columns:
                dfs.sort_values("今日涨跌幅", ascending=False, inplace=True)
            dfs.drop(columns=["序号"], inplace=True, errors='ignore')
            dfs = pd.concat([dfs.head(20), dfs.tail(20)])
            lines = [f"# {cate}\n", f"数据来源: akshare (东方财富)\n"]
            lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)
    except Exception as e:
        _LOGGER.debug(f"东方财富板块资金流获取失败: {e}")

    # 备用数据源：行业板块实时行情（仅支持行业板块+今日）
    if cate == "行业资金流":
        try:
            time.sleep(1)  # 防止请求过快
            if days == "今日":
                dfs = ak.stock_board_industry_name_em()
            else:
                dfs = ak.stock_board_industry_hist_em(period=days.replace("日", ""))
            if dfs is not None and not dfs.empty:
                if "涨跌幅" in dfs.columns:
                    dfs.sort_values("涨跌幅", ascending=False, inplace=True)
                elif "涨幅" in dfs.columns:
                    dfs.sort_values("涨幅", ascending=False, inplace=True)
                dfs.drop(columns=["排名"], inplace=True, errors='ignore')
                dfs = pd.concat([dfs.head(20), dfs.tail(20)])
                lines = [f"# {cate}\n", f"数据来源: akshare (东方财富-板块行情)\n"]
                lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
                return "\n".join(lines)
        except Exception as e:
            _LOGGER.debug(f"备用板块行情获取失败: {e}")

    # 第三备用：概念板块
    if cate == "概念资金流":
        try:
            time.sleep(1)
            dfs = ak.stock_board_concept_name_em()
            if dfs is not None and not dfs.empty:
                if "涨跌幅" in dfs.columns:
                    dfs.sort_values("涨跌幅", ascending=False, inplace=True)
                dfs.drop(columns=["排名"], inplace=True, errors='ignore')
                dfs = pd.concat([dfs.head(20), dfs.tail(20)])
                lines = [f"# {cate}\n", f"数据来源: akshare (东方财富-概念板块)\n"]
                lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
                return "\n".join(lines)
        except Exception as e:
            _LOGGER.debug(f"概念板块获取失败: {e}")

    return f"获取{cate}数据失败（数据源可能暂时不可用，请稍后重试）"


@mcp.tool(
    title="沪深港通北向资金",
    description="获取沪深港通北向资金(外资)流向数据，包括沪股通、深股通的资金净流入情况。北向资金是A股重要的风向标。",
)
def stock_north_flow(
    indicator: str = Field("北向资金", description="指标类型，可选: '北向资金', '沪股通', '深股通'"),
):
    """获取北向资金流向数据"""
    try:
        # 获取沪深港通资金流向历史数据
        df = ak_cache(ak.stock_hsgt_fund_flow_summary_em, ttl=600)
        if df is None or df.empty:
            return "获取北向资金数据失败"

        # 根据指标筛选
        if indicator == "沪股通":
            cols = ["日期", "沪股通-净流入"]
            if "沪股通-净流入" in df.columns:
                df = df[["日期", "沪股通-净流入"]].copy()
                df.columns = ["日期", "净流入(亿)"]
        elif indicator == "深股通":
            cols = ["日期", "深股通-净流入"]
            if "深股通-净流入" in df.columns:
                df = df[["日期", "深股通-净流入"]].copy()
                df.columns = ["日期", "净流入(亿)"]
        else:
            # 北向资金 = 沪股通 + 深股通
            if "北向资金-净流入" in df.columns:
                df = df[["日期", "北向资金-净流入"]].copy()
                df.columns = ["日期", "净流入(亿)"]
            elif "沪股通-净流入" in df.columns and "深股通-净流入" in df.columns:
                df["净流入(亿)"] = df["沪股通-净流入"] + df["深股通-净流入"]
                df = df[["日期", "净流入(亿)"]].copy()

        # 返回最近30天数据
        df = df.head(30)
        lines = [f"# {indicator}流向\n", f"数据来源: akshare\n"]
        lines.append(df.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as exc:
        _LOGGER.warning(f"获取北向资金失败: {exc}")
        return f"获取北向资金数据失败: {exc}"


def _detect_stock_market(symbol: str) -> str:
    """根据股票代码判断市场"""
    if symbol.startswith(('6', '5')):
        return 'sh'
    elif symbol.startswith(('0', '3', '1', '2')):
        return 'sz'
    return 'sh'


@mcp.tool(
    title="A股融资融券",
    description="获取A股市场融资融券数据，包括融资余额、融券余额等。融资融券是衡量市场杠杆资金的重要指标。",
)
def stock_margin_trading(
    symbol: str = Field("", description="股票代码（可选），留空则获取市场整体数据"),
    market: str = Field("sh", description="市场: 'sh'(沪市), 'sz'(深市)"),
    limit: int = Field(30, description="返回数据条数"),
):
    """获取融资融券数据（使用多数据源自动故障转移）"""
    try:
        if symbol:
            # 个股融资融券数据：使用多数据源管理器
            stock_market = _detect_stock_market(symbol)
            manager = get_data_manager()
            df = manager.get_margin_detail(symbol, stock_market)

            if df is not None and not df.empty:
                source = format_source_name(df.attrs.get('source', ''))
                is_ratio = df.attrs.get('is_ratio_data', False)

                if is_ratio:
                    result = f"# {symbol} 融资融券比例\n\n"
                    result += f"数据来源: {source}\n"
                    result += "注：交易所明细接口暂不可用，以下为融资融券比例数据\n\n"
                    result += df.head(limit).to_csv(index=False, float_format="%.2f").strip()
                    return result
                else:
                    lines = [f"# {symbol} 融资融券\n", f"数据来源: {source}\n"]
                    lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())
                    return "\n".join(lines)

            # 所有数据源都失败
            return (
                f"获取个股 {symbol} 融资融券数据失败\n\n"
                f"可能原因:\n"
                f"1. 该股票不在融资融券标的范围内\n"
                f"2. akshare深交所接口存在兼容性问题（建议升级akshare）\n"
                f"3. 网络连接问题"
            )
        else:
            # 市场整体融资融券数据
            if market == "sh":
                df = ak_cache(ak.stock_margin_sse, start_date="", end_date="", ttl=1800)
            else:
                df = ak_cache(ak.stock_margin_szse, start_date="", end_date="", ttl=1800)

            if df is None or df.empty:
                return f"获取{market}市场融资融券数据失败"

            # 返回最近的数据
            df = df.tail(limit)
            return df.to_csv(index=False, float_format="%.2f").strip()
    except Exception as exc:
        _LOGGER.warning(f"获取融资融券失败: {exc}")
        return f"获取融资融券数据失败: {exc}"


@mcp.tool(
    title="A股大宗交易",
    description="获取A股大宗交易数据，包括成交价、成交量、溢价率等。大宗交易反映机构大额交易动向。",
)
def stock_block_trade(
    symbol: str = Field("", description="股票代码（可选），留空则获取当日全市场数据"),
    limit: int = Field(50, description="返回数据条数"),
):
    """获取大宗交易数据"""
    try:
        if symbol:
            # 个股大宗交易历史
            try:
                df = ak_cache(ak.stock_dzjy_mrmx, symbol=symbol, ttl=1800)
                if df is not None and not df.empty:
                    df = df.head(limit)
                    lines = [f"# {symbol} 大宗交易\n", f"数据来源: akshare\n"]
                    lines.append(df.to_csv(index=False, float_format="%.2f").strip())
                    return "\n".join(lines)
            except Exception as e:
                _LOGGER.debug(f"stock_dzjy_mrmx 获取失败: {e}")
            # 尝试另一个接口
            try:
                df = ak_cache(ak.stock_dzjy_mrtj, start_date="", end_date="", ttl=1800)
                if df is not None and not df.empty:
                    if "证券代码" in df.columns:
                        df = df[df["证券代码"].astype(str).str.contains(symbol)]
                    if not df.empty:
                        lines = [f"# {symbol} 大宗交易\n", f"数据来源: akshare\n"]
                        lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())
                        return "\n".join(lines)
            except Exception as e:
                _LOGGER.debug(f"stock_dzjy_mrtj 获取失败: {e}")
            return f"未找到股票 {symbol} 的大宗交易数据"
        else:
            # 全市场大宗交易每日统计
            df = ak_cache(ak.stock_dzjy_mrtj, start_date="", end_date="", ttl=1800)
            if df is None or df.empty:
                return "获取大宗交易数据失败"
            df = df.head(limit)
            lines = ["# 大宗交易统计\n", "数据来源: akshare\n"]
            lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)
    except Exception as exc:
        _LOGGER.warning(f"获取大宗交易失败: {exc}")
        return f"获取大宗交易数据失败: {exc}"


@mcp.tool(
    title="A股股东人数",
    description="获取A股股东户数变化数据，筹码集中度的重要指标。股东人数减少通常意味着筹码趋于集中。",
)
def stock_holder_num(
    symbol: str = Field(description="股票代码，如: 300058, 600036"),
):
    """获取股东人数变化数据"""
    try:
        # 使用单股查询接口（stock_zh_a_gdhs 的 symbol 参数是日期，不是股票代码）
        df = ak_cache(ak.stock_zh_a_gdhs_detail_em, symbol=symbol, ttl=3600)
        if df is not None and not df.empty:
            lines = [f"# {symbol} 股东人数\n", f"数据来源: akshare\n"]
            lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)

        return f"未找到股票 {symbol} 的股东人数数据"
    except Exception as exc:
        _LOGGER.warning(f"获取股东人数失败: {exc}")
        return f"获取股东人数数据失败: {exc}"


@mcp.tool(
    title="全球财经快讯",
    description="获取最新的全球财经快讯",
)
def stock_news_global():
    news = ["# 全球财经快讯\n", "数据来源: 新浪财经, NewsNow\n"]
    try:
        dfs = ak.stock_info_global_sina()
        csv = dfs.to_csv(index=False, float_format="%.2f").strip()
        csv = csv.replace(datetime.now().strftime("%Y-%m-%d "), "")
        lines = csv.split("\n")
        # 第一行是标题，保留；后续行添加来源标识
        if lines:
            news.append(lines[0] + ",来源")  # 添加来源列标题
            for line in lines[1:]:
                news.append(f"{line},新浪财经")
    except Exception as e:
        _LOGGER.debug(f"获取新浪财经快讯失败: {e}")
    news.extend(newsnow_news())
    return "\n".join(news)


# NewsNow 频道名称映射
_NEWSNOW_CHANNEL_NAMES = {
    "wallstreetcn-quick": "华尔街见闻",
    "cls-telegraph": "财联社",
    "jin10": "金十数据",
    "gelonghui": "格隆汇",
    "fastbull-express": "快讯通",
    "yicai": "第一财经",
    "caixin": "财新",
    "36kr-newsflash": "36氪",
}


def newsnow_news(channels=None):
    base = os.getenv("NEWSNOW_BASE_URL")
    if not base:
        _LOGGER.debug("NEWSNOW_BASE_URL 未配置，跳过 NewsNow 数据源")
        return []
    if not channels:
        channels = os.getenv("NEWSNOW_CHANNELS") or "wallstreetcn-quick,cls-telegraph,jin10"
    if isinstance(channels, str):
        channels = channels.split(",")
    _LOGGER.debug(f"NewsNow 请求: base={base}, channels={channels}")
    all = []
    try:
        res = requests.post(
            f"{base}/api/s/entire",
            json={"sources": channels},
            headers={
                "User-Agent": USER_AGENT,
                "Referer": base,
            },
            timeout=60,
        )
        _LOGGER.debug(f"NewsNow 响应状态: {res.status_code}")
        lst = res.json() or []
        _LOGGER.debug(f"NewsNow 获取到 {len(lst)} 个频道数据")
        for item in lst:
            source_id = item.get("id", "")
            source_name = _NEWSNOW_CHANNEL_NAMES.get(source_id, source_id)
            for v in item.get("items", [])[0:15]:
                title = v.get("title", "")
                extra = v.get("extra") or {}
                hover = extra.get("hover") or title
                info = extra.get("info") or ""
                content = f"{hover} {info}".strip().replace("\n", " ")
                # 提取时间 (格式: 2026-02-06 13:26:11 -> 13:26:11)
                pub_date = str(v.get("pubDate", ""))
                time_str = pub_date.split(" ")[-1] if " " in pub_date else ""
                all.append(f"{time_str},{content},{source_name}")
    except Exception as e:
        _LOGGER.warning(f"NewsNow 请求失败: {e}")
    return all


@mcp.tool(
    title="获取加密货币历史价格",
    description="获取OKX加密货币的历史K线数据，包括价格、交易量和技术指标。支持自动重试。",
)
def okx_prices(
    instId: str = Field("BTC-USDT", description="产品ID，格式: BTC-USDT"),
    bar: str = Field("1H", description="K线时间粒度，仅支持: [1m/3m/5m/15m/30m/1H/2H/4H/6H/12H/1D/2D/3D/1W/1M/3M] 除分钟为小写m外,其余均为大写"),
    limit: int = Field(100, description="返回数量(int)，最大300，最小建议30", strict=False),
):
    if not bar.endswith("m"):
        bar = bar.upper()

    try:
        res = _http_get_with_retry(
            f"{OKX_BASE_URL}/api/v5/market/candles",
            params={
                "instId": instId,
                "bar": bar,
                "limit": max(300, limit + 62),
            },
        )
        if res is None:
            return f"OKX API 请求失败"
        data = res.json() or {}
        dfs = pd.DataFrame(data.get("data", []))
    except Exception as e:
        return f"OKX API 请求失败: {e}"

    if dfs.empty:
        return f"未获取到 {instId} 数据"

    dfs.columns = ["时间", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "成交额USDT", "K线已完结"]
    dfs.sort_values("时间", inplace=True)
    dfs["时间"] = pd.to_datetime(pd.to_numeric(dfs["时间"], errors="coerce"), unit="ms")
    dfs["开盘"] = pd.to_numeric(dfs["开盘"], errors="coerce")
    dfs["最高"] = pd.to_numeric(dfs["最高"], errors="coerce")
    dfs["最低"] = pd.to_numeric(dfs["最低"], errors="coerce")
    dfs["收盘"] = pd.to_numeric(dfs["收盘"], errors="coerce")
    dfs["成交量"] = pd.to_numeric(dfs["成交量"], errors="coerce")
    dfs["成交额"] = pd.to_numeric(dfs["成交额"], errors="coerce")
    add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
    all = dfs.to_csv(columns=CRYPTO_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
    lines = [f"# {instId} K线数据\n", f"数据来源: OKX\n"]
    lines.append("\n".join([all[0], *all[-limit:]]))
    return "\n".join(lines)


@mcp.tool(
    title="获取加密货币杠杆多空比",
    description="获取OKX加密货币借入计价货币与借入交易货币的累计数额比值。支持自动重试。",
)
def okx_loan_ratios(
    symbol: str = Field("BTC", description="币种，格式: BTC 或 ETH"),
    period: str = Field("1h", description="时间粒度，仅支持: [5m/1H/1D] 注意大小写，仅分钟为小写m"),
):
    try:
        res = _http_get_with_retry(
            f"{OKX_BASE_URL}/api/v5/rubik/stat/margin/loan-ratio",
            params={"ccy": symbol, "period": period},
        )
        if res is None:
            return f"OKX API 请求失败"
        data = res.json() or {}
    except Exception as e:
        return f"OKX API 请求失败: {e}"

    dfs = pd.DataFrame(data.get("data", []))
    if dfs.empty:
        return f"未获取到 {symbol} 多空比数据"
    dfs.columns = ["时间", "多空比"]
    dfs["时间"] = pd.to_datetime(pd.to_numeric(dfs["时间"], errors="coerce"), unit="ms")
    dfs["多空比"] = pd.to_numeric(dfs["多空比"], errors="coerce")
    lines = [f"# {symbol} 多空比\n", f"数据来源: OKX\n"]
    lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
    return "\n".join(lines)


@mcp.tool(
    title="获取加密货币主动买卖情况",
    description="获取OKX加密货币主动买入和卖出的交易量。支持自动重试。",
)
def okx_taker_volume(
    symbol: str = Field("BTC", description="币种，格式: BTC 或 ETH"),
    period: str = Field("1h", description="时间粒度，仅支持: [5m/1H/1D] 注意大小写，仅分钟为小写m"),
    instType: str = Field("SPOT", description="产品类型 SPOT:现货 CONTRACTS:衍生品"),
):
    try:
        res = _http_get_with_retry(
            f"{OKX_BASE_URL}/api/v5/rubik/stat/taker-volume",
            params={"ccy": symbol, "period": period, "instType": instType},
        )
        if res is None:
            return f"OKX API 请求失败"
        data = res.json() or {}
    except Exception as e:
        return f"OKX API 请求失败: {e}"

    dfs = pd.DataFrame(data.get("data", []))
    if dfs.empty:
        return f"未获取到 {symbol} 主动买卖数据"
    dfs.columns = ["时间", "卖出量", "买入量"]
    dfs["时间"] = pd.to_datetime(pd.to_numeric(dfs["时间"], errors="coerce"), unit="ms")
    dfs["卖出量"] = pd.to_numeric(dfs["卖出量"], errors="coerce")
    dfs["买入量"] = pd.to_numeric(dfs["买入量"], errors="coerce")
    lines = [f"# {symbol} 主动买卖\n", f"数据来源: OKX\n"]
    lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
    return "\n".join(lines)


@mcp.tool(
    title="获取加密货币分析报告",
    description="获取币安对加密货币的AI分析报告，此工具对分析加密货币非常有用。支持自动重试。",
)
def binance_ai_report(
    symbol: str = Field("BTC", description="加密货币币种，格式: BTC 或 ETH"),
):
    try:
        res = _http_post_with_retry(
            f"{BINANCE_BASE_URL}/bapi/bigdata/v3/friendly/bigdata/search/ai-report/report",
            json={
                'lang': 'zh-CN',
                'token': symbol,
                'symbol': f'{symbol}USDT',
                'product': 'web-spot',
                'timestamp': int(time.time() * 1000),
                'translateToken': None,
            },
            headers={
                'User-Agent': USER_AGENT,
                'Referer': f'https://www.binance.com/zh-CN/trade/{symbol}_USDT?type=spot',
                'lang': 'zh-CN',
            },
        )
    except Exception as e:
        return f"Binance API 请求失败: {e}"

    if res is None:
        return f"未获取到 {symbol} 分析报告"

    try:
        resp = res.json() or {}
    except Exception as e:
        _LOGGER.debug(f"JSON 解析失败，尝试文本解析: {e}")
        try:
            resp = json.loads(res.text.strip()) or {}
        except Exception as e2:
            _LOGGER.debug(f"文本解析也失败: {e2}")
            return res.text
    data = resp.get('data') or {}
    report = data.get('report') or {}
    translated = report.get('translated') or report.get('original') or {}
    modules = translated.get('modules') or []
    txts = [f"# {symbol} AI分析报告\n", f"数据来源: Binance\n"]
    for module in modules:
        if tit := module.get('overview'):
            txts.append(tit)
        for point in module.get('points', []):
            txts.append(point.get('content', ''))
    return '\n'.join(txts)


@mcp.tool(
    title="获取股票实时行情",
    description="获取A股/港股实时行情数据，包括最新价、涨跌幅、成交量、换手率、市盈率等。支持多数据源自动故障转移。",
)
def stock_realtime(
    symbol: str = field_symbol,
    market: str = Field("sh", description="股票市场，仅支持: sh(上证), sz(深证), hk(港股)"),
):
    try:
        # 综合校验股票类型：结合用户指定的 market 和自动检测
        stock_type, validated_market = validate_stock_type(symbol, market)

        manager = get_data_manager()
        quote = manager.get_realtime_quote(symbol, stock_type=stock_type)
        if quote is None:
            return f"Not Found for {symbol}.{validated_market}"

        # 统一输出为 CSV 格式（与 batch_realtime 一致）
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
        lines = [f"# {symbol} 实时行情\n", f"数据来源: {source}\n"]
        lines.append(df.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取实时行情失败: {e}")
        return f"获取 {symbol} 实时行情失败: {e}"


@mcp.tool(
    title="获取筹码分布",
    description="获取A股筹码分布数据，包括获利比例、平均成本、成本区间、筹码集中度等。",
)
def stock_chip(
    symbol: str = field_symbol,
):
    # ETF/LOF/基金等产品不支持筹码分布
    if symbol.startswith(('51', '15', '16', '50', '52', '56', '58', '11', '12')):
        return f"{symbol} 是ETF/LOF/基金/可转债等产品，不支持筹码分布查询。筹码分布仅适用于普通A股。"

    try:
        manager = get_data_manager()
        chip = manager.get_chip_distribution(symbol)
        if chip is None:
            return f"未找到 {symbol} 的筹码分布数据，请确认是有效的A股代码"

        # 格式化输出（Markdown）
        lines = [
            f"# {chip.code} 筹码分布\n",
            f"数据来源: {chip.source}\n",
            f"日期: {chip.date or '-'}\n",
            "## 筹码数据",
            f"- 获利比例: {chip.profit_ratio or '-'}%",
            f"- 平均成本: {chip.avg_cost or '-'}",
            f"- 90%成本区间: {chip.cost_90_low or '-'} - {chip.cost_90_high or '-'}",
            f"- 90%集中度: {chip.concentration_90 or '-'}%",
            f"- 70%成本区间: {chip.cost_70_low or '-'} - {chip.cost_70_high or '-'}",
            f"- 70%集中度: {chip.concentration_70 or '-'}%",
        ]

        # 添加筹码状态分析
        status = chip.get_chip_status()
        if 'chip_level' in status:
            lines.append(f"筹码状态: {status['chip_level']}")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取筹码分布失败: {e}")
        return f"获取 {symbol} 筹码分布失败: {e}"


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

        codes = codes[:limit]  # 限制数量
        manager = get_data_manager()
        quotes = manager.prefetch_realtime_quotes(codes)

        if not quotes:
            return "未获取到任何行情数据"

        # 转换为 DataFrame 输出（与 stock_realtime 字段一致）
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
        lines = [f"# 批量实时行情\n", f"数据来源: {source_str}\n"]
        lines.append(df.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"批量获取实时行情失败: {e}")
        return f"批量获取实时行情失败: {e}"


@mcp.tool(
    title="查看数据源状态",
    description="查看多数据源的状态和熔断器信息",
)
def data_source_status():
    try:
        manager = get_data_manager()
        status = manager.get_status()

        lines = ["# 数据源状态\n"]
        for fetcher in status.get('fetchers', []):
            available = "OK" if fetcher['available'] else "FAIL"
            lines.append(f"- [{available}] {fetcher['name']} (优先级: {fetcher['priority']})")

        lines.append("\n## 熔断器状态")

        for name, breaker_status in [
            ("日线数据", status.get('daily_circuit_breaker', {})),
            ("实时行情", status.get('realtime_circuit_breaker', {})),
            ("筹码分布", status.get('chip_circuit_breaker', {})),
            ("资金流向", status.get('fund_flow_circuit_breaker', {})),
            ("板块数据", status.get('board_circuit_breaker', {})),
            ("龙虎榜", status.get('billboard_circuit_breaker', {})),
            ("融资融券", status.get('margin_circuit_breaker', {})),
            ("美股基本面", status.get('us_financials_circuit_breaker', {})),
        ]:
            if breaker_status:
                lines.append(f"\n### {name}")
                for source, state in breaker_status.items():
                    state_label = "正常" if state['state'] == 'closed' else "已熔断"
                    lines.append(f"- {source}: {state_label} (失败次数: {state['failure_count']})")
            else:
                lines.append(f"\n### {name}: 无熔断记录")

        return "\n".join(lines)
    except Exception as e:
        return f"获取数据源状态失败: {e}"


@mcp.tool(
    title="获取股票多周期统计",
    description="获取A股多周期统计数据，包括累计涨跌幅、振幅、换手率等，支持5日、10日、20日、60日、120日等周期",
)
def stock_period_stats(
    symbol: str = field_symbol,
    market: str = Field("sh", description="股票市场，仅支持: sh(上证), sz(深证)"),
):
    try:
        manager = get_data_manager()
        # 获取足够多的历史数据用于计算
        df = manager.get_daily_data(symbol, days=180)
        if df is None or df.empty:
            return f"Not Found for {symbol}.{market}"

        # 获取数据来源
        source = format_source_name(df.attrs.get('source', ''))

        df = to_chinese_columns(df)
        close = df["收盘"]
        high = df["最高"]
        low = df["最低"]
        volume = df.get("成交量")

        periods = [5, 10, 20, 60, 120]
        available_periods = [p for p in periods if len(close) >= p]

        lines = [f"# {symbol} 多周期统计\n", f"数据来源: {source}\n"]

        # 价格统计
        lines.append("## 价格统计")
        lines.append(f"- 最新价: {close.iloc[-1]:.2f}")
        for p in available_periods:
            avg_price = close.iloc[-p:].mean()
            max_price = high.iloc[-p:].max()
            min_price = low.iloc[-p:].min()
            lines.append(f"- {p}日均价: {avg_price:.2f}, 最高: {max_price:.2f}, 最低: {min_price:.2f}")

        # 涨跌幅统计
        lines.append("\n## 涨跌幅统计")
        if len(close) >= 2:
            today_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            lines.append(f"- 当日涨跌: {today_change:.2f}%")
        for p in available_periods:
            if len(close) > p:
                change = (close.iloc[-1] / close.iloc[-p-1] - 1) * 100
                lines.append(f"- {p}日累计涨跌: {change:.2f}%")

        # 振幅统计
        lines.append("\n## 振幅统计")
        if len(high) >= 1:
            today_amp = (high.iloc[-1] / low.iloc[-1] - 1) * 100
            lines.append(f"- 当日振幅: {today_amp:.2f}%")
        for p in available_periods:
            amp = (high.iloc[-p:].max() / low.iloc[-p:].min() - 1) * 100
            lines.append(f"- {p}日振幅: {amp:.2f}%")

        # 换手率统计（如果有成交量数据）
        if volume is not None and "换手率" in df.columns:
            turnover = df["换手率"]
            lines.append("\n## 换手率统计")
            if len(turnover) >= 1 and turnover.iloc[-1] is not None:
                lines.append(f"- 当日换手: {turnover.iloc[-1]:.2f}%")
            for p in available_periods:
                avg_turn = turnover.iloc[-p:].mean()
                total_turn = turnover.iloc[-p:].sum()
                if avg_turn is not None:
                    lines.append(f"- {p}日均换手: {avg_turn:.2f}%, 累计换手: {total_turn:.2f}%")

        # 成交量统计
        if volume is not None:
            lines.append("\n## 成交量统计(万手)")
            lines.append(f"- 当日成交: {volume.iloc[-1] / 10000:.2f}")
            for p in available_periods:
                avg_vol = volume.iloc[-p:].mean() / 10000
                lines.append(f"- {p}日均量: {avg_vol:.2f}")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取多周期统计失败: {e}")
        return f"获取 {symbol} 多周期统计失败: {e}"


@mcp.tool(
    title="获取个股资金流向",
    description="获取A股个股的资金流向数据，包括主力、超大单、大单、中单、小单的流入流出情况。支持多数据源自动故障转移。",
)
def stock_fund_flow(
    symbol: str = field_symbol,
):
    try:
        manager = get_data_manager()
        dfs = manager.get_fund_flow(symbol)

        if dfs is None or dfs.empty:
            return f"Not Found for {symbol}"

        source = format_source_name(dfs.attrs.get('source', ''))
        # 获取最近几天的数据
        dfs = dfs.tail(10)

        lines = [f"# {symbol} 资金流向\n"]
        lines.append(f"数据来源: {source}\n")
        lines.append("## 近期资金流向")
        lines.append("")

        # 转换为CSV格式输出
        cols_to_show = [c for c in dfs.columns if c not in ["序号"]]
        csv_data = dfs.to_csv(columns=cols_to_show, index=False, float_format="%.2f").strip()
        return "\n".join(lines) + "\n" + csv_data
    except Exception as e:
        _LOGGER.warning(f"获取资金流向失败: {e}")
        return f"获取 {symbol} 资金流向失败: {e}"


@mcp.tool(
    title="获取个股所属板块",
    description="获取A股个股所属的行业和概念板块信息",
)
def stock_sector_spot(
    symbol: str = field_symbol,
):
    try:
        manager = get_data_manager()
        boards = manager.get_belong_board(symbol)

        lines = [f"# {symbol} 所属板块\n"]

        if boards is not None and not boards.empty:
            source = format_source_name(boards.attrs.get('source', ''))
            lines.append(f"数据来源: {source}\n")
            lines.append("## 所属板块")
            lines.append(boards.to_csv(index=False, float_format="%.2f").strip())
        else:
            lines.append("未获取到板块数据")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取板块信息失败: {e}")
        return f"获取 {symbol} 板块信息失败: {e}"


@mcp.tool(
    title="获取板块成分股",
    description="获取行业或概念板块的成分股列表。支持多数据源自动故障转移。",
)
def stock_board_cons(
    board_name: str = Field(description="板块名称，如: 酿酒行业、新能源、人工智能"),
    board_type: str = Field("industry", description="板块类型: industry(行业), concept(概念)"),
    limit: int = Field(30, description="返回数量(int)", strict=False),
):
    try:
        manager = get_data_manager()
        dfs = manager.get_board_cons(board_name, board_type)

        if dfs is None or dfs.empty:
            return f"Not Found for {board_name}"

        source = format_source_name(dfs.attrs.get('source', ''))
        dfs = dfs.head(int(limit))
        dfs = dfs.drop(columns=["序号"], errors='ignore')

        lines = [f"# {board_name} 成分股\n", f"数据来源: {source}\n"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取板块成分股失败: {e}")
        return f"获取 {board_name} 成分股失败: {e}"


def _search_us_stock_fast(symbol: str) -> pd.Series | None:
    """使用 yfinance 快速验证美股代码，避免遍历全部数据"""
    import yfinance as yf
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # 验证是否为有效股票（检查关键字段）
        if info and info.get("symbol") and info.get("shortName"):
            return pd.Series({
                "symbol": info.get("symbol", symbol),
                "name": info.get("shortName", ""),
                "cname": info.get("longName", info.get("shortName", "")),
                "market": "us",
            })
    except Exception as e:
        _LOGGER.debug(f"yfinance 快速搜索失败 [{symbol}]: {e}")
    return None


def ak_search(symbol=None, keyword=None, market=None):
    # 美股快速路径：使用 yfinance 验证，避免遍历 843 页数据
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


def ak_cache(fun, *args, **kwargs) -> pd.DataFrame | None:
    key = kwargs.pop("key", None)
    if not key:
        key = f"{fun.__name__}-{args}-{kwargs}"
    ttl1 = kwargs.pop("ttl", 86400)
    ttl2 = kwargs.pop("ttl2", None)
    cache = CacheKey.init(key, ttl1, ttl2)
    all = cache.get()
    if all is None:
        try:
            _LOGGER.debug("Request akshare: %s", [key, args, kwargs])
            all = fun(*args, **kwargs)
            cache.set(all)
        except Exception as exc:
            _LOGGER.exception(str(exc))
    return all


def fetch_with_retry(func, max_retries: int = 3, delay: float = 1.0, initial_delay: float = 0.5, **kwargs):
    """
    带重试的数据获取（含反爬虫延迟）

    Args:
        func: 获取函数
        max_retries: 最大重试次数
        delay: 重试间隔基数（秒）
        initial_delay: 首次请求前延迟（秒）
        **kwargs: 传递给函数的参数

    Returns:
        函数返回值或 None
    """
    last_error = None
    for i in range(max_retries):
        try:
            # 每次尝试前增加随机延迟（避免反爬虫）
            sleep_time = initial_delay + random.uniform(0.5, 1.5) * (i + 1)
            time.sleep(sleep_time)

            result = func(**kwargs)
            if result is not None:
                return result
        except (ConnectionError, TimeoutError, OSError) as e:
            last_error = e
            _LOGGER.warning(f"[{func.__name__}] 第{i+1}次尝试失败(网络): {type(e).__name__}")
            if i < max_retries - 1:
                time.sleep(delay * (i + 1))  # 递增延迟
        except Exception as e:
            last_error = e
            _LOGGER.warning(f"[{func.__name__}] 第{i+1}次尝试失败: {e}")
            if i < max_retries - 1:
                time.sleep(delay * (i + 1))
    return None

def add_technical_indicators(df, clos, lows, high, volume=None):
    # 计算多周期均线
    for period in [5, 10, 20, 30, 60]:
        df[f"MA{period}"] = clos.rolling(window=period, min_periods=1).mean()

    # 计算MACD指标
    ema12 = clos.ewm(span=12, adjust=False).mean()
    ema26 = clos.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2

    # 计算KDJ指标
    low_min  = lows.rolling(window=9, min_periods=1).min()
    high_max = high.rolling(window=9, min_periods=1).max()
    rsv = (clos - low_min) / (high_max - low_min) * 100
    df["KDJ.K"] = rsv.ewm(com=2, adjust=False).mean()
    df["KDJ.D"] = df["KDJ.K"].ewm(com=2, adjust=False).mean()
    df["KDJ.J"] = 3 * df["KDJ.K"] - 2 * df["KDJ.D"]

    # 计算多周期RSI指标
    delta = clos.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    for period in [6, 12, 14, 24]:
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        col_name = "RSI" if period == 14 else f"RSI{period}"
        df[col_name] = 100 - (100 / (1 + rs))

    # 计算布林带指标
    df["BOLL.M"] = clos.rolling(window=20).mean()
    std = clos.rolling(window=20).std()
    df["BOLL.U"] = df["BOLL.M"] + 2 * std
    df["BOLL.L"] = df["BOLL.M"] - 2 * std
    # 布林带宽度 - 波动率收缩/扩张指标
    df["BOLL.W"] = (df["BOLL.U"] - df["BOLL.L"]) / df["BOLL.M"] * 100

    # 计算OBV（能量潮指标）- 向量化实现
    if volume is not None:
        price_diff = clos.diff()
        direction = np.sign(price_diff).fillna(0)
        df["OBV"] = (direction * volume).fillna(0).cumsum()
        # 成交量均线 - 量能趋势判断
        df["VMA5"] = volume.rolling(window=5, min_periods=1).mean()
        df["VMA10"] = volume.rolling(window=10, min_periods=1).mean()

    # 计算ATR（真实波幅）
    tr1 = high - lows
    tr2 = abs(high - clos.shift(1))
    tr3 = abs(lows - clos.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    # 计算ADX（平均趋向指标）- 趋势强度
    # +DM 和 -DM
    high_diff = high.diff()
    low_diff = lows.diff()
    plus_dm = high_diff.where((high_diff > low_diff.abs()) & (high_diff > 0), 0)
    minus_dm = low_diff.abs().where((low_diff.abs() > high_diff) & (low_diff < 0), 0)
    # 平滑的 TR, +DM, -DM
    atr14 = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
    # DX 和 ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.rolling(window=14).mean()
    # 添加 +DI 和 -DI（DMI 方向指标）- 配合 ADX 判断多空方向
    df["+DI"] = plus_di
    df["-DI"] = minus_di

    # 计算CCI（商品通道指标）
    tp = (high + lows + clos) / 3  # 典型价格
    tp_sma = tp.rolling(window=20).mean()
    tp_mad = tp.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - tp_sma) / (0.015 * tp_mad)

    # 计算Williams %R（威廉指标）
    highest_high = high.rolling(window=14).max()
    lowest_low = lows.rolling(window=14).min()
    df["WR"] = -100 * (highest_high - clos) / (highest_high - lowest_low)

    # 计算VWAP（成交量加权平均价）
    if volume is not None:
        tp_vol = tp * volume
        df["VWAP"] = tp_vol.cumsum() / volume.cumsum()


# ==================== Alpha Vantage 美股工具 ====================

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


@mcp.tool(
    title="美股公司概览",
    description="获取美股公司基本面概览，包括市值、PE、EPS、股息率、52周高低点、分析师评级等。支持多数据源: Alpha Vantage (需API key) -> yfinance (免费)。",
)
def stock_overview_us(
    symbol: str = Field(description="美股代码，如: AAPL, MSFT, GOOGL, TSLA"),
):
    try:
        manager = get_data_manager()
        overview = manager.get_us_company_overview(symbol)
        if overview is None:
            return f"未获取到 {symbol} 的公司概览数据"
        return manager.format_us_overview_report(overview)
    except Exception as e:
        _LOGGER.warning(f"获取美股公司概览失败: {e}")
        return f"获取 {symbol} 公司概览失败: {e}"


@mcp.tool(
    title="美股财务报表",
    description="获取美股财务报表数据，包括资产负债表、利润表、现金流量表。支持多数据源: Alpha Vantage (需API key) -> yfinance (免费)。",
)
def stock_financials_us(
    symbol: str = Field(description="美股代码，如: AAPL, MSFT, GOOGL"),
    report_type: str = Field("balance_sheet", description="报表类型: balance_sheet(资产负债表), income_statement(利润表), cash_flow(现金流量表)"),
    quarterly: bool = Field(True, description="是否获取季度数据，False则获取年度数据"),
):
    try:
        manager = get_data_manager()

        if report_type == "balance_sheet":
            data = manager.get_us_balance_sheet(symbol, quarterly)
            title = "资产负债表"
        elif report_type == "income_statement":
            data = manager.get_us_income_statement(symbol, quarterly)
            title = "利润表"
        elif report_type == "cash_flow":
            data = manager.get_us_cash_flow(symbol, quarterly)
            title = "现金流量表"
        else:
            return f"不支持的报表类型: {report_type}"

        if data is None or not data.get("reports"):
            return f"未获取到 {symbol} 的{title}数据"

        # 格式化输出
        period_type = "季度" if quarterly else "年度"
        lines = [f"# {symbol} {title} ({period_type})\n"]

        for i, report in enumerate(data["reports"][:4]):
            fiscal_date = report.get("fiscalDateEnding", "-")
            lines.append(f"## {fiscal_date}")

            # 根据报表类型选择关键字段
            if report_type == "balance_sheet":
                key_fields = [
                    ("totalAssets", "总资产"),
                    ("totalLiabilities", "总负债"),
                    ("totalShareholderEquity", "股东权益"),
                    ("cashAndCashEquivalentsAtCarryingValue", "现金及等价物"),
                    ("currentDebt", "短期债务"),
                    ("longTermDebt", "长期债务"),
                ]
            elif report_type == "income_statement":
                key_fields = [
                    ("totalRevenue", "总收入"),
                    ("grossProfit", "毛利润"),
                    ("operatingIncome", "营业利润"),
                    ("netIncome", "净利润"),
                    ("ebitda", "EBITDA"),
                ]
            else:  # cash_flow
                key_fields = [
                    ("operatingCashflow", "经营现金流"),
                    ("capitalExpenditures", "资本支出"),
                    ("dividendPayout", "股息支出"),
                    ("netIncome", "净利润"),
                ]

            for field, label in key_fields:
                value = report.get(field, "-")
                if value and value != "None":
                    try:
                        num = float(value)
                        if abs(num) >= 1e9:
                            value = f"${num/1e9:.2f}B"
                        elif abs(num) >= 1e6:
                            value = f"${num/1e6:.2f}M"
                        else:
                            value = f"${num:,.0f}"
                    except (ValueError, TypeError):
                        pass
                lines.append(f"- {label}: {value}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股财务报表失败: {e}")
        return f"获取 {symbol} 财务报表失败: {e}"


@mcp.tool(
    title="美股新闻情绪",
    description="获取美股相关新闻及情绪分析数据。需要配置 ALPHA_VANTAGE_API_KEY 环境变量。",
)
def stock_news_us(
    symbol: str = Field("", description="美股代码（可选），如: AAPL, MSFT。留空则获取市场整体新闻"),
    topics: str = Field("", description="主题过滤（可选），如: technology, earnings, ipo, mergers_and_acquisitions"),
    limit: int = Field(20, description="返回数量限制，最大50"),
):
    if not ALPHA_VANTAGE_API_KEY:
        return "错误: 未配置 ALPHA_VANTAGE_API_KEY 环境变量，无法使用此功能"

    try:
        manager = get_data_manager()
        news_data = manager.get_us_news_sentiment(
            symbol=symbol if symbol else None,
            topics=topics if topics else None,
            limit=min(limit, 50)
        )
        if news_data is None:
            return "未获取到新闻数据"
        return manager.format_us_news_report(news_data, limit)
    except Exception as e:
        _LOGGER.warning(f"获取美股新闻情绪失败: {e}")
        return f"获取新闻情绪失败: {e}"


@mcp.tool(
    title="美股盈利数据",
    description="获取美股历史盈利数据和分析师预期。支持多数据源: Alpha Vantage (需API key) -> yfinance (免费)。",
)
def stock_earnings_us(
    symbol: str = Field(description="美股代码，如: AAPL, MSFT, GOOGL"),
):
    try:
        manager = get_data_manager()
        data = manager.get_us_earnings(symbol)
        if data is None:
            return f"未获取到 {symbol} 的盈利数据"

        lines = [f"# {symbol} 盈利数据\n"]

        # 年度盈利
        annual = data.get("annualEarnings", [])
        if annual:
            lines.append("## 年度盈利")
            for item in annual[:5]:
                year = item.get("fiscalDateEnding", "-")
                eps = item.get("reportedEPS", "-")
                lines.append(f"- {year}: EPS ${eps}")
            lines.append("")

        # 季度盈利
        quarterly = data.get("quarterlyEarnings", [])
        if quarterly:
            lines.append("## 季度盈利")
            for item in quarterly[:8]:
                date = item.get("fiscalDateEnding", "-")
                reported = item.get("reportedEPS", "-")
                estimated = item.get("estimatedEPS", "-")
                surprise = item.get("surprisePercentage", "-")
                lines.append(f"- {date}: 实际 ${reported}, 预期 ${estimated}, 惊喜 {surprise}%")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股盈利数据失败: {e}")
        return f"获取 {symbol} 盈利数据失败: {e}"


@mcp.tool(
    title="美股内部交易",
    description="获取美股公司内部人交易记录。支持多数据源: Alpha Vantage (需API key) -> yfinance (免费)。",
)
def stock_insider_us(
    symbol: str = Field(description="美股代码，如: AAPL, MSFT, GOOGL"),
    limit: int = Field(20, description="返回数量限制"),
):
    try:
        manager = get_data_manager()
        data = manager.get_us_insider_transactions(symbol)
        if data is None:
            return f"未获取到 {symbol} 的内部交易数据"

        transactions = data.get("data", [])
        if not transactions:
            return f"{symbol} 暂无内部交易记录"

        lines = [f"# {symbol} 内部交易记录\n"]

        for item in transactions[:limit]:
            date = item.get("transaction_date", "-")
            owner = item.get("owner_name", "-")
            position = item.get("owner_title", "-")
            trans_type = item.get("acquisition_or_disposition", "-")
            shares = item.get("shares", "-")
            value = item.get("transaction_value", "-")

            type_label = "买入" if trans_type == "A" else "卖出" if trans_type == "D" else trans_type

            lines.append(f"## {date}")
            lines.append(f"- 内部人: {owner} ({position})")
            lines.append(f"- 类型: {type_label}")
            lines.append(f"- 股数: {shares}")
            if value and value != "-":
                try:
                    value_num = float(value)
                    if value_num >= 1e6:
                        value = f"${value_num/1e6:.2f}M"
                    else:
                        value = f"${value_num:,.0f}"
                except (ValueError, TypeError):
                    pass
            lines.append(f"- 金额: {value}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股内部交易失败: {e}")
        return f"获取 {symbol} 内部交易失败: {e}"


@mcp.tool(
    title="美股技术指标",
    description="获取美股技术分析指标数据，如SMA、EMA、RSI、MACD、布林带等。需要配置 ALPHA_VANTAGE_API_KEY 环境变量。",
)
def stock_tech_indicators_us(
    symbol: str = Field(description="美股代码，如: AAPL, MSFT, GOOGL"),
    indicator: str = Field("RSI", description="指标类型: SMA(简单移动平均), EMA(指数移动平均), RSI(相对强弱), MACD(指数平滑移动平均), BBANDS(布林带), STOCH(随机指标), ADX(趋向指标), ATR(真实波幅)"),
    interval: str = Field("daily", description="时间间隔: daily(日), weekly(周), monthly(月)"),
    time_period: int = Field(14, description="计算周期，如RSI常用14，SMA常用20"),
    limit: int = Field(30, description="返回数量限制"),
):
    if not ALPHA_VANTAGE_API_KEY:
        return "错误: 未配置 ALPHA_VANTAGE_API_KEY 环境变量，无法使用此功能"

    try:
        manager = get_data_manager()
        data = manager.get_us_technical_indicator(symbol, indicator, interval, time_period)

        if data is None or not data.get("data"):
            return f"未获取到 {symbol} 的 {indicator} 指标数据"

        # 格式化输出
        lines = [
            f"# {symbol} {indicator.upper()} 技术指标",
            f"",
            f"- 时间间隔: {interval}",
            f"- 计算周期: {time_period}",
            f"",
            "## 数据",
        ]

        # 根据指标类型构建表头
        indicator_upper = indicator.upper()
        if indicator_upper == "MACD":
            lines.append("| 日期 | MACD | Signal | Histogram |")
            lines.append("|------|------|--------|-----------|")
        elif indicator_upper == "BBANDS":
            lines.append("| 日期 | Upper | Middle | Lower |")
            lines.append("|------|-------|--------|-------|")
        elif indicator_upper == "STOCH":
            lines.append("| 日期 | SlowK | SlowD |")
            lines.append("|------|-------|-------|")
        else:
            lines.append(f"| 日期 | {indicator_upper} |")
            lines.append("|------|--------|")

        for entry in data["data"][:limit]:
            date = entry.get("date", "-")

            if indicator_upper == "MACD":
                macd = entry.get("MACD", "-")
                signal = entry.get("MACD_Signal", "-")
                hist = entry.get("MACD_Hist", "-")
                lines.append(f"| {date} | {macd} | {signal} | {hist} |")
            elif indicator_upper == "BBANDS":
                upper = entry.get("Real Upper Band", "-")
                middle = entry.get("Real Middle Band", "-")
                lower = entry.get("Real Lower Band", "-")
                lines.append(f"| {date} | {upper} | {middle} | {lower} |")
            elif indicator_upper == "STOCH":
                slowk = entry.get("SlowK", "-")
                slowd = entry.get("SlowD", "-")
                lines.append(f"| {date} | {slowk} | {slowd} |")
            else:
                # 单值指标 (SMA, EMA, RSI, ADX 等)
                value = entry.get(indicator_upper, "-")
                lines.append(f"| {date} | {value} |")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股技术指标失败: {e}")
        return f"获取 {symbol} {indicator} 指标失败: {e}"


@mcp.tool(
    title="A股估值对比",
    description="获取A股个股估值与行业对比数据，包括PE/PB在行业中的分位数。",
)
def stock_valuation_compare(
    symbol: str = field_symbol,
):
    """获取个股估值与同行业对比"""
    try:
        # 1. 获取个股基本信息（含估值）
        base_info = ef.stock.get_base_info(symbol)
        if base_info is None or base_info.empty:
            return f"未获取到 {symbol} 基本信息"

        stock_name = base_info.get("股票名称", "-")
        stock_pe = base_info.get("市盈率(动)", None)
        stock_pb = base_info.get("市净率", None)
        stock_roe = base_info.get("ROE", None)
        industry = base_info.get("所处行业", "-")

        # 2. 获取所属行业板块
        boards = ef.stock.get_belong_board(symbol)
        if boards is None or boards.empty:
            return f"未获取到 {symbol} 板块信息"

        # 找到行业板块（板块代码以 BK04 开头的通常是行业分类）
        industry_board = boards[boards["板块代码"].str.startswith("BK04")]
        if industry_board.empty:
            # 回退到第一个板块
            industry_board = boards.head(1)

        board_code = industry_board.iloc[0]["板块代码"]
        board_name = industry_board.iloc[0]["板块名称"]

        # 3. 获取板块成分股（使用 manager）
        manager = get_data_manager()
        peers_df = manager.get_board_cons(board_name, "industry")

        if peers_df is None or peers_df.empty:
            # 尝试概念板块
            peers_df = manager.get_board_cons(board_name, "concept")

        if peers_df is None or peers_df.empty:
            # 无法获取同行业数据，只返回个股信息
            lines = [
                f"# {stock_name} ({symbol}) 估值信息\n",
                f"数据来源: efinance\n",
                f"所属行业: {industry}\n",
                "",
                f"## 估值指标",
                f"- 市盈率(动态): {stock_pe or '-'}",
                f"- 市净率: {stock_pb or '-'}",
                f"- ROE: {stock_roe or '-'}%",
                "",
                "注：未能获取同行业数据进行对比",
            ]
            return "\n".join(lines)

        # 4. 获取同行业股票估值
        peer_codes = []
        code_col = None
        for col in ["代码", "股票代码", "证券代码"]:
            if col in peers_df.columns:
                code_col = col
                break
        if code_col:
            peer_codes = peers_df[code_col].astype(str).tolist()[:20]  # 限制20只，加快速度

        # 5. 批量获取实时行情（包含PE/PB），带容错
        peer_quotes = {}
        if peer_codes:
            try:
                peer_quotes = manager.prefetch_realtime_quotes(peer_codes)
            except Exception as e:
                _LOGGER.warning(f"批量获取同行业行情失败: {e}")

        # 6. 计算分位数
        pe_values = []
        pb_values = []
        for code, quote in peer_quotes.items():
            if quote.pe_ratio is not None and quote.pe_ratio > 0:
                pe_values.append(quote.pe_ratio)
            if quote.pb_ratio is not None and quote.pb_ratio > 0:
                pb_values.append(quote.pb_ratio)

        pe_percentile = None
        pb_percentile = None
        pe_median = None
        pb_median = None

        if stock_pe and pe_values:
            pe_values_sorted = sorted(pe_values)
            pe_median = pe_values_sorted[len(pe_values_sorted) // 2]
            count_below = sum(1 for v in pe_values if v < stock_pe)
            pe_percentile = count_below / len(pe_values) * 100

        if stock_pb and pb_values:
            pb_values_sorted = sorted(pb_values)
            pb_median = pb_values_sorted[len(pb_values_sorted) // 2]
            count_below = sum(1 for v in pb_values if v < stock_pb)
            pb_percentile = count_below / len(pb_values) * 100

        # 7. 格式化输出
        lines = [
            f"# {stock_name} ({symbol}) 估值对比分析\n",
            f"数据来源: efinance\\n",
            f"所属行业: {board_name} (共{len(peer_codes)}只股票)\n",
            "",
            "## 个股估值",
            f"- 市盈率(动态): {stock_pe or '-'}",
            f"- 市净率: {stock_pb or '-'}",
            f"- ROE: {stock_roe or '-'}%",
            "",
        ]

        if not peer_quotes:
            lines.append("## 行业对比")
            lines.append("暂无同行业数据（网络问题），请稍后重试")
            return "\n".join(lines)

        lines.append("## 行业对比")

        if pe_percentile is not None:
            pe_level = "高估" if pe_percentile > 70 else "低估" if pe_percentile < 30 else "中性"
            lines.append(f"- PE分位: {pe_percentile:.1f}% ({pe_level})")
            lines.append(f"- 行业PE中位数: {pe_median:.2f}")

        if pb_percentile is not None:
            pb_level = "高估" if pb_percentile > 70 else "低估" if pb_percentile < 30 else "中性"
            lines.append(f"- PB分位: {pb_percentile:.1f}% ({pb_level})")
            lines.append(f"- 行业PB中位数: {pb_median:.2f}")

        # 8. 估值建议
        lines.append("")
        lines.append("## 估值评估")
        if pe_percentile is not None and pb_percentile is not None:
            avg_percentile = (pe_percentile + pb_percentile) / 2
            if avg_percentile < 30:
                lines.append("综合评估: **低估** - PE/PB均低于行业70%的股票")
            elif avg_percentile > 70:
                lines.append("综合评估: **高估** - PE/PB均高于行业70%的股票")
            else:
                lines.append("综合评估: **合理** - PE/PB处于行业中等水平")
        else:
            lines.append("综合评估: 数据不足，无法评估")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取估值对比失败: {e}")
        return f"获取 {symbol} 估值对比失败: {e}"


@mcp.tool(
    title="A股市场PE分位",
    description="获取A股市场整体PE/PB的历史分位数，用于判断市场整体估值水平。",
)
def stock_market_pe_percentile():
    """获取A股市场PE历史分位"""
    try:
        # 获取市场PE历史数据
        pe_df = ak_cache(ak.stock_a_ttm_lyr, ttl=3600)
        pb_df = ak_cache(ak.stock_a_all_pb, ttl=3600)

        if pe_df is None or pe_df.empty:
            return "获取市场PE数据失败"

        # 获取最新数据
        latest_pe = pe_df.iloc[-1]
        pe_ttm_median = latest_pe.get("middlePETTM", None)
        pe_ttm_avg = latest_pe.get("averagePETTM", None)
        pe_percentile_all = latest_pe.get("quantileInAllHistoryMiddlePeTtm", None)
        pe_percentile_10y = latest_pe.get("quantileInRecent10YearsMiddlePeTtm", None)

        lines = [
            "# A股市场估值分位\n",
            "数据来源: akshare (乐咕乐股)\n",
            "",
            "## 市盈率(PE-TTM)",
            f"- 中位数PE: {pe_ttm_median:.2f}" if pe_ttm_median else "- 中位数PE: -",
            f"- 平均PE: {pe_ttm_avg:.2f}" if pe_ttm_avg else "- 平均PE: -",
        ]

        if pe_percentile_all is not None:
            pct = pe_percentile_all * 100
            level = "极度高估" if pct > 80 else "高估" if pct > 60 else "合理" if pct > 40 else "低估" if pct > 20 else "极度低估"
            lines.append(f"- 历史分位(全部): {pct:.1f}% ({level})")

        if pe_percentile_10y is not None:
            pct = pe_percentile_10y * 100
            level = "极度高估" if pct > 80 else "高估" if pct > 60 else "合理" if pct > 40 else "低估" if pct > 20 else "极度低估"
            lines.append(f"- 历史分位(近10年): {pct:.1f}% ({level})")

        if pb_df is not None and not pb_df.empty:
            latest_pb = pb_df.iloc[-1]
            pb_median = latest_pb.get("middlePB", None)
            pb_percentile_all = latest_pb.get("quantileInAllHistoryMiddlePB", None)
            pb_percentile_10y = latest_pb.get("quantileInRecent10YearsMiddlePB", None)

            lines.append("")
            lines.append("## 市净率(PB)")
            if pb_median:
                lines.append(f"- 中位数PB: {pb_median:.2f}")
            if pb_percentile_all is not None:
                pct = pb_percentile_all * 100
                level = "极度高估" if pct > 80 else "高估" if pct > 60 else "合理" if pct > 40 else "低估" if pct > 20 else "极度低估"
                lines.append(f"- 历史分位(全部): {pct:.1f}% ({level})")
            if pb_percentile_10y is not None:
                pct = pb_percentile_10y * 100
                lines.append(f"- 历史分位(近10年): {pct:.1f}%")

        # 综合评估
        lines.append("")
        lines.append("## 市场估值建议")
        if pe_percentile_10y is not None:
            pct = pe_percentile_10y * 100
            if pct < 30:
                lines.append("当前市场估值处于**历史低位**，长期投资价值凸显")
            elif pct > 70:
                lines.append("当前市场估值处于**历史高位**，需注意回调风险")
            else:
                lines.append("当前市场估值处于**历史中位**，选股重于择时")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取市场PE分位失败: {e}")
        return f"获取市场PE分位失败: {e}"


@mcp.tool(
    title="A股行业PE对比",
    description="获取A股各行业PE对比数据，用于行业估值比较和行业轮动分析。",
)
def stock_industry_pe(
    date: str = Field("", description="日期(可选)，格式: 20250210，默认最新"),
):
    """获取行业PE对比"""
    try:
        if not date:
            date = recent_trade_date().strftime("%Y%m%d")

        # 获取行业PE数据
        df = ak_cache(
            ak.stock_industry_pe_ratio_cninfo,
            symbol="证监会行业分类",
            date=date,
            ttl=3600
        )

        if df is None or df.empty:
            return f"获取行业PE数据失败，日期: {date}"

        # 筛选一级行业
        df_l1 = df[df["行业层级"] == 1.0].copy()
        if df_l1.empty:
            df_l1 = df.head(20)

        # 按PE排序
        df_l1 = df_l1.sort_values("静态市盈率-加权平均", ascending=True)

        lines = [
            "# A股行业PE对比\n",
            f"数据来源: akshare (巨潮资讯)\n",
            f"数据日期: {date}\n",
            "",
            "## 行业PE排名 (按加权PE升序)",
        ]

        # 输出CSV格式
        cols = ["行业名称", "公司数量", "静态市盈率-加权平均", "静态市盈率-中位数"]
        df_out = df_l1[cols].copy()
        df_out.columns = ["行业", "公司数", "加权PE", "中位PE"]
        lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())

        # 添加分析
        lines.append("")
        lines.append("## 估值提示")
        low_pe = df_l1.head(3)["行业名称"].tolist()
        high_pe = df_l1.tail(3)["行业名称"].tolist()
        lines.append(f"- 低估值行业: {', '.join(low_pe)}")
        lines.append(f"- 高估值行业: {', '.join(high_pe)}")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取行业PE对比失败: {e}")
        return f"获取行业PE对比失败: {e}"


@mcp.tool(
    title="A股分红历史",
    description="获取A股个股历史分红送转数据，包括派息、送股、转增等。用于分析股息率和分红政策。",
)
def stock_dividend_history(
    symbol: str = field_symbol,
    limit: int = Field(10, description="返回数量限制"),
):
    """获取个股分红历史"""
    try:
        df = ak_cache(ak.stock_history_dividend_detail, symbol=symbol, indicator="分红", ttl=86400)
        if df is None or df.empty:
            return f"未获取到 {symbol} 的分红历史数据"

        # 计算股息率（需要当前价格）
        manager = get_data_manager()
        quote = manager.get_realtime_quote(symbol)
        current_price = quote.price if quote else None

        lines = [f"# {symbol} 分红历史\n", "数据来源: akshare\n"]

        # 格式化分红数据
        df = df.head(limit)
        lines.append("## 历史分红记录")
        lines.append("| 公告日期 | 送股 | 转增 | 派息(元/10股) | 进度 | 除权除息日 |")
        lines.append("|---------|------|------|--------------|------|-----------|")

        total_dividend = 0
        recent_dividend = 0
        for _, row in df.iterrows():
            date = str(row.get("公告日期", "-"))[:10]
            song = row.get("送股", 0) or 0
            zhuan = row.get("转增", 0) or 0
            pai = row.get("派息", 0) or 0
            status = row.get("进度", "-")
            ex_date = str(row.get("除权除息日", "-"))[:10] if pd.notna(row.get("除权除息日")) else "-"

            lines.append(f"| {date} | {song} | {zhuan} | {pai:.2f} | {status} | {ex_date} |")

            if status == "实施" and pai > 0:
                total_dividend += pai
                if recent_dividend == 0:
                    recent_dividend = pai

        lines.append("")

        # 计算股息率
        if current_price and recent_dividend > 0:
            dividend_yield = (recent_dividend / 10) / current_price * 100
            lines.append(f"## 股息率分析")
            lines.append(f"- 当前股价: {current_price:.2f}")
            lines.append(f"- 最近一次派息: {recent_dividend:.2f}元/10股")
            lines.append(f"- 股息率: {dividend_yield:.2f}%")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取分红历史失败: {e}")
        return f"获取 {symbol} 分红历史失败: {e}"


@mcp.tool(
    title="A股基金持仓",
    description="获取A股基金重仓股数据，显示公募基金持仓最多的股票及持仓变化。用于跟踪机构动向。",
)
def stock_institutional_holdings(
    date: str = Field("", description="报告期，格式: 20240930，默认最新季度"),
    limit: int = Field(30, description="返回数量限制"),
):
    """获取基金重仓股"""
    try:
        if not date:
            # 计算最近季度末
            now = datetime.now()
            quarter_ends = ["0331", "0630", "0930", "1231"]
            year = now.year
            month = now.month
            if month <= 4:
                date = f"{year-1}1231"
            elif month <= 7:
                date = f"{year}0331"
            elif month <= 10:
                date = f"{year}0630"
            else:
                date = f"{year}0930"

        df = ak_cache(ak.stock_report_fund_hold, symbol="基金持仓", date=date, ttl=86400)
        if df is None or df.empty:
            return f"未获取到基金持仓数据，报告期: {date}"

        df = df.head(limit)

        lines = [
            f"# 基金重仓股 ({date})\n",
            "数据来源: akshare\n",
            "",
            "## 持仓排名",
        ]

        # 格式化输出
        cols = ["股票代码", "股票简称", "持有基金家数", "持股总数", "持股市值", "持股变化", "持股变动比例"]
        df_out = df[cols].copy()
        df_out["持股市值"] = (df_out["持股市值"] / 1e8).round(2)
        df_out["持股总数"] = (df_out["持股总数"] / 1e4).round(2)
        df_out.columns = ["代码", "名称", "基金数", "持股(万)", "市值(亿)", "变化", "变动%"]
        lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())

        # 统计
        lines.append("")
        lines.append("## 持仓变化统计")
        increase = len(df[df["持股变化"] == "增仓"])
        decrease = len(df[df["持股变化"] == "减仓"])
        new_hold = len(df[df["持股变化"] == "新进"])
        lines.append(f"- 增仓: {increase}只, 减仓: {decrease}只, 新进: {new_hold}只")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取基金持仓失败: {e}")
        return f"获取基金持仓失败: {e}"


@mcp.tool(
    title="A股财报日历",
    description="获取A股财报披露时间表，查看即将披露财报的公司。用于跟踪财报季。",
)
def stock_earnings_calendar(
    period: str = Field("", description="报告期，如: 2024年报、2024三季报，默认最新"),
    limit: int = Field(50, description="返回数量限制"),
):
    """获取财报披露日历"""
    try:
        if not period:
            # 自动确定当前报告期
            now = datetime.now()
            year = now.year
            month = now.month
            if month <= 4:
                period = f"{year-1}年报"
            elif month <= 8:
                period = f"{year}半年报"
            elif month <= 10:
                period = f"{year}三季报"
            else:
                period = f"{year}年报"

        df = ak_cache(ak.stock_report_disclosure, market="沪深京", period=period, ttl=3600)
        if df is None or df.empty:
            return f"未获取到财报披露数据，报告期: {period}"

        lines = [
            f"# 财报披露日历 ({period})\n",
            "数据来源: akshare (巨潮资讯)\n",
        ]

        # 按披露日期排序，筛选未来的
        if "首次预约时间" in df.columns:
            df = df.sort_values("首次预约时间")

        # 统计今日披露
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = len(df[df.get("首次预约时间", "").astype(str).str.startswith(today)]) if "首次预约时间" in df.columns else 0

        lines.append(f"今日披露: {today_count}家\n")
        lines.append("## 即将披露")

        df = df.head(limit)
        cols_available = [c for c in ["股票代码", "股票简称", "首次预约时间", "实际披露时间", "修改次数"] if c in df.columns]
        if cols_available:
            lines.append(df[cols_available].to_csv(index=False).strip())
        else:
            lines.append(df.to_csv(index=False).strip())

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取财报日历失败: {e}")
        return f"获取财报日历失败: {e}"


@mcp.tool(
    title="A股财务指标对比",
    description="获取A股个股详细财务指标，包括盈利能力、偿债能力、运营能力等多维度分析。",
)
def stock_financial_compare(
    symbol: str = field_symbol,
):
    """获取个股财务指标详情"""
    try:
        # 获取财务分析指标
        df = ak_cache(
            ak.stock_financial_analysis_indicator,
            symbol=symbol,
            start_year=str(datetime.now().year - 2),
            ttl=3600
        )
        if df is None or df.empty:
            return f"未获取到 {symbol} 的财务指标数据"

        lines = [f"# {symbol} 财务指标分析\n", "数据来源: akshare\n"]

        # 取最近4个季度
        df = df.head(4)

        # 盈利能力
        lines.append("## 盈利能力")
        profit_cols = ["日期", "净资产收益率(%)", "销售毛利率(%)", "销售净利率(%)", "总资产利润率(%)"]
        profit_cols = [c for c in profit_cols if c in df.columns]
        if profit_cols:
            lines.append(df[profit_cols].to_csv(index=False, float_format="%.2f").strip())

        # 成长能力
        lines.append("\n## 成长能力")
        growth_cols = ["日期", "主营业务收入增长率(%)", "净利润增长率(%)", "净资产增长率(%)", "总资产增长率(%)"]
        growth_cols = [c for c in growth_cols if c in df.columns]
        if growth_cols:
            lines.append(df[growth_cols].to_csv(index=False, float_format="%.2f").strip())

        # 偿债能力
        lines.append("\n## 偿债能力")
        debt_cols = ["日期", "流动比率", "速动比率", "资产负债率(%)", "股东权益比率(%)"]
        debt_cols = [c for c in debt_cols if c in df.columns]
        if debt_cols:
            lines.append(df[debt_cols].to_csv(index=False, float_format="%.2f").strip())

        # 运营能力
        lines.append("\n## 运营能力")
        ops_cols = ["日期", "应收账款周转率(次)", "存货周转率(次)", "总资产周转率(次)"]
        ops_cols = [c for c in ops_cols if c in df.columns]
        if ops_cols:
            lines.append(df[ops_cols].to_csv(index=False, float_format="%.2f").strip())

        # 每股指标
        lines.append("\n## 每股指标")
        share_cols = ["日期", "摊薄每股收益(元)", "每股净资产_调整前(元)", "每股经营性现金流(元)"]
        share_cols = [c for c in share_cols if c in df.columns]
        if share_cols:
            lines.append(df[share_cols].to_csv(index=False, float_format="%.4f").strip())

        # 趋势分析
        if len(df) >= 2:
            lines.append("\n## 趋势分析")
            latest = df.iloc[0]
            prev = df.iloc[1]

            if "净资产收益率(%)" in df.columns:
                roe_change = (latest["净资产收益率(%)"] or 0) - (prev["净资产收益率(%)"] or 0)
                trend = "↑" if roe_change > 0 else "↓" if roe_change < 0 else "→"
                lines.append(f"- ROE变化: {trend} {abs(roe_change):.2f}%")

            if "净利润增长率(%)" in df.columns and pd.notna(latest.get("净利润增长率(%)")):
                growth = latest["净利润增长率(%)"]
                level = "高速增长" if growth > 30 else "稳定增长" if growth > 0 else "下滑"
                lines.append(f"- 净利润增速: {growth:.2f}% ({level})")

            if "资产负债率(%)" in df.columns:
                debt_ratio = latest["资产负债率(%)"]
                risk = "高" if debt_ratio > 70 else "中" if debt_ratio > 50 else "低"
                lines.append(f"- 资产负债率: {debt_ratio:.2f}% (风险{risk})")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取财务指标失败: {e}")
        return f"获取 {symbol} 财务指标失败: {e}"


@mcp.tool(
    title="投资组合风险分析",
    description="分析投资组合的风险指标，包括波动率、最大回撤、相关性矩阵、夏普比率等。",
)
def portfolio_risk_analysis(
    symbols: str = Field(description="股票代码列表，逗号分隔，如: 600519,000858,601318"),
    days: int = Field(60, description="分析周期(天)"),
):
    """分析投资组合风险"""
    try:
        codes = [s.strip() for s in symbols.split(",") if s.strip()]
        if len(codes) < 2:
            return "请提供至少2只股票进行组合分析"

        manager = get_data_manager()
        price_data = {}
        names = {}

        # 获取各股票历史价格
        for code in codes[:10]:  # 限制最多10只
            df = manager.get_daily_data(code, days=days + 10)
            if df is not None and not df.empty:
                df = to_chinese_columns(df)
                price_data[code] = df["收盘"].tail(days)

                # 尝试获取股票名称
                quote = manager.get_realtime_quote(code)
                names[code] = quote.name if quote and quote.name else code

        if len(price_data) < 2:
            return "有效股票数据不足，请检查股票代码"

        # 构建价格DataFrame
        prices_df = pd.DataFrame(price_data)
        returns_df = prices_df.pct_change().dropna()

        lines = [
            f"# 投资组合风险分析\n",
            f"分析周期: {days}天\n",
            f"股票数量: {len(price_data)}只\n",
        ]

        # 1. 个股风险指标
        lines.append("## 个股风险指标")
        lines.append("| 代码 | 名称 | 年化波动率 | 最大回撤 | 夏普比率 |")
        lines.append("|------|------|-----------|---------|---------|")

        risk_free_rate = 0.02  # 无风险利率假设2%
        for code in price_data.keys():
            ret = returns_df[code]
            prices = prices_df[code]

            # 年化波动率
            volatility = ret.std() * np.sqrt(252) * 100

            # 最大回撤
            cummax = prices.cummax()
            drawdown = (prices - cummax) / cummax
            max_drawdown = drawdown.min() * 100

            # 夏普比率
            annual_return = ret.mean() * 252
            sharpe = (annual_return - risk_free_rate) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

            name = names.get(code, code)
            lines.append(f"| {code} | {name} | {volatility:.2f}% | {max_drawdown:.2f}% | {sharpe:.2f} |")

        # 2. 相关性矩阵
        lines.append("\n## 相关性矩阵")
        corr_matrix = returns_df.corr()
        # 替换代码为名称
        corr_display = corr_matrix.copy()
        corr_display.index = [names.get(c, c) for c in corr_display.index]
        corr_display.columns = [names.get(c, c) for c in corr_display.columns]
        lines.append(corr_display.to_csv(float_format="%.2f").strip())

        # 3. 组合风险分析 (等权重)
        lines.append("\n## 等权组合分析")
        n = len(price_data)
        weights = np.array([1/n] * n)
        portfolio_return = returns_df.mean().values @ weights * 252
        portfolio_vol = np.sqrt(weights @ returns_df.cov().values @ weights) * np.sqrt(252)
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        # 组合最大回撤
        portfolio_value = (1 + returns_df @ weights).cumprod()
        portfolio_cummax = portfolio_value.cummax()
        portfolio_drawdown = (portfolio_value - portfolio_cummax) / portfolio_cummax
        portfolio_max_dd = portfolio_drawdown.min() * 100

        lines.append(f"- 预期年化收益: {portfolio_return*100:.2f}%")
        lines.append(f"- 年化波动率: {portfolio_vol*100:.2f}%")
        lines.append(f"- 夏普比率: {portfolio_sharpe:.2f}")
        lines.append(f"- 最大回撤: {portfolio_max_dd:.2f}%")

        # 4. 风险提示
        lines.append("\n## 风险提示")

        # 高相关性警告
        high_corr = []
        for i, c1 in enumerate(corr_matrix.columns):
            for c2 in corr_matrix.columns[i+1:]:
                if corr_matrix.loc[c1, c2] > 0.7:
                    high_corr.append(f"{names.get(c1, c1)}-{names.get(c2, c2)}")
        if high_corr:
            lines.append(f"- 高相关性组合(>0.7): {', '.join(high_corr[:3])}")
            lines.append("  建议: 考虑分散投资以降低集中风险")

        # 高波动股票
        high_vol = [names.get(c, c) for c in returns_df.columns if returns_df[c].std() * np.sqrt(252) > 0.4]
        if high_vol:
            lines.append(f"- 高波动股票(>40%): {', '.join(high_vol[:3])}")

        # 负夏普股票
        for code in price_data.keys():
            ret = returns_df[code]
            annual_return = ret.mean() * 252
            sharpe = (annual_return - risk_free_rate) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
            if sharpe < 0:
                lines.append(f"- {names.get(code, code)} 夏普比率为负，风险收益不匹配")
                break

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"组合风险分析失败: {e}")
        return f"组合风险分析失败: {e}"


@mcp.tool(
    title="A股限售解禁日历",
    description="获取A股限售股解禁日历，查看即将解禁的股票及解禁规模。限售解禁是重要的市场供给压力指标。",
)
def stock_locked_shares(
    start_date: str = Field("", description="开始日期，格式: 20250211，默认今日"),
    end_date: str = Field("", description="结束日期，格式: 20250311，默认未来30天"),
    mode: str = Field("detail", description="模式: 'detail'(个股明细), 'summary'(每日汇总)"),
    limit: int = Field(50, description="返回数量限制"),
):
    """获取限售解禁日历"""
    try:
        # 处理 Field 对象作为默认值的情况（直接调用时）
        if hasattr(start_date, 'default'):
            start_date = start_date.default or ""
        if hasattr(end_date, 'default'):
            end_date = end_date.default or ""
        if hasattr(mode, 'default'):
            mode = mode.default or "detail"

        # 默认日期范围：今日起未来30天
        if not start_date:
            start_date = datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")

        if mode == "summary":
            # 每日解禁汇总
            df = ak_cache(
                ak.stock_restricted_release_summary_em,
                start_date=start_date,
                end_date=end_date,
                ttl=3600
            )
            if df is None or df.empty:
                return f"未获取到限售解禁汇总数据 ({start_date} ~ {end_date})"

            lines = [
                f"# 限售解禁日历 (汇总)\n",
                f"数据来源: akshare (东方财富)\n",
                f"日期范围: {start_date} ~ {end_date}\n",
                "",
                "## 每日解禁汇总",
            ]

            # 选择关键列
            cols = ["解禁时间", "当日解禁股票家数", "解禁数量", "实际解禁数量", "实际解禁市值"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df = df[available_cols].head(limit)
                # 格式化市值
                if "实际解禁市值" in df.columns:
                    df["实际解禁市值(亿)"] = (df["实际解禁市值"] / 1e8).round(2)
                    df = df.drop(columns=["实际解禁市值"])
                lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

            return "\n".join(lines)

        else:
            # 个股解禁明细
            df = ak_cache(
                ak.stock_restricted_release_detail_em,
                start_date=start_date,
                end_date=end_date,
                ttl=3600
            )
            if df is None or df.empty:
                return f"未获取到限售解禁明细数据 ({start_date} ~ {end_date})"

            lines = [
                f"# 限售解禁日历 (明细)\n",
                f"数据来源: akshare (东方财富)\n",
                f"日期范围: {start_date} ~ {end_date}\n",
                f"共 {len(df)} 只股票即将解禁\n",
            ]

            # 按解禁市值排序
            if "实际解禁市值" in df.columns:
                df = df.sort_values("实际解禁市值", ascending=False)

            # 选择关键列
            cols = ["股票代码", "股票简称", "解禁时间", "限售股类型", "实际解禁数量", "实际解禁市值", "占解禁前流通市值比例"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df_out = df[available_cols].head(limit).copy()
                # 格式化市值
                if "实际解禁市值" in df_out.columns:
                    df_out["实际解禁市值(万)"] = (df_out["实际解禁市值"] / 1e4).round(2)
                    df_out = df_out.drop(columns=["实际解禁市值"])
                lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

            # 添加风险提示
            if "占解禁前流通市值比例" in df.columns:
                high_impact = df[df["占解禁前流通市值比例"] > 10].head(5)
                if not high_impact.empty:
                    lines.append("\n## 高冲击风险股票 (解禁占比>10%)")
                    for _, row in high_impact.iterrows():
                        code = row.get("股票代码", "-")
                        name = row.get("股票简称", "-")
                        ratio = row.get("占解禁前流通市值比例", 0)
                        unlock_date = row.get("解禁时间", "-")
                        lines.append(f"- {code} {name}: 解禁占比 {ratio:.1f}%, 解禁日 {unlock_date}")

            return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取限售解禁日历失败: {e}")
        return f"获取限售解禁日历失败: {e}"


@mcp.tool(
    title="A股股权质押",
    description="获取A股股权质押数据，包括行业质押统计和市场整体质押比例。股权质押是衡量大股东杠杆风险的重要指标。",
)
def stock_pledge_ratio(
    mode: str = Field("industry", description="模式: 'industry'(行业统计), 'market'(市场整体趋势)"),
    limit: int = Field(30, description="返回数量限制"),
):
    """获取股权质押数据"""
    try:
        if mode == "industry":
            # 行业质押统计（快速）
            df = ak_cache(ak.stock_gpzy_industry_data_em, ttl=3600)
            if df is None or df.empty:
                return "获取行业质押数据失败"

            # 按平均质押比例排序
            if "平均质押比例" in df.columns:
                df = df.sort_values("平均质押比例", ascending=False)

            lines = [
                "# 行业股权质押统计\n",
                "数据来源: akshare (东方财富)\n",
                "",
                "## 各行业质押情况 (按质押比例降序)",
            ]

            # 选择关键列
            cols = ["行业", "公司家数", "质押总笔数", "平均质押比例", "质押总股本", "最新质押市值"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df_out = df[available_cols].head(limit).copy()
                # 格式化市值
                if "最新质押市值" in df_out.columns:
                    df_out["质押市值(亿)"] = (df_out["最新质押市值"] / 1e8).round(2)
                    df_out = df_out.drop(columns=["最新质押市值"])
                if "质押总股本" in df_out.columns:
                    df_out["质押股本(亿股)"] = (df_out["质押总股本"] / 1e8).round(2)
                    df_out = df_out.drop(columns=["质押总股本"])
                lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

            # 风险分析
            if "平均质押比例" in df.columns:
                high_pledge = df[df["平均质押比例"] > 20].head(5)
                if not high_pledge.empty:
                    lines.append("\n## 高质押风险行业 (平均质押比例>20%)")
                    for _, row in high_pledge.iterrows():
                        industry = row.get("行业", "-")
                        ratio = row.get("平均质押比例", 0)
                        count = row.get("公司家数", 0)
                        lines.append(f"- {industry}: 平均质押 {ratio:.1f}%, 涉及 {count} 家公司")

            return "\n".join(lines)

        else:
            # 市场整体质押趋势
            df = ak_cache(ak.stock_gpzy_profile_em, ttl=3600)
            if df is None or df.empty:
                return "获取市场质押趋势数据失败"

            lines = [
                "# A股市场股权质押趋势\n",
                "数据来源: akshare (东方财富)\n",
            ]

            # 取最近数据
            df = df.tail(limit)

            # 选择关键列
            cols = ["统计时间", "A股质押总比例", "A股质押总股数", "A股质押总市值", "A股质押公司数量"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df_out = df[available_cols].copy()
                # 格式化
                if "A股质押总市值" in df_out.columns:
                    df_out["质押市值(万亿)"] = (df_out["A股质押总市值"] / 1e12).round(2)
                    df_out = df_out.drop(columns=["A股质押总市值"])
                if "A股质押总股数" in df_out.columns:
                    df_out["质押股数(亿股)"] = (df_out["A股质押总股数"] / 1e8).round(2)
                    df_out = df_out.drop(columns=["A股质押总股数"])
                lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.to_csv(index=False, float_format="%.2f").strip())

            # 趋势分析
            if "A股质押总比例" in df.columns and len(df) >= 2:
                latest = df.iloc[-1]["A股质押总比例"]
                prev = df.iloc[-2]["A股质押总比例"]
                change = latest - prev
                trend = "上升" if change > 0 else "下降" if change < 0 else "持平"
                lines.append(f"\n## 趋势分析")
                lines.append(f"- 最新质押比例: {latest:.2f}%")
                lines.append(f"- 变化趋势: {trend} ({change:+.2f}%)")

            return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取股权质押数据失败: {e}")
        return f"获取股权质押数据失败: {e}"


@mcp.tool(
    title="A股十大股东",
    description="获取A股个股十大股东或十大流通股东信息，用于分析股权结构和机构持仓变化。",
)
def stock_top10_holders(
    symbol: str = field_symbol,
    holder_type: str = Field("main", description="股东类型: 'main'(十大股东), 'circulate'(十大流通股东)"),
    limit: int = Field(30, description="返回数量限制（多期数据）"),
):
    """获取十大股东信息"""
    try:
        if holder_type == "circulate":
            # 十大流通股东
            df = ak_cache(ak.stock_circulate_stock_holder, symbol=symbol, ttl=3600)
            title = "十大流通股东"
            date_col = "截止日期"
        else:
            # 十大股东
            df = ak_cache(ak.stock_main_stock_holder, stock=symbol, ttl=3600)
            title = "十大股东"
            date_col = "截至日期"

        if df is None or df.empty:
            return f"未获取到 {symbol} 的{title}数据"

        lines = [
            f"# {symbol} {title}\n",
            "数据来源: akshare (东方财富)\n",
        ]

        # 获取最近几期数据
        if date_col in df.columns:
            dates = df[date_col].unique()[:3]  # 最近3期
            for date in dates:
                period_df = df[df[date_col] == date].head(10)  # 每期10个股东

                lines.append(f"\n## {date}")

                if holder_type == "circulate":
                    # 流通股东列
                    cols = ["编号", "股东名称", "持股数量", "占流通股比例", "股本性质"]
                else:
                    # 大股东列
                    cols = ["编号", "股东名称", "持股数量", "持股比例", "股本性质"]

                available_cols = [c for c in cols if c in period_df.columns]
                if available_cols:
                    df_out = period_df[available_cols].copy()
                    # 格式化持股数量
                    if "持股数量" in df_out.columns:
                        df_out["持股(万股)"] = (df_out["持股数量"] / 1e4).round(2)
                        df_out = df_out.drop(columns=["持股数量"])
                    lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
                else:
                    lines.append(period_df.to_csv(index=False, float_format="%.2f").strip())

            # 股东人数统计（如果有）
            if "股东总数" in df.columns:
                latest = df.iloc[0]
                holder_count = latest.get("股东总数")
                avg_shares = latest.get("平均持股数")
                if holder_count:
                    lines.append(f"\n## 股东统计")
                    lines.append(f"- 股东总数: {holder_count}")
                    if avg_shares:
                        lines.append(f"- 平均持股: {avg_shares}")

            # 变化分析（比较最近两期）
            if len(dates) >= 2:
                latest_date = dates[0]
                prev_date = dates[1]
                latest_holders = set(df[df[date_col] == latest_date]["股东名称"].tolist())
                prev_holders = set(df[df[date_col] == prev_date]["股东名称"].tolist())

                new_holders = latest_holders - prev_holders
                exit_holders = prev_holders - latest_holders

                if new_holders or exit_holders:
                    lines.append(f"\n## 股东变化 ({prev_date} → {latest_date})")
                    if new_holders:
                        lines.append(f"- 新进股东: {', '.join(list(new_holders)[:5])}")
                    if exit_holders:
                        lines.append(f"- 退出股东: {', '.join(list(exit_holders)[:5])}")
        else:
            lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取十大股东失败: {e}")
        return f"获取 {symbol} 十大股东失败: {e}"


def main():
    port = int(os.getenv("PORT", 0)) or 80
    parser = argparse.ArgumentParser(description="Stock Data MCP Server")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP mode instead of stdio")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=port, help=f"Port to listen on (default: {port})")

    args = parser.parse_args()
    mode = os.getenv("TRANSPORT") or ("http" if args.http else None)
    if mode in ["http", "sse"]:
        app = mcp.http_app(transport=mode)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id", "mcp-protocol-version"],
            max_age=86400,
        )
        mcp.run(transport=mode, host=args.host, port=args.port)
    else:
        mcp.run()

if __name__ == "__main__":
    main()
