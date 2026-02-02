import os
import time
import json
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
INDICATOR_COLUMNS = ["MACD", "DIF", "DEA", "KDJ.K", "KDJ.D", "KDJ.J", "RSI6", "RSI", "RSI24", "BOLL.U", "BOLL.M", "BOLL.L", "OBV", "ATR"]
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
        suffix = f"交易市场: {market}"
        return "\n".join([info.to_string(), suffix])
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
        return all.to_string()

    info = ak_search(symbol, market)
    if info is not None:
        return info.to_string()
    return f"Not Found for {symbol}.{market}"


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
                # 转换为中文列名
                df = to_chinese_columns(df)
                # 添加换手率列（如果没有）
                if "换手率" not in df.columns:
                    df["换手率"] = None
                # 添加技术指标
                add_technical_indicators(df, df["收盘"], df["最低"], df["最高"], df.get("成交量"))
                available_cols = [c for c in STOCK_PRICE_COLUMNS if c in df.columns]
                all_lines = df.to_csv(columns=available_cols, index=False, float_format="%.2f").strip().split("\n")
                return "\n".join([all_lines[0], *all_lines[-limit:]])
        except Exception as e:
            _LOGGER.warning(f"多数据源获取失败，回退到原有逻辑: {e}")

    # 回退到原有逻辑（港股、美股、ETF 等）
    if period == "weekly":
        delta = {"weeks": limit + 62}
    else:
        delta = {"days": limit + 62}
    start_date = (datetime.now() - timedelta(**delta)).strftime("%Y%m%d")
    markets = [
        ["sh", ak.stock_zh_a_hist, {}],
        ["sz", ak.stock_zh_a_hist, {}],
        ["hk", ak.stock_hk_hist, {}],
        ["us", stock_us_daily, {}],
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
        return "\n".join([all[0], *all[-limit:]])
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
    news = list(dict.fromkeys([
        v["新闻内容"]
        for v in ak_cache(stock_news_em, symbol=symbol, ttl=3600).to_dict(orient="records")
        if isinstance(v, dict)
    ]))
    if news:
        return "\n".join(news[0:limit])
    return f"Not Found for {symbol}"

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
    title="A股关键指标",
    description="获取中国A股市场(上证、深证)的股票财务报告关键指标",
)
def stock_indicators_a(
    symbol: str = field_symbol,
):
    dfs = ak_cache(ak.stock_financial_abstract_ths, symbol=symbol)
    keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
    return "\n".join([keys[0], *keys[-15:]])


@mcp.tool(
    title="港股关键指标",
    description="获取港股市场的股票财务报告关键指标",
)
def stock_indicators_hk(
    symbol: str = field_symbol,
):
    dfs = ak_cache(ak.stock_financial_hk_analysis_indicator_em, symbol=symbol, indicator="报告期")
    keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
    return "\n".join(keys[0:15])


@mcp.tool(
    title="美股关键指标",
    description="获取美股市场的股票财务报告关键指标",
)
def stock_indicators_us(
    symbol: str = field_symbol,
):
    dfs = ak_cache(ak.stock_financial_us_analysis_indicator_em, symbol=symbol, indicator="单季报")
    keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
    return "\n".join(keys[0:15])


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
    title="A股涨停股池",
    description="获取中国A股市场(上证、深证)的所有涨停股票",
)
def stock_zt_pool_em(
    date: str = Field("", description="交易日日期(可选)，默认为最近的交易日，格式: 20251231"),
    limit: int = Field(50, description="返回数量(int,30-100)", strict=False),
):
    if not date:
        date = recent_trade_date().strftime("%Y%m%d")
    dfs = ak_cache(ak.stock_zt_pool_em, date=date, ttl=1200)
    if dfs is None or dfs.empty:
        return "获取涨停股池数据失败"
    cnt = len(dfs)
    dfs.drop(columns=["序号", "流通市值", "总市值"], inplace=True, errors='ignore')
    if "成交额" in dfs.columns:
        dfs.sort_values("成交额", ascending=False, inplace=True)
    dfs = dfs.head(int(limit))
    desc = f"共{cnt}只涨停股\n"
    return desc + dfs.to_csv(index=False, float_format="%.2f").strip()


@mcp.tool(
    title="A股强势股池",
    description="获取中国A股市场(上证、深证)的强势股池数据",
)
def stock_zt_pool_strong_em(
    date: str = Field("", description="交易日日期(可选)，默认为最近的交易日，格式: 20251231"),
    limit: int = Field(50, description="返回数量(int,30-100)", strict=False),
):
    if not date:
        date = recent_trade_date().strftime("%Y%m%d")
    dfs = ak_cache(ak.stock_zt_pool_strong_em, date=date, ttl=1200)
    if dfs is None or dfs.empty:
        return "获取强势股池数据失败"
    dfs.drop(columns=["序号", "流通市值", "总市值"], inplace=True, errors='ignore')
    if "成交额" in dfs.columns:
        dfs.sort_values("成交额", ascending=False, inplace=True)
    dfs = dfs.head(int(limit))
    return dfs.to_csv(index=False, float_format="%.2f").strip()


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

        source = dfs.attrs.get('source', '-')
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
    dfs = ak_cache(ak.stock_sector_fund_flow_rank, indicator=days, sector_type=cate, ttl=1200)
    if dfs is None:
        return "获取数据失败"
    if "今日涨跌幅" in dfs.columns:
        dfs.sort_values("今日涨跌幅", ascending=False, inplace=True)
    dfs.drop(columns=["序号"], inplace=True, errors='ignore')
    try:
        dfs = pd.concat([dfs.head(20), dfs.tail(20)])
        return dfs.to_csv(index=False, float_format="%.2f").strip()
    except Exception as exc:
        return str(exc)


@mcp.tool(
    title="全球财经快讯",
    description="获取最新的全球财经快讯",
)
def stock_news_global():
    news = []
    try:
        dfs = ak.stock_info_global_sina()
        csv = dfs.to_csv(index=False, float_format="%.2f").strip()
        csv = csv.replace(datetime.now().strftime("%Y-%m-%d "), "")
        news.extend(csv.split("\n"))
    except Exception:
        pass
    news.extend(newsnow_news())
    return "\n".join(news)


def newsnow_news(channels=None):
    base = os.getenv("NEWSNOW_BASE_URL")
    if not base:
        return []
    if not channels:
        channels = os.getenv("NEWSNOW_CHANNELS") or "wallstreetcn-quick,cls-telegraph,jin10"
    if isinstance(channels, str):
        channels = channels.split(",")
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
        lst = res.json() or []
        for item in lst:
            for v in item.get("items", [])[0:15]:
                title = v.get("title", "")
                extra = v.get("extra") or {}
                hover = extra.get("hover") or title
                info = extra.get("info") or ""
                all.append(f"{hover} {info}".strip().replace("\n", " "))
    except Exception:
        pass
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
    return "\n".join([all[0], *all[-limit:]])


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
    return dfs.to_csv(index=False, float_format="%.2f").strip()


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
    return dfs.to_csv(index=False, float_format="%.2f").strip()


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
    except Exception:
        try:
            resp = json.loads(res.text.strip()) or {}
        except Exception:
            return res.text
    data = resp.get('data') or {}
    report = data.get('report') or {}
    translated = report.get('translated') or report.get('original') or {}
    modules = translated.get('modules') or []
    txts = []
    for module in modules:
        if tit := module.get('overview'):
            txts.append(tit)
        for point in module.get('points', []):
            txts.append(point.get('content', ''))
    return '\n'.join(txts)


@mcp.tool(
    title="给出投资建议",
    description="基于AI对其他工具提供的数据分析结果给出具体投资建议",
)
def trading_suggest(
    symbol: str = Field(description="股票代码或加密币种"),
    action: str = Field(description="推荐操作: buy/sell/hold"),
    score: int = Field(description="置信度，范围: 0-100"),
    reason: str = Field(description="推荐理由"),
):
    return {
        "symbol": symbol,
        "action": action,
        "score": score,
        "reason": reason,
    }


@mcp.tool(
    title="获取股票实时行情",
    description="获取A股/港股实时行情数据，包括最新价、涨跌幅、成交量、换手率、市盈率等。支持多数据源自动故障转移。",
)
def stock_realtime(
    symbol: str = field_symbol,
    market: str = Field("sh", description="股票市场，仅支持: sh(上证), sz(深证), hk(港股)"),
):
    try:
        manager = get_data_manager()
        quote = manager.get_realtime_quote(symbol)
        if quote is None:
            return f"Not Found for {symbol}.{market}"

        # 格式化输出（Markdown）
        lines = [
            f"# {quote.name or symbol} ({quote.code}) 实时行情\n",
            f"数据来源: {quote.source.value if quote.source else '-'}\n",
            "## 价格",
            f"- 最新价: {quote.price or '-'}",
            f"- 涨跌幅: {quote.change_pct or '-'}%",
            f"- 涨跌额: {quote.change_amount or '-'}",
            f"- 今开: {quote.open_price or '-'}",
            f"- 最高: {quote.high or '-'}",
            f"- 最低: {quote.low or '-'}",
            f"- 昨收: {quote.pre_close or '-'}",
            f"- 振幅: {quote.amplitude or '-'}%",
            "",
            "## 成交",
            f"- 成交量: {quote.volume or '-'}",
            f"- 成交额: {quote.amount or '-'}",
            f"- 换手率: {quote.turnover_rate or '-'}%",
            f"- 量比: {quote.volume_ratio or '-'}",
            "",
            "## 估值",
            f"- 市盈率: {quote.pe_ratio or '-'}",
            f"- 市净率: {quote.pb_ratio or '-'}",
            f"- 总市值: {quote.total_mv or '-'}",
            f"- 流通市值: {quote.circ_mv or '-'}",
        ]
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
    try:
        manager = get_data_manager()
        chip = manager.get_chip_distribution(symbol)
        if chip is None:
            return f"Not Found for {symbol}"

        # 格式化输出（Markdown）
        lines = [
            f"# {chip.code} 筹码分布\n",
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

        # 转换为 DataFrame 输出
        rows = []
        for code, quote in quotes.items():
            rows.append({
                "代码": quote.code,
                "名称": quote.name or "-",
                "最新价": quote.price,
                "涨跌幅": quote.change_pct,
                "成交量": quote.volume,
                "成交额": quote.amount,
                "换手率": quote.turnover_rate,
                "市盈率": quote.pe_ratio,
                "市净率": quote.pb_ratio,
            })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False, float_format="%.2f").strip()
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

        df = to_chinese_columns(df)
        close = df["收盘"]
        high = df["最高"]
        low = df["最低"]
        volume = df.get("成交量")

        periods = [5, 10, 20, 60, 120]
        available_periods = [p for p in periods if len(close) >= p]

        lines = [f"# {symbol} 多周期统计\n"]

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

        source = dfs.attrs.get('source', '-')
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
            source = boards.attrs.get('source', '-')
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

        source = dfs.attrs.get('source', '-')
        dfs = dfs.head(int(limit))
        dfs = dfs.drop(columns=["序号"], errors='ignore')

        lines = [f"# {board_name} 成分股\n", f"数据来源: {source}\n"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取板块成分股失败: {e}")
        return f"获取 {board_name} 成分股失败: {e}"


def _fetch_board_cons_direct(board_name: str, board_type: str) -> pd.DataFrame | None:
    """直接调用东财API获取板块成分股"""
    # 先获取板块代码
    try:
        if board_type == "concept":
            boards = ak_cache(ak.stock_board_concept_name_em, ttl=3600)
            code_col = "板块代码"
        else:
            boards = ak_cache(ak.stock_board_industry_name_em, ttl=3600)
            code_col = "板块代码"

        if boards is None or boards.empty:
            return None

        matched = boards[boards["板块名称"] == board_name]
        if matched.empty:
            return None

        board_code = matched[code_col].values[0]
    except Exception:
        return None

    # 调用成分股API
    url = "http://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1,
        "pz": 100,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": f"b:{board_code}+t:2",
        "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f15,f16,f17,f18",
    }

    max_retries = 3
    for i in range(max_retries):
        try:
            res = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
            data = res.json()
            if data and data.get("data") and data["data"].get("diff"):
                df = pd.DataFrame(data["data"]["diff"])
                df = df.rename(columns={
                    "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅",
                    "f4": "涨跌额", "f5": "成交量", "f6": "成交额", "f7": "振幅",
                    "f15": "最高", "f16": "最低", "f17": "今开", "f18": "昨收"
                })
                return df
        except Exception as e:
            _LOGGER.warning(f"直接API第{i+1}次尝试失败: {e}")
            if i < max_retries - 1:
                time.sleep(1 * (i + 1))

    return None


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
            _LOGGER.info("Request akshare: %s", [key, args, kwargs])
            all = fun(*args, **kwargs)
            cache.set(all)
        except Exception as exc:
            _LOGGER.exception(str(exc))
    return all


def multi_source_fetch(
    sources: list[tuple[callable, dict]],
    ttl: int = 3600,
    cache_key: str = None,
) -> pd.DataFrame | None:
    """
    多数据源获取数据，自动故障转移

    Args:
        sources: [(函数, 参数字典), ...] 按优先级排序
        ttl: 缓存时间（秒）
        cache_key: 缓存键（可选）

    Returns:
        DataFrame 或 None
    """
    # 尝试从缓存获取
    if cache_key:
        cache = CacheKey.init(cache_key, ttl, ttl * 7)
        cached = cache.get()
        if cached is not None:
            return cached

    last_error = None
    for func, kwargs in sources:
        try:
            _LOGGER.info(f"多数据源获取: {func.__name__} {kwargs}")
            result = func(**kwargs)
            if result is not None and not (hasattr(result, 'empty') and result.empty):
                # 缓存成功结果
                if cache_key:
                    cache.set(result)
                return result
        except Exception as e:
            last_error = e
            _LOGGER.warning(f"[{func.__name__}] 获取失败: {e}")
            continue

    if last_error:
        _LOGGER.error(f"所有数据源均失败，最后错误: {last_error}")
    return None


def fetch_with_retry(func, max_retries: int = 3, delay: float = 1.0, **kwargs):
    """
    带重试的数据获取

    Args:
        func: 获取函数
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
        **kwargs: 传递给函数的参数

    Returns:
        函数返回值或 None
    """
    import time
    last_error = None
    for i in range(max_retries):
        try:
            result = func(**kwargs)
            if result is not None:
                return result
        except Exception as e:
            last_error = e
            _LOGGER.warning(f"[{func.__name__}] 第{i+1}次尝试失败: {e}")
            if i < max_retries - 1:
                time.sleep(delay * (i + 1))  # 递增延迟
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

    # 计算OBV（能量潮指标）- 向量化实现
    if volume is not None:
        price_diff = clos.diff()
        direction = np.sign(price_diff).fillna(0)
        df["OBV"] = (direction * volume).fillna(0).cumsum()

    # 计算ATR（真实波幅）
    tr1 = high - lows
    tr2 = abs(high - clos.shift(1))
    tr3 = abs(lows - clos.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()


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
