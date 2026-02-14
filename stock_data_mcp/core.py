"""
MCP 核心模块

包含 MCP 实例、全局数据管理器、公共辅助函数
"""

import os
import time
import json
import random
import logging
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta
from fastmcp import FastMCP
from pydantic import Field

from ._version import __version__
from .cache import CacheKey
from .data_provider import DataFetcherManager

_LOGGER = logging.getLogger(__name__)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOGGER.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

# MCP 实例
mcp = FastMCP(name="stock-data-mcp", version=__version__)

# 全局数据获取管理器（支持多数据源自动故障转移）
_data_manager = None
_data_manager_lock = threading.Lock()


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
    # Fetcher 类名映射
    "EfinanceFetcher": "efinance",
    "AkshareFetcher": "akshare",
    "TushareFetcher": "tushare",
    "BaostockFetcher": "baostock",
    "YfinanceFetcher": "yfinance",
    "AlphaVantage": "alphavantage",
    "AlphaVantageFetcher": "alphavantage",
}

# akshare 函数后缀到数据源的映射
_AKSHARE_SUFFIX_MAP = {
    "_em": "东方财富",
    "_sina": "新浪",
    "_ths": "同花顺",
    "_cninfo": "巨潮资讯",
    "_qq": "腾讯",
    "_163": "网易",
    "_szse": "深交所",
    "_sse": "上交所",
}

# akshare 无后缀但数据源已知的函数（部分列表）
_AKSHARE_KNOWN_SOURCES = {
    "stock_sector_fund_flow_rank": "东方财富",
    "stock_board_industry_name": "东方财富",
    "stock_board_concept_name": "东方财富",
    "stock_zh_a_spot": "东方财富",
    "stock_zh_a_hist": "东方财富",
    "stock_individual_info": "东方财富",
    "stock_circulate_stock_holder": "东方财富",
    "stock_main_stock_holder": "东方财富",
    "stock_dzjy": "东方财富",
    "stock_hsgt_fund_flow": "东方财富",
    "stock_margin": "交易所",
    "stock_report_fund_hold": "东方财富",
    "stock_report_disclosure": "巨潮资讯",
    "stock_financial_abstract": "同花顺",
    "stock_financial_analysis_indicator": "同花顺",
    "stock_history_dividend": "东方财富",
    "stock_a_ttm_lyr": "乐咕乐股",
    "stock_a_all_pb": "乐咕乐股",
    "stock_industry_pe_ratio": "巨潮资讯",
    "stock_zh_a_gdhs": "东方财富",
}


def format_source_name(source: str) -> str:
    """格式化数据源名称为友好显示格式"""
    if not source:
        return "-"
    base_source = source.split("_")[0]
    friendly_name = _SOURCE_NAME_MAP.get(base_source, source)
    if "_" in source:
        suffix = source.split("_", 1)[1]
        friendly_name = f"{friendly_name} ({suffix})"
    return friendly_name


def get_akshare_source(func) -> str:
    """
    从 akshare 函数判断数据来源

    Args:
        func: akshare 函数对象或函数名字符串

    Returns:
        格式化的数据来源字符串，如 "akshare (东方财富)"
    """
    if func is None:
        return "akshare"

    # 获取函数名
    func_name = func.__name__ if callable(func) else str(func)

    # 1. 先检查已知函数映射（精确匹配）
    for known_func, source_name in _AKSHARE_KNOWN_SOURCES.items():
        if func_name.startswith(known_func):
            return f"akshare ({source_name})"

    # 2. 再检查后缀映射
    for suffix, source_name in _AKSHARE_SUFFIX_MAP.items():
        if suffix in func_name:
            return f"akshare ({source_name})"

    return "akshare"


# 公共 Field 定义
field_symbol = Field(description="股票代码")
field_market = Field("sh", description="股票市场，仅支持: sh(上证), sz(深证), hk(港股), us(美股), 不支持加密货币")

# API 基础 URL
OKX_BASE_URL = os.getenv("OKX_BASE_URL") or "https://www.okx.com"
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL") or "https://www.binance.com"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
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


def recent_trade_date():
    """获取最近交易日"""
    import akshare as ak
    now = datetime.now().date()
    dfs = ak_cache(ak.tool_trade_date_hist_sina, ttl=43200)
    if dfs is None:
        return now
    dfs.sort_values("trade_date", ascending=False, inplace=True)
    for d in dfs["trade_date"]:
        if d <= now:
            return d
    return now


def ak_cache(fun, *args, **kwargs) -> pd.DataFrame | None:
    """带缓存的 akshare 数据获取"""
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
            sleep_time = initial_delay + random.uniform(0.5, 1.5) * (i + 1)
            time.sleep(sleep_time)
            result = func(**kwargs)
            if result is not None:
                return result
        except (ConnectionError, TimeoutError, OSError) as e:
            last_error = e
            _LOGGER.warning(f"[{func.__name__}] 第{i+1}次尝试失败(网络): {type(e).__name__}")
            if i < max_retries - 1:
                time.sleep(delay * (i + 1))
        except Exception as e:
            last_error = e
            _LOGGER.warning(f"[{func.__name__}] 第{i+1}次尝试失败: {e}")
            if i < max_retries - 1:
                time.sleep(delay * (i + 1))
    return None


def _detect_stock_market(symbol: str) -> str:
    """根据股票代码判断市场"""
    if symbol.startswith(('6', '5')):
        return 'sh'
    elif symbol.startswith(('0', '3', '1', '2')):
        return 'sz'
    return 'sh'
