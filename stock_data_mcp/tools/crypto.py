"""
加密货币工具模块

包含 OKX、Binance 等加密货币市场相关的 MCP 工具
"""

import time
import json
import logging
import pandas as pd
from pydantic import Field

from ..core import (
    mcp,
    OKX_BASE_URL,
    BINANCE_BASE_URL,
    USER_AGENT,
    _http_get_with_retry,
    _http_post_with_retry,
)
from ..indicators import add_technical_indicators, CRYPTO_PRICE_COLUMNS

_LOGGER = logging.getLogger(__name__)


# ==================== OKX K线数据 ====================

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
    all_lines = dfs.to_csv(columns=CRYPTO_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
    lines = [f"# {instId} K线数据", f"# 数据来源: OKX"]
    lines.append("\n".join([all_lines[0], *all_lines[-limit:]]))
    return "\n".join(lines)


# ==================== OKX 多空比 ====================

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
    lines = [f"# {symbol} 多空比", f"# 数据来源: OKX"]
    lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
    return "\n".join(lines)


# ==================== OKX 主动买卖 ====================

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
    lines = [f"# {symbol} 主动买卖", f"# 数据来源: OKX"]
    lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
    return "\n".join(lines)


# ==================== Binance AI 报告 ====================

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
    lines = [f"# {symbol} AI分析报告", f"# 数据来源: Binance"]
    # AI报告内容以纯文本形式输出，每个观点一行
    lines.append("# 报告内容")
    lines.append("类型,内容")
    for module in modules:
        if tit := module.get('overview'):
            lines.append(f"概述,{tit.replace(',', '，')}")
        for point in module.get('points', []):
            content = point.get('content', '').replace(',', '，').replace('\n', ' ')
            lines.append(f"观点,{content}")
    return '\n'.join(lines)
