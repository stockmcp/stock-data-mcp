"""
市场概览与系统工具模块

包含全球财经新闻、数据源状态等 MCP 工具
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime
from pydantic import Field

from ..core import (
    mcp,
    get_data_manager,
    ak_cache,
    USER_AGENT,
)

_LOGGER = logging.getLogger(__name__)


# ==================== 个股新闻 ====================

@mcp.tool(
    title="获取股票/加密货币相关新闻",
    description="根据股票代码或加密货币符号获取近期相关新闻",
)
def stock_news(
    symbol: str = Field(description="股票代码/加密货币符号"),
    limit: int = Field(15, description="返回数量(int)", strict=False),
):
    try:
        result = ak_cache(_stock_news_em, symbol=symbol, ttl=3600)
        if result is None or (hasattr(result, 'empty') and result.empty):
            return f"未找到 {symbol} 相关新闻"
        news = list(dict.fromkeys([
            v["新闻内容"]
            for v in result.to_dict(orient="records")
            if isinstance(v, dict)
        ]))
        if news:
            lines = [f"# {symbol} 相关新闻", f"# 数据来源: 东方财经"]
            lines.append("新闻内容")
            lines.extend(news[0:limit])
            return "\n".join(lines)
        return f"未找到 {symbol} 相关新闻"
    except Exception as e:
        _LOGGER.warning(f"获取新闻失败: {e}")
        return f"获取 {symbol} 新闻失败: {e}"


def _stock_news_em(symbol, limit=20):
    """从东方财富获取个股新闻"""
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


# ==================== 全球财经快讯 ====================

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


@mcp.tool(
    title="全球财经快讯",
    description="获取最新的全球财经快讯",
)
def stock_news_global():
    import akshare as ak
    news = ["# 全球财经快讯", "# 数据来源: 新浪财经, NewsNow"]
    news.append("时间,内容,来源")
    try:
        dfs = ak.stock_info_global_sina()
        csv = dfs.to_csv(index=False, float_format="%.2f").strip()
        csv = csv.replace(datetime.now().strftime("%Y-%m-%d "), "")
        lines = csv.split("\n")
        if lines:
            for line in lines[1:]:
                news.append(f"{line},新浪财经")
    except Exception as e:
        _LOGGER.debug(f"获取新浪财经快讯失败: {e}")
    news.extend(_newsnow_news())
    return "\n".join(news)


def _newsnow_news(channels=None):
    """从 NewsNow 获取财经快讯"""
    base = os.getenv("NEWSNOW_BASE_URL")
    if not base:
        _LOGGER.debug("NEWSNOW_BASE_URL 未配置，跳过 NewsNow 数据源")
        return []
    if not channels:
        channels = os.getenv("NEWSNOW_CHANNELS") or "wallstreetcn-quick,cls-telegraph,jin10"
    if isinstance(channels, str):
        channels = channels.split(",")
    _LOGGER.debug(f"NewsNow 请求: base={base}, channels={channels}")
    all_news = []
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
                pub_date = str(v.get("pubDate", ""))
                time_str = pub_date.split(" ")[-1] if " " in pub_date else ""
                all_news.append(f"{time_str},{content},{source_name}")
    except Exception as e:
        _LOGGER.warning(f"NewsNow 请求失败: {e}")
    return all_news


# ==================== 数据源状态 ====================

@mcp.tool(
    title="查看数据源状态",
    description="查看多数据源的状态和熔断器信息",
)
def data_source_status():
    try:
        manager = get_data_manager()
        status = manager.get_status()

        lines = ["# 数据源状态"]

        # 数据源列表
        lines.append("# 数据源")
        lines.append("名称,状态,优先级")
        for fetcher in status.get('fetchers', []):
            available = "OK" if fetcher['available'] else "FAIL"
            lines.append(f"{fetcher['name']},{available},{fetcher['priority']}")

        # 熔断器状态
        lines.append("# 熔断器状态")
        lines.append("类型,数据源,状态,失败次数")

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
                for source, state in breaker_status.items():
                    state_label = "正常" if state['state'] == 'closed' else "已熔断"
                    lines.append(f"{name},{source},{state_label},{state['failure_count']}")

        return "\n".join(lines)
    except Exception as e:
        return f"获取数据源状态失败: {e}"
