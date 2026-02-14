"""
A股工具模块

包含 A 股市场相关的 MCP 工具
"""

import time
import json
import pandas as pd
import numpy as np
import akshare as ak
import efinance as ef
from datetime import datetime, timedelta
from pydantic import Field

from ..core import (
    mcp,
    get_data_manager,
    format_source_name,
    get_akshare_source,
    field_symbol,
    field_market,
    ak_cache,
    recent_trade_date,
    fetch_with_retry,
    _detect_stock_market,
    USER_AGENT,
)
from ..data_provider import to_chinese_columns, validate_stock_type
from ..indicators import add_technical_indicators, STOCK_PRICE_COLUMNS


# ==================== 历史价格 ====================

def _fund_etf_hist_sina(symbol, market="sh", start_date="2025-01-01", period="daily"):
    """获取 ETF 历史数据"""
    dfs = ak.fund_etf_hist_sina(symbol=f"{market}{symbol}")
    if dfs is None or dfs.empty:
        return None
    dfs.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}, inplace=True)
    dfs["换手率"] = None
    dfs.index = pd.to_datetime(dfs["日期"], errors="coerce")
    return dfs.loc[start_date:]


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
    if market in ("hk", "us"):
        from .us_stock import _fetch_global_prices
        dfs = _fetch_global_prices(symbol, market, start_date, period)
        if dfs is not None and not dfs.empty:
            add_technical_indicators(dfs, dfs["收盘"], dfs["最低"], dfs["最高"], dfs.get("成交量"))
            all_lines = dfs.to_csv(columns=STOCK_PRICE_COLUMNS, index=False, float_format="%.2f").strip().split("\n")
            source = dfs.attrs.get('source', 'unknown')
            market_label = "港股" if market == "hk" else "美股"
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
        if market in ["sh", "sz"]:
            dfs = ak_cache(ak.stock_financial_abstract_ths, symbol=symbol)
            if dfs is None or dfs.empty:
                return f"获取A股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标", f"# 数据来源: akshare", f"# 市场: A股"]
            lines.append("\n".join([keys[0], *keys[-15:]]))
            return "\n".join(lines)
        elif market == "hk":
            dfs = ak_cache(ak.stock_financial_hk_analysis_indicator_em, symbol=symbol, indicator="报告期")
            if dfs is None or dfs.empty:
                return f"获取港股指标失败: {symbol}"
            keys = dfs.to_csv(index=False, float_format="%.3f").strip().split("\n")
            lines = [f"# {symbol} 财务指标", f"# 数据来源: akshare", f"# 市场: 港股"]
            lines.append("\n".join(keys[0:15]))
            return "\n".join(lines)
        elif market == "us":
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


# ==================== 涨停/强势股池 ====================

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
        lines = [f"# {title}", f"# 数据来源: akshare", f"# 共{cnt}只股票"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as exc:
        return f"获取股池数据失败: {exc}"


# ==================== 龙虎榜 ====================

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
        lines = [f"# 龙虎榜统计", f"# 数据来源: {source}"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        return f"获取龙虎榜数据失败: {e}"


# ==================== 板块资金流 ====================

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
            lines = [f"# {cate}", f"# 数据来源: {get_akshare_source(ak.stock_sector_fund_flow_rank)}"]
            lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)
    except Exception as e:
        pass

    # 备用数据源：行业板块实时行情（仅支持行业板块+今日）
    if cate == "行业资金流":
        try:
            time.sleep(1)
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
                lines = [f"# {cate}", f"# 数据来源: {get_akshare_source(ak.stock_board_industry_name_em)}"]
                lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
                return "\n".join(lines)
        except Exception:
            pass

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
                lines = [f"# {cate}", f"# 数据来源: {get_akshare_source(ak.stock_board_concept_name_em)}"]
                lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
                return "\n".join(lines)
        except Exception:
            pass

    return f"获取{cate}数据失败（数据源可能暂时不可用，请稍后重试）"


# ==================== 北向资金 ====================

@mcp.tool(
    title="沪深港通北向资金",
    description="获取沪深港通北向资金(外资)流向数据，包括沪股通、深股通的资金净流入情况。北向资金是A股重要的风向标。",
)
def stock_north_flow(
    indicator: str = Field("北向资金", description="指标类型，可选: '北向资金', '沪股通', '深股通'"),
):
    try:
        df = ak_cache(ak.stock_hsgt_fund_flow_summary_em, ttl=600)
        if df is None or df.empty:
            return "获取北向资金数据失败"

        if indicator == "沪股通":
            if "沪股通-净流入" in df.columns:
                df = df[["日期", "沪股通-净流入"]].copy()
                df.columns = ["日期", "净流入(亿)"]
        elif indicator == "深股通":
            if "深股通-净流入" in df.columns:
                df = df[["日期", "深股通-净流入"]].copy()
                df.columns = ["日期", "净流入(亿)"]
        else:
            if "北向资金-净流入" in df.columns:
                df = df[["日期", "北向资金-净流入"]].copy()
                df.columns = ["日期", "净流入(亿)"]
            elif "沪股通-净流入" in df.columns and "深股通-净流入" in df.columns:
                df["净流入(亿)"] = df["沪股通-净流入"] + df["深股通-净流入"]
                df = df[["日期", "净流入(亿)"]].copy()

        df = df.head(30)
        lines = [f"# {indicator}流向", f"# 数据来源: akshare"]
        lines.append(df.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as exc:
        return f"获取北向资金数据失败: {exc}"


# ==================== 融资融券 ====================

@mcp.tool(
    title="A股融资融券",
    description="获取A股市场融资融券数据，包括融资余额、融券余额等。融资融券是衡量市场杠杆资金的重要指标。",
)
def stock_margin_trading(
    symbol: str = Field("", description="股票代码（可选），留空则获取市场整体数据"),
    market: str = Field("sh", description="市场: 'sh'(沪市), 'sz'(深市)"),
    limit: int = Field(30, description="返回数据条数"),
):
    try:
        if symbol:
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
                    lines = [f"# {symbol} 融资融券", f"# 数据来源: {source}"]
                    lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())
                    return "\n".join(lines)

            return (
                f"获取个股 {symbol} 融资融券数据失败\n\n"
                f"可能原因:\n"
                f"1. 该股票不在融资融券标的范围内\n"
                f"2. akshare深交所接口存在兼容性问题（建议升级akshare）\n"
                f"3. 网络连接问题"
            )
        else:
            if market == "sh":
                df = ak_cache(ak.stock_margin_sse, start_date="", end_date="", ttl=1800)
            else:
                df = ak_cache(ak.stock_margin_szse, start_date="", end_date="", ttl=1800)

            if df is None or df.empty:
                return f"获取{market}市场融资融券数据失败"

            market_name = "沪市" if market == "sh" else "深市"
            df = df.tail(limit)
            lines = [f"# {market_name}融资融券", f"# 数据来源: akshare"]
            lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)
    except Exception as exc:
        return f"获取融资融券数据失败: {exc}"


# ==================== 大宗交易 ====================

@mcp.tool(
    title="A股大宗交易",
    description="获取A股大宗交易数据，包括成交价、成交量、溢价率等。大宗交易反映机构大额交易动向。",
)
def stock_block_trade(
    symbol: str = Field("", description="股票代码（可选），留空则获取当日全市场数据"),
    limit: int = Field(50, description="返回数据条数"),
):
    try:
        if symbol:
            try:
                df = ak_cache(ak.stock_dzjy_mrmx, symbol=symbol, ttl=1800)
                if df is not None and not df.empty:
                    df = df.head(limit)
                    lines = [f"# {symbol} 大宗交易", f"# 数据来源: akshare"]
                    lines.append(df.to_csv(index=False, float_format="%.2f").strip())
                    return "\n".join(lines)
            except Exception:
                pass
            try:
                df = ak_cache(ak.stock_dzjy_mrtj, start_date="", end_date="", ttl=1800)
                if df is not None and not df.empty:
                    if "证券代码" in df.columns:
                        df = df[df["证券代码"].astype(str).str.contains(symbol)]
                    if not df.empty:
                        lines = [f"# {symbol} 大宗交易", f"# 数据来源: akshare"]
                        lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())
                        return "\n".join(lines)
            except Exception:
                pass
            return f"未找到股票 {symbol} 的大宗交易数据"
        else:
            df = ak_cache(ak.stock_dzjy_mrtj, start_date="", end_date="", ttl=1800)
            if df is None or df.empty:
                return "获取大宗交易数据失败"
            df = df.head(limit)
            lines = ["# 大宗交易统计", "# 数据来源: akshare"]
            lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)
    except Exception as exc:
        return f"获取大宗交易数据失败: {exc}"


# ==================== 股东人数 ====================

@mcp.tool(
    title="A股股东人数",
    description="获取A股股东户数变化数据，筹码集中度的重要指标。股东人数减少通常意味着筹码趋于集中。",
)
def stock_holder_num(
    symbol: str = Field(description="股票代码，如: 300058, 600036"),
):
    try:
        df = ak_cache(ak.stock_zh_a_gdhs_detail_em, symbol=symbol, ttl=3600)
        if df is not None and not df.empty:
            lines = [f"# {symbol} 股东人数", f"# 数据来源: akshare"]
            lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            return "\n".join(lines)
        return f"未找到股票 {symbol} 的股东人数数据"
    except Exception as exc:
        return f"获取股东人数数据失败: {exc}"


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


# ==================== 筹码分布 ====================

@mcp.tool(
    title="获取筹码分布",
    description="获取A股筹码分布数据，包括获利比例、平均成本、成本区间、筹码集中度等。",
)
def stock_chip(
    symbol: str = field_symbol,
):
    if symbol.startswith(('51', '15', '16', '50', '52', '56', '58', '11', '12')):
        return f"{symbol} 是ETF/LOF/基金/可转债等产品，不支持筹码分布查询。筹码分布仅适用于普通A股。"

    try:
        manager = get_data_manager()
        chip = manager.get_chip_distribution(symbol)
        if chip is None:
            return f"未找到 {symbol} 的筹码分布数据，请确认是有效的A股代码"

        status = chip.get_chip_status()
        chip_level = status.get('chip_level', '-') if status else '-'

        lines = [
            f"# {chip.code} 筹码分布",
            f"# 数据来源: {chip.source}",
            f"# 日期: {chip.date or '-'}",
            "获利比例(%),平均成本,90%成本低,90%成本高,90%集中度(%),70%成本低,70%成本高,70%集中度(%),筹码状态",
            f"{chip.profit_ratio or '-'},{chip.avg_cost or '-'},{chip.cost_90_low or '-'},{chip.cost_90_high or '-'},{chip.concentration_90 or '-'},{chip.cost_70_low or '-'},{chip.cost_70_high or '-'},{chip.concentration_70 or '-'},{chip_level}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 筹码分布失败: {e}"


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


# ==================== 多周期统计 ====================

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
        df = manager.get_daily_data(symbol, days=180)
        if df is None or df.empty:
            return f"Not Found for {symbol}.{market}"

        source = format_source_name(df.attrs.get('source', ''))
        df = to_chinese_columns(df)
        close = df["收盘"]
        high = df["最高"]
        low = df["最低"]
        volume = df.get("成交量")

        periods = [5, 10, 20, 60, 120]
        available_periods = [p for p in periods if len(close) >= p]

        lines = [f"# {symbol} 多周期统计", f"# 数据来源: {source}"]

        # 价格统计表
        lines.append("# 价格统计")
        price_header = ["周期", "均价", "最高", "最低"]
        price_rows = [[f"当日", f"{close.iloc[-1]:.2f}", f"{high.iloc[-1]:.2f}", f"{low.iloc[-1]:.2f}"]]
        for p in available_periods:
            avg_price = close.iloc[-p:].mean()
            max_price = high.iloc[-p:].max()
            min_price = low.iloc[-p:].min()
            price_rows.append([f"{p}日", f"{avg_price:.2f}", f"{max_price:.2f}", f"{min_price:.2f}"])
        lines.append(",".join(price_header))
        lines.extend([",".join(row) for row in price_rows])

        # 涨跌幅统计表
        lines.append("# 涨跌幅统计")
        change_header = ["周期", "涨跌幅(%)"]
        change_rows = []
        if len(close) >= 2:
            today_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            change_rows.append(["当日", f"{today_change:.2f}"])
        for p in available_periods:
            if len(close) > p:
                change = (close.iloc[-1] / close.iloc[-p-1] - 1) * 100
                change_rows.append([f"{p}日", f"{change:.2f}"])
        lines.append(",".join(change_header))
        lines.extend([",".join(row) for row in change_rows])

        # 振幅统计表
        lines.append("# 振幅统计")
        amp_header = ["周期", "振幅(%)"]
        amp_rows = []
        if len(high) >= 1:
            today_amp = (high.iloc[-1] / low.iloc[-1] - 1) * 100
            amp_rows.append(["当日", f"{today_amp:.2f}"])
        for p in available_periods:
            amp = (high.iloc[-p:].max() / low.iloc[-p:].min() - 1) * 100
            amp_rows.append([f"{p}日", f"{amp:.2f}"])
        lines.append(",".join(amp_header))
        lines.extend([",".join(row) for row in amp_rows])

        # 换手率统计表
        if volume is not None and "换手率" in df.columns:
            turnover = df["换手率"]
            lines.append("# 换手率统计")
            turn_header = ["周期", "均换手(%)", "累计换手(%)"]
            turn_rows = []
            if len(turnover) >= 1 and turnover.iloc[-1] is not None:
                turn_rows.append(["当日", f"{turnover.iloc[-1]:.2f}", f"{turnover.iloc[-1]:.2f}"])
            for p in available_periods:
                avg_turn = turnover.iloc[-p:].mean()
                total_turn = turnover.iloc[-p:].sum()
                if avg_turn is not None:
                    turn_rows.append([f"{p}日", f"{avg_turn:.2f}", f"{total_turn:.2f}"])
            lines.append(",".join(turn_header))
            lines.extend([",".join(row) for row in turn_rows])

        # 成交量统计表
        if volume is not None:
            lines.append("# 成交量统计(万手)")
            vol_header = ["周期", "成交量"]
            vol_rows = [[f"当日", f"{volume.iloc[-1] / 10000:.2f}"]]
            for p in available_periods:
                avg_vol = volume.iloc[-p:].mean() / 10000
                vol_rows.append([f"{p}日", f"{avg_vol:.2f}"])
            lines.append(",".join(vol_header))
            lines.extend([",".join(row) for row in vol_rows])

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 多周期统计失败: {e}"


# ==================== 资金流向 ====================

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
        dfs = dfs.tail(10)

        lines = [f"# {symbol} 资金流向"]
        lines.append(f"# 数据来源: {source}")
        lines.append("# 近期资金流向")

        cols_to_show = [c for c in dfs.columns if c not in ["序号"]]
        csv_data = dfs.to_csv(columns=cols_to_show, index=False, float_format="%.2f").strip()
        return "\n".join(lines) + "\n" + csv_data
    except Exception as e:
        return f"获取 {symbol} 资金流向失败: {e}"


# ==================== 所属板块 ====================

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

        lines = [f"# {symbol} 所属板块"]

        if boards is not None and not boards.empty:
            source = format_source_name(boards.attrs.get('source', ''))
            lines.append(f"# 数据来源: {source}")
            lines.append("# 所属板块")
            lines.append(boards.to_csv(index=False, float_format="%.2f").strip())
        else:
            lines.append("未获取到板块数据")

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 板块信息失败: {e}"


# ==================== 板块成分股 ====================

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

        lines = [f"# {board_name} 成分股", f"# 数据来源: {source}"]
        lines.append(dfs.to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        return f"获取 {board_name} 成分股失败: {e}"


# ==================== 估值对比 ====================

@mcp.tool(
    title="A股估值对比",
    description="获取A股个股估值与行业对比数据，包括PE/PB在行业中的分位数。",
)
def stock_valuation_compare(
    symbol: str = field_symbol,
):
    try:
        base_info = ef.stock.get_base_info(symbol)
        if base_info is None or base_info.empty:
            return f"未获取到 {symbol} 基本信息"

        stock_name = base_info.get("股票名称", "-")
        stock_pe = base_info.get("市盈率(动)", None)
        stock_pb = base_info.get("市净率", None)
        stock_roe = base_info.get("ROE", None)
        industry = base_info.get("所处行业", "-")

        boards = ef.stock.get_belong_board(symbol)
        if boards is None or boards.empty:
            return f"未获取到 {symbol} 板块信息"

        industry_board = boards[boards["板块代码"].str.startswith("BK04")]
        if industry_board.empty:
            industry_board = boards.head(1)

        board_code = industry_board.iloc[0]["板块代码"]
        board_name = industry_board.iloc[0]["板块名称"]

        manager = get_data_manager()
        peers_df = manager.get_board_cons(board_name, "industry")

        if peers_df is None or peers_df.empty:
            peers_df = manager.get_board_cons(board_name, "concept")

        if peers_df is None or peers_df.empty:
            lines = [
                f"# {stock_name} ({symbol}) 估值信息",
                f"# 数据来源: efinance",
                f"# 所属行业: {industry}",
            ]
            lines.append("# 估值指标")
            lines.append("市盈率(动态),市净率,ROE(%)")
            lines.append(f"{stock_pe or '-'},{stock_pb or '-'},{stock_roe or '-'}")
            lines.append("# 注意: 未能获取同行业数据进行对比")
            return "\n".join(lines)

        peer_codes = []
        code_col = None
        for col in ["代码", "股票代码", "证券代码"]:
            if col in peers_df.columns:
                code_col = col
                break
        if code_col:
            peer_codes = peers_df[code_col].astype(str).tolist()[:20]

        peer_quotes = {}
        if peer_codes:
            try:
                peer_quotes = manager.prefetch_realtime_quotes(peer_codes)
            except Exception:
                pass

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

        lines = [
            f"# {stock_name} ({symbol}) 估值对比分析",
            f"# 数据来源: efinance",
            f"# 所属行业: {board_name} (共{len(peer_codes)}只股票)",
        ]

        # 个股估值表
        lines.append("# 个股估值")
        lines.append("市盈率(动态),市净率,ROE(%)")
        lines.append(f"{stock_pe or '-'},{stock_pb or '-'},{stock_roe or '-'}")

        if not peer_quotes:
            lines.append("# 行业对比: 暂无同行业数据（网络问题），请稍后重试")
            return "\n".join(lines)

        # 行业对比表
        lines.append("# 行业对比")
        compare_header = ["指标", "个股值", "行业中位数", "分位数(%)", "估值水平"]
        compare_rows = []
        if pe_percentile is not None:
            pe_level = "高估" if pe_percentile > 70 else "低估" if pe_percentile < 30 else "中性"
            compare_rows.append(["PE", f"{stock_pe:.2f}", f"{pe_median:.2f}", f"{pe_percentile:.1f}", pe_level])
        if pb_percentile is not None:
            pb_level = "高估" if pb_percentile > 70 else "低估" if pb_percentile < 30 else "中性"
            compare_rows.append(["PB", f"{stock_pb:.2f}", f"{pb_median:.2f}", f"{pb_percentile:.1f}", pb_level])
        lines.append(",".join(compare_header))
        lines.extend([",".join(row) for row in compare_rows])

        # 综合评估
        lines.append("# 综合评估")
        if pe_percentile is not None and pb_percentile is not None:
            avg_percentile = (pe_percentile + pb_percentile) / 2
            if avg_percentile < 30:
                conclusion = "低估 - PE/PB均低于行业70%的股票"
            elif avg_percentile > 70:
                conclusion = "高估 - PE/PB均高于行业70%的股票"
            else:
                conclusion = "合理 - PE/PB处于行业中等水平"
        else:
            conclusion = "数据不足无法评估"
        lines.append("结论")
        lines.append(conclusion)

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 估值对比失败: {e}"


# ==================== 市场PE分位 ====================

@mcp.tool(
    title="A股市场PE分位",
    description="获取A股市场整体PE/PB的历史分位数，用于判断市场整体估值水平。",
)
def stock_market_pe_percentile():
    try:
        pe_df = ak_cache(ak.stock_a_ttm_lyr, ttl=3600)
        pb_df = ak_cache(ak.stock_a_all_pb, ttl=3600)

        if pe_df is None or pe_df.empty:
            return "获取市场PE数据失败"

        latest_pe = pe_df.iloc[-1]
        pe_ttm_median = latest_pe.get("middlePETTM", None)
        pe_ttm_avg = latest_pe.get("averagePETTM", None)
        pe_percentile_all = latest_pe.get("quantileInAllHistoryMiddlePeTtm", None)
        pe_percentile_10y = latest_pe.get("quantileInRecent10YearsMiddlePeTtm", None)

        lines = [
            "# A股市场估值分位",
            "# 数据来源: akshare (乐咕乐股)",
        ]

        # 市盈率数据表
        lines.append("# 市盈率(PE-TTM)")
        pe_header = ["指标", "值", "估值水平"]
        pe_rows = []
        pe_rows.append(["中位数PE", f"{pe_ttm_median:.2f}" if pe_ttm_median else "-", "-"])
        pe_rows.append(["平均PE", f"{pe_ttm_avg:.2f}" if pe_ttm_avg else "-", "-"])
        if pe_percentile_all is not None:
            pct = pe_percentile_all * 100
            level = "极度高估" if pct > 80 else "高估" if pct > 60 else "合理" if pct > 40 else "低估" if pct > 20 else "极度低估"
            pe_rows.append(["历史分位(全部)", f"{pct:.1f}%", level])
        if pe_percentile_10y is not None:
            pct = pe_percentile_10y * 100
            level = "极度高估" if pct > 80 else "高估" if pct > 60 else "合理" if pct > 40 else "低估" if pct > 20 else "极度低估"
            pe_rows.append(["历史分位(近10年)", f"{pct:.1f}%", level])
        lines.append(",".join(pe_header))
        lines.extend([",".join(row) for row in pe_rows])

        # 市净率数据表
        if pb_df is not None and not pb_df.empty:
            latest_pb = pb_df.iloc[-1]
            pb_median = latest_pb.get("middlePB", None)
            pb_percentile_all = latest_pb.get("quantileInAllHistoryMiddlePB", None)
            pb_percentile_10y = latest_pb.get("quantileInRecent10YearsMiddlePB", None)

            lines.append("# 市净率(PB)")
            pb_header = ["指标", "值", "估值水平"]
            pb_rows = []
            if pb_median:
                pb_rows.append(["中位数PB", f"{pb_median:.2f}", "-"])
            if pb_percentile_all is not None:
                pct = pb_percentile_all * 100
                level = "极度高估" if pct > 80 else "高估" if pct > 60 else "合理" if pct > 40 else "低估" if pct > 20 else "极度低估"
                pb_rows.append(["历史分位(全部)", f"{pct:.1f}%", level])
            if pb_percentile_10y is not None:
                pct = pb_percentile_10y * 100
                pb_rows.append(["历史分位(近10年)", f"{pct:.1f}%", "-"])
            lines.append(",".join(pb_header))
            lines.extend([",".join(row) for row in pb_rows])

        # 市场估值建议
        lines.append("# 市场估值建议")
        if pe_percentile_10y is not None:
            pct = pe_percentile_10y * 100
            if pct < 30:
                suggestion = "当前市场估值处于历史低位，长期投资价值凸显"
            elif pct > 70:
                suggestion = "当前市场估值处于历史高位，需注意回调风险"
            else:
                suggestion = "当前市场估值处于历史中位，选股重于择时"
            lines.append("建议")
            lines.append(suggestion)

        return "\n".join(lines)
    except Exception as e:
        return f"获取市场PE分位失败: {e}"


# ==================== 行业PE对比 ====================

@mcp.tool(
    title="A股行业PE对比",
    description="获取A股各行业PE对比数据，用于行业估值比较和行业轮动分析。",
)
def stock_industry_pe(
    date: str = Field("", description="日期(可选)，格式: 20250210，默认最新"),
):
    try:
        if not date:
            date = recent_trade_date().strftime("%Y%m%d")

        df = ak_cache(
            ak.stock_industry_pe_ratio_cninfo,
            symbol="证监会行业分类",
            date=date,
            ttl=3600
        )

        if df is None or df.empty:
            return f"获取行业PE数据失败，日期: {date}"

        df_l1 = df[df["行业层级"] == 1.0].copy()
        if df_l1.empty:
            df_l1 = df.head(20)

        df_l1 = df_l1.sort_values("静态市盈率-加权平均", ascending=True)

        lines = [
            "# A股行业PE对比",
            f"# 数据来源: akshare (巨潮资讯)",
            f"# 数据日期: {date}",
        ]

        cols = ["行业名称", "公司数量", "静态市盈率-加权平均", "静态市盈率-中位数"]
        df_out = df_l1[cols].copy()
        df_out.columns = ["行业", "公司数", "加权PE", "中位PE"]
        lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())

        # 估值提示
        low_pe = df_l1.head(3)["行业名称"].tolist()
        high_pe = df_l1.tail(3)["行业名称"].tolist()
        lines.append("# 估值提示")
        lines.append("类型,行业")
        lines.append(f"低估值行业,{' '.join(low_pe)}")
        lines.append(f"高估值行业,{' '.join(high_pe)}")

        return "\n".join(lines)
    except Exception as e:
        return f"获取行业PE对比失败: {e}"


# ==================== 分红历史 ====================

@mcp.tool(
    title="A股分红历史",
    description="获取A股个股历史分红送转数据，包括派息、送股、转增等。用于分析股息率和分红政策。",
)
def stock_dividend_history(
    symbol: str = field_symbol,
    limit: int = Field(10, description="返回数量限制"),
):
    try:
        df = ak_cache(ak.stock_history_dividend_detail, symbol=symbol, indicator="分红", ttl=86400)
        if df is None or df.empty:
            return f"未获取到 {symbol} 的分红历史数据"

        manager = get_data_manager()
        quote = manager.get_realtime_quote(symbol)
        current_price = quote.price if quote else None

        lines = [f"# {symbol} 分红历史", "# 数据来源: akshare"]

        df = df.head(limit)

        # 分红记录表
        dividend_header = ["公告日期", "送股", "转增", "派息(元/10股)", "进度", "除权除息日"]
        dividend_rows = []
        total_dividend = 0
        recent_dividend = 0
        for _, row in df.iterrows():
            date = str(row.get("公告日期", "-"))[:10]
            song = row.get("送股", 0) or 0
            zhuan = row.get("转增", 0) or 0
            pai = row.get("派息", 0) or 0
            status = row.get("进度", "-")
            ex_date = str(row.get("除权除息日", "-"))[:10] if pd.notna(row.get("除权除息日")) else "-"
            dividend_rows.append([date, str(song), str(zhuan), f"{pai:.2f}", str(status), ex_date])
            if status == "实施" and pai > 0:
                total_dividend += pai
                if recent_dividend == 0:
                    recent_dividend = pai

        lines.append(",".join(dividend_header))
        lines.extend([",".join(row) for row in dividend_rows])

        # 股息率分析
        if current_price and recent_dividend > 0:
            dividend_yield = (recent_dividend / 10) / current_price * 100
            lines.append("# 股息率分析")
            lines.append("当前股价,最近派息(元/10股),股息率(%)")
            lines.append(f"{current_price:.2f},{recent_dividend:.2f},{dividend_yield:.2f}")

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 分红历史失败: {e}"


# ==================== 基金持仓 ====================

@mcp.tool(
    title="A股基金持仓",
    description="获取A股基金重仓股数据，显示公募基金持仓最多的股票及持仓变化。用于跟踪机构动向。",
)
def stock_institutional_holdings(
    date: str = Field("", description="报告期，格式: 20240930，默认最新季度"),
    limit: int = Field(30, description="返回数量限制"),
):
    try:
        if not date:
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
            f"# 基金重仓股 ({date})",
            "# 数据来源: akshare",
        ]

        cols = ["股票代码", "股票简称", "持有基金家数", "持股总数", "持股市值", "持股变化", "持股变动比例"]
        df_out = df[cols].copy()
        df_out["持股市值"] = (df_out["持股市值"] / 1e8).round(2)
        df_out["持股总数"] = (df_out["持股总数"] / 1e4).round(2)
        df_out.columns = ["代码", "名称", "基金数", "持股(万)", "市值(亿)", "变化", "变动%"]
        lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())

        # 持仓变化统计
        increase = len(df[df["持股变化"] == "增仓"])
        decrease = len(df[df["持股变化"] == "减仓"])
        new_hold = len(df[df["持股变化"] == "新进"])
        lines.append("# 持仓变化统计")
        lines.append("增仓,减仓,新进")
        lines.append(f"{increase},{decrease},{new_hold}")

        return "\n".join(lines)
    except Exception as e:
        return f"获取基金持仓失败: {e}"


# ==================== 财报日历 ====================

@mcp.tool(
    title="A股财报日历",
    description="获取A股财报披露时间表，查看即将披露财报的公司。用于跟踪财报季。",
)
def stock_earnings_calendar(
    period: str = Field("", description="报告期，如: 2024年报、2024三季报，默认最新"),
    limit: int = Field(50, description="返回数量限制"),
):
    try:
        if not period:
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
            f"# 财报披露日历 ({period})",
            "# 数据来源: akshare (巨潮资讯)",
        ]

        if "首次预约时间" in df.columns:
            df = df.sort_values("首次预约时间")

        today = datetime.now().strftime("%Y-%m-%d")
        today_count = len(df[df.get("首次预约时间", "").astype(str).str.startswith(today)]) if "首次预约时间" in df.columns else 0

        lines.append(f"# 今日披露: {today_count}家")
        lines.append("# 即将披露")

        df = df.head(limit)
        cols_available = [c for c in ["股票代码", "股票简称", "首次预约时间", "实际披露时间", "修改次数"] if c in df.columns]
        if cols_available:
            lines.append(df[cols_available].to_csv(index=False).strip())
        else:
            lines.append(df.to_csv(index=False).strip())

        return "\n".join(lines)
    except Exception as e:
        return f"获取财报日历失败: {e}"


# ==================== 财务指标对比 ====================

@mcp.tool(
    title="A股财务指标对比",
    description="获取A股个股详细财务指标，包括盈利能力、偿债能力、运营能力等多维度分析。",
)
def stock_financial_compare(
    symbol: str = field_symbol,
):
    try:
        df = ak_cache(
            ak.stock_financial_analysis_indicator,
            symbol=symbol,
            start_year=str(datetime.now().year - 2),
            ttl=3600
        )
        if df is None or df.empty:
            return f"未获取到 {symbol} 的财务指标数据"

        lines = [f"# {symbol} 财务指标分析", "# 数据来源: akshare"]

        df = df.head(4)

        # 盈利能力
        lines.append("# 盈利能力")
        profit_cols = ["日期", "净资产收益率(%)", "销售毛利率(%)", "销售净利率(%)", "总资产利润率(%)"]
        profit_cols = [c for c in profit_cols if c in df.columns]
        if profit_cols:
            lines.append(df[profit_cols].to_csv(index=False, float_format="%.2f").strip())

        # 成长能力
        lines.append("# 成长能力")
        growth_cols = ["日期", "主营业务收入增长率(%)", "净利润增长率(%)", "净资产增长率(%)", "总资产增长率(%)"]
        growth_cols = [c for c in growth_cols if c in df.columns]
        if growth_cols:
            lines.append(df[growth_cols].to_csv(index=False, float_format="%.2f").strip())

        # 偿债能力
        lines.append("# 偿债能力")
        debt_cols = ["日期", "流动比率", "速动比率", "资产负债率(%)", "股东权益比率(%)"]
        debt_cols = [c for c in debt_cols if c in df.columns]
        if debt_cols:
            lines.append(df[debt_cols].to_csv(index=False, float_format="%.2f").strip())

        # 运营能力
        lines.append("# 运营能力")
        ops_cols = ["日期", "应收账款周转率(次)", "存货周转率(次)", "总资产周转率(次)"]
        ops_cols = [c for c in ops_cols if c in df.columns]
        if ops_cols:
            lines.append(df[ops_cols].to_csv(index=False, float_format="%.2f").strip())

        # 每股指标
        lines.append("# 每股指标")
        share_cols = ["日期", "摊薄每股收益(元)", "每股净资产_调整前(元)", "每股经营性现金流(元)"]
        share_cols = [c for c in share_cols if c in df.columns]
        if share_cols:
            lines.append(df[share_cols].to_csv(index=False, float_format="%.4f").strip())

        # 趋势分析
        if len(df) >= 2:
            latest = df.iloc[0]
            prev = df.iloc[1]
            lines.append("# 趋势分析")
            trend_header = ["指标", "变化", "值", "评价"]
            trend_rows = []

            if "净资产收益率(%)" in df.columns:
                roe_change = (latest["净资产收益率(%)"] or 0) - (prev["净资产收益率(%)"] or 0)
                trend = "↑" if roe_change > 0 else "↓" if roe_change < 0 else "→"
                trend_rows.append(["ROE变化", trend, f"{abs(roe_change):.2f}%", "-"])

            if "净利润增长率(%)" in df.columns and pd.notna(latest.get("净利润增长率(%)")):
                growth = latest["净利润增长率(%)"]
                level = "高速增长" if growth > 30 else "稳定增长" if growth > 0 else "下滑"
                trend_rows.append(["净利润增速", "-", f"{growth:.2f}%", level])

            if "资产负债率(%)" in df.columns:
                debt_ratio = latest["资产负债率(%)"]
                risk = "高" if debt_ratio > 70 else "中" if debt_ratio > 50 else "低"
                trend_rows.append(["资产负债率", "-", f"{debt_ratio:.2f}%", f"风险{risk}"])

            if trend_rows:
                lines.append(",".join(trend_header))
                lines.extend([",".join(row) for row in trend_rows])

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 财务指标失败: {e}"


# ==================== 投资组合风险分析 ====================

@mcp.tool(
    title="投资组合风险分析",
    description="分析投资组合的风险指标，包括波动率、最大回撤、相关性矩阵、夏普比率等。",
)
def portfolio_risk_analysis(
    symbols: str = Field(description="股票代码列表，逗号分隔，如: 600519,000858,601318"),
    days: int = Field(60, description="分析周期(天)"),
):
    try:
        codes = [s.strip() for s in symbols.split(",") if s.strip()]
        if len(codes) < 2:
            return "请提供至少2只股票进行组合分析"

        manager = get_data_manager()
        price_data = {}
        names = {}

        for code in codes[:10]:
            df = manager.get_daily_data(code, days=days + 10)
            if df is not None and not df.empty:
                df = to_chinese_columns(df)
                price_data[code] = df["收盘"].tail(days)

                quote = manager.get_realtime_quote(code)
                names[code] = quote.name if quote and quote.name else code

        if len(price_data) < 2:
            return "有效股票数据不足，请检查股票代码"

        prices_df = pd.DataFrame(price_data)
        returns_df = prices_df.pct_change().dropna()

        lines = [
            f"# 投资组合风险分析",
            f"# 分析周期: {days}天",
            f"# 股票数量: {len(price_data)}只",
        ]

        # 个股风险指标表
        lines.append("# 个股风险指标")
        risk_header = ["代码", "名称", "年化波动率(%)", "最大回撤(%)", "夏普比率"]
        risk_rows = []

        risk_free_rate = 0.02
        for code in price_data.keys():
            ret = returns_df[code]
            prices = prices_df[code]

            volatility = ret.std() * np.sqrt(252) * 100
            cummax = prices.cummax()
            drawdown = (prices - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            annual_return = ret.mean() * 252
            sharpe = (annual_return - risk_free_rate) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

            name = names.get(code, code)
            risk_rows.append([code, name, f"{volatility:.2f}", f"{max_drawdown:.2f}", f"{sharpe:.2f}"])

        lines.append(",".join(risk_header))
        lines.extend([",".join(row) for row in risk_rows])

        # 相关性矩阵
        lines.append("# 相关性矩阵")
        corr_matrix = returns_df.corr()
        corr_display = corr_matrix.copy()
        corr_display.index = [names.get(c, c) for c in corr_display.index]
        corr_display.columns = [names.get(c, c) for c in corr_display.columns]
        lines.append(corr_display.to_csv(float_format="%.2f").strip())

        # 等权组合分析
        lines.append("# 等权组合分析")
        n = len(price_data)
        weights = np.array([1/n] * n)
        portfolio_return = returns_df.mean().values @ weights * 252
        portfolio_vol = np.sqrt(weights @ returns_df.cov().values @ weights) * np.sqrt(252)
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        portfolio_value = (1 + returns_df @ weights).cumprod()
        portfolio_cummax = portfolio_value.cummax()
        portfolio_drawdown = (portfolio_value - portfolio_cummax) / portfolio_cummax
        portfolio_max_dd = portfolio_drawdown.min() * 100

        lines.append("预期年化收益(%),年化波动率(%),夏普比率,最大回撤(%)")
        lines.append(f"{portfolio_return*100:.2f},{portfolio_vol*100:.2f},{portfolio_sharpe:.2f},{portfolio_max_dd:.2f}")

        # 风险提示
        lines.append("# 风险提示")
        risk_header = ["风险类型", "说明"]
        risk_rows = []

        high_corr = []
        for i, c1 in enumerate(corr_matrix.columns):
            for c2 in corr_matrix.columns[i+1:]:
                if corr_matrix.loc[c1, c2] > 0.7:
                    high_corr.append(f"{names.get(c1, c1)}-{names.get(c2, c2)}")
        if high_corr:
            risk_rows.append(["高相关性(>0.7)", " ".join(high_corr[:3])])

        high_vol = [names.get(c, c) for c in returns_df.columns if returns_df[c].std() * np.sqrt(252) > 0.4]
        if high_vol:
            risk_rows.append(["高波动(>40%)", " ".join(high_vol[:3])])

        for code in price_data.keys():
            ret = returns_df[code]
            annual_return = ret.mean() * 252
            sharpe = (annual_return - risk_free_rate) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
            if sharpe < 0:
                risk_rows.append(["负夏普比率", f"{names.get(code, code)} 风险收益不匹配"])
                break

        if risk_rows:
            lines.append(",".join(risk_header))
            lines.extend([",".join(row) for row in risk_rows])

        return "\n".join(lines)
    except Exception as e:
        return f"组合风险分析失败: {e}"


# ==================== 限售解禁 ====================

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
    try:
        if hasattr(start_date, 'default'):
            start_date = start_date.default or ""
        if hasattr(end_date, 'default'):
            end_date = end_date.default or ""
        if hasattr(mode, 'default'):
            mode = mode.default or "detail"

        if not start_date:
            start_date = datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")

        if mode == "summary":
            df = ak_cache(
                ak.stock_restricted_release_summary_em,
                start_date=start_date,
                end_date=end_date,
                ttl=3600
            )
            if df is None or df.empty:
                return f"未获取到限售解禁汇总数据 ({start_date} ~ {end_date})"

            lines = [
                f"# 限售解禁日历 (汇总)",
                f"# 数据来源: {get_akshare_source(ak.stock_restricted_release_summary_em)}",
                f"# 日期范围: {start_date} ~ {end_date}",
                "# 每日解禁汇总",
            ]

            cols = ["解禁时间", "当日解禁股票家数", "解禁数量", "实际解禁数量", "实际解禁市值"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df = df[available_cols].head(limit)
                if "实际解禁市值" in df.columns:
                    df["实际解禁市值(亿)"] = (df["实际解禁市值"] / 1e8).round(2)
                    df = df.drop(columns=["实际解禁市值"])
                lines.append(df.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

            return "\n".join(lines)

        else:
            df = ak_cache(
                ak.stock_restricted_release_detail_em,
                start_date=start_date,
                end_date=end_date,
                ttl=3600
            )
            if df is None or df.empty:
                return f"未获取到限售解禁明细数据 ({start_date} ~ {end_date})"

            lines = [
                f"# 限售解禁日历 (明细)",
                f"# 数据来源: {get_akshare_source(ak.stock_restricted_release_detail_em)}",
                f"# 日期范围: {start_date} ~ {end_date}",
                f"# 共 {len(df)} 只股票即将解禁",
            ]

            if "实际解禁市值" in df.columns:
                df = df.sort_values("实际解禁市值", ascending=False)

            cols = ["股票代码", "股票简称", "解禁时间", "限售股类型", "实际解禁数量", "实际解禁市值", "占解禁前流通市值比例"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df_out = df[available_cols].head(limit).copy()
                if "实际解禁市值" in df_out.columns:
                    df_out["实际解禁市值(万)"] = (df_out["实际解禁市值"] / 1e4).round(2)
                    df_out = df_out.drop(columns=["实际解禁市值"])
                lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

            if "占解禁前流通市值比例" in df.columns:
                high_impact = df[df["占解禁前流通市值比例"] > 10].head(5)
                if not high_impact.empty:
                    lines.append("# 高冲击风险股票(解禁占比>10%)")
                    hi_header = ["代码", "名称", "解禁占比(%)", "解禁日"]
                    hi_rows = []
                    for _, row in high_impact.iterrows():
                        code = row.get("股票代码", "-")
                        name = row.get("股票简称", "-")
                        ratio = row.get("占解禁前流通市值比例", 0)
                        unlock_date = row.get("解禁时间", "-")
                        hi_rows.append([str(code), str(name), f"{ratio:.1f}", str(unlock_date)])
                    lines.append(",".join(hi_header))
                    lines.extend([",".join(row) for row in hi_rows])

            return "\n".join(lines)
    except Exception as e:
        return f"获取限售解禁日历失败: {e}"


# ==================== 股权质押 ====================

@mcp.tool(
    title="A股股权质押",
    description="获取A股股权质押数据，包括行业质押统计和市场整体质押比例。股权质押是衡量大股东杠杆风险的重要指标。",
)
def stock_pledge_ratio(
    mode: str = Field("industry", description="模式: 'industry'(行业统计), 'market'(市场整体趋势)"),
    limit: int = Field(30, description="返回数量限制"),
):
    try:
        if mode == "industry":
            df = ak_cache(ak.stock_gpzy_industry_data_em, ttl=3600)
            if df is None or df.empty:
                return "获取行业质押数据失败"

            if "平均质押比例" in df.columns:
                df = df.sort_values("平均质押比例", ascending=False)

            lines = [
                "# 行业股权质押统计",
                f"# 数据来源: {get_akshare_source(ak.stock_gpzy_industry_data_em)}",
                "# 各行业质押情况 (按质押比例降序)",
            ]

            cols = ["行业", "公司家数", "质押总笔数", "平均质押比例", "质押总股本", "最新质押市值"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df_out = df[available_cols].head(limit).copy()
                if "最新质押市值" in df_out.columns:
                    df_out["质押市值(亿)"] = (df_out["最新质押市值"] / 1e8).round(2)
                    df_out = df_out.drop(columns=["最新质押市值"])
                if "质押总股本" in df_out.columns:
                    df_out["质押股本(亿股)"] = (df_out["质押总股本"] / 1e8).round(2)
                    df_out = df_out.drop(columns=["质押总股本"])
                lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

            if "平均质押比例" in df.columns:
                high_pledge = df[df["平均质押比例"] > 20].head(5)
                if not high_pledge.empty:
                    lines.append("# 高质押风险行业(平均质押比例>20%)")
                    hp_header = ["行业", "平均质押(%)", "公司家数"]
                    hp_rows = []
                    for _, row in high_pledge.iterrows():
                        industry = row.get("行业", "-")
                        ratio = row.get("平均质押比例", 0)
                        count = row.get("公司家数", 0)
                        hp_rows.append([str(industry), f"{ratio:.1f}", str(count)])
                    lines.append(",".join(hp_header))
                    lines.extend([",".join(row) for row in hp_rows])

            return "\n".join(lines)

        else:
            df = ak_cache(ak.stock_gpzy_profile_em, ttl=3600)
            if df is None or df.empty:
                return "获取市场质押趋势数据失败"

            lines = [
                "# A股市场股权质押趋势",
                f"# 数据来源: {get_akshare_source(ak.stock_gpzy_profile_em)}",
            ]

            df = df.tail(limit)

            cols = ["统计时间", "A股质押总比例", "A股质押总股数", "A股质押总市值", "A股质押公司数量"]
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df_out = df[available_cols].copy()
                if "A股质押总市值" in df_out.columns:
                    df_out["质押市值(万亿)"] = (df_out["A股质押总市值"] / 1e12).round(2)
                    df_out = df_out.drop(columns=["A股质押总市值"])
                if "A股质押总股数" in df_out.columns:
                    df_out["质押股数(亿股)"] = (df_out["A股质押总股数"] / 1e8).round(2)
                    df_out = df_out.drop(columns=["A股质押总股数"])
                lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
            else:
                lines.append(df.to_csv(index=False, float_format="%.2f").strip())

            if "A股质押总比例" in df.columns and len(df) >= 2:
                latest = df.iloc[-1]["A股质押总比例"]
                prev = df.iloc[-2]["A股质押总比例"]
                change = latest - prev
                trend = "上升" if change > 0 else "下降" if change < 0 else "持平"
                lines.append("# 趋势分析")
                lines.append("最新质押比例(%),变化趋势,变化幅度(%)")
                lines.append(f"{latest:.2f},{trend},{change:+.2f}")

            return "\n".join(lines)
    except Exception as e:
        return f"获取股权质押数据失败: {e}"


# ==================== 十大股东 ====================

@mcp.tool(
    title="A股十大股东",
    description="获取A股个股十大股东或十大流通股东信息，用于分析股权结构和机构持仓变化。",
)
def stock_top10_holders(
    symbol: str = field_symbol,
    holder_type: str = Field("main", description="股东类型: 'main'(十大股东), 'circulate'(十大流通股东)"),
    limit: int = Field(30, description="返回数量限制（多期数据）"),
):
    try:
        if holder_type == "circulate":
            ak_func = ak.stock_circulate_stock_holder
            df = ak_cache(ak_func, symbol=symbol, ttl=3600)
            title = "十大流通股东"
            date_col = "截止日期"
        else:
            ak_func = ak.stock_main_stock_holder
            df = ak_cache(ak_func, stock=symbol, ttl=3600)
            title = "十大股东"
            date_col = "截至日期"

        if df is None or df.empty:
            return f"未获取到 {symbol} 的{title}数据"

        lines = [
            f"# {symbol} {title}",
            f"# 数据来源: {get_akshare_source(ak_func)}",
        ]

        if date_col in df.columns:
            dates = df[date_col].unique()[:3]
            for date in dates:
                period_df = df[df[date_col] == date].head(10)

                lines.append(f"# {date}")

                if holder_type == "circulate":
                    cols = ["编号", "股东名称", "持股数量", "占流通股比例", "股本性质"]
                else:
                    cols = ["编号", "股东名称", "持股数量", "持股比例", "股本性质"]

                available_cols = [c for c in cols if c in period_df.columns]
                if available_cols:
                    df_out = period_df[available_cols].copy()
                    if "持股数量" in df_out.columns:
                        df_out["持股(万股)"] = (df_out["持股数量"] / 1e4).round(2)
                        df_out = df_out.drop(columns=["持股数量"])
                    lines.append(df_out.to_csv(index=False, float_format="%.2f").strip())
                else:
                    lines.append(period_df.to_csv(index=False, float_format="%.2f").strip())

            if "股东总数" in df.columns:
                latest = df.iloc[0]
                holder_count = latest.get("股东总数")
                avg_shares = latest.get("平均持股数")
                if holder_count:
                    lines.append("# 股东统计")
                    lines.append("股东总数,平均持股")
                    lines.append(f"{holder_count},{avg_shares or '-'}")

            if len(dates) >= 2:
                latest_date = dates[0]
                prev_date = dates[1]
                latest_holders = set(df[df[date_col] == latest_date]["股东名称"].tolist())
                prev_holders = set(df[df[date_col] == prev_date]["股东名称"].tolist())

                new_holders = latest_holders - prev_holders
                exit_holders = prev_holders - latest_holders

                if new_holders or exit_holders:
                    lines.append(f"# 股东变化({prev_date}→{latest_date})")
                    lines.append("类型,股东")
                    if new_holders:
                        lines.append(f"新进,{' '.join(list(new_holders)[:5])}")
                    if exit_holders:
                        lines.append(f"退出,{' '.join(list(exit_holders)[:5])}")
        else:
            lines.append(df.head(limit).to_csv(index=False, float_format="%.2f").strip())

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 十大股东失败: {e}"


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


# ==================== 选股器 ====================

@mcp.tool(
    title="A股选股器",
    description="根据条件筛选A股股票，支持按PE、PB、市值、涨跌幅、换手率等指标筛选。",
)
def stock_screener(
    pe_min: float = Field(0, description="最小市盈率(动态)"),
    pe_max: float = Field(100, description="最大市盈率(动态)，0表示不限"),
    pb_min: float = Field(0, description="最小市净率"),
    pb_max: float = Field(50, description="最大市净率，0表示不限"),
    mc_min: float = Field(0, description="最小总市值(亿元)"),
    mc_max: float = Field(0, description="最大总市值(亿元)，0表示不限"),
    change_min: float = Field(-100, description="最小涨跌幅(%)"),
    change_max: float = Field(100, description="最大涨跌幅(%)"),
    turnover_min: float = Field(0, description="最小换手率(%)"),
    volume_ratio_min: float = Field(0, description="最小量比"),
    sort_by: str = Field("涨跌幅", description="排序字段: 涨跌幅/换手率/量比/市盈率/市净率/总市值"),
    ascending: bool = Field(False, description="是否升序排列"),
    limit: int = Field(30, description="返回数量"),
):
    try:
        df = ak_cache(ak.stock_zh_a_spot_em, ttl=300)
        if df is None or df.empty:
            return "获取A股行情数据失败"

        # 转换数值列
        for col in ["市盈率-动态", "市净率", "总市值", "涨跌幅", "换手率", "量比"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 筛选条件
        mask = pd.Series([True] * len(df), index=df.index)

        # PE筛选
        if pe_min > 0:
            mask &= df["市盈率-动态"] >= pe_min
        if pe_max > 0:
            mask &= df["市盈率-动态"] <= pe_max

        # PB筛选
        if pb_min > 0:
            mask &= df["市净率"] >= pb_min
        if pb_max > 0:
            mask &= df["市净率"] <= pb_max

        # 市值筛选 (转换为亿)
        if "总市值" in df.columns:
            df["总市值_亿"] = df["总市值"] / 1e8
            if mc_min > 0:
                mask &= df["总市值_亿"] >= mc_min
            if mc_max > 0:
                mask &= df["总市值_亿"] <= mc_max

        # 涨跌幅筛选
        mask &= df["涨跌幅"] >= change_min
        mask &= df["涨跌幅"] <= change_max

        # 换手率筛选
        if turnover_min > 0:
            mask &= df["换手率"] >= turnover_min

        # 量比筛选
        if volume_ratio_min > 0:
            mask &= df["量比"] >= volume_ratio_min

        # 应用筛选
        result = df[mask].copy()

        if result.empty:
            return "未找到符合条件的股票"

        # 排序字段映射
        sort_col = {
            "涨跌幅": "涨跌幅",
            "换手率": "换手率",
            "量比": "量比",
            "市盈率": "市盈率-动态",
            "市净率": "市净率",
            "总市值": "总市值",
        }.get(sort_by, "涨跌幅")

        if sort_col in result.columns:
            result = result.sort_values(sort_col, ascending=ascending)

        result = result.head(int(limit))

        # 选择输出列
        output_cols = ["代码", "名称", "最新价", "涨跌幅", "换手率", "量比", "市盈率-动态", "市净率"]
        if "总市值_亿" in result.columns:
            result["市值(亿)"] = result["总市值_亿"].round(2)
            output_cols.append("市值(亿)")
        available_cols = [c for c in output_cols if c in result.columns]

        lines = [
            f"# A股选股结果",
            f"# 数据来源: {get_akshare_source(ak.stock_zh_a_spot_em)}",
            f"# 筛选条件: PE[{pe_min},{pe_max}] PB[{pb_min},{pb_max}] 涨跌幅[{change_min}%,{change_max}%]",
            f"# 共找到 {len(result)} 只符合条件的股票",
        ]
        lines.append(result[available_cols].to_csv(index=False, float_format="%.2f").strip())
        return "\n".join(lines)
    except Exception as e:
        return f"选股失败: {e}"


# ==================== 交易信号 ====================

@mcp.tool(
    title="A股交易信号",
    description="根据技术指标生成股票交易信号，综合MACD、KDJ、RSI、布林带等指标判断买卖时机。",
)
def trading_signals(
    symbol: str = field_symbol,
    days: int = Field(60, description="分析周期(天)"),
):
    try:
        manager = get_data_manager()
        df = manager.get_daily_data(symbol, days=days + 30)
        if df is None or df.empty:
            return f"未获取到 {symbol} 的历史数据"

        source = format_source_name(df.attrs.get('source', ''))
        df = to_chinese_columns(df)
        close = df["收盘"]
        high = df["最高"]
        low = df["最低"]
        volume = df.get("成交量")

        add_technical_indicators(df, close, low, high, volume)

        # 获取最新数据
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        signals = []
        buy_score = 0
        sell_score = 0

        # 1. MACD 信号
        macd = latest.get("MACD")
        signal_line = latest.get("信号线")
        prev_macd = prev.get("MACD")
        prev_signal = prev.get("信号线")
        if all(v is not None for v in [macd, signal_line, prev_macd, prev_signal]):
            if prev_macd < prev_signal and macd > signal_line:
                signals.append(["MACD", "金叉↑", "买入信号"])
                buy_score += 2
            elif prev_macd > prev_signal and macd < signal_line:
                signals.append(["MACD", "死叉↓", "卖出信号"])
                sell_score += 2
            elif macd > 0 and macd > signal_line:
                signals.append(["MACD", "多头排列", "偏多"])
                buy_score += 1
            elif macd < 0 and macd < signal_line:
                signals.append(["MACD", "空头排列", "偏空"])
                sell_score += 1

        # 2. KDJ 信号
        k = latest.get("K")
        d = latest.get("D")
        j = latest.get("J")
        prev_k = prev.get("K")
        prev_d = prev.get("D")
        if all(v is not None for v in [k, d, j]):
            if prev_k and prev_d and prev_k < prev_d and k > d:
                signals.append(["KDJ", f"金叉 K={k:.1f} D={d:.1f} J={j:.1f}", "买入"])
                buy_score += 2
            elif prev_k and prev_d and prev_k > prev_d and k < d:
                signals.append(["KDJ", f"死叉 K={k:.1f} D={d:.1f} J={j:.1f}", "卖出"])
                sell_score += 2
            elif j < 20:
                signals.append(["KDJ", f"超卖区 J={j:.1f}", "关注反弹"])
                buy_score += 1
            elif j > 80:
                signals.append(["KDJ", f"超买区 J={j:.1f}", "注意回调"])
                sell_score += 1

        # 3. RSI 信号
        rsi14 = latest.get("RSI14")
        if rsi14 is not None:
            if rsi14 < 30:
                signals.append(["RSI14", f"{rsi14:.1f} 超卖", "反弹机会"])
                buy_score += 2
            elif rsi14 > 70:
                signals.append(["RSI14", f"{rsi14:.1f} 超买", "回调风险"])
                sell_score += 2
            elif 40 <= rsi14 <= 60:
                signals.append(["RSI14", f"{rsi14:.1f} 中性区间", "中性"])

        # 4. 布林带信号
        boll_upper = latest.get("BOLL_上轨")
        boll_mid = latest.get("BOLL_中轨")
        boll_lower = latest.get("BOLL_下轨")
        price = close.iloc[-1]
        if all(v is not None for v in [boll_upper, boll_mid, boll_lower]):
            if price <= boll_lower:
                signals.append(["布林带", f"触及下轨 {boll_lower:.2f}", "超卖"])
                buy_score += 2
            elif price >= boll_upper:
                signals.append(["布林带", f"触及上轨 {boll_upper:.2f}", "超买"])
                sell_score += 2
            elif price > boll_mid:
                signals.append(["布林带", "在中轨上方运行", "偏多"])
                buy_score += 1
            else:
                signals.append(["布林带", "在中轨下方运行", "偏空"])
                sell_score += 1

        # 5. 均线系统
        ma5 = latest.get("MA5")
        ma20 = latest.get("MA20")
        ma60 = latest.get("MA60")
        if ma5 and ma20:
            if ma5 > ma20:
                signals.append(["均线", f"MA5({ma5:.2f})>MA20({ma20:.2f})", "多头"])
                buy_score += 1
            else:
                signals.append(["均线", f"MA5({ma5:.2f})<MA20({ma20:.2f})", "空头"])
                sell_score += 1
            if ma60 and price > ma60:
                signals.append(["趋势", f"价格在MA60({ma60:.2f})上方", "中期多头"])
                buy_score += 1
            elif ma60 and price < ma60:
                signals.append(["趋势", f"价格在MA60({ma60:.2f})下方", "中期空头"])
                sell_score += 1

        # 6. 量能分析
        if volume is not None and len(volume) >= 5:
            vol_ma5 = volume.iloc[-5:].mean()
            current_vol = volume.iloc[-1]
            vol_ratio = current_vol / vol_ma5 if vol_ma5 > 0 else 1
            if vol_ratio > 2:
                signals.append(["量能", f"放量 {vol_ratio:.1f}倍", "关注突破"])
            elif vol_ratio < 0.5:
                signals.append(["量能", f"缩量 {vol_ratio:.1f}倍", "观望"])

        # 综合建议
        total_score = buy_score - sell_score
        if total_score >= 4:
            suggestion = "强烈买入 - 多项技术指标共振看多"
        elif total_score >= 2:
            suggestion = "买入 - 技术面偏多，可考虑建仓"
        elif total_score <= -4:
            suggestion = "强烈卖出 - 多项技术指标共振看空"
        elif total_score <= -2:
            suggestion = "卖出 - 技术面偏空，考虑减仓"
        else:
            suggestion = "观望 - 信号不明确，等待方向确认"

        lines = [
            f"# {symbol} 交易信号分析",
            f"# 数据来源: {source}",
            f"# 最新价格: {price:.2f}",
        ]

        # 技术指标信号表
        lines.append("# 技术指标信号")
        signal_header = ["指标", "状态", "建议"]
        lines.append(",".join(signal_header))
        lines.extend([",".join(row) for row in signals])

        # 综合评分表
        lines.append("# 综合评分")
        lines.append("多头得分,空头得分,综合得分")
        lines.append(f"{buy_score},{sell_score},{total_score:+d}")

        # 交易建议
        lines.append("# 交易建议")
        lines.append("建议,注意")
        lines.append(f"{suggestion},技术分析仅供参考需结合基本面和市场环境综合判断")

        return "\n".join(lines)
    except Exception as e:
        return f"获取 {symbol} 交易信号失败: {e}"


# ==================== 辅助函数 ====================

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
