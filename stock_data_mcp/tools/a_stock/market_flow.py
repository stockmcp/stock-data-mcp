"""
A股市场资金流模块

包含涨停股池、龙虎榜、板块资金流、北向资金、融资融券、大宗交易等工具
"""

import time
import pandas as pd
import akshare as ak
from pydantic import Field

from ...core import (
    mcp,
    get_data_manager,
    format_source_name,
    get_akshare_source,
    field_symbol,
    ak_cache,
    recent_trade_date,
    fetch_with_retry,
    _detect_stock_market,
)


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
