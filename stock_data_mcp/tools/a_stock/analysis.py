"""
A股个股分析模块

包含筹码分布、多周期统计、资金流向、所属板块、板块成分股等工具
"""

import pandas as pd
from pydantic import Field

from ...core import (
    mcp,
    get_data_manager,
    format_source_name,
    field_symbol,
)
from ...data_provider import to_chinese_columns


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
