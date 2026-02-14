"""
A股估值与财务模块

包含估值对比、市场PE分位、行业PE对比、分红历史、基金持仓、财报日历、财务指标等工具
"""

import pandas as pd
import akshare as ak
import efinance as ef
from datetime import datetime
from pydantic import Field

from ...core import (
    mcp,
    get_data_manager,
    field_symbol,
    ak_cache,
    recent_trade_date,
)


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
