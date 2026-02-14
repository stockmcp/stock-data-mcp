"""
A股股东与风险模块

包含限售解禁、股权质押、十大股东、投资组合风险分析等工具
"""

import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from pydantic import Field

from ...core import (
    mcp,
    get_data_manager,
    get_akshare_source,
    field_symbol,
    ak_cache,
)
from ...data_provider import to_chinese_columns


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
