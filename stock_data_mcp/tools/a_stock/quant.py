"""
A股量化分析模块

包含选股器、交易信号、策略回测等工具
"""

import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from pydantic import Field

from ...core import (
    mcp,
    get_data_manager,
    format_source_name,
    get_akshare_source,
    field_symbol,
    ak_cache,
)
from ...data_provider import to_chinese_columns
from ...indicators import add_technical_indicators


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
        # 处理 FieldInfo 默认值
        if hasattr(days, 'default'):
            days = days.default or 60

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
        signal_line = latest.get("DEA")  # DEA即信号线
        prev_macd = prev.get("MACD")
        prev_signal = prev.get("DEA")
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
        k = latest.get("KDJ.K")
        d = latest.get("KDJ.D")
        j = latest.get("KDJ.J")
        prev_k = prev.get("KDJ.K")
        prev_d = prev.get("KDJ.D")
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
        rsi14 = latest.get("RSI")  # 默认14周期RSI
        if rsi14 is not None:
            if rsi14 < 30:
                signals.append(["RSI", f"{rsi14:.1f} 超卖", "反弹机会"])
                buy_score += 2
            elif rsi14 > 70:
                signals.append(["RSI", f"{rsi14:.1f} 超买", "回调风险"])
                sell_score += 2
            elif 40 <= rsi14 <= 60:
                signals.append(["RSI", f"{rsi14:.1f} 中性区间", "中性"])

        # 4. 布林带信号
        boll_upper = latest.get("BOLL.U")
        boll_mid = latest.get("BOLL.M")
        boll_lower = latest.get("BOLL.L")
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


# ==================== 策略回测 ====================

@mcp.tool(
    title="A股策略回测",
    description="对股票进行简单策略回测，支持均线交叉、MACD、KDJ等策略，返回收益率、最大回撤、胜率等指标。",
)
def backtest_strategy(
    symbol: str = field_symbol,
    strategy: str = Field("ma_cross", description="策略类型: 'ma_cross'(均线交叉), 'macd'(MACD金叉死叉), 'kdj'(KDJ超买超卖), 'rsi'(RSI超买超卖), 'boll'(布林带突破)"),
    start_date: str = Field("", description="开始日期，格式: 20240101，默认一年前"),
    end_date: str = Field("", description="结束日期，格式: 20241231，默认今天"),
    initial_capital: float = Field(100000, description="初始资金(元)"),
    ma_short: int = Field(5, description="短期均线周期(ma_cross策略)"),
    ma_long: int = Field(20, description="长期均线周期(ma_cross策略)"),
):
    try:
        # 处理 FieldInfo 默认值
        if hasattr(start_date, 'default'):
            start_date = start_date.default or ""
        if hasattr(end_date, 'default'):
            end_date = end_date.default or ""
        if hasattr(strategy, 'default'):
            strategy = strategy.default or "ma_cross"
        if hasattr(initial_capital, 'default'):
            initial_capital = initial_capital.default or 100000
        if hasattr(ma_short, 'default'):
            ma_short = ma_short.default or 5
        if hasattr(ma_long, 'default'):
            ma_long = ma_long.default or 20

        # 默认日期范围
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

        # 获取历史数据
        manager = get_data_manager()
        days = (datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")).days + 60
        df = manager.get_daily_data(symbol, days=days)

        if df is None or df.empty:
            return f"未获取到 {symbol} 的历史数据"

        source = format_source_name(df.attrs.get('source', ''))
        df = to_chinese_columns(df)

        # 过滤日期范围
        df.index = pd.to_datetime(df["日期"], errors="coerce")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()

        if len(df) < 30:
            return f"数据量不足（{len(df)}条），至少需要30个交易日"

        close = df["收盘"]
        high = df["最高"]
        low = df["最低"]

        # 计算技术指标
        add_technical_indicators(df, close, low, high, df.get("成交量"))

        # 生成交易信号
        signals = pd.Series(0, index=df.index)  # 0=持币, 1=买入, -1=卖出

        if strategy == "ma_cross":
            # 均线交叉策略
            ma_s = close.rolling(window=ma_short, min_periods=1).mean()
            ma_l = close.rolling(window=ma_long, min_periods=1).mean()
            # 金叉买入
            signals[(ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))] = 1
            # 死叉卖出
            signals[(ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))] = -1
            strategy_name = f"MA{ma_short}/MA{ma_long}交叉"

        elif strategy == "macd":
            # MACD金叉死叉策略
            macd = df.get("MACD")
            signal_line = df.get("DEA")  # DEA即信号线
            if macd is None or signal_line is None:
                return "MACD指标计算失败"
            # 金叉买入
            signals[(macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))] = 1
            # 死叉卖出
            signals[(macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))] = -1
            strategy_name = "MACD金叉死叉"

        elif strategy == "kdj":
            # KDJ超买超卖策略
            k = df.get("KDJ.K")
            d = df.get("KDJ.D")
            j = df.get("KDJ.J")
            if k is None or d is None or j is None:
                return "KDJ指标计算失败"
            # J<20 金叉买入
            signals[(j < 20) & (k > d) & (k.shift(1) <= d.shift(1))] = 1
            # J>80 死叉卖出
            signals[(j > 80) & (k < d) & (k.shift(1) >= d.shift(1))] = -1
            strategy_name = "KDJ超买超卖"

        elif strategy == "rsi":
            # RSI超买超卖策略
            rsi = df.get("RSI")  # 默认14周期RSI
            if rsi is None:
                return "RSI指标计算失败"
            # RSI<30 买入
            signals[(rsi < 30) & (rsi.shift(1) >= 30)] = 1
            # RSI>70 卖出
            signals[(rsi > 70) & (rsi.shift(1) <= 70)] = -1
            strategy_name = "RSI超买超卖"

        elif strategy == "boll":
            # 布林带突破策略
            boll_upper = df.get("BOLL.U")
            boll_lower = df.get("BOLL.L")
            if boll_upper is None or boll_lower is None:
                return "布林带指标计算失败"
            # 触及下轨买入
            signals[(close <= boll_lower) & (close.shift(1) > boll_lower.shift(1))] = 1
            # 触及上轨卖出
            signals[(close >= boll_upper) & (close.shift(1) < boll_upper.shift(1))] = -1
            strategy_name = "布林带突破"

        else:
            return f"不支持的策略类型: {strategy}"

        # 回测计算
        capital = initial_capital
        position = 0  # 持仓数量
        entry_price = 0
        trades = []  # 交易记录
        equity_curve = []  # 权益曲线

        for i, (date, row) in enumerate(df.iterrows()):
            price = row["收盘"]
            signal = signals.iloc[i]

            # 记录当前权益
            current_equity = capital + position * price
            equity_curve.append({"日期": date, "权益": current_equity})

            if signal == 1 and position == 0:  # 买入信号且空仓
                shares = int(capital / price / 100) * 100  # 按手买入
                if shares > 0:
                    cost = shares * price
                    capital -= cost
                    position = shares
                    entry_price = price
                    trades.append({
                        "日期": date.strftime("%Y-%m-%d"),
                        "类型": "买入",
                        "价格": price,
                        "数量": shares,
                        "金额": cost,
                    })

            elif signal == -1 and position > 0:  # 卖出信号且持仓
                revenue = position * price
                profit = (price - entry_price) * position
                profit_pct = (price / entry_price - 1) * 100
                capital += revenue
                trades.append({
                    "日期": date.strftime("%Y-%m-%d"),
                    "类型": "卖出",
                    "价格": price,
                    "数量": position,
                    "金额": revenue,
                    "盈亏": profit,
                    "盈亏%": profit_pct,
                })
                position = 0
                entry_price = 0

        # 最终清仓
        final_price = close.iloc[-1]
        final_equity = capital + position * final_price

        # 计算回测指标
        total_return = (final_equity / initial_capital - 1) * 100
        equity_df = pd.DataFrame(equity_curve)
        equity_series = equity_df["权益"]

        # 最大回撤
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # 年化收益率
        trading_days = len(df)
        annual_return = total_return * (252 / trading_days) if trading_days > 0 else 0

        # 夏普比率 (假设无风险利率2%)
        if len(equity_df) > 1:
            daily_returns = equity_series.pct_change().dropna()
            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() * 252 - 0.02) / (daily_returns.std() * np.sqrt(252))
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 胜率
        sell_trades = [t for t in trades if t["类型"] == "卖出"]
        win_trades = [t for t in sell_trades if t.get("盈亏", 0) > 0]
        win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0

        # 盈亏比
        wins = [t["盈亏"] for t in sell_trades if t.get("盈亏", 0) > 0]
        losses = [-t["盈亏"] for t in sell_trades if t.get("盈亏", 0) < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # 买入持有收益
        buy_hold_return = (final_price / close.iloc[0] - 1) * 100

        lines = [
            f"# {symbol} 策略回测报告",
            f"# 策略: {strategy_name}",
            f"# 数据来源: {source}",
            f"# 回测区间: {start_date} ~ {end_date} ({trading_days}个交易日)",
            "",
            "# 收益指标",
            "初始资金,最终权益,总收益率(%),年化收益率(%),买入持有收益(%),超额收益(%)",
            f"{initial_capital:.0f},{final_equity:.0f},{total_return:.2f},{annual_return:.2f},{buy_hold_return:.2f},{total_return - buy_hold_return:.2f}",
            "",
            "# 风险指标",
            "最大回撤(%),夏普比率,波动率(%)",
            f"{max_drawdown:.2f},{sharpe:.2f},{daily_returns.std() * np.sqrt(252) * 100:.2f}" if len(equity_df) > 1 else f"{max_drawdown:.2f},0,0",
            "",
            "# 交易统计",
            "交易次数,胜率(%),盈亏比,平均盈利,平均亏损",
            f"{len(sell_trades)},{win_rate:.1f},{profit_factor:.2f},{avg_win:.0f},{avg_loss:.0f}",
        ]

        # 最近交易记录
        if trades:
            lines.append("")
            lines.append("# 最近交易记录")
            lines.append("日期,类型,价格,数量,金额,盈亏,盈亏%")
            for t in trades[-10:]:
                profit_str = f"{t.get('盈亏', 0):.0f}" if "盈亏" in t else "-"
                pct_str = f"{t.get('盈亏%', 0):.2f}" if "盈亏%" in t else "-"
                lines.append(f"{t['日期']},{t['类型']},{t['价格']:.2f},{t['数量']},{t['金额']:.0f},{profit_str},{pct_str}")

        # 策略评价
        lines.append("")
        lines.append("# 策略评价")
        if total_return > buy_hold_return and max_drawdown > -20:
            evaluation = "优秀 - 跑赢大盘且回撤可控"
        elif total_return > 0 and max_drawdown > -30:
            evaluation = "良好 - 盈利且风险适中"
        elif total_return > buy_hold_return:
            evaluation = "一般 - 跑赢大盘但回撤较大"
        else:
            evaluation = "较差 - 未能跑赢买入持有策略"
        lines.append(f"评价: {evaluation}")
        lines.append("注意: 回测结果仅供参考，历史表现不代表未来收益")

        return "\n".join(lines)
    except Exception as e:
        return f"回测 {symbol} 失败: {e}"
