"""
美股工具模块

包含美股市场相关的 MCP 工具
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pydantic import Field

from ..core import (
    mcp,
    get_data_manager,
    format_source_name,
    field_symbol,
    field_market,
    ak_cache,
    ALPHA_VANTAGE_API_KEY,
)
from ..data_provider import to_chinese_columns
from ..indicators import add_technical_indicators, STOCK_PRICE_COLUMNS

_LOGGER = logging.getLogger(__name__)


def _fetch_global_prices(symbol: str, market: str, start_date: str, period: str = "daily") -> pd.DataFrame | None:
    """统一的港股/美股价格获取（带故障转移）"""
    import akshare as ak
    import yfinance as yf

    if hasattr(period, 'default'):
        period = period.default or "daily"

    label = "港股" if market == "hk" else "美股"

    # 1. akshare
    try:
        if market == "hk":
            dfs = ak_cache(ak.stock_hk_hist, symbol=symbol, period=period, start_date=start_date, ttl=3600)
        else:
            dfs = ak_cache(_stock_us_daily, symbol=symbol, start_date=start_date, period=period, ttl=3600)
        if dfs is not None and not dfs.empty:
            dfs.attrs['source'] = 'akshare'
            return dfs
    except Exception as e:
        _LOGGER.warning(f"[{label}] akshare 获取失败 {symbol}: {e}")

    # 2. yfinance
    try:
        yf_symbol = f"{symbol.lstrip('0').zfill(4)}.HK" if market == "hk" else symbol.upper()
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}" if len(start_date) == 8 else start_date
        df = yf.download(yf_symbol, start=start_dt, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.rename(columns={"Date": "日期", "Open": "开盘", "Close": "收盘", "High": "最高", "Low": "最低", "Volume": "成交量"}, inplace=True)
            df["换手率"] = None
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
            df.attrs['source'] = 'yfinance'
            return df
    except Exception as e:
        _LOGGER.warning(f"[{label}] yfinance 获取失败 {symbol}: {e}")

    # 3. Alpha Vantage（仅美股）
    if market == "us" and ALPHA_VANTAGE_API_KEY:
        try:
            from ..data_provider import AlphaVantageFetcher
            av = AlphaVantageFetcher()
            df = av._fetch_raw_data(symbol, start_date, datetime.now().strftime("%Y%m%d"))
            if df is not None and not df.empty:
                df = av._normalize_data(df, symbol)
                df.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}, inplace=True)
                df["换手率"] = None
                df.attrs['source'] = 'alphavantage'
                return df
        except Exception as e:
            _LOGGER.warning(f"[美股] Alpha Vantage 获取失败 {symbol}: {e}")

    return None


def _stock_us_daily(symbol, start_date="2025-01-01", period="daily"):
    """获取美股日线数据"""
    import akshare as ak
    dfs = ak.stock_us_daily(symbol=symbol)
    if dfs is None or dfs.empty:
        return None
    dfs.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}, inplace=True)
    dfs["换手率"] = None
    dfs.index = pd.to_datetime(dfs["日期"], errors="coerce")
    return dfs.loc[start_date:]


# ==================== 美股历史价格 ====================

@mcp.tool(
    title="获取美股/港股历史价格",
    description="获取美股或港股的历史价格数据及技术指标。支持多数据源自动故障转移。",
)
def stock_prices_global(
    symbol: str = field_symbol,
    market: str = Field("us", description="市场: us(美股), hk(港股)"),
    period: str = Field("daily", description="周期: daily(日线), weekly(周线)"),
    limit: int = Field(30, description="返回数量(int)", strict=False),
):
    if market not in ("us", "hk"):
        return f"不支持的市场类型: {market}，仅支持 us(美股) 和 hk(港股)"

    if period == "weekly":
        delta = {"weeks": limit + 62}
    else:
        delta = {"days": limit + 62}
    start_date = (datetime.now() - timedelta(**delta)).strftime("%Y%m%d")

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


# ==================== 美股公司概览 ====================

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


# ==================== 美股财务报表 ====================

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

        period_type = "季度" if quarterly else "年度"
        lines = [f"# {symbol} {title} ({period_type})"]

        for i, report in enumerate(data["reports"][:4]):
            fiscal_date = report.get("fiscalDateEnding", "-")
            lines.append(f"# {fiscal_date}")

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
            else:
                key_fields = [
                    ("operatingCashflow", "经营现金流"),
                    ("capitalExpenditures", "资本支出"),
                    ("dividendPayout", "股息支出"),
                    ("netIncome", "净利润"),
                ]

            header = [label for _, label in key_fields]
            values = []
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
                values.append(str(value))
            lines.append(",".join(header))
            lines.append(",".join(values))

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股财务报表失败: {e}")
        return f"获取 {symbol} 财务报表失败: {e}"


# ==================== 美股新闻情绪 ====================

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


# ==================== 美股盈利数据 ====================

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

        lines = [f"# {symbol} 盈利数据"]

        annual = data.get("annualEarnings", [])
        if annual:
            lines.append("# 年度盈利")
            lines.append("年度,EPS($)")
            for item in annual[:5]:
                year = item.get("fiscalDateEnding", "-")
                eps = item.get("reportedEPS", "-")
                lines.append(f"{year},{eps}")

        quarterly = data.get("quarterlyEarnings", [])
        if quarterly:
            lines.append("# 季度盈利")
            lines.append("日期,实际EPS($),预期EPS($),惊喜(%)")
            for item in quarterly[:8]:
                date = item.get("fiscalDateEnding", "-")
                reported = item.get("reportedEPS", "-")
                estimated = item.get("estimatedEPS", "-")
                surprise = item.get("surprisePercentage", "-")
                lines.append(f"{date},{reported},{estimated},{surprise}")

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股盈利数据失败: {e}")
        return f"获取 {symbol} 盈利数据失败: {e}"


# ==================== 美股内部交易 ====================

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

        lines = [f"# {symbol} 内部交易记录"]

        insider_header = ["日期", "内部人", "职位", "类型", "股数", "金额"]
        insider_rows = []

        for item in transactions[:limit]:
            date = item.get("transaction_date", "-")
            owner = item.get("owner_name", "-")
            position = item.get("owner_title", "-")
            trans_type = item.get("acquisition_or_disposition", "-")
            shares = item.get("shares", "-")
            value = item.get("transaction_value", "-")

            type_label = "买入" if trans_type == "A" else "卖出" if trans_type == "D" else trans_type

            if value and value != "-":
                try:
                    value_num = float(value)
                    if value_num >= 1e6:
                        value = f"${value_num/1e6:.2f}M"
                    else:
                        value = f"${value_num:,.0f}"
                except (ValueError, TypeError):
                    pass
            insider_rows.append([str(date), str(owner), str(position), str(type_label), str(shares), str(value)])

        lines.append(",".join(insider_header))
        lines.extend([",".join(row) for row in insider_rows])

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股内部交易失败: {e}")
        return f"获取 {symbol} 内部交易失败: {e}"


# ==================== 美股技术指标 ====================

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

        lines = [
            f"# {symbol} {indicator.upper()} 技术指标",
            f"# 时间间隔: {interval}",
            f"# 计算周期: {time_period}",
        ]

        indicator_upper = indicator.upper()
        if indicator_upper == "MACD":
            header = ["日期", "MACD", "Signal", "Histogram"]
        elif indicator_upper == "BBANDS":
            header = ["日期", "Upper", "Middle", "Lower"]
        elif indicator_upper == "STOCH":
            header = ["日期", "SlowK", "SlowD"]
        else:
            header = ["日期", indicator_upper]

        rows = []
        for entry in data["data"][:limit]:
            date = entry.get("date", "-")

            if indicator_upper == "MACD":
                macd_val = entry.get("MACD", "-")
                signal = entry.get("MACD_Signal", "-")
                hist = entry.get("MACD_Hist", "-")
                rows.append([str(date), str(macd_val), str(signal), str(hist)])
            elif indicator_upper == "BBANDS":
                upper = entry.get("Real Upper Band", "-")
                middle = entry.get("Real Middle Band", "-")
                lower = entry.get("Real Lower Band", "-")
                rows.append([str(date), str(upper), str(middle), str(lower)])
            elif indicator_upper == "STOCH":
                slowk = entry.get("SlowK", "-")
                slowd = entry.get("SlowD", "-")
                rows.append([str(date), str(slowk), str(slowd)])
            else:
                value = entry.get(indicator_upper, "-")
                rows.append([str(date), str(value)])

        lines.append(",".join(header))
        lines.extend([",".join(row) for row in rows])

        return "\n".join(lines)
    except Exception as e:
        _LOGGER.warning(f"获取美股技术指标失败: {e}")
        return f"获取 {symbol} {indicator} 指标失败: {e}"
