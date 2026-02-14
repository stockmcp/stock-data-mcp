"""
A股工具包

包含所有A股相关的MCP工具
"""

# 价格与行情模块
from .prices import (
    stock_prices,
    stock_realtime,
    stock_batch_realtime,
)

# 基本信息与搜索模块
from .info import (
    search,
    stock_info,
    stock_indicators,
    get_current_time,
)

# 市场资金流模块
from .market_flow import (
    stock_zt_pool,
    stock_lhb_ggtj_sina,
    stock_sector_fund_flow_rank,
    stock_north_flow,
    stock_margin_trading,
    stock_block_trade,
    stock_holder_num,
)

# 个股分析模块
from .analysis import (
    stock_chip,
    stock_period_stats,
    stock_fund_flow,
    stock_sector_spot,
    stock_board_cons,
)

# 估值与财务模块
from .valuation import (
    stock_valuation_compare,
    stock_market_pe_percentile,
    stock_industry_pe,
    stock_dividend_history,
    stock_institutional_holdings,
    stock_earnings_calendar,
    stock_financial_compare,
)

# 股东与风险模块
from .shareholders import (
    stock_locked_shares,
    stock_pledge_ratio,
    stock_top10_holders,
    portfolio_risk_analysis,
)

# 量化分析模块
from .quant import (
    stock_screener,
    trading_signals,
    backtest_strategy,
)

__all__ = [
    # 价格与行情
    "stock_prices",
    "stock_realtime",
    "stock_batch_realtime",
    # 基本信息与搜索
    "search",
    "stock_info",
    "stock_indicators",
    "get_current_time",
    # 市场资金流
    "stock_zt_pool",
    "stock_lhb_ggtj_sina",
    "stock_sector_fund_flow_rank",
    "stock_north_flow",
    "stock_margin_trading",
    "stock_block_trade",
    "stock_holder_num",
    # 个股分析
    "stock_chip",
    "stock_period_stats",
    "stock_fund_flow",
    "stock_sector_spot",
    "stock_board_cons",
    # 估值与财务
    "stock_valuation_compare",
    "stock_market_pe_percentile",
    "stock_industry_pe",
    "stock_dividend_history",
    "stock_institutional_holdings",
    "stock_earnings_calendar",
    "stock_financial_compare",
    # 股东与风险
    "stock_locked_shares",
    "stock_pledge_ratio",
    "stock_top10_holders",
    "portfolio_risk_analysis",
    # 量化分析
    "stock_screener",
    "trading_signals",
    "backtest_strategy",
]
