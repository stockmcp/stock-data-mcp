"""
多数据源数据提供层

支持自动故障转移的多数据源股票数据获取：
- TushareFetcher (优先级 0): Tushare Pro A 股数据（需要 token）
- EfinanceFetcher (优先级 1): 东方财富 A 股数据
- AkshareFetcher (优先级 2): Akshare 多市场数据
- BaostockFetcher (优先级 3): Baostock A 股免费数据
- AlphaVantageFetcher (优先级 4): Alpha Vantage 美股基本面和新闻（需要 API key）
- YfinanceFetcher (优先级 5): Yahoo Finance 全局后备

使用示例:
    from stock_data_mcp.data_provider import DataFetcherManager

    manager = DataFetcherManager()
    df = manager.get_daily_data("600519", days=30)
    quote = manager.get_realtime_quote("600519")
    chip = manager.get_chip_distribution("600519")
"""

from .types import (
    # 数据类型
    UnifiedRealtimeQuote,
    ChipDistribution,
    RealtimeSource,
    CircuitBreaker,
    CircuitBreakerState,
    StockType,
    # 工具函数
    safe_float,
    safe_int,
    to_chinese_columns,
    to_english_columns,
    is_etf_code,
    is_hk_code,
    is_us_code,
    is_a_stock_code,
    detect_stock_type,
    validate_stock_type,
    # 常量
    STANDARD_COLUMNS,
    COLUMN_MAPPING_TO_CN,
    COLUMN_MAPPING_TO_EN,
    # 熔断器
    get_realtime_circuit_breaker,
    get_chip_circuit_breaker,
    get_daily_circuit_breaker,
    get_fund_flow_circuit_breaker,
    get_board_circuit_breaker,
    get_billboard_circuit_breaker,
    get_us_financials_circuit_breaker,
)

from .base import (
    BaseFetcher,
    DataFetcherManager,
    DataFetchError,
    RateLimitError,
    DataSourceUnavailableError,
)

from .efinance_fetcher import EfinanceFetcher
from .akshare_fetcher import AkshareFetcher
from .tushare_fetcher import TushareFetcher
from .baostock_fetcher import BaostockFetcher
from .yfinance_fetcher import YfinanceFetcher
from .alphavantage_fetcher import AlphaVantageFetcher, AlphaVantageRateLimitError

__all__ = [
    # 管理器
    "DataFetcherManager",
    # 数据获取器
    "BaseFetcher",
    "EfinanceFetcher",
    "AkshareFetcher",
    "TushareFetcher",
    "BaostockFetcher",
    "YfinanceFetcher",
    "AlphaVantageFetcher",
    # 数据类型
    "UnifiedRealtimeQuote",
    "ChipDistribution",
    "RealtimeSource",
    "CircuitBreaker",
    "CircuitBreakerState",
    "StockType",
    # 异常
    "DataFetchError",
    "RateLimitError",
    "DataSourceUnavailableError",
    "AlphaVantageRateLimitError",
    # 熔断器
    "get_realtime_circuit_breaker",
    "get_chip_circuit_breaker",
    "get_daily_circuit_breaker",
    "get_fund_flow_circuit_breaker",
    "get_board_circuit_breaker",
    "get_billboard_circuit_breaker",
    "get_us_financials_circuit_breaker",
    # 工具函数
    "safe_float",
    "safe_int",
    "to_chinese_columns",
    "to_english_columns",
    "is_etf_code",
    "is_hk_code",
    "is_us_code",
    "is_a_stock_code",
    "detect_stock_type",
    "validate_stock_type",
    # 常量
    "STANDARD_COLUMNS",
    "COLUMN_MAPPING_TO_CN",
    "COLUMN_MAPPING_TO_EN",
]
