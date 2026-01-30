"""
多数据源数据提供层

支持自动故障转移的多数据源股票数据获取：
- EfinanceFetcher (优先级 0): 东方财富数据
- AkshareFetcher (优先级 1): Akshare 多源数据
- TushareFetcher (优先级 0/2): Tushare Pro 数据（需要 token）
- BaostockFetcher (优先级 3): Baostock 免费数据
- YfinanceFetcher (优先级 4): Yahoo Finance 国际数据

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
    # 工具函数
    safe_float,
    safe_int,
    to_chinese_columns,
    to_english_columns,
    # 常量
    STANDARD_COLUMNS,
    COLUMN_MAPPING_TO_CN,
    COLUMN_MAPPING_TO_EN,
    # 熔断器
    get_realtime_circuit_breaker,
    get_chip_circuit_breaker,
    get_daily_circuit_breaker,
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
    # 数据类型
    "UnifiedRealtimeQuote",
    "ChipDistribution",
    "RealtimeSource",
    "CircuitBreaker",
    "CircuitBreakerState",
    # 异常
    "DataFetchError",
    "RateLimitError",
    "DataSourceUnavailableError",
    # 工具函数
    "safe_float",
    "safe_int",
    "to_chinese_columns",
    "to_english_columns",
    # 常量
    "STANDARD_COLUMNS",
    "COLUMN_MAPPING_TO_CN",
    "COLUMN_MAPPING_TO_EN",
]
