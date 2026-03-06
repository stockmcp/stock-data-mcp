"""
市场交易时间配置
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time
from typing import List, Tuple, Optional


@dataclass
class TradingHours:
    """交易时间段"""
    name: str              # 时间段名称，如 "上午"
    start: time            # 开盘时间
    end: time              # 闭盘时间


class MarketType(Enum):
    """市场类型"""
    A_STOCK = "a_stock"    # A股
    HK_STOCK = "hk"        # 港股
    US_STOCK = "us"        # 美股
    CRYPTO = "crypto"      # 加密货币


class MarketHoursConfig:
    """市场交易时间配置"""

    # A股交易时间：9:30-11:30, 13:00-15:00
    A_STOCK_HOURS = [
        TradingHours("上午", time(9, 30), time(11, 30)),
        TradingHours("下午", time(13, 0), time(15, 0)),
    ]

    # 港股交易时间：9:30-12:00, 13:00-16:00
    HK_STOCK_HOURS = [
        TradingHours("上午", time(9, 30), time(12, 0)),
        TradingHours("下午", time(13, 0), time(16, 0)),
    ]

    # 美股交易时间：9:30-16:00 (EST，注意夏令时)
    US_STOCK_HOURS = [
        TradingHours("常规", time(9, 30), time(16, 0)),
    ]

    # 加密货币：24/7 交易
    CRYPTO_HOURS = [
        TradingHours("全天", time(0, 0), time(23, 59)),
    ]

    # 市场配置映射
    _MARKET_CONFIG = {
        MarketType.A_STOCK: A_STOCK_HOURS,
        MarketType.HK_STOCK: HK_STOCK_HOURS,
        MarketType.US_STOCK: US_STOCK_HOURS,
        MarketType.CRYPTO: CRYPTO_HOURS,
    }

    @classmethod
    def get_trading_hours(cls, market: MarketType) -> List[TradingHours]:
        """获取指定市场的交易时间段"""
        return cls._MARKET_CONFIG.get(market, [])

    @classmethod
    def is_trading_time(
        cls,
        market: MarketType,
        dt: Optional[datetime] = None
    ) -> bool:
        """判断指定时间是否在交易时间内"""
        if dt is None:
            dt = datetime.now()

        current_time = dt.time()
        trading_hours = cls.get_trading_hours(market)

        for hours in trading_hours:
            if hours.start <= current_time < hours.end:
                return True
        return False

    @classmethod
    def get_time_to_next_trading(
        cls,
        market: MarketType,
        dt: Optional[datetime] = None
    ) -> Optional[float]:
        """
        获取距离下一个交易时间的秒数

        Returns:
            秒数，如果当前已在交易时间内则返回 None
        """
        if dt is None:
            dt = datetime.now()

        current_time = dt.time()
        trading_hours = cls.get_trading_hours(market)

        # 按时间顺序检查
        for hours in trading_hours:
            if hours.start <= current_time < hours.end:
                # 已在交易时间内
                return None

        # 查找下一个交易时间段
        for hours in trading_hours:
            if current_time < hours.start:
                # 今天的下一个交易时间
                next_time = datetime.combine(dt.date(), hours.start)
                return (next_time - dt).total_seconds()

        # 没有更多交易时间，返回明天第一个交易时间
        if trading_hours:
            first_hours = trading_hours[0]
            next_time = datetime.combine(
                dt.date(),
                first_hours.start
            )
            # 加上一天
            next_time = next_time.replace(day=next_time.day + 1)
            if next_time.day > 31:
                next_time = next_time.replace(day=1, month=next_time.month + 1)
            return (next_time - dt).total_seconds()

        return None

    @classmethod
    def get_default_cache_ttl(
        cls,
        market: MarketType,
        trading_cache_ttl: float = 60.0,
        non_trading_cache_ttl: float = 300.0,
    ) -> float:
        """
        获取默认的 TTL（按交易状态动态计算）

        Args:
            market: 市场类型
            trading_cache_ttl: 交易时间内的 TTL（秒）
            non_trading_cache_ttl: 非交易时间的 TTL（秒）

        Returns:
            推荐的 TTL 秒数
        """
        if cls.is_trading_time(market):
            return trading_cache_ttl
        return non_trading_cache_ttl

    @classmethod
    def detect_market_type(cls, market_str: str) -> Optional[MarketType]:
        """从市场字符串检测市场类型"""
        market_map = {
            "sh": MarketType.A_STOCK,
            "sz": MarketType.A_STOCK,
            "a_stock": MarketType.A_STOCK,
            "hk": MarketType.HK_STOCK,
            "hk_stock": MarketType.HK_STOCK,
            "us": MarketType.US_STOCK,
            "us_stock": MarketType.US_STOCK,
            "crypto": MarketType.CRYPTO,
        }
        return market_map.get(market_str.lower())
