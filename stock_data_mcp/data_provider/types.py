"""
统一数据类型定义和熔断器
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)


def safe_float(val, default: Optional[float] = None) -> Optional[float]:
    """安全转换为浮点数"""
    if val is None:
        return default
    if isinstance(val, float):
        if np.isnan(val):
            return default
        return val
    if isinstance(val, (int, np.integer)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if not val or val in ('--', '-', 'nan', 'NaN', 'None'):
            return default
        try:
            return float(val.replace(',', ''))
        except ValueError:
            return default
    try:
        result = float(val)
        if np.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_int(val, default: Optional[int] = None) -> Optional[int]:
    """安全转换为整数"""
    result = safe_float(val, None)
    if result is None:
        return default
    return int(result)


class RealtimeSource(Enum):
    """实时行情数据源"""
    TUSHARE = "tushare"
    EFINANCE = "efinance"
    AKSHARE_EM = "akshare_em"
    AKSHARE_SINA = "akshare_sina"
    AKSHARE_QQ = "akshare_qq"
    TENCENT = "tencent"
    SINA = "sina"
    YFINANCE = "yfinance"
    FALLBACK = "fallback"


@dataclass
class UnifiedRealtimeQuote:
    """统一实时行情数据结构"""
    # 核心标识
    code: str
    name: Optional[str] = None
    source: RealtimeSource = RealtimeSource.FALLBACK

    # 核心价格数据
    price: Optional[float] = None
    change_pct: Optional[float] = None
    change_amount: Optional[float] = None

    # 成交数据
    volume: Optional[float] = None
    amount: Optional[float] = None
    volume_ratio: Optional[float] = None
    turnover_rate: Optional[float] = None
    amplitude: Optional[float] = None

    # 价格区间
    open_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    pre_close: Optional[float] = None

    # 估值数据
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    total_mv: Optional[float] = None
    circ_mv: Optional[float] = None

    # 52周数据
    change_60d: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'source': self.source.value if self.source else None,
            'price': self.price,
            'change_pct': self.change_pct,
            'change_amount': self.change_amount,
            'volume': self.volume,
            'amount': self.amount,
            'volume_ratio': self.volume_ratio,
            'turnover_rate': self.turnover_rate,
            'amplitude': self.amplitude,
            'open_price': self.open_price,
            'high': self.high,
            'low': self.low,
            'pre_close': self.pre_close,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'total_mv': self.total_mv,
            'circ_mv': self.circ_mv,
        }

    def has_basic_data(self) -> bool:
        """是否有基本数据"""
        return self.price is not None and self.change_pct is not None

    def has_volume_data(self) -> bool:
        """是否有成交量数据"""
        return self.volume is not None and self.amount is not None


@dataclass
class ChipDistribution:
    """筹码分布数据结构"""
    code: str
    date: Optional[str] = None
    source: str = "akshare"

    # 获利比例
    profit_ratio: Optional[float] = None
    # 平均成本
    avg_cost: Optional[float] = None

    # 90%成本区间
    cost_90_low: Optional[float] = None
    cost_90_high: Optional[float] = None
    concentration_90: Optional[float] = None

    # 70%成本区间
    cost_70_low: Optional[float] = None
    cost_70_high: Optional[float] = None
    concentration_70: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'date': self.date,
            'source': self.source,
            'profit_ratio': self.profit_ratio,
            'avg_cost': self.avg_cost,
            'cost_90_low': self.cost_90_low,
            'cost_90_high': self.cost_90_high,
            'concentration_90': self.concentration_90,
            'cost_70_low': self.cost_70_low,
            'cost_70_high': self.cost_70_high,
            'concentration_70': self.concentration_70,
        }

    def get_chip_status(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """获取筹码状态分析"""
        status = {
            'profit_ratio': self.profit_ratio,
            'avg_cost': self.avg_cost,
            'concentration_90': self.concentration_90,
            'concentration_70': self.concentration_70,
        }

        if current_price and self.avg_cost:
            status['price_vs_avg_cost'] = (current_price - self.avg_cost) / self.avg_cost * 100

        # 集中度判断
        if self.concentration_90:
            if self.concentration_90 < 10:
                status['chip_level'] = '高度集中'
            elif self.concentration_90 < 20:
                status['chip_level'] = '相对集中'
            elif self.concentration_90 < 30:
                status['chip_level'] = '中等分散'
            else:
                status['chip_level'] = '高度分散'

        return status


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断
    HALF_OPEN = "half_open" # 恢复中


@dataclass
class SourceState:
    """数据源状态"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    last_error: Optional[str] = None


class CircuitBreaker:
    """熔断器实现"""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 300.0,
        half_open_max_calls: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        self._states: Dict[str, SourceState] = {}

    def _get_state(self, source: str) -> SourceState:
        """获取或创建数据源状态"""
        if source not in self._states:
            self._states[source] = SourceState()
        return self._states[source]

    def is_available(self, source: str) -> bool:
        """检查数据源是否可用"""
        state = self._get_state(source)

        if state.state == CircuitBreakerState.CLOSED:
            return True

        if state.state == CircuitBreakerState.OPEN:
            # 检查是否可以进入半开状态
            if time.time() - state.last_failure_time >= self.cooldown_seconds:
                state.state = CircuitBreakerState.HALF_OPEN
                state.half_open_calls = 0
                _LOGGER.info(f"数据源 {source} 进入恢复状态")
                return True
            return False

        if state.state == CircuitBreakerState.HALF_OPEN:
            return state.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self, source: str):
        """记录成功"""
        state = self._get_state(source)

        if state.state == CircuitBreakerState.HALF_OPEN:
            state.state = CircuitBreakerState.CLOSED
            state.failure_count = 0
            state.last_error = None
            _LOGGER.info(f"数据源 {source} 恢复正常")
        elif state.state == CircuitBreakerState.CLOSED:
            state.failure_count = 0

    def record_failure(self, source: str, error: Optional[str] = None):
        """记录失败"""
        state = self._get_state(source)
        state.failure_count += 1
        state.last_failure_time = time.time()
        state.last_error = error

        if state.state == CircuitBreakerState.HALF_OPEN:
            state.half_open_calls += 1
            if state.half_open_calls >= self.half_open_max_calls:
                state.state = CircuitBreakerState.OPEN
                _LOGGER.warning(f"数据源 {source} 恢复失败，重新熔断")
        elif state.state == CircuitBreakerState.CLOSED:
            if state.failure_count >= self.failure_threshold:
                state.state = CircuitBreakerState.OPEN
                _LOGGER.warning(f"数据源 {source} 触发熔断，错误: {error}")

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有数据源状态"""
        return {
            source: {
                'state': state.state.value,
                'failure_count': state.failure_count,
                'last_error': state.last_error,
            }
            for source, state in self._states.items()
        }

    def reset(self, source: Optional[str] = None):
        """重置数据源状态"""
        if source:
            if source in self._states:
                self._states[source] = SourceState()
        else:
            self._states.clear()


# 全局熔断器实例
_circuit_breaker_lock = threading.Lock()
_realtime_circuit_breaker: Optional[CircuitBreaker] = None
_chip_circuit_breaker: Optional[CircuitBreaker] = None
_daily_circuit_breaker: Optional[CircuitBreaker] = None
_fund_flow_circuit_breaker: Optional[CircuitBreaker] = None
_board_circuit_breaker: Optional[CircuitBreaker] = None
_billboard_circuit_breaker: Optional[CircuitBreaker] = None
_us_financials_circuit_breaker: Optional[CircuitBreaker] = None


def _get_or_create_circuit_breaker(
    attr_name: str,
    failure_threshold: int = 3,
    cooldown_seconds: float = 300.0
) -> CircuitBreaker:
    """线程安全地获取或创建熔断器实例"""
    current = globals().get(attr_name)
    if current is None:
        with _circuit_breaker_lock:
            current = globals().get(attr_name)
            if current is None:
                current = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    cooldown_seconds=cooldown_seconds,
                )
                globals()[attr_name] = current
    return current


def get_realtime_circuit_breaker() -> CircuitBreaker:
    """获取实时行情熔断器（5分钟冷却）"""
    return _get_or_create_circuit_breaker("_realtime_circuit_breaker", 3, 300.0)


def get_chip_circuit_breaker() -> CircuitBreaker:
    """获取筹码分布熔断器（10分钟冷却）"""
    return _get_or_create_circuit_breaker("_chip_circuit_breaker", 2, 600.0)


def get_daily_circuit_breaker() -> CircuitBreaker:
    """获取日线数据熔断器（5分钟冷却）"""
    return _get_or_create_circuit_breaker("_daily_circuit_breaker", 3, 300.0)


def get_fund_flow_circuit_breaker() -> CircuitBreaker:
    """获取资金流向熔断器（5分钟冷却）"""
    return _get_or_create_circuit_breaker("_fund_flow_circuit_breaker", 3, 300.0)


def get_board_circuit_breaker() -> CircuitBreaker:
    """获取板块数据熔断器（10分钟冷却）"""
    return _get_or_create_circuit_breaker("_board_circuit_breaker", 3, 600.0)


def get_billboard_circuit_breaker() -> CircuitBreaker:
    """获取龙虎榜熔断器（5分钟冷却）"""
    return _get_or_create_circuit_breaker("_billboard_circuit_breaker", 3, 300.0)


def get_us_financials_circuit_breaker() -> CircuitBreaker:
    """获取美股基本面熔断器（10分钟冷却，适应 API 限流）"""
    return _get_or_create_circuit_breaker("_us_financials_circuit_breaker", 3, 600.0)


# 标准列名定义
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

# 英文列名到中文列名的映射
COLUMN_MAPPING_TO_CN = {
    'date': '日期',
    'open': '开盘',
    'high': '最高',
    'low': '最低',
    'close': '收盘',
    'volume': '成交量',
    'amount': '成交额',
    'pct_chg': '涨跌幅',
    'turnover_rate': '换手率',
    'pe_ratio': '市盈率',
    'pb_ratio': '市净率',
    'total_mv': '总市值',
    'circ_mv': '流通市值',
    'volume_ratio': '量比',
    'amplitude': '振幅',
}

# 中文列名到英文列名的映射
COLUMN_MAPPING_TO_EN = {v: k for k, v in COLUMN_MAPPING_TO_CN.items()}


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 的英文列名转换为中文"""
    return df.rename(columns=COLUMN_MAPPING_TO_CN)


def to_english_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 的中文列名转换为英文"""
    return df.rename(columns=COLUMN_MAPPING_TO_EN)


def is_etf_code(stock_code: str) -> bool:
    """判断是否为 ETF 代码"""
    code = stock_code.lstrip('0')
    if len(code) == 6:
        prefix = code[:2]
        # 上交所 ETF: 51, 52, 56, 58
        # 深交所 ETF: 15, 16, 18
        return prefix in ('51', '52', '56', '58', '15', '16', '18')
    return False


def is_hk_code(stock_code: str) -> bool:
    """判断是否为港股代码"""
    code = stock_code.lower()
    if code.startswith('hk'):
        return True
    # 港股代码：5位或更少的纯数字（考虑前导零）
    if code.isdigit() and len(code) <= 5:
        return True
    return False


def is_us_code(stock_code: str) -> bool:
    """判断是否为美股代码"""
    # 美股代码通常是1-5个大写字母，可能带有后缀如 .O .N
    code = stock_code.upper().split('.')[0]
    return len(code) <= 5 and code.isalpha()


def is_a_stock_code(stock_code: str) -> bool:
    """判断是否为A股代码（含ETF）"""
    code = stock_code.lstrip('0')
    if len(code) != 6 or not code.isdigit():
        return False
    prefix = code[:2]
    # 上交所: 60, 68 (科创板)
    # 深交所: 00, 30 (创业板)
    # ETF: 51, 52, 56, 58, 15, 16, 18
    # 可转债: 11, 12
    a_stock_prefixes = ('60', '68', '00', '30', '51', '52', '56', '58', '15', '16', '18', '11', '12')
    return prefix in a_stock_prefixes


class StockType(Enum):
    """股票类型枚举"""
    A_STOCK = "a"       # A股个股
    ETF = "etf"         # ETF基金
    HK = "hk"           # 港股
    US = "us"           # 美股
    UNKNOWN = "unknown"


def detect_stock_type(stock_code: str) -> StockType:
    """自动检测股票类型"""
    if is_hk_code(stock_code):
        return StockType.HK
    if is_us_code(stock_code):
        return StockType.US
    if is_etf_code(stock_code):
        return StockType.ETF
    if is_a_stock_code(stock_code):
        return StockType.A_STOCK
    return StockType.UNKNOWN


def validate_stock_type(stock_code: str, user_market: str) -> tuple[StockType, str]:
    """
    综合校验股票类型：结合用户传入的 market 和自动检测结果

    Args:
        stock_code: 股票代码
        user_market: 用户传入的市场类型 (sh/sz/hk/us)

    Returns:
        (StockType, market_str): 最终确定的股票类型和市场字符串
    """
    detected_type = detect_stock_type(stock_code)

    # 将用户市场映射到 StockType
    market_to_type = {
        'sh': StockType.A_STOCK,
        'sz': StockType.A_STOCK,
        'hk': StockType.HK,
        'us': StockType.US,
    }
    user_type = market_to_type.get(user_market.lower(), StockType.UNKNOWN)

    # A股市场需要区分 ETF 和个股
    if user_type == StockType.A_STOCK and detected_type == StockType.ETF:
        # 用户指定 A股市场，检测为 ETF，这是合理的（ETF 也在 A股市场交易）
        final_type = StockType.ETF
        final_market = user_market
    elif detected_type == StockType.UNKNOWN:
        # 无法自动检测时，信任用户输入
        _LOGGER.warning(
            f"无法自动检测股票类型: code={stock_code}, 使用用户指定的 market={user_market}"
        )
        final_type = user_type
        final_market = user_market
    elif user_type != detected_type and user_type != StockType.UNKNOWN:
        # 用户指定的市场与检测结果不一致
        _LOGGER.warning(
            f"股票类型不一致: code={stock_code}, user_market={user_market}({user_type.value}), "
            f"detected={detected_type.value}, 优先使用检测结果"
        )
        final_type = detected_type
        # 根据检测结果修正市场
        if detected_type == StockType.HK:
            final_market = 'hk'
        elif detected_type == StockType.US:
            final_market = 'us'
        elif detected_type in (StockType.A_STOCK, StockType.ETF):
            # 保持用户的 sh/sz 选择，或默认 sh
            final_market = user_market if user_market in ('sh', 'sz') else 'sh'
        else:
            final_market = user_market
    else:
        final_type = detected_type
        final_market = user_market

    return final_type, final_market
