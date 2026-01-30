"""
统一数据类型定义和熔断器
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
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
    EFINANCE = "efinance"
    AKSHARE_EM = "akshare_em"
    AKSHARE_SINA = "akshare_sina"
    AKSHARE_QQ = "akshare_qq"
    TENCENT = "tencent"
    SINA = "sina"
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
_realtime_circuit_breaker: Optional[CircuitBreaker] = None
_chip_circuit_breaker: Optional[CircuitBreaker] = None
_daily_circuit_breaker: Optional[CircuitBreaker] = None


def get_realtime_circuit_breaker() -> CircuitBreaker:
    """获取实时行情熔断器（5分钟冷却）"""
    global _realtime_circuit_breaker
    if _realtime_circuit_breaker is None:
        _realtime_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=300.0
        )
    return _realtime_circuit_breaker


def get_chip_circuit_breaker() -> CircuitBreaker:
    """获取筹码分布熔断器（10分钟冷却）"""
    global _chip_circuit_breaker
    if _chip_circuit_breaker is None:
        _chip_circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=600.0
        )
    return _chip_circuit_breaker


def get_daily_circuit_breaker() -> CircuitBreaker:
    """获取日线数据熔断器（5分钟冷却）"""
    global _daily_circuit_breaker
    if _daily_circuit_breaker is None:
        _daily_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=300.0
        )
    return _daily_circuit_breaker


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
