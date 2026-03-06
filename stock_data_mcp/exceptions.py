"""
股票数据 MCP 异常类定义

提供细粒度的异常分类，便于针对性处理和重试。
"""

from typing import Optional


class StockDataError(Exception):
    """股票数据错误基类"""

    def __init__(self, message: str, code: Optional[str] = None, source: Optional[str] = None):
        self.message = message
        self.code = code  # 股票代码
        self.source = source  # 数据源名称
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[代码: {self.code}]")
        if self.source:
            parts.append(f"[数据源: {self.source}]")
        return " ".join(parts)


# ==================== 数据获取错误 ====================

class DataFetchError(StockDataError):
    """数据获取错误基类"""
    pass


class NetworkError(DataFetchError):
    """网络连接错误（可重试）"""
    pass


class RequestTimeoutError(DataFetchError):
    """请求超时错误（可重试）"""
    pass


class RateLimitError(DataFetchError):
    """API 限流错误（需等待后重试）"""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        source: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        self.retry_after = retry_after  # 建议等待秒数
        super().__init__(message, code, source)


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用错误（需切换数据源）"""
    pass


class AuthenticationError(DataFetchError):
    """认证错误（API Key 无效等）"""
    pass


# ==================== 数据验证错误 ====================

class DataValidationError(StockDataError):
    """数据验证错误基类"""
    pass


class InvalidSymbolError(DataValidationError):
    """无效的股票代码"""
    pass


class InvalidMarketError(DataValidationError):
    """无效的市场类型"""
    pass


class InvalidDateRangeError(DataValidationError):
    """无效的日期范围"""
    pass


class EmptyDataError(DataValidationError):
    """返回数据为空"""
    pass


class DataParseError(DataValidationError):
    """数据解析错误"""
    pass


# ==================== 业务逻辑错误 ====================

class BusinessError(StockDataError):
    """业务逻辑错误基类"""
    pass


class MarketClosedError(BusinessError):
    """市场已关闭"""
    pass


class SymbolNotFoundError(BusinessError):
    """股票代码不存在"""
    pass


class FeatureNotSupportedError(BusinessError):
    """功能不支持"""
    pass


# ==================== 异常判断辅助函数 ====================

def is_retryable(error: Exception) -> bool:
    """
    判断异常是否可重试

    Args:
        error: 异常实例

    Returns:
        True 如果可以重试，False 否则
    """
    retryable_types = (
        NetworkError,
        RequestTimeoutError,
        RateLimitError,
        DataSourceUnavailableError,
        ConnectionError,
        OSError,
    )
    return isinstance(error, retryable_types)


def should_switch_source(error: Exception) -> bool:
    """
    判断是否应该切换数据源

    Args:
        error: 异常实例

    Returns:
        True 如果应该切换数据源，False 否则
    """
    switch_types = (
        DataSourceUnavailableError,
        AuthenticationError,
        RateLimitError,
        EmptyDataError,
    )
    return isinstance(error, switch_types)


def get_retry_delay(error: Exception, attempt: int = 1) -> float:
    """
    获取建议的重试延迟时间

    Args:
        error: 异常实例
        attempt: 当前尝试次数（从1开始）

    Returns:
        建议的延迟秒数
    """
    if isinstance(error, RateLimitError) and error.retry_after:
        return float(error.retry_after)

    # 指数退避：1s, 2s, 4s, 8s...
    base_delay = 1.0
    max_delay = 30.0
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

    return delay


def classify_exception(error: Exception, source: str = None, code: str = None) -> StockDataError:
    """
    将通用异常转换为特定的 StockDataError 子类

    Args:
        error: 原始异常
        source: 数据源名称
        code: 股票代码

    Returns:
        对应的 StockDataError 子类实例
    """
    import requests

    error_msg = str(error)

    # 已经是 StockDataError，直接返回
    if isinstance(error, StockDataError):
        return error

    # 网络相关错误
    if isinstance(error, (ConnectionError, requests.exceptions.ConnectionError)):
        return NetworkError(f"网络连接失败: {error_msg}", code=code, source=source)

    if isinstance(error, (RequestTimeoutError, requests.exceptions.Timeout)):
        return NetworkError(f"请求超时: {error_msg}", code=code, source=source)

    if isinstance(error, requests.exceptions.RequestException):
        return NetworkError(f"请求失败: {error_msg}", code=code, source=source)

    # 数据解析错误
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return DataParseError(f"数据解析失败: {error_msg}", code=code, source=source)

    if isinstance(error, (IndexError,)):
        return EmptyDataError(f"数据为空或索引越界: {error_msg}", code=code, source=source)

    # 限流错误（通过错误消息判断）
    if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
        return RateLimitError(f"API限流: {error_msg}", code=code, source=source)

    if "unauthorized" in error_msg.lower() or "invalid api" in error_msg.lower():
        return AuthenticationError(f"认证失败: {error_msg}", code=code, source=source)

    # 默认返回通用数据获取错误
    return DataFetchError(error_msg, code=code, source=source)


def get_error_category(error: Exception) -> str:
    """
    获取异常的分类标签，用于日志和监控

    Args:
        error: 异常实例

    Returns:
        分类标签字符串
    """
    if isinstance(error, NetworkError):
        return "NETWORK"
    if isinstance(error, RateLimitError):
        return "RATE_LIMIT"
    if isinstance(error, AuthenticationError):
        return "AUTH"
    if isinstance(error, DataParseError):
        return "PARSE"
    if isinstance(error, EmptyDataError):
        return "EMPTY"
    if isinstance(error, InvalidSymbolError):
        return "INVALID_SYMBOL"
    if isinstance(error, DataSourceUnavailableError):
        return "SOURCE_DOWN"
    if isinstance(error, StockDataError):
        return "DATA_ERROR"
    return "UNKNOWN"
