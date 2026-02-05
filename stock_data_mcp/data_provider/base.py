"""
数据获取基类和管理器
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import pandas as pd
import numpy as np

from .types import (
    STANDARD_COLUMNS,
    UnifiedRealtimeQuote,
    ChipDistribution,
    StockType,
    get_daily_circuit_breaker,
    get_realtime_circuit_breaker,
    get_chip_circuit_breaker,
    get_fund_flow_circuit_breaker,
    get_board_circuit_breaker,
    get_billboard_circuit_breaker,
    get_us_financials_circuit_breaker,
)

if TYPE_CHECKING:
    from .efinance_fetcher import EfinanceFetcher
    from .akshare_fetcher import AkshareFetcher
    from .tushare_fetcher import TushareFetcher
    from .baostock_fetcher import BaostockFetcher
    from .yfinance_fetcher import YfinanceFetcher

_LOGGER = logging.getLogger(__name__)


class DataFetchError(Exception):
    """数据获取错误"""
    pass


class RateLimitError(DataFetchError):
    """API 限流错误"""
    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用错误"""
    pass


class BaseFetcher(ABC):
    """数据获取器基类"""

    name: str = "BaseFetcher"
    priority: int = 99

    # User-Agent 池用于反爬
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(self):
        self._available = True

    @property
    def is_available(self) -> bool:
        """数据源是否可用"""
        return self._available

    def random_sleep(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """随机延迟，用于反爬"""
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_random_user_agent(self) -> str:
        """获取随机 User-Agent"""
        return random.choice(self.USER_AGENTS)

    @abstractmethod
    def _fetch_raw_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取原始数据（子类实现）"""
        pass

    @abstractmethod
    def _normalize_data(
        self,
        df: pd.DataFrame,
        stock_code: str
    ) -> pd.DataFrame:
        """标准化数据（子类实现）"""
        pass

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        获取日线数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            days: 获取天数（当 start_date 未指定时使用）

        Returns:
            标准化的 DataFrame，列名为 STANDARD_COLUMNS
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")

        try:
            df = self._fetch_raw_data(stock_code, start_date, end_date)
            if df is None or df.empty:
                return None

            df = self._normalize_data(df, stock_code)
            df = self._clean_data(df)
            df = self._calculate_indicators(df)

            return df

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 {stock_code} 数据失败: {e}")
            raise DataFetchError(f"获取数据失败: {e}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        if df is None or df.empty:
            return df

        # 确保有标准列
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        # 日期格式化
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # 数值类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除收盘价为空的行
        df = df.dropna(subset=['close'])

        # 按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=True)

        return df.reset_index(drop=True)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df is None or df.empty or 'close' not in df.columns:
            return df

        close = df['close']

        # 计算移动平均线
        df['MA5'] = close.rolling(window=5, min_periods=1).mean()
        df['MA10'] = close.rolling(window=10, min_periods=1).mean()
        df['MA20'] = close.rolling(window=20, min_periods=1).mean()

        # 计算成交量比率
        if 'volume' in df.columns:
            vol = df['volume']
            df['volume_ratio'] = vol / vol.rolling(window=5, min_periods=1).mean()

        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """获取实时行情（子类可覆盖）"""
        return None

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """获取筹码分布（子类可覆盖）"""
        return None

    def get_fund_flow(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取资金流向（子类可覆盖）"""
        return None

    def get_belong_board(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取所属板块（子类可覆盖）"""
        return None

    def get_board_cons(self, board_name: str, board_type: str = "industry") -> Optional[pd.DataFrame]:
        """获取板块成分股（子类可覆盖）"""
        return None

    def get_billboard(self, days: str = "5") -> Optional[pd.DataFrame]:
        """获取龙虎榜统计（子类可覆盖）"""
        return None


class DataFetcherManager:
    """数据获取管理器，支持多数据源自动故障转移"""

    def __init__(self, auto_init: bool = True):
        self._fetchers: List[BaseFetcher] = []
        self._realtime_cache: Dict[str, tuple[UnifiedRealtimeQuote, float]] = {}  # (quote, timestamp)
        self._realtime_cache_ttl: float = 60.0  # 1分钟缓存
        self._realtime_cache_maxsize: int = 500  # 最多缓存500只股票

        if auto_init:
            self._init_default_fetchers()

    def _init_default_fetchers(self):
        """初始化默认数据源"""
        try:
            from .efinance_fetcher import EfinanceFetcher
            self.add_fetcher(EfinanceFetcher())
        except Exception as e:
            _LOGGER.warning(f"EfinanceFetcher 初始化失败: {e}")

        try:
            from .akshare_fetcher import AkshareFetcher
            self.add_fetcher(AkshareFetcher())
        except Exception as e:
            _LOGGER.warning(f"AkshareFetcher 初始化失败: {e}")

        try:
            from .tushare_fetcher import TushareFetcher
            fetcher = TushareFetcher()
            if fetcher.is_available:
                self.add_fetcher(fetcher)
        except Exception as e:
            _LOGGER.warning(f"TushareFetcher 初始化失败: {e}")

        try:
            from .baostock_fetcher import BaostockFetcher
            self.add_fetcher(BaostockFetcher())
        except Exception as e:
            _LOGGER.warning(f"BaostockFetcher 初始化失败: {e}")

        try:
            from .yfinance_fetcher import YfinanceFetcher
            self.add_fetcher(YfinanceFetcher())
        except Exception as e:
            _LOGGER.warning(f"YfinanceFetcher 初始化失败: {e}")

        try:
            from .alphavantage_fetcher import AlphaVantageFetcher
            fetcher = AlphaVantageFetcher()
            if fetcher.is_available:
                self.add_fetcher(fetcher)
        except Exception as e:
            _LOGGER.warning(f"AlphaVantageFetcher 初始化失败: {e}")

        _LOGGER.info(f"已初始化 {len(self._fetchers)} 个数据源: {[f.name for f in self._fetchers]}")

    def add_fetcher(self, fetcher: BaseFetcher):
        """添加数据源并按优先级排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)

    def get_fetchers(self) -> List[BaseFetcher]:
        """获取所有数据源"""
        return self._fetchers.copy()

    def _get_fetchers_for_realtime(self, stock_type: Optional[StockType] = None) -> List[BaseFetcher]:
        """
        获取实时行情数据源列表（按实时行情优先级）

        Args:
            stock_type: 股票类型，用于优化数据源选择

        Returns:
            按优先级排序的 fetcher 列表
        """
        # 根据股票类型调整优先级
        if stock_type == StockType.HK:
            # 港股：Akshare > Yfinance (fallback)
            priority_order = {
                "AkshareFetcher": 0,
                "YfinanceFetcher": 1,
            }
        elif stock_type == StockType.ETF:
            # ETF：Akshare > Yfinance (fallback)，Efinance 不支持
            priority_order = {
                "AkshareFetcher": 0,
                "YfinanceFetcher": 1,
                "EfinanceFetcher": 10,  # 降低优先级，因为不支持 ETF
                "TushareFetcher": 10,
            }
        elif stock_type == StockType.US:
            # 美股：YFinance 支持实时行情
            priority_order = {
                "YfinanceFetcher": 0,
                "AlphaVantageFetcher": 1,
            }
        else:
            # A股个股：默认优先级
            priority_order = {
                "EfinanceFetcher": 0,   # 批量获取全市场，缓存后查询快
                "AkshareFetcher": 1,    # 支持多数据源
                "TushareFetcher": 2,    # 单个查询
            }
        return sorted(
            self._fetchers,
            key=lambda f: priority_order.get(f.name, f.priority + 10)
        )

    def _evict_realtime_cache(self):
        """清理过期和超量的实时行情缓存"""
        now = time.time()
        # 先清理过期条目
        expired = [k for k, (_, ts) in self._realtime_cache.items() if now - ts >= self._realtime_cache_ttl]
        for k in expired:
            del self._realtime_cache[k]
        # 如果仍超过上限，按时间戳淘汰最旧的
        if len(self._realtime_cache) > self._realtime_cache_maxsize:
            sorted_keys = sorted(self._realtime_cache, key=lambda k: self._realtime_cache[k][1])
            for k in sorted_keys[:len(self._realtime_cache) - self._realtime_cache_maxsize]:
                del self._realtime_cache[k]

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        获取日线数据，自动故障转移

        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            days: 获取天数

        Returns:
            标准化的 DataFrame
        """
        circuit_breaker = get_daily_circuit_breaker()
        last_error = None

        for fetcher in self._fetchers:
            source_name = fetcher.name

            # 检查熔断器
            if not circuit_breaker.is_available(source_name):
                _LOGGER.debug(f"[{source_name}] 熔断中，跳过")
                continue

            try:
                df = fetcher.get_daily_data(stock_code, start_date, end_date, days)
                if df is not None and not df.empty:
                    circuit_breaker.record_success(source_name)
                    _LOGGER.debug(f"[{source_name}] 成功获取 {stock_code} 数据")
                    return df
            except Exception as e:
                last_error = str(e)
                circuit_breaker.record_failure(source_name, last_error)
                _LOGGER.warning(f"[{source_name}] 获取 {stock_code} 失败: {e}")
                continue

        _LOGGER.error(f"所有数据源均无法获取 {stock_code} 数据")
        return None

    def get_realtime_quote(
        self,
        stock_code: str,
        stock_type: Optional[StockType] = None
    ) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情，自动故障转移

        Args:
            stock_code: 股票代码
            stock_type: 股票类型（可选），用于优化数据源选择

        Returns:
            UnifiedRealtimeQuote 或 None
        """
        # 检查缓存（每个股票独立的时间戳）
        if stock_code in self._realtime_cache:
            quote, cached_time = self._realtime_cache[stock_code]
            if time.time() - cached_time < self._realtime_cache_ttl:
                return quote

        circuit_breaker = get_realtime_circuit_breaker()

        # 使用实时行情专用优先级，根据股票类型优化
        for fetcher in self._get_fetchers_for_realtime(stock_type):
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                quote = fetcher.get_realtime_quote(stock_code)
                if quote is not None and quote.has_basic_data():
                    circuit_breaker.record_success(source_name)
                    self._realtime_cache[stock_code] = (quote, time.time())
                    self._evict_realtime_cache()
                    return quote
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {stock_code} 实时行情失败: {e}")
                continue

        return None

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """
        获取筹码分布，自动故障转移

        Args:
            stock_code: 股票代码

        Returns:
            ChipDistribution 或 None
        """
        circuit_breaker = get_chip_circuit_breaker()

        for fetcher in self._fetchers:
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                chip = fetcher.get_chip_distribution(stock_code)
                if chip is not None:
                    circuit_breaker.record_success(source_name)
                    return chip
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {stock_code} 筹码分布失败: {e}")
                continue

        return None

    def prefetch_realtime_quotes(
        self,
        stock_codes: List[str],
        stock_type: Optional[StockType] = None
    ) -> Dict[str, UnifiedRealtimeQuote]:
        """
        批量预取实时行情

        Args:
            stock_codes: 股票代码列表
            stock_type: 股票类型（可选），如果为None则自动检测

        Returns:
            股票代码 -> UnifiedRealtimeQuote 的映射
        """
        from .types import detect_stock_type

        result: Dict[str, UnifiedRealtimeQuote] = {}

        # 如果指定了 stock_type，使用该类型的优先数据源
        # 使用实时行情专用优先级，优先尝试支持批量获取的数据源
        for fetcher in self._get_fetchers_for_realtime(stock_type):
            if hasattr(fetcher, 'get_batch_realtime_quotes'):
                try:
                    batch_result = fetcher.get_batch_realtime_quotes(stock_codes)
                    if batch_result:
                        result.update(batch_result)
                        # 更新缓存（每个股票独立时间戳）
                        current_time = time.time()
                        for code, quote in batch_result.items():
                            self._realtime_cache[code] = (quote, current_time)
                        return result
                except Exception as e:
                    _LOGGER.warning(f"[{fetcher.name}] 批量获取实时行情失败: {e}")
                    continue

        # 回退到逐个获取，自动检测每个代码的类型
        for code in stock_codes:
            code_type = stock_type or detect_stock_type(code)
            quote = self.get_realtime_quote(code, stock_type=code_type)
            if quote:
                result[code] = quote

        return result

    def get_status(self) -> Dict[str, Any]:
        """获取数据源状态"""
        return {
            'fetchers': [
                {
                    'name': f.name,
                    'priority': f.priority,
                    'available': f.is_available,
                }
                for f in self._fetchers
            ],
            'daily_circuit_breaker': get_daily_circuit_breaker().get_status(),
            'realtime_circuit_breaker': get_realtime_circuit_breaker().get_status(),
            'chip_circuit_breaker': get_chip_circuit_breaker().get_status(),
            'fund_flow_circuit_breaker': get_fund_flow_circuit_breaker().get_status(),
            'board_circuit_breaker': get_board_circuit_breaker().get_status(),
            'billboard_circuit_breaker': get_billboard_circuit_breaker().get_status(),
            'us_financials_circuit_breaker': get_us_financials_circuit_breaker().get_status(),
        }

    def get_fund_flow(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取资金流向，自动故障转移

        Args:
            stock_code: 股票代码

        Returns:
            DataFrame 或 None，包含 source 属性标记来源
        """
        circuit_breaker = get_fund_flow_circuit_breaker()

        for fetcher in self._fetchers:
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                _LOGGER.debug(f"[{source_name}] 熔断中，跳过资金流向")
                continue

            try:
                df = fetcher.get_fund_flow(stock_code)
                if df is not None and not df.empty:
                    circuit_breaker.record_success(source_name)
                    df.attrs['source'] = source_name
                    _LOGGER.debug(f"[{source_name}] 成功获取 {stock_code} 资金流向")
                    return df
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {stock_code} 资金流向失败: {e}")
                continue

        _LOGGER.error(f"所有数据源均无法获取 {stock_code} 资金流向")
        return None

    def get_belong_board(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取所属板块，自动故障转移

        Args:
            stock_code: 股票代码

        Returns:
            DataFrame 或 None，包含 source 属性标记来源
        """
        circuit_breaker = get_board_circuit_breaker()

        for fetcher in self._fetchers:
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                _LOGGER.debug(f"[{source_name}] 熔断中，跳过所属板块")
                continue

            try:
                df = fetcher.get_belong_board(stock_code)
                if df is not None and not df.empty:
                    circuit_breaker.record_success(source_name)
                    df.attrs['source'] = source_name
                    _LOGGER.debug(f"[{source_name}] 成功获取 {stock_code} 所属板块")
                    return df
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {stock_code} 所属板块失败: {e}")
                continue

        _LOGGER.error(f"所有数据源均无法获取 {stock_code} 所属板块")
        return None

    def get_board_cons(self, board_name: str, board_type: str = "industry") -> Optional[pd.DataFrame]:
        """
        获取板块成分股，自动故障转移

        Args:
            board_name: 板块名称
            board_type: 板块类型 industry/concept

        Returns:
            DataFrame 或 None，包含 source 属性标记来源
        """
        circuit_breaker = get_board_circuit_breaker()

        for fetcher in self._fetchers:
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                _LOGGER.debug(f"[{source_name}] 熔断中，跳过板块成分股")
                continue

            try:
                df = fetcher.get_board_cons(board_name, board_type)
                if df is not None and not df.empty:
                    circuit_breaker.record_success(source_name)
                    df.attrs['source'] = source_name
                    _LOGGER.debug(f"[{source_name}] 成功获取 {board_name} 成分股")
                    return df
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {board_name} 成分股失败: {e}")
                continue

        _LOGGER.error(f"所有数据源均无法获取 {board_name} 成分股")
        return None

    def get_billboard(self, days: str = "5") -> Optional[pd.DataFrame]:
        """
        获取龙虎榜统计，自动故障转移

        Args:
            days: 统计天数 (5/10/30/60)

        Returns:
            DataFrame 或 None，包含 source 属性标记来源
        """
        circuit_breaker = get_billboard_circuit_breaker()

        for fetcher in self._fetchers:
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                _LOGGER.debug(f"[{source_name}] 熔断中，跳过龙虎榜")
                continue

            try:
                df = fetcher.get_billboard(days)
                if df is not None and not df.empty:
                    circuit_breaker.record_success(source_name)
                    df.attrs['source'] = source_name
                    _LOGGER.debug(f"[{source_name}] 成功获取龙虎榜数据")
                    return df
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取龙虎榜失败: {e}")
                continue

        _LOGGER.error(f"所有数据源均无法获取龙虎榜数据")
        return None

    # ==================== 美股多数据源方法 ====================

    def _get_us_fetcher(self, fetcher_name: str):
        """获取指定的美股数据源"""
        for fetcher in self._fetchers:
            if fetcher.name == fetcher_name and fetcher.is_available:
                return fetcher
        return None

    def _get_us_fetchers_for_financials(self) -> List[BaseFetcher]:
        """
        获取美股基本面数据源列表（按优先级）
        AlphaVantage (需要 API key) -> YfinanceFetcher (免费)
        """
        fetchers = []

        # AlphaVantage 优先（如果可用）
        av = self._get_us_fetcher("AlphaVantage")
        if av:
            fetchers.append(av)

        # YFinance 作为后备
        yf = self._get_us_fetcher("YfinanceFetcher")
        if yf:
            fetchers.append(yf)

        return fetchers

    def get_us_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取美股公司概览（多数据源自动故障转移）
        优先级: AlphaVantage -> yfinance
        """
        circuit_breaker = get_us_financials_circuit_breaker()

        for fetcher in self._get_us_fetchers_for_financials():
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                _LOGGER.debug(f"[{source_name}] 熔断中，跳过公司概览")
                continue

            try:
                result = fetcher.get_company_overview(symbol)
                if result is not None:
                    circuit_breaker.record_success(source_name)
                    result['_data_source'] = source_name
                    _LOGGER.debug(f"[{source_name}] 成功获取 {symbol} 公司概览")
                    return result
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {symbol} 公司概览失败: {e}")
                continue

        _LOGGER.error(f"所有数据源均无法获取 {symbol} 公司概览")
        return None

    def get_us_balance_sheet(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取美股资产负债表（多数据源自动故障转移）
        优先级: AlphaVantage -> yfinance
        """
        circuit_breaker = get_us_financials_circuit_breaker()

        for fetcher in self._get_us_fetchers_for_financials():
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                result = fetcher.get_balance_sheet(symbol, quarterly)
                if result is not None and result.get("reports"):
                    circuit_breaker.record_success(source_name)
                    result['_data_source'] = source_name
                    return result
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {symbol} 资产负债表失败: {e}")
                continue

        return None

    def get_us_income_statement(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取美股利润表（多数据源自动故障转移）
        优先级: AlphaVantage -> yfinance
        """
        circuit_breaker = get_us_financials_circuit_breaker()

        for fetcher in self._get_us_fetchers_for_financials():
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                result = fetcher.get_income_statement(symbol, quarterly)
                if result is not None and result.get("reports"):
                    circuit_breaker.record_success(source_name)
                    result['_data_source'] = source_name
                    return result
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {symbol} 利润表失败: {e}")
                continue

        return None

    def get_us_cash_flow(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取美股现金流量表（多数据源自动故障转移）
        优先级: AlphaVantage -> yfinance
        """
        circuit_breaker = get_us_financials_circuit_breaker()

        for fetcher in self._get_us_fetchers_for_financials():
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                result = fetcher.get_cash_flow(symbol, quarterly)
                if result is not None and result.get("reports"):
                    circuit_breaker.record_success(source_name)
                    result['_data_source'] = source_name
                    return result
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {symbol} 现金流量表失败: {e}")
                continue

        return None

    def get_us_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取美股盈利数据（多数据源自动故障转移）
        优先级: AlphaVantage -> yfinance
        """
        circuit_breaker = get_us_financials_circuit_breaker()

        for fetcher in self._get_us_fetchers_for_financials():
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                result = fetcher.get_earnings(symbol)
                if result is not None:
                    circuit_breaker.record_success(source_name)
                    result['_data_source'] = source_name
                    return result
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {symbol} 盈利数据失败: {e}")
                continue

        return None

    def get_us_news_sentiment(
        self,
        symbol: str = None,
        topics: str = None,
        limit: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        获取美股新闻情绪（通过 Alpha Vantage，无后备源）

        Args:
            symbol: 股票代码（可选）
            topics: 主题过滤（可选）
            limit: 返回数量限制

        Returns:
            新闻情绪数据
        """
        fetcher = self._get_us_fetcher("AlphaVantage")
        if fetcher is None:
            _LOGGER.warning("AlphaVantage 数据源未配置或不可用")
            return None

        try:
            return fetcher.get_news_sentiment(symbol, topics, limit)
        except Exception as e:
            _LOGGER.warning(f"获取美股新闻情绪失败: {e}")
            return None

    def get_us_insider_transactions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取美股内部交易（多数据源自动故障转移）
        优先级: AlphaVantage -> yfinance
        """
        circuit_breaker = get_us_financials_circuit_breaker()

        for fetcher in self._get_us_fetchers_for_financials():
            source_name = fetcher.name

            if not circuit_breaker.is_available(source_name):
                continue

            try:
                result = fetcher.get_insider_transactions(symbol)
                if result is not None and result.get("data"):
                    circuit_breaker.record_success(source_name)
                    result['_data_source'] = source_name
                    return result
            except Exception as e:
                circuit_breaker.record_failure(source_name, str(e))
                _LOGGER.warning(f"[{source_name}] 获取 {symbol} 内部交易失败: {e}")
                continue

        return None

    def get_us_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
    ) -> Optional[Dict[str, Any]]:
        """
        获取美股技术指标（通过 Alpha Vantage）

        Args:
            symbol: 美股代码
            indicator: 指标类型 (SMA, EMA, RSI, MACD, BBANDS, STOCH, ADX)
            interval: 时间间隔 (daily, weekly, monthly)
            time_period: 计算周期

        Returns:
            技术指标数据
        """
        fetcher = self._get_us_fetcher("AlphaVantage")
        if fetcher is None:
            _LOGGER.warning("AlphaVantage 数据源未配置或不可用")
            return None

        try:
            return fetcher.get_technical_indicator(symbol, indicator, interval, time_period)
        except Exception as e:
            _LOGGER.warning(f"获取美股技术指标失败: {e}")
            return None

    def format_us_overview_report(self, overview: Dict[str, Any]) -> str:
        """格式化美股公司概览报告（支持多数据源）"""
        if not overview:
            return "无数据"

        source = overview.get('_data_source', 'unknown')

        # AlphaVantage 有专用格式化方法
        av = self._get_us_fetcher("AlphaVantage")
        if av and source == "AlphaVantage":
            return av.format_overview_report(overview)

        # 通用格式（yfinance 或其他）
        lines = [
            f"# {overview.get('Name', '')} ({overview.get('Symbol', '')})",
            f"",
            f"数据来源: {source}",
            f"",
            f"## 基本信息",
            f"- 行业: {overview.get('Industry', '-')}",
            f"- 板块: {overview.get('Sector', '-')}",
            f"- 国家: {overview.get('Country', '-')}",
            f"",
            f"## 估值指标",
            f"- 市值: ${self._format_large_number(overview.get('MarketCapitalization'))}",
            f"- 市盈率(PE): {overview.get('PERatio', '-')}",
            f"- 远期市盈率: {overview.get('ForwardPE', '-')}",
            f"- 市净率(PB): {overview.get('PriceToBookRatio', '-')}",
            f"",
            f"## 盈利指标",
            f"- 每股收益(EPS): ${overview.get('EPS', '-')}",
            f"- 净利润率: {overview.get('ProfitMargin', '-')}",
            f"",
            f"## 股息信息",
            f"- 股息率: {overview.get('DividendYield', '-')}",
            f"",
            f"## 价格区间",
            f"- 52周最高: ${overview.get('52WeekHigh', '-')}",
            f"- 52周最低: ${overview.get('52WeekLow', '-')}",
        ]
        return "\n".join(lines)

    def _format_large_number(self, value) -> str:
        """格式化大数字"""
        if not value or value in ("", "None", "nan"):
            return "-"
        try:
            num = float(value)
            if num >= 1e12:
                return f"{num/1e12:.2f}T"
            elif num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            else:
                return f"{num:,.0f}"
        except (ValueError, TypeError):
            return str(value)

    def format_us_news_report(self, news_data: Dict[str, Any], limit: int = 10) -> str:
        """格式化美股新闻报告"""
        fetcher = self._get_us_fetcher("AlphaVantage")
        if fetcher is None:
            return "AlphaVantage 数据源未配置"
        return fetcher.format_news_report(news_data, limit)
