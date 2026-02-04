"""
Efinance 数据获取器 (优先级 1)
使用 efinance 库获取东方财富 A 股数据
"""

import logging
import time
from typing import Optional, Dict, List

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseFetcher, DataFetchError
from .types import (
    UnifiedRealtimeQuote,
    RealtimeSource,
    safe_float,
    is_etf_code,
)

_LOGGER = logging.getLogger(__name__)


class EfinanceFetcher(BaseFetcher):
    """Efinance 数据获取器"""

    name = "EfinanceFetcher"
    priority = 1  # A 股次选

    def __init__(self):
        super().__init__()
        self._realtime_cache: Dict[str, UnifiedRealtimeQuote] = {}
        self._realtime_cache_time: float = 0
        self._cache_ttl: float = 600.0  # 10分钟缓存

        # 延迟导入
        try:
            import efinance as ef
            self._ef = ef
            self._available = True
        except ImportError:
            _LOGGER.warning("efinance 库未安装")
            self._available = False

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def _fetch_raw_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取原始数据"""
        if not self._available:
            return None

        self.random_sleep(1.5, 3.0)

        try:
            if is_etf_code(stock_code):
                return self._fetch_etf_data(stock_code, start_date, end_date)
            else:
                return self._fetch_stock_data(stock_code, start_date, end_date)
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 {stock_code} 原始数据失败: {e}")
            raise DataFetchError(f"获取数据失败: {e}")

    def _fetch_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # efinance 日期格式: YYYYMMDD
            df = self._ef.stock.get_quote_history(
                stock_codes=stock_code,
                beg=start_date,
                end=end_date,
                klt=101,  # 日线
                fqt=1,    # 前复权
            )
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取股票数据失败: {e}")
            return None

    def _fetch_etf_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取 ETF 数据"""
        try:
            df = self._ef.fund.get_quote_history(
                fund_code=stock_code,
                beg=start_date,
                end=end_date,
                klt=101,  # 日线
                fqt=1,    # 前复权
            )
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 ETF 数据失败: {e}")
            return None

    def _normalize_data(
        self,
        df: pd.DataFrame,
        stock_code: str
    ) -> pd.DataFrame:
        """标准化数据"""
        if df is None or df.empty:
            return pd.DataFrame()

        # efinance 列名映射（可能有 股票名称/基金名称 前缀）
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '换手率': 'turnover_rate',
        }

        # 处理可能的列名变体
        for old_col in list(df.columns):
            for key, new_col in column_mapping.items():
                if key in old_col:
                    df = df.rename(columns={old_col: new_col})
                    break

        # 选择标准列
        result_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        available_cols = [col for col in result_cols if col in df.columns]
        df = df[available_cols].copy()

        return df

    def _create_quote_from_row(self, row) -> Optional[UnifiedRealtimeQuote]:
        """从 DataFrame 行创建 UnifiedRealtimeQuote"""
        code = str(row.get('股票代码', ''))
        if not code:
            return None
        return UnifiedRealtimeQuote(
            code=code,
            name=row.get('股票名称'),
            source=RealtimeSource.EFINANCE,
            price=safe_float(row.get('最新价')),
            change_pct=safe_float(row.get('涨跌幅')),
            change_amount=safe_float(row.get('涨跌额')),
            volume=safe_float(row.get('成交量')),
            amount=safe_float(row.get('成交额')),
            turnover_rate=safe_float(row.get('换手率')),
            amplitude=safe_float(row.get('振幅')),
            open_price=safe_float(row.get('今开')),
            high=safe_float(row.get('最高')),
            low=safe_float(row.get('最低')),
            pre_close=safe_float(row.get('昨收')),
        )

    def _refresh_realtime_cache(self) -> bool:
        """刷新实时行情缓存，返回是否成功"""
        try:
            self.random_sleep(0.5, 1.5)
            df = self._ef.stock.get_realtime_quotes()
            if df is None or df.empty:
                return False

            self._realtime_cache.clear()
            self._realtime_cache_time = time.time()

            for _, row in df.iterrows():
                quote = self._create_quote_from_row(row)
                if quote:
                    self._realtime_cache[quote.code] = quote

            return True
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 刷新实时行情缓存失败: {e}")
            return False

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """获取实时行情（仅支持A股个股，不支持ETF和港股）"""
        if not self._available:
            return None

        # ETF 不支持，直接返回 None 让其他数据源处理
        if is_etf_code(stock_code):
            _LOGGER.debug(f"[{self.name}] ETF {stock_code} 不支持实时行情，跳过")
            return None

        # 检查缓存
        if (
            stock_code in self._realtime_cache
            and time.time() - self._realtime_cache_time < self._cache_ttl
        ):
            return self._realtime_cache[stock_code]

        # 刷新缓存
        if self._refresh_realtime_cache():
            return self._realtime_cache.get(stock_code)
        return None

    def get_batch_realtime_quotes(
        self,
        stock_codes: List[str]
    ) -> Dict[str, UnifiedRealtimeQuote]:
        """批量获取实时行情（仅支持A股个股）"""
        if not self._available:
            return {}

        # 过滤掉 ETF，只处理 A 股个股
        a_stock_codes = [code for code in stock_codes if not is_etf_code(code)]
        if not a_stock_codes:
            _LOGGER.debug(f"[{self.name}] 批量查询中无A股个股，跳过")
            return {}

        # 先尝试从缓存获取
        if time.time() - self._realtime_cache_time < self._cache_ttl:
            result = {}
            for code in a_stock_codes:
                if code in self._realtime_cache:
                    result[code] = self._realtime_cache[code]
            if len(result) == len(a_stock_codes):
                return result

        # 刷新缓存
        if self._refresh_realtime_cache():
            return {code: self._realtime_cache[code] for code in a_stock_codes if code in self._realtime_cache}
        return {}

    def get_base_info(self, stock_code: str) -> Optional[Dict]:
        """获取股票基本信息"""
        if not self._available:
            return None

        try:
            self.random_sleep(0.5, 1.0)
            info = self._ef.stock.get_base_info(stock_code)
            if info is None:
                return None

            return {
                'code': stock_code,
                'pe_ratio': safe_float(info.get('市盈率(动)')),
                'pb_ratio': safe_float(info.get('市净率')),
                'industry': info.get('行业'),
                'total_mv': safe_float(info.get('总市值')),
                'circ_mv': safe_float(info.get('流通市值')),
                'roe': safe_float(info.get('ROE')),
                'net_profit_margin': safe_float(info.get('净利率')),
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取基本信息失败: {e}")
            return None

    def get_belong_board(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取所属板块"""
        if not self._available:
            return None

        try:
            self.random_sleep(0.5, 1.0)
            df = self._ef.stock.get_belong_board(stock_code)
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取所属板块失败: {e}")
            return None

    def get_fund_flow(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取资金流向"""
        if not self._available:
            return None

        try:
            self.random_sleep(0.5, 1.5)
            df = self._ef.stock.get_history_bill(stock_code)
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取资金流向失败: {e}")
            return None

    def get_billboard(self, days: str = "5") -> Optional[pd.DataFrame]:
        """获取龙虎榜统计（当日龙虎榜）"""
        if not self._available:
            return None

        try:
            self.random_sleep(0.5, 1.5)
            df = self._ef.stock.get_daily_billboard()
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取龙虎榜失败: {e}")
            return None

    def get_board_cons(self, board_name: str, board_type: str = "industry") -> Optional[pd.DataFrame]:
        """获取板块成分股"""
        if not self._available:
            return None

        try:
            self.random_sleep(0.5, 1.5)
            # efinance 通过板块代码获取成分股，需要先获取板块列表
            if board_type == "concept":
                # 获取概念板块列表
                boards = self._ef.stock.get_realtime_quotes()
                # efinance 可能没有直接的概念板块成分股接口，返回 None 以使用其他数据源
                return None
            else:
                # 获取行业板块成分股
                # efinance 的 get_members_stocks 需要板块代码
                # 尝试通过板块名称查找
                return None  # efinance 需要板块代码而非名称，回退到 akshare
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取板块成分股失败: {e}")
            return None
