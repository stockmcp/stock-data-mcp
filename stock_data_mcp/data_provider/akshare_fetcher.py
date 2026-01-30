"""
Akshare 数据获取器 (优先级 1)
使用 akshare 库获取股票数据，支持多数据源（东财、新浪、腾讯）
"""

import logging
import time
from typing import Optional, Dict, List

import pandas as pd
import akshare as ak
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseFetcher, DataFetchError
from .types import (
    UnifiedRealtimeQuote,
    ChipDistribution,
    RealtimeSource,
    safe_float,
    safe_int,
)

_LOGGER = logging.getLogger(__name__)


def _is_etf_code(stock_code: str) -> bool:
    """判断是否为 ETF 代码"""
    code = stock_code.lstrip('0')
    if len(code) == 6:
        prefix = code[:2]
        return prefix in ('51', '52', '56', '58', '15', '16', '18')
    return False


def _is_hk_code(stock_code: str) -> bool:
    """判断是否为港股代码"""
    code = stock_code.lower()
    if code.startswith('hk'):
        return True
    # 5位数字可能是港股
    clean_code = code.lstrip('0')
    return len(clean_code) == 5 and clean_code.isdigit()


def _is_us_code(stock_code: str) -> bool:
    """判断是否为美股代码"""
    # 美股代码通常是1-5个大写字母，可能带有后缀如 .O .N
    code = stock_code.upper().split('.')[0]
    return len(code) <= 5 and code.isalpha()


class AkshareFetcher(BaseFetcher):
    """Akshare 数据获取器"""

    name = "AkshareFetcher"
    priority = 1

    def __init__(self):
        super().__init__()
        self._realtime_cache: Dict[str, UnifiedRealtimeQuote] = {}
        self._realtime_cache_time: float = 0
        self._cache_ttl: float = 1200.0  # 20分钟缓存

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
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
        self.random_sleep(2.0, 5.0)

        try:
            if _is_hk_code(stock_code):
                return self._fetch_hk_data(stock_code, start_date, end_date)
            elif _is_etf_code(stock_code):
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
        """获取 A 股数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 A 股数据失败: {e}")
            return None

    def _fetch_etf_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取 ETF 数据"""
        try:
            df = ak.fund_etf_hist_em(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 ETF 数据失败: {e}")
            return None

    def _fetch_hk_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取港股数据"""
        try:
            # 清理港股代码
            code = stock_code.lower().replace('hk', '').lstrip('0')
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取港股数据失败: {e}")
            return None

    def _normalize_data(
        self,
        df: pd.DataFrame,
        stock_code: str
    ) -> pd.DataFrame:
        """标准化数据"""
        if df is None or df.empty:
            return pd.DataFrame()

        # Akshare 列名映射
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

        df = df.rename(columns=column_mapping)

        # 选择标准列
        result_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        available_cols = [col for col in result_cols if col in df.columns]
        df = df[available_cols].copy()

        return df

    def get_realtime_quote(
        self,
        stock_code: str,
        source: str = "em"
    ) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情

        Args:
            stock_code: 股票代码
            source: 数据源，可选 "em"(东财), "sina"(新浪), "tencent"(腾讯)
        """
        # 检查缓存
        cache_key = f"{stock_code}_{source}"
        if (
            cache_key in self._realtime_cache
            and time.time() - self._realtime_cache_time < self._cache_ttl
        ):
            return self._realtime_cache[cache_key]

        try:
            self.random_sleep(0.5, 1.5)

            if _is_hk_code(stock_code):
                return self._get_hk_realtime_quote(stock_code)
            elif _is_etf_code(stock_code):
                return self._get_etf_realtime_quote(stock_code)
            else:
                if source == "sina":
                    return self._get_stock_realtime_quote_sina(stock_code)
                elif source == "tencent":
                    return self._get_stock_realtime_quote_tencent(stock_code)
                else:
                    return self._get_stock_realtime_quote_em(stock_code)

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 {stock_code} 实时行情失败: {e}")
            return None

    def _get_stock_realtime_quote_em(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """从东财获取 A 股实时行情（完整数据）"""
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return None

            # 查找股票
            row = df[df['代码'] == stock_code]
            if row.empty:
                return None

            row = row.iloc[0]
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=row.get('名称'),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                change_amount=safe_float(row.get('涨跌额')),
                volume=safe_float(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                volume_ratio=safe_float(row.get('量比')),
                turnover_rate=safe_float(row.get('换手率')),
                amplitude=safe_float(row.get('振幅')),
                open_price=safe_float(row.get('今开')),
                high=safe_float(row.get('最高')),
                low=safe_float(row.get('最低')),
                pre_close=safe_float(row.get('昨收')),
                pe_ratio=safe_float(row.get('市盈率-动态')),
                pb_ratio=safe_float(row.get('市净率')),
                total_mv=safe_float(row.get('总市值')),
                circ_mv=safe_float(row.get('流通市值')),
                change_60d=safe_float(row.get('60日涨跌幅')),
                high_52w=safe_float(row.get('52周最高')),
                low_52w=safe_float(row.get('52周最低')),
            )

            # 更新缓存
            self._realtime_cache[f"{stock_code}_em"] = quote
            self._realtime_cache_time = time.time()

            return quote

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 东财实时行情获取失败: {e}")
            return None

    def _get_stock_realtime_quote_sina(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """从新浪获取 A 股实时行情（轻量级）"""
        try:
            # 确定市场前缀
            if stock_code.startswith(('6', '9')):
                symbol = f"sh{stock_code}"
            else:
                symbol = f"sz{stock_code}"

            df = ak.stock_zh_a_spot()
            if df is None or df.empty:
                return None

            row = df[df['代码'] == stock_code]
            if row.empty:
                return None

            row = row.iloc[0]
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=row.get('名称'),
                source=RealtimeSource.AKSHARE_SINA,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                change_amount=safe_float(row.get('涨跌额')),
                volume=safe_float(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                open_price=safe_float(row.get('今开')),
                high=safe_float(row.get('最高')),
                low=safe_float(row.get('最低')),
                pre_close=safe_float(row.get('昨收')),
            )

            self._realtime_cache[f"{stock_code}_sina"] = quote
            self._realtime_cache_time = time.time()

            return quote

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 新浪实时行情获取失败: {e}")
            return None

    def _get_stock_realtime_quote_tencent(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """从腾讯获取 A 股实时行情"""
        try:
            # 腾讯行情通过新浪接口代理
            return self._get_stock_realtime_quote_sina(stock_code)
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 腾讯实时行情获取失败: {e}")
            return None

    def _get_etf_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """获取 ETF 实时行情"""
        try:
            df = ak.fund_etf_spot_em()
            if df is None or df.empty:
                return None

            row = df[df['代码'] == stock_code]
            if row.empty:
                return None

            row = row.iloc[0]
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=row.get('名称'),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                volume=safe_float(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                open_price=safe_float(row.get('今开')),
                high=safe_float(row.get('最高')),
                low=safe_float(row.get('最低')),
                pre_close=safe_float(row.get('昨收')),
            )

            self._realtime_cache[f"{stock_code}_em"] = quote
            self._realtime_cache_time = time.time()

            return quote

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] ETF 实时行情获取失败: {e}")
            return None

    def _get_hk_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """获取港股实时行情"""
        try:
            df = ak.stock_hk_spot_em()
            if df is None or df.empty:
                return None

            # 清理港股代码
            code = stock_code.lower().replace('hk', '').lstrip('0')
            row = df[df['代码'] == code]
            if row.empty:
                return None

            row = row.iloc[0]
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=row.get('名称'),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                volume=safe_float(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                open_price=safe_float(row.get('今开')),
                high=safe_float(row.get('最高')),
                low=safe_float(row.get('最低')),
                pre_close=safe_float(row.get('昨收')),
            )

            return quote

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 港股实时行情获取失败: {e}")
            return None

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """获取筹码分布"""
        try:
            self.random_sleep(1.0, 2.0)

            df = ak.stock_cyq_em(symbol=stock_code)
            if df is None or df.empty:
                return None

            # 取最新一天数据
            latest = df.iloc[-1]

            chip = ChipDistribution(
                code=stock_code,
                date=str(latest.get('日期', '')),
                source="akshare",
                profit_ratio=safe_float(latest.get('获利比例')),
                avg_cost=safe_float(latest.get('平均成本')),
                cost_90_low=safe_float(latest.get('90成本-低')),
                cost_90_high=safe_float(latest.get('90成本-高')),
                concentration_90=safe_float(latest.get('90集中度')),
                cost_70_low=safe_float(latest.get('70成本-低')),
                cost_70_high=safe_float(latest.get('70成本-高')),
                concentration_70=safe_float(latest.get('70集中度')),
            )

            return chip

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取筹码分布失败: {e}")
            return None
