"""
YFinance 数据获取器 (优先级 4 - 最低)
使用 yfinance 库获取国际股票数据
作为最后的后备数据源
"""

import logging
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseFetcher, DataFetchError

_LOGGER = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """YFinance 数据获取器"""

    name = "YfinanceFetcher"
    priority = 4  # 最低优先级，作为后备

    def __init__(self):
        super().__init__()
        self._yf = None

        # 延迟导入
        try:
            import yfinance as yf
            self._yf = yf
            self._available = True
        except ImportError:
            _LOGGER.warning("yfinance 库未安装")
            self._available = False

    def _convert_stock_code(self, stock_code: str) -> str:
        """转换股票代码为 YFinance 格式"""
        code = stock_code.upper()

        # 已经是 yfinance 格式
        if '.' in code:
            return code

        # 港股
        if code.startswith('HK') or code.startswith('0'):
            # hk00700 -> 0700.HK
            clean = code.replace('HK', '').lstrip('0')
            if len(clean) <= 5:
                return f"{clean.zfill(4)}.HK"

        # A股沪市
        if code.startswith('6') or code.startswith('9'):
            return f"{code}.SS"

        # A股深市
        if code.startswith('0') or code.startswith('3') or code.startswith('2'):
            return f"{code}.SZ"

        # 美股（字母代码）
        if code.isalpha():
            return code

        # 默认作为沪市处理
        return f"{code}.SS"

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
        if not self._available:
            return None

        try:
            yf_code = self._convert_stock_code(stock_code)

            # 日期格式转换 YYYYMMDD -> YYYY-MM-DD
            if len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            df = self._yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            if df is None or df.empty:
                return None

            # 重置索引，将日期变为列
            df = df.reset_index()

            return df

        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取 {stock_code} 数据失败: {e}")
            raise DataFetchError(f"获取数据失败: {e}")

    def _normalize_data(
        self,
        df: pd.DataFrame,
        stock_code: str
    ) -> pd.DataFrame:
        """标准化数据"""
        if df is None or df.empty:
            return pd.DataFrame()

        # YFinance 列名映射（大写）
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        }

        df = df.rename(columns=column_mapping)

        # 计算涨跌幅
        if 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100

        # 估算成交额（volume × close）
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']

        # 日期格式化
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 选择标准列
        result_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        available_cols = [col for col in result_cols if col in df.columns]
        df = df[available_cols].copy()

        return df
