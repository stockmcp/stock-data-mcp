"""
Tushare 数据获取器 (优先级 0/2)
使用 tushare 库获取股票数据
需要 TUSHARE_TOKEN 环境变量，有 token 时优先级为 0，无 token 时为 2
"""

import os
import logging
import time
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseFetcher, DataFetchError, RateLimitError

_LOGGER = logging.getLogger(__name__)


class TushareFetcher(BaseFetcher):
    """Tushare 数据获取器"""

    name = "TushareFetcher"
    priority = 2  # 默认优先级

    # 限流配置：免费版 80次/分钟
    RATE_LIMIT = 80
    RATE_WINDOW = 60  # 秒

    def __init__(self):
        super().__init__()
        self._api = None
        self._call_count = 0
        self._window_start = time.time()

        # 从环境变量获取 token
        token = os.getenv("TUSHARE_TOKEN")
        if token:
            try:
                import tushare as ts
                ts.set_token(token)
                self._api = ts.pro_api()
                self._available = True
                self.priority = 0  # 有 token 提升为最高优先级
                _LOGGER.info("✅ Tushare API 初始化成功，优先级提升为 0")
            except Exception as e:
                _LOGGER.warning(f"Tushare API 初始化失败: {e}")
                self._available = False
        else:
            _LOGGER.info("未配置 TUSHARE_TOKEN，TushareFetcher 降级为优先级 2")
            self._available = False

    def _check_rate_limit(self):
        """检查限流"""
        current_time = time.time()
        if current_time - self._window_start >= self.RATE_WINDOW:
            # 重置窗口
            self._call_count = 0
            self._window_start = current_time

        if self._call_count >= self.RATE_LIMIT:
            # 等待到下一个窗口
            wait_time = self.RATE_WINDOW - (current_time - self._window_start)
            if wait_time > 0:
                _LOGGER.warning(f"[{self.name}] 达到限流，等待 {wait_time:.1f}s")
                time.sleep(wait_time)
                self._call_count = 0
                self._window_start = time.time()

        self._call_count += 1

    def _convert_stock_code(self, stock_code: str) -> str:
        """转换股票代码为 Tushare 格式"""
        code = stock_code.upper()
        if '.' in code:
            return code

        # 根据代码前缀判断市场
        if code.startswith(('6', '9')):
            return f"{code}.SH"
        elif code.startswith(('0', '3', '2')):
            return f"{code}.SZ"
        elif code.startswith('4') or code.startswith('8'):
            return f"{code}.BJ"  # 北交所

        return f"{code}.SH"  # 默认上交所

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
        if not self._available or self._api is None:
            return None

        self._check_rate_limit()

        try:
            ts_code = self._convert_stock_code(stock_code)

            df = self._api.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                # 尝试获取复权数据
                df = self._api.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    adj='qfq'  # 前复权
                )

            return df

        except Exception as e:
            error_msg = str(e).lower()
            if 'quota' in error_msg or 'limit' in error_msg:
                raise RateLimitError(f"Tushare 配额超限: {e}")
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

        # Tushare 列名映射
        column_mapping = {
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'pct_chg': 'pct_chg',
        }

        df = df.rename(columns=column_mapping)

        # 单位转换
        # vol: 手 -> 股 (× 100)
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * 100

        # amount: 千元 -> 元 (× 1000)
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 1000

        # 日期格式转换 YYYYMMDD -> YYYY-MM-DD
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

        # 按日期排序（Tushare 默认降序）
        df = df.sort_values('date', ascending=True)

        # 选择标准列
        result_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        available_cols = [col for col in result_cols if col in df.columns]
        df = df[available_cols].copy()

        return df

    def get_fund_flow(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取资金流向"""
        if not self._available or self._api is None:
            return None

        try:
            self._check_rate_limit()
            ts_code = self._convert_stock_code(stock_code)

            # 获取最近10天的资金流向
            df = self._api.moneyflow(ts_code=ts_code)
            if df is None or df.empty:
                return None

            # 重命名列为中文
            column_mapping = {
                'trade_date': '日期',
                'buy_sm_vol': '小单买入量',
                'buy_sm_amount': '小单买入金额',
                'sell_sm_vol': '小单卖出量',
                'sell_sm_amount': '小单卖出金额',
                'buy_md_vol': '中单买入量',
                'buy_md_amount': '中单买入金额',
                'sell_md_vol': '中单卖出量',
                'sell_md_amount': '中单卖出金额',
                'buy_lg_vol': '大单买入量',
                'buy_lg_amount': '大单买入金额',
                'sell_lg_vol': '大单卖出量',
                'sell_lg_amount': '大单卖出金额',
                'buy_elg_vol': '超大单买入量',
                'buy_elg_amount': '超大单买入金额',
                'sell_elg_vol': '超大单卖出量',
                'sell_elg_amount': '超大单卖出金额',
            }
            df = df.rename(columns=column_mapping)
            return df.head(10)
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取资金流向失败: {e}")
            return None

    def get_billboard(self, days: str = "5") -> Optional[pd.DataFrame]:
        """获取龙虎榜统计"""
        if not self._available or self._api is None:
            return None

        try:
            self._check_rate_limit()
            from datetime import datetime, timedelta

            # 获取最近交易日的龙虎榜
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=int(days) + 5)).strftime('%Y%m%d')

            df = self._api.top_list(start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return None

            # 重命名列为中文
            column_mapping = {
                'trade_date': '上榜日期',
                'ts_code': '股票代码',
                'name': '股票名称',
                'close': '收盘价',
                'pct_change': '涨跌幅',
                'turnover_rate': '换手率',
                'amount': '龙虎榜成交额',
                'l_sell': '龙虎榜卖出额',
                'l_buy': '龙虎榜买入额',
                'net_amount': '龙虎榜净买额',
                'reason': '上榜原因',
            }
            df = df.rename(columns=column_mapping)
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取龙虎榜失败: {e}")
            return None

    def get_belong_board(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取所属板块（通过股票基本信息获取行业）"""
        if not self._available or self._api is None:
            return None

        try:
            self._check_rate_limit()
            ts_code = self._convert_stock_code(stock_code)

            # 获取股票基本信息
            df = self._api.stock_basic(ts_code=ts_code, fields='ts_code,name,industry,market,list_date')
            if df is None or df.empty:
                return None

            # 重命名列为中文
            column_mapping = {
                'ts_code': '股票代码',
                'name': '股票名称',
                'industry': '行业',
                'market': '市场',
                'list_date': '上市日期',
            }
            df = df.rename(columns=column_mapping)
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取所属板块失败: {e}")
            return None

    def get_board_cons(self, board_name: str, board_type: str = "industry") -> Optional[pd.DataFrame]:
        """获取板块成分股"""
        if not self._available or self._api is None:
            return None

        try:
            self._check_rate_limit()

            # Tushare 通过行业获取成分股
            if board_type == "industry":
                df = self._api.stock_basic(
                    industry=board_name,
                    fields='ts_code,symbol,name,industry,market,list_date'
                )
            else:
                # 概念板块需要使用 concept_detail 接口
                # 先获取概念代码
                concepts = self._api.concept()
                if concepts is None or concepts.empty:
                    return None

                matched = concepts[concepts['name'].str.contains(board_name, na=False)]
                if matched.empty:
                    return None

                concept_code = matched.iloc[0]['code']
                df = self._api.concept_detail(id=concept_code)

            if df is None or df.empty:
                return None

            # 重命名列为中文
            column_mapping = {
                'ts_code': '代码',
                'symbol': '股票代码',
                'name': '名称',
                'industry': '行业',
                'market': '市场',
            }
            df = df.rename(columns=column_mapping)
            return df
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取板块成分股失败: {e}")
            return None
