"""
YFinance 数据获取器 (优先级 5 - 全局后备)
使用 yfinance 库获取国际股票数据
作为所有市场的最终后备数据源
"""

import logging
from typing import Optional, Dict, Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseFetcher, DataFetchError

_LOGGER = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """YFinance 数据获取器"""

    name = "YfinanceFetcher"
    priority = 5  # 全局后备

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

        # 美股（纯字母代码，1-5位）
        if code.isalpha() and len(code) <= 5:
            return code

        # 港股（明确以 HK 开头）
        if code.startswith('HK'):
            clean = code.replace('HK', '').lstrip('0')
            return f"{clean.zfill(4)}.HK"

        # A股：6位数字代码
        if len(code) == 6 and code.isdigit():
            # 沪市：6, 9 开头
            if code.startswith(('6', '9')):
                return f"{code}.SS"
            # 深市：0, 2, 3 开头
            if code.startswith(('0', '2', '3')):
                return f"{code}.SZ"

        # 港股：5位或更少的纯数字（如 00700 -> 0700.HK）
        if code.isdigit() and len(code) <= 5:
            return f"{code.lstrip('0').zfill(4)}.HK"

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

    # ==================== 美股财务数据方法（兼容 AlphaVantage 格式）====================

    def _get_ticker(self, symbol: str):
        """获取 yfinance Ticker 对象"""
        if not self._available:
            return None
        return self._yf.Ticker(symbol.upper())

    def _safe_str(self, df: pd.DataFrame, index_name: str, col) -> Optional[str]:
        """安全提取 DataFrame 值并转为字符串"""
        try:
            if index_name in df.index:
                val = df.loc[index_name, col]
                if pd.notna(val):
                    return str(int(val)) if isinstance(val, (int, float)) else str(val)
        except Exception:
            pass
        return None

    def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取美股公司概览（通过 yfinance）
        返回格式与 AlphaVantage 兼容
        """
        try:
            ticker = self._get_ticker(symbol)
            if ticker is None:
                return None

            info = ticker.info
            if not info or 'symbol' not in info:
                return None

            # 标准化为 AlphaVantage 格式
            return {
                "Symbol": info.get("symbol", symbol),
                "Name": info.get("longName") or info.get("shortName", ""),
                "Industry": info.get("industry", ""),
                "Sector": info.get("sector", ""),
                "Country": info.get("country", ""),
                "Exchange": info.get("exchange", ""),
                "MarketCapitalization": str(info.get("marketCap", "")),
                "PERatio": str(info.get("trailingPE", "")),
                "ForwardPE": str(info.get("forwardPE", "")),
                "PriceToBookRatio": str(info.get("priceToBook", "")),
                "PriceToSalesRatioTTM": str(info.get("priceToSalesTrailing12Months", "")),
                "EPS": str(info.get("trailingEps", "")),
                "BookValue": str(info.get("bookValue", "")),
                "DividendYield": str(info.get("dividendYield", "")),
                "DividendPerShare": str(info.get("dividendRate", "")),
                "52WeekHigh": str(info.get("fiftyTwoWeekHigh", "")),
                "52WeekLow": str(info.get("fiftyTwoWeekLow", "")),
                "50DayMovingAverage": str(info.get("fiftyDayAverage", "")),
                "200DayMovingAverage": str(info.get("twoHundredDayAverage", "")),
                "ProfitMargin": str(info.get("profitMargins", "")),
                "OperatingMarginTTM": str(info.get("operatingMargins", "")),
                "ReturnOnEquityTTM": str(info.get("returnOnEquity", "")),
                "ReturnOnAssetsTTM": str(info.get("returnOnAssets", "")),
                "AnalystTargetPrice": str(info.get("targetMeanPrice", "")),
                "_source": "yfinance",
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取公司概览失败 {symbol}: {e}")
            return None

    def get_balance_sheet(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取美股资产负债表（通过 yfinance）
        返回格式与 AlphaVantage 兼容
        """
        try:
            ticker = self._get_ticker(symbol)
            if ticker is None:
                return None

            if quarterly:
                df = ticker.quarterly_balance_sheet
            else:
                df = ticker.balance_sheet

            if df is None or df.empty:
                return None

            reports = []
            for date_col in df.columns:
                report = {
                    "fiscalDateEnding": date_col.strftime("%Y-%m-%d"),
                    "totalAssets": self._safe_str(df, "Total Assets", date_col),
                    "totalLiabilities": self._safe_str(df, "Total Liabilities Net Minority Interest", date_col),
                    "totalShareholderEquity": self._safe_str(df, "Stockholders Equity", date_col),
                    "cashAndCashEquivalentsAtCarryingValue": self._safe_str(df, "Cash And Cash Equivalents", date_col),
                    "currentDebt": self._safe_str(df, "Current Debt", date_col),
                    "longTermDebt": self._safe_str(df, "Long Term Debt", date_col),
                    "totalCurrentAssets": self._safe_str(df, "Current Assets", date_col),
                    "totalCurrentLiabilities": self._safe_str(df, "Current Liabilities", date_col),
                    "inventory": self._safe_str(df, "Inventory", date_col),
                    "propertyPlantEquipment": self._safe_str(df, "Net PPE", date_col),
                }
                reports.append(report)

            return {
                "symbol": symbol,
                "reports": reports[:4] if quarterly else reports[:3],
                "_source": "yfinance",
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取资产负债表失败 {symbol}: {e}")
            return None

    def get_income_statement(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取美股利润表（通过 yfinance）
        返回格式与 AlphaVantage 兼容
        """
        try:
            ticker = self._get_ticker(symbol)
            if ticker is None:
                return None

            if quarterly:
                df = ticker.quarterly_income_stmt
            else:
                df = ticker.income_stmt

            if df is None or df.empty:
                return None

            reports = []
            for date_col in df.columns:
                report = {
                    "fiscalDateEnding": date_col.strftime("%Y-%m-%d"),
                    "totalRevenue": self._safe_str(df, "Total Revenue", date_col),
                    "grossProfit": self._safe_str(df, "Gross Profit", date_col),
                    "operatingIncome": self._safe_str(df, "Operating Income", date_col),
                    "netIncome": self._safe_str(df, "Net Income", date_col),
                    "ebitda": self._safe_str(df, "EBITDA", date_col),
                    "costOfRevenue": self._safe_str(df, "Cost Of Revenue", date_col),
                    "operatingExpenses": self._safe_str(df, "Operating Expense", date_col),
                }
                reports.append(report)

            return {
                "symbol": symbol,
                "reports": reports[:4] if quarterly else reports[:3],
                "_source": "yfinance",
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取利润表失败 {symbol}: {e}")
            return None

    def get_cash_flow(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取美股现金流量表（通过 yfinance）
        返回格式与 AlphaVantage 兼容
        """
        try:
            ticker = self._get_ticker(symbol)
            if ticker is None:
                return None

            if quarterly:
                df = ticker.quarterly_cashflow
            else:
                df = ticker.cashflow

            if df is None or df.empty:
                return None

            reports = []
            for date_col in df.columns:
                report = {
                    "fiscalDateEnding": date_col.strftime("%Y-%m-%d"),
                    "operatingCashflow": self._safe_str(df, "Operating Cash Flow", date_col),
                    "capitalExpenditures": self._safe_str(df, "Capital Expenditure", date_col),
                    "dividendPayout": self._safe_str(df, "Common Stock Dividend Paid", date_col),
                    "netIncome": self._safe_str(df, "Net Income", date_col),
                    "changeInCashAndCashEquivalents": self._safe_str(df, "Changes In Cash", date_col),
                }
                reports.append(report)

            return {
                "symbol": symbol,
                "reports": reports[:4] if quarterly else reports[:3],
                "_source": "yfinance",
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取现金流量表失败 {symbol}: {e}")
            return None

    def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取美股盈利数据（通过 yfinance）
        返回格式与 AlphaVantage 兼容
        """
        try:
            ticker = self._get_ticker(symbol)
            if ticker is None:
                return None

            # 尝试获取盈利历史
            earnings = None
            try:
                earnings = ticker.earnings_history
            except Exception:
                pass

            if earnings is None or earnings.empty:
                try:
                    earnings = ticker.quarterly_earnings
                except Exception:
                    pass

            if earnings is None or earnings.empty:
                return None

            quarterly_earnings = []
            for idx, row in earnings.iterrows():
                quarterly_earnings.append({
                    "fiscalDateEnding": str(idx) if hasattr(idx, 'strftime') else str(idx),
                    "reportedEPS": str(row.get('epsActual', row.get('Reported EPS', ''))),
                    "estimatedEPS": str(row.get('epsEstimate', '')),
                    "surprisePercentage": str(row.get('epsDifference', '')),
                })

            return {
                "quarterlyEarnings": quarterly_earnings[:8],
                "annualEarnings": [],
                "_source": "yfinance",
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取盈利数据失败 {symbol}: {e}")
            return None

    def get_insider_transactions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取美股内部交易（通过 yfinance）
        返回格式与 AlphaVantage 兼容
        """
        try:
            ticker = self._get_ticker(symbol)
            if ticker is None:
                return None

            insider = ticker.insider_transactions
            if insider is None or insider.empty:
                return None

            transactions = []
            for _, row in insider.iterrows():
                # 映射交易类型
                trans_type = str(row.get('Transaction', ''))
                if 'Sale' in trans_type or 'Sell' in trans_type:
                    acq_disp = 'D'  # Disposition
                elif 'Purchase' in trans_type or 'Buy' in trans_type:
                    acq_disp = 'A'  # Acquisition
                else:
                    acq_disp = trans_type

                transactions.append({
                    "transaction_date": str(row.get('Start Date', '')),
                    "owner_name": str(row.get('Insider', '')),
                    "owner_title": str(row.get('Position', '')),
                    "acquisition_or_disposition": acq_disp,
                    "shares": str(row.get('Shares', '')),
                    "transaction_value": str(row.get('Value', '')),
                })

            return {
                "data": transactions,
                "_source": "yfinance",
            }
        except Exception as e:
            _LOGGER.warning(f"[{self.name}] 获取内部交易失败 {symbol}: {e}")
            return None
