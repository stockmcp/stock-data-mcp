"""
Alpha Vantage 数据获取器

提供美股基本面数据和新闻数据，需要配置 ALPHA_VANTAGE_API_KEY 环境变量。
参考 TradingAgents 项目实现: https://github.com/TauricResearch/TradingAgents
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

import requests
import pandas as pd

from .base import BaseFetcher, DataFetchError, RateLimitError

_LOGGER = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageRateLimitError(RateLimitError):
    """Alpha Vantage API 限流错误"""
    pass


class AlphaVantageFetcher(BaseFetcher):
    """
    Alpha Vantage 数据获取器

    提供美股基本面数据和新闻数据：
    - 公司概览 (Company Overview)
    - 资产负债表 (Balance Sheet)
    - 现金流量表 (Cash Flow)
    - 利润表 (Income Statement)
    - 新闻情绪 (News Sentiment)
    - 内部交易 (Insider Transactions)
    """

    name: str = "AlphaVantage"
    priority: int = 4  # 美股首选（需配置 API key）

    def __init__(self):
        super().__init__()
        self._api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self._available = bool(self._api_key)
        self._last_request_time = 0
        self._min_request_interval = 12.5  # 免费版每分钟 5 次请求限制

        if not self._available:
            _LOGGER.info("Alpha Vantage API key not configured, fetcher disabled")
        else:
            _LOGGER.info("Alpha Vantage fetcher initialized")

    @property
    def is_available(self) -> bool:
        """检查是否可用（需要配置 API key）"""
        return self._available and bool(self._api_key)

    def _rate_limit_wait(self):
        """限流等待"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            wait_time = self._min_request_interval - elapsed
            _LOGGER.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        self._last_request_time = time.time()

    def _make_request(self, function: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        发起 Alpha Vantage API 请求

        Args:
            function: API 函数名
            params: 额外参数

        Returns:
            API 响应 JSON
        """
        if not self._api_key:
            raise DataFetchError("Alpha Vantage API key not configured")

        self._rate_limit_wait()

        request_params = {
            "function": function,
            "apikey": self._api_key,
        }
        if params:
            request_params.update(params)

        try:
            response = requests.get(
                ALPHA_VANTAGE_BASE_URL,
                params=request_params,
                headers={"User-Agent": self.get_random_user_agent()},
                timeout=30
            )
            response.raise_for_status()

            # 尝试解析 JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                # 可能是 CSV 格式
                return {"csv_data": response.text}

            # 检查错误响应
            if "Error Message" in data:
                raise DataFetchError(f"Alpha Vantage API error: {data['Error Message']}")

            if "Information" in data:
                info_msg = data["Information"]
                if "rate limit" in info_msg.lower() or "api key" in info_msg.lower():
                    raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit: {info_msg}")
                _LOGGER.warning(f"Alpha Vantage info: {info_msg}")

            if "Note" in data:
                note_msg = data["Note"]
                if "API call frequency" in note_msg:
                    raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit: {note_msg}")

            return data

        except requests.exceptions.RequestException as e:
            raise DataFetchError(f"Alpha Vantage request failed: {e}")

    def _fetch_raw_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取日线数据（Alpha Vantage 主要用于基本面，日线可用但有限制）"""
        try:
            data = self._make_request("TIME_SERIES_DAILY", {
                "symbol": stock_code,
                "outputsize": "compact"  # 最近 100 天
            })

            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                return None

            rows = []
            for date_str, values in time_series.items():
                rows.append({
                    "date": date_str,
                    "open": float(values.get("1. open", 0)),
                    "high": float(values.get("2. high", 0)),
                    "low": float(values.get("3. low", 0)),
                    "close": float(values.get("4. close", 0)),
                    "volume": float(values.get("5. volume", 0)),
                })

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # 过滤日期范围
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df["date"] >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df["date"] <= end_dt]

            return df

        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage daily data fetch failed: {e}")
            return None

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化数据"""
        df["code"] = stock_code
        df["amount"] = None
        df["pct_chg"] = df["close"].pct_change() * 100
        return df

    # ==================== 基本面数据 ====================

    def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取公司概览

        包含：市值、PE、EPS、股息率、52周高低点等

        Args:
            symbol: 美股代码，如 AAPL, MSFT

        Returns:
            公司概览数据字典
        """
        try:
            data = self._make_request("OVERVIEW", {"symbol": symbol})
            if not data or "Symbol" not in data:
                return None
            return data
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage company overview failed for {symbol}: {e}")
            return None

    def get_balance_sheet(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取资产负债表

        Args:
            symbol: 美股代码
            quarterly: 是否获取季度数据（默认 True）

        Returns:
            资产负债表数据
        """
        try:
            data = self._make_request("BALANCE_SHEET", {"symbol": symbol})
            if not data:
                return None

            key = "quarterlyReports" if quarterly else "annualReports"
            reports = data.get(key, [])

            return {
                "symbol": symbol,
                "reports": reports[:4] if quarterly else reports[:3]  # 最近几期
            }
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage balance sheet failed for {symbol}: {e}")
            return None

    def get_income_statement(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取利润表

        Args:
            symbol: 美股代码
            quarterly: 是否获取季度数据（默认 True）

        Returns:
            利润表数据
        """
        try:
            data = self._make_request("INCOME_STATEMENT", {"symbol": symbol})
            if not data:
                return None

            key = "quarterlyReports" if quarterly else "annualReports"
            reports = data.get(key, [])

            return {
                "symbol": symbol,
                "reports": reports[:4] if quarterly else reports[:3]
            }
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage income statement failed for {symbol}: {e}")
            return None

    def get_cash_flow(self, symbol: str, quarterly: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取现金流量表

        Args:
            symbol: 美股代码
            quarterly: 是否获取季度数据（默认 True）

        Returns:
            现金流量表数据
        """
        try:
            data = self._make_request("CASH_FLOW", {"symbol": symbol})
            if not data:
                return None

            key = "quarterlyReports" if quarterly else "annualReports"
            reports = data.get(key, [])

            return {
                "symbol": symbol,
                "reports": reports[:4] if quarterly else reports[:3]
            }
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage cash flow failed for {symbol}: {e}")
            return None

    def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取盈利数据

        Args:
            symbol: 美股代码

        Returns:
            盈利数据（含历史 EPS 和分析师预期）
        """
        try:
            data = self._make_request("EARNINGS", {"symbol": symbol})
            if not data:
                return None
            return data
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage earnings failed for {symbol}: {e}")
            return None

    # ==================== 新闻数据 ====================

    def get_news_sentiment(
        self,
        symbol: str = None,
        topics: str = None,
        limit: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        获取新闻情绪数据

        Args:
            symbol: 股票代码（可选）
            topics: 主题过滤，如 technology, earnings, ipo（可选）
            limit: 返回数量限制

        Returns:
            新闻情绪数据
        """
        try:
            params = {
                "sort": "LATEST",
                "limit": str(min(limit, 200))  # API 最大 200
            }

            if symbol:
                params["tickers"] = symbol
            if topics:
                params["topics"] = topics

            data = self._make_request("NEWS_SENTIMENT", params)
            if not data or "feed" not in data:
                return None

            return {
                "items_count": data.get("items", 0),
                "sentiment_score_definition": data.get("sentiment_score_definition", ""),
                "relevance_score_definition": data.get("relevance_score_definition", ""),
                "feed": data.get("feed", [])
            }
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage news sentiment failed: {e}")
            return None

    def get_insider_transactions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取内部交易数据

        Args:
            symbol: 美股代码

        Returns:
            内部交易数据
        """
        try:
            data = self._make_request("INSIDER_TRANSACTIONS", {"symbol": symbol})
            if not data:
                return None
            return data
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage insider transactions failed for {symbol}: {e}")
            return None

    # ==================== 技术指标 ====================

    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close"
    ) -> Optional[Dict[str, Any]]:
        """
        获取技术指标数据

        Args:
            symbol: 美股代码
            indicator: 指标类型 (SMA, EMA, RSI, MACD, BBANDS, STOCH, ADX, ATR, OBV 等)
            interval: 时间间隔 (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period: 计算周期
            series_type: 价格类型 (close, open, high, low)

        Returns:
            技术指标数据
        """
        try:
            # 映射常用指标名称到 Alpha Vantage 函数名
            indicator_map = {
                "SMA": "SMA",
                "EMA": "EMA",
                "RSI": "RSI",
                "MACD": "MACD",
                "BBANDS": "BBANDS",
                "STOCH": "STOCH",
                "ADX": "ADX",
                "CCI": "CCI",
                "AROON": "AROON",
                "OBV": "OBV",
                "ATR": "ATR",
            }

            func_name = indicator_map.get(indicator.upper(), indicator.upper())

            params = {
                "symbol": symbol,
                "interval": interval,
            }

            # 不同指标需要不同参数
            if func_name in ("SMA", "EMA", "RSI", "ADX", "CCI", "ATR"):
                params["time_period"] = str(time_period)
                params["series_type"] = series_type
            elif func_name == "MACD":
                params["series_type"] = series_type
                # MACD 使用默认周期: fast=12, slow=26, signal=9
            elif func_name == "BBANDS":
                params["time_period"] = str(time_period)
                params["series_type"] = series_type
                params["nbdevup"] = "2"
                params["nbdevdn"] = "2"
            elif func_name == "STOCH":
                pass  # 使用默认 K 和 D 周期

            data = self._make_request(func_name, params)

            if not data:
                return None

            # 查找响应中的技术指标键
            tech_key = None
            for key in data.keys():
                if "Technical Analysis" in key or func_name in key:
                    tech_key = key
                    break

            if not tech_key or not data.get(tech_key):
                return None

            # 转换为列表格式便于使用
            indicators_data = data[tech_key]
            result = []
            for date_str, values in sorted(indicators_data.items(), reverse=True)[:100]:
                entry = {"date": date_str}
                entry.update(values)
                result.append(entry)

            return {
                "symbol": symbol,
                "indicator": func_name,
                "interval": interval,
                "data": result,
            }
        except Exception as e:
            _LOGGER.warning(f"Alpha Vantage technical indicator failed for {symbol}/{indicator}: {e}")
            return None

    # ==================== 辅助方法 ====================

    def format_overview_report(self, overview: Dict[str, Any]) -> str:
        """格式化公司概览报告"""
        if not overview:
            return "无数据"

        lines = [
            f"# {overview.get('Name', '')} ({overview.get('Symbol', '')})",
            f"",
            f"## 基本信息",
            f"- 行业: {overview.get('Industry', '-')}",
            f"- 板块: {overview.get('Sector', '-')}",
            f"- 国家: {overview.get('Country', '-')}",
            f"- 交易所: {overview.get('Exchange', '-')}",
            f"",
            f"## 估值指标",
            f"- 市值: ${self._format_number(overview.get('MarketCapitalization'))}",
            f"- 市盈率(PE): {overview.get('PERatio', '-')}",
            f"- 远期市盈率: {overview.get('ForwardPE', '-')}",
            f"- 市净率(PB): {overview.get('PriceToBookRatio', '-')}",
            f"- 市销率(PS): {overview.get('PriceToSalesRatioTTM', '-')}",
            f"- PEG比率: {overview.get('PEGRatio', '-')}",
            f"",
            f"## 盈利指标",
            f"- 每股收益(EPS): ${overview.get('EPS', '-')}",
            f"- 每股净资产: ${overview.get('BookValue', '-')}",
            f"- 净利润率: {overview.get('ProfitMargin', '-')}",
            f"- 营业利润率: {overview.get('OperatingMarginTTM', '-')}",
            f"- ROE: {overview.get('ReturnOnEquityTTM', '-')}",
            f"- ROA: {overview.get('ReturnOnAssetsTTM', '-')}",
            f"",
            f"## 股息信息",
            f"- 股息率: {overview.get('DividendYield', '-')}",
            f"- 每股股息: ${overview.get('DividendPerShare', '-')}",
            f"- 除息日: {overview.get('ExDividendDate', '-')}",
            f"",
            f"## 价格区间",
            f"- 52周最高: ${overview.get('52WeekHigh', '-')}",
            f"- 52周最低: ${overview.get('52WeekLow', '-')}",
            f"- 50日均价: ${overview.get('50DayMovingAverage', '-')}",
            f"- 200日均价: ${overview.get('200DayMovingAverage', '-')}",
            f"",
            f"## 分析师评级",
            f"- 目标价: ${overview.get('AnalystTargetPrice', '-')}",
            f"- 强烈买入: {overview.get('AnalystRatingStrongBuy', '-')}",
            f"- 买入: {overview.get('AnalystRatingBuy', '-')}",
            f"- 持有: {overview.get('AnalystRatingHold', '-')}",
            f"- 卖出: {overview.get('AnalystRatingSell', '-')}",
            f"- 强烈卖出: {overview.get('AnalystRatingStrongSell', '-')}",
        ]
        return "\n".join(lines)

    def format_news_report(self, news_data: Dict[str, Any], limit: int = 10) -> str:
        """格式化新闻报告"""
        if not news_data or not news_data.get("feed"):
            return "无新闻数据"

        lines = [
            f"# 新闻情绪分析",
            f"",
            f"共 {news_data.get('items_count', 0)} 条新闻",
            f""
        ]

        for i, item in enumerate(news_data["feed"][:limit]):
            title = item.get("title", "")
            source = item.get("source", "")
            time_published = item.get("time_published", "")
            overall_sentiment = item.get("overall_sentiment_label", "")
            sentiment_score = item.get("overall_sentiment_score", 0)

            # 格式化时间
            if time_published:
                try:
                    dt = datetime.strptime(time_published[:8], "%Y%m%d")
                    time_published = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

            lines.extend([
                f"## {i + 1}. {title}",
                f"- 来源: {source}",
                f"- 时间: {time_published}",
                f"- 情绪: {overall_sentiment} ({sentiment_score:.3f})",
                f""
            ])

            # 添加相关股票情绪
            ticker_sentiments = item.get("ticker_sentiment", [])
            if ticker_sentiments:
                lines.append("- 相关股票:")
                for ts in ticker_sentiments[:3]:
                    ticker = ts.get("ticker", "")
                    relevance = ts.get("relevance_score", 0)
                    sent_score = ts.get("ticker_sentiment_score", 0)
                    sent_label = ts.get("ticker_sentiment_label", "")
                    lines.append(f"  - {ticker}: {sent_label} ({sent_score:.3f}), 相关度: {relevance:.3f}")
                lines.append("")

        return "\n".join(lines)

    def _format_number(self, value: str) -> str:
        """格式化大数字"""
        if not value or value == "None":
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
            return value
