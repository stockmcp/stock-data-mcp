"""
技术指标计算模块

提供股票和加密货币的技术分析指标计算功能。
所有指标计算均采用向量化实现，性能优异。
"""

import pandas as pd
import numpy as np
from typing import Optional, List


# ==================== 列定义 ====================

MA_COLUMNS = ["MA5", "MA10", "MA20", "MA30", "MA60"]

INDICATOR_COLUMNS = [
    "MACD", "DIF", "DEA",
    "KDJ.K", "KDJ.D", "KDJ.J",
    "RSI6", "RSI", "RSI24",
    "BOLL.U", "BOLL.M", "BOLL.L", "BOLL.W",
    "OBV", "VMA5", "VMA10",
    "ATR",
    "ADX", "+DI", "-DI",
    "CCI", "WR", "VWAP",
]

STOCK_PRICE_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "换手率"] + MA_COLUMNS + INDICATOR_COLUMNS
CRYPTO_PRICE_COLUMNS = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额"] + MA_COLUMNS + INDICATOR_COLUMNS


# ==================== 单独指标计算函数 ====================

def calc_ma(series: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    计算多周期移动平均线

    Args:
        series: 价格序列（通常是收盘价）
        periods: 周期列表，默认 [5, 10, 20, 30, 60]

    Returns:
        包含各周期 MA 的 DataFrame
    """
    if periods is None:
        periods = [5, 10, 20, 30, 60]

    result = pd.DataFrame(index=series.index)
    for period in periods:
        result[f"MA{period}"] = series.rolling(window=period, min_periods=1).mean()
    return result


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    计算 MACD 指标

    Args:
        close: 收盘价序列
        fast: 快线 EMA 周期，默认 12
        slow: 慢线 EMA 周期，默认 26
        signal: 信号线 EMA 周期，默认 9

    Returns:
        包含 DIF, DEA, MACD 的 DataFrame
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2

    return pd.DataFrame({
        "DIF": dif,
        "DEA": dea,
        "MACD": macd
    }, index=close.index)


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> pd.DataFrame:
    """
    计算 KDJ 指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: RSV 计算周期，默认 9

    Returns:
        包含 KDJ.K, KDJ.D, KDJ.J 的 DataFrame
    """
    low_min = low.rolling(window=period, min_periods=1).min()
    high_max = high.rolling(window=period, min_periods=1).max()

    rsv = (close - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)  # 处理除零情况

    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({
        "KDJ.K": k,
        "KDJ.D": d,
        "KDJ.J": j
    }, index=close.index)


def calc_rsi(close: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    计算多周期 RSI 指标

    Args:
        close: 收盘价序列
        periods: 周期列表，默认 [6, 12, 14, 24]

    Returns:
        包含各周期 RSI 的 DataFrame
    """
    if periods is None:
        periods = [6, 12, 14, 24]

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    result = pd.DataFrame(index=close.index)
    for period in periods:
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        col_name = "RSI" if period == 14 else f"RSI{period}"
        result[col_name] = 100 - (100 / (1 + rs))

    return result


def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    计算布林带指标

    Args:
        close: 收盘价序列
        period: 移动平均周期，默认 20
        std_dev: 标准差倍数，默认 2.0

    Returns:
        包含 BOLL.U, BOLL.M, BOLL.L, BOLL.W 的 DataFrame
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle * 100  # 布林带宽度百分比

    return pd.DataFrame({
        "BOLL.U": upper,
        "BOLL.M": middle,
        "BOLL.L": lower,
        "BOLL.W": width
    }, index=close.index)


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算 OBV（能量潮指标）

    Args:
        close: 收盘价序列
        volume: 成交量序列

    Returns:
        OBV 序列
    """
    price_diff = close.diff()
    direction = np.sign(price_diff).fillna(0)
    return (direction * volume).fillna(0).cumsum()


def calc_vma(volume: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    计算成交量移动平均线

    Args:
        volume: 成交量序列
        periods: 周期列表，默认 [5, 10]

    Returns:
        包含 VMA 的 DataFrame
    """
    if periods is None:
        periods = [5, 10]

    result = pd.DataFrame(index=volume.index)
    for period in periods:
        result[f"VMA{period}"] = volume.rolling(window=period, min_periods=1).mean()
    return result


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 ATR（真实波幅）

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期，默认 14

    Returns:
        ATR 序列
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """
    计算 ADX（平均趋向指标）及 +DI/-DI

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期，默认 14

    Returns:
        包含 ADX, +DI, -DI 的 DataFrame
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # +DM 和 -DM
    high_diff = high.diff()
    low_diff = low.diff()
    plus_dm = high_diff.where((high_diff > low_diff.abs()) & (high_diff > 0), 0)
    minus_dm = low_diff.abs().where((low_diff.abs() > high_diff) & (low_diff < 0), 0)

    # +DI 和 -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # DX 和 ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return pd.DataFrame({
        "ADX": adx,
        "+DI": plus_di,
        "-DI": minus_di
    }, index=close.index)


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    计算 CCI（商品通道指标）

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期，默认 20

    Returns:
        CCI 序列
    """
    tp = (high + low + close) / 3  # 典型价格
    tp_sma = tp.rolling(window=period).mean()
    tp_mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    return (tp - tp_sma) / (0.015 * tp_mad)


def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 Williams %R（威廉指标）

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期，默认 14

    Returns:
        WR 序列
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def calc_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算 VWAP（成交量加权平均价）

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列

    Returns:
        VWAP 序列
    """
    tp = (high + low + close) / 3  # 典型价格
    tp_vol = tp * volume
    return tp_vol.cumsum() / volume.cumsum()


# ==================== 综合计算函数 ====================

def add_technical_indicators(
    df: pd.DataFrame,
    close: pd.Series,
    low: pd.Series,
    high: pd.Series,
    volume: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    为 DataFrame 添加所有技术指标

    Args:
        df: 目标 DataFrame（原地修改）
        close: 收盘价序列
        low: 最低价序列
        high: 最高价序列
        volume: 成交量序列（可选）

    Returns:
        添加了技术指标的 DataFrame
    """
    # 移动平均线
    ma_df = calc_ma(close)
    for col in ma_df.columns:
        df[col] = ma_df[col]

    # MACD
    macd_df = calc_macd(close)
    for col in macd_df.columns:
        df[col] = macd_df[col]

    # KDJ
    kdj_df = calc_kdj(high, low, close)
    for col in kdj_df.columns:
        df[col] = kdj_df[col]

    # RSI
    rsi_df = calc_rsi(close)
    for col in rsi_df.columns:
        df[col] = rsi_df[col]

    # 布林带
    boll_df = calc_bollinger(close)
    for col in boll_df.columns:
        df[col] = boll_df[col]

    # OBV 和 VMA（需要成交量）
    if volume is not None:
        df["OBV"] = calc_obv(close, volume)
        vma_df = calc_vma(volume)
        for col in vma_df.columns:
            df[col] = vma_df[col]

    # ATR
    df["ATR"] = calc_atr(high, low, close)

    # ADX 和 DMI
    adx_df = calc_adx(high, low, close)
    for col in adx_df.columns:
        df[col] = adx_df[col]

    # CCI
    df["CCI"] = calc_cci(high, low, close)

    # Williams %R
    df["WR"] = calc_williams_r(high, low, close)

    # VWAP（需要成交量）
    if volume is not None:
        df["VWAP"] = calc_vwap(high, low, close, volume)

    return df


# ==================== 信号生成辅助函数 ====================

def detect_macd_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测 MACD 金叉/死叉信号

    Args:
        df: 包含 DIF 和 DEA 的 DataFrame

    Returns:
        添加了 MACD_CROSS 列的 DataFrame（1=金叉, -1=死叉, 0=无信号）
    """
    if "DIF" not in df.columns or "DEA" not in df.columns:
        return df

    prev_dif = df["DIF"].shift(1)
    prev_dea = df["DEA"].shift(1)

    # 金叉：DIF 从下向上穿越 DEA
    golden_cross = (prev_dif <= prev_dea) & (df["DIF"] > df["DEA"])
    # 死叉：DIF 从上向下穿越 DEA
    death_cross = (prev_dif >= prev_dea) & (df["DIF"] < df["DEA"])

    df["MACD_CROSS"] = 0
    df.loc[golden_cross, "MACD_CROSS"] = 1
    df.loc[death_cross, "MACD_CROSS"] = -1

    return df


def detect_kdj_signal(df: pd.DataFrame, overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
    """
    检测 KDJ 超买/超卖信号

    Args:
        df: 包含 KDJ.K 和 KDJ.D 的 DataFrame
        overbought: 超买阈值，默认 80
        oversold: 超卖阈值，默认 20

    Returns:
        添加了 KDJ_SIGNAL 列的 DataFrame（1=超卖反弹, -1=超买回落, 0=中性）
    """
    if "KDJ.K" not in df.columns or "KDJ.D" not in df.columns:
        return df

    df["KDJ_SIGNAL"] = 0

    # 超卖区金叉
    oversold_zone = (df["KDJ.K"] < oversold) | (df["KDJ.D"] < oversold)
    k_cross_d_up = (df["KDJ.K"].shift(1) <= df["KDJ.D"].shift(1)) & (df["KDJ.K"] > df["KDJ.D"])
    df.loc[oversold_zone & k_cross_d_up, "KDJ_SIGNAL"] = 1

    # 超买区死叉
    overbought_zone = (df["KDJ.K"] > overbought) | (df["KDJ.D"] > overbought)
    k_cross_d_down = (df["KDJ.K"].shift(1) >= df["KDJ.D"].shift(1)) & (df["KDJ.K"] < df["KDJ.D"])
    df.loc[overbought_zone & k_cross_d_down, "KDJ_SIGNAL"] = -1

    return df


def detect_ma_trend(df: pd.DataFrame) -> str:
    """
    检测均线趋势（多头/空头/震荡）

    Args:
        df: 包含 MA5, MA10, MA20 的 DataFrame

    Returns:
        趋势描述字符串
    """
    if len(df) < 1:
        return "数据不足"

    latest = df.iloc[-1]

    ma5 = latest.get("MA5")
    ma10 = latest.get("MA10")
    ma20 = latest.get("MA20")

    if ma5 is None or ma10 is None or ma20 is None:
        return "指标缺失"

    if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
        return "指标缺失"

    # 多头排列：MA5 > MA10 > MA20
    if ma5 > ma10 > ma20:
        return "多头排列"

    # 空头排列：MA5 < MA10 < MA20
    if ma5 < ma10 < ma20:
        return "空头排列"

    return "震荡整理"
