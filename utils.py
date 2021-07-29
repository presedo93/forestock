import argparse
import pandas as pd


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def SMA(series: pd.Series, periods: int = 9, fillna: bool = False) -> pd.Series:
    """
    Simple Moving Average (EMA).
    Rolling mean in pandas lingo. Also known as 'MA'.
    The simple moving average (SMA) is the most basic of the moving averages used for trading.
    """
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def EMA(series: pd.Series, periods: int = 9, fillna: bool = False) -> pd.Series:
    """
    Exponential Moving Average (EMA).
    Like all moving average indicators, they are much better suited for trending markets.
    When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and vice-versa for a down trend.
    EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge their validity.
    """
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def BBANDS(
    series, periods: int = 20, ma: pd.Series = None, stds: float = 2
) -> pd.DataFrame:
    """
    Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
    Volatility is based on the standard deviation, which changes as volatility increases and decreases.
    The bands automatically widen when volatility increases and narrow when volatility decreases.
    This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
    Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
    """
    std = pd.Series(series).rolling(window=periods).std()

    if not isinstance(ma, pd.Series):
        middle_band = pd.Series(SMA(series, periods), name="BB_MIDDLE")
    else:
        middle_band = pd.Series(ma, name="BB_MIDDLE")

    upper_bb = pd.Series(middle_band + (stds * std), name="BB_UPPER")
    lower_bb = pd.Series(middle_band - (stds * std), name="BB_LOWER")

    return pd.concat([upper_bb, middle_band, lower_bb], axis=1)
