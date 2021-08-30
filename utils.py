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
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def EMA(series: pd.Series, periods: int = 9, fillna: bool = False) -> pd.Series:
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def BBANDS(series, periods: int = 20, stds: float = 2) -> pd.DataFrame:
    std = pd.Series(series).rolling(window=periods).std()

    middle_band = pd.Series(SMA(series, periods), name="BB_MIDDLE")
    upper_bb = pd.Series(middle_band + (stds * std), name="BB_UPPER")
    lower_bb = pd.Series(middle_band - (stds * std), name="BB_LOWER")

    return pd.concat([upper_bb, middle_band, lower_bb], axis=1)
