import pandas as pd


def apply_ta(df: pd.DataFrame) -> pd.DataFrame:
    # Discard everything except Open High Low Close and Volume - Columns 0 to 4
    df = df[df.columns[:5]]

    # Set index to datetime
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

    # Add Bollinger Bands - Columns 5 to 7
    bbands = BBANDS(df.Close).fillna(0)
    df = pd.concat([df, bbands], axis=1)

    # Add percentage change - Column 8
    df["PCT"] = df["Close"].pct_change(fill_method="ffill")

    # Add EMA 50 & EMA 200 - Columns 9 to 10
    df["EMA50"] = EMA(df["Close"], 50, fillna=True)
    df["EMA200"] = EMA(df["Close"], 200, fillna=True)

    return df


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
