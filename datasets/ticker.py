import torch
import numpy as np
import pandas as pd
import yfinance as yf
import pytorch_lightning as pl

from tools.ta import BBANDS, EMA
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split


class TickerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mode: str,
        window: int,
        *,
        csv: str = None,
        ticker: str = None,
        interval: str = None,
        period: str = None,
        target_idx: int = 3,
        split: int = 0.8,
        workers: int = 4,
        batch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        # Clf or reg mode and target column index.
        self.mode = mode.lower()
        self.target_idx = target_idx

        # Data fetch parameters.
        self.csv = csv

        self.ticker = ticker
        self.interval = interval
        self.period = period

        # Window size.
        self.window = window

        # Train/test stage parameters.
        self.split = split
        self.batch_size = batch_size
        self.workers = workers

        # Scalers
        self.sc = MinMaxScaler()

    def prepare_data(self) -> None:
        # Fetch the data
        if self.csv is None:
            self.df = (
                yf.Ticker(self.ticker).history(self.period, self.interval).interpolate()
            )
        else:
            self.df = pd.read_csv(self.csv).set_index("Date")

        if self.df.empty:
            raise ValueError(
                f"\033[1;31m{self.ticker} data couldn't be fetched for these period and intervals.\033[0m"
            )
            exit()

        # And discard everything except Open High Low Close and Volume - Columns 0 to 4
        self.df = self.df[self.df.columns[:5]]

        # Set index to datetime
        self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")

        # Add Bollinger Bands - Columns 5 to 7
        bbands = BBANDS(self.df.Close).fillna(0)
        self.df = pd.concat([self.df, bbands], axis=1)

        # Add percentage change - Column 8
        self.df["PCT"] = self.df["Close"].pct_change(fill_method="ffill")

        # Add EMA 50 & EMA 200 - Columns 9 to 10
        self.df["EMA50"] = EMA(self.df["Close"], 50, fillna=True)
        self.df["EMA200"] = EMA(self.df["Close"], 200, fillna=True)

        # Normalize the data
        self.data = self.sc.fit_transform(self.df)

        self.data = self.sc.inverse_transform(self.data)

        # Get the pd.Series that is going to be used as target
        self.target = self.data[..., self.target_idx]

        if self.mode == "clf":
            self.target = self.target_clf(self.target)

    def setup(self, stage: Optional[str]) -> None:
        ticker_data = self.window_series(self.data, self.target, self.window, self.mode)

        test_size = int(len(ticker_data) * self.split)

        if stage == "fit" or stage is None:
            train_set = Subset(ticker_data, range(0, test_size))
            train_size = int(0.8 * len(train_set))
            val_size = len(train_set) - train_size
            self.ticker_train, self.ticker_val = random_split(
                train_set, [train_size, val_size]
            )

        if stage == "test":
            self.ticker_test = Subset(ticker_data, range(test_size, len(ticker_data)))

        if stage == "predict":
            self.ticker_pred = ticker_data

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.ticker_train, batch_size=self.batch_size, num_workers=self.workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ticker_val, batch_size=self.batch_size, num_workers=self.workers
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ticker_test, batch_size=self.batch_size, num_workers=self.workers
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ticker_pred, batch_size=self.batch_size, num_workers=self.workers
        )

    @staticmethod
    def window_series(
        data: np.array, target: np.array, window: int, mode: str
    ) -> TensorDataset:
        x = torch.tensor(data, dtype=torch.float)
        x = x.unfold(0, window, 1)[:-1]
        y = torch.tensor(target, dtype=torch.float).unsqueeze(1)[window:]

        return TensorDataset(x, y)

    @staticmethod
    def target_clf(array: np.array) -> np.array:
        target = pd.Series(array).pct_change(1).shift(-1)
        target[target > 0] = 1
        target[target <= 0] = 0

        return target.to_numpy()
