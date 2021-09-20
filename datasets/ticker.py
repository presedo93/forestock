import torch
import numpy as np
import pandas as pd
from torch._C import Value
import yfinance as yf
import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

from utils import BBANDS


class TickerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ticker: str,
        interval: str,
        period: str,
        window: int,
        steps: int,
        workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        # Data fetch parameters.
        self.ticker = ticker
        self.interval = interval
        self.period = period

        # Model size parameters.
        self.window = window
        self.steps = steps

        # Train/test stage parameters.
        self.batch_size = batch_size
        self.workers = workers

        self.sc = MinMaxScaler()

    def prepare_data(self) -> None:
        # Fetch the data
        df = yf.Ticker(self.ticker).history(self.period, self.interval).interpolate()

        if df.empty:
            raise ValueError(f"{self.ticker}'s data couldn't be fetched for these period and intervals.")
            exit()

        # And discard everything except Open High Low Close and Volume
        df = df[df.columns[:5]]

        # Set index to datetime
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

        # Add Bollinger Bands
        bbands = BBANDS(df.Close).fillna(0)
        df = pd.concat([df, bbands], axis=1)

        # Normalize the data
        self.df_sc = self.sc.fit_transform(df)

    def setup(self, stage: Optional[str]) -> None:
        ticker_full = self.window_series(self.df_sc, self.window, self.steps)

        test_size = int(len(ticker_full) * 0.8)
        if stage == "test" or stage is None:
            self.ticker_test = Subset(ticker_full, range(test_size, len(ticker_full)))

        if stage == "fit" or stage is None:
            train_set = Subset(ticker_full, range(0, test_size))
            train_size = int(0.8 * len(train_set))
            val_size = len(train_set) - train_size
            self.ticker_train, self.ticker_val = random_split(
                train_set, [train_size, val_size]
            )

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
            self.ticker_test, batch_size=self.batch_size, num_workers=self.workers
        )

    @staticmethod
    def window_series(data: np.array, window: int, steps: int) -> TensorDataset:
        x = torch.tensor(data, dtype=torch.float)
        x = x.unfold(0, window, 1)[:-steps]

        y = torch.tensor(data[..., 3], dtype=torch.float).unsqueeze(1)[window:]

        return TensorDataset(x, y)
