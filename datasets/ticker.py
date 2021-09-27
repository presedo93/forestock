import torch
import numpy as np
import pandas as pd
from torch._C import Value
import yfinance as yf
import pytorch_lightning as pl

from tools.ta import BBANDS
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split


class TickerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window: int,
        steps: int,
        *,
        csv_path: str = None,
        ticker: str = None,
        interval: str = None,
        period: str = None,
        split: int = 0.8,
        workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        # Data fetch parameters.
        self.csv_path = csv_path

        self.ticker = ticker
        self.interval = interval
        self.period = period

        # Model size parameters.
        self.window = window
        self.steps = steps

        # Train/test stage parameters.
        self.split = split
        self.batch_size = batch_size
        self.workers = workers

        # Scalers
        self.sc = MinMaxScaler()

    def prepare_data(self) -> None:
        # Fetch the data
        if self.csv_path is None:
            self.df = (
                yf.Ticker(self.ticker).history(self.period, self.interval).interpolate()
            )
        else:
            self.df = pd.read_csv(self.csv_path).set_index("Date")

        if self.df.empty:
            raise ValueError(
                f"\033[1;31m{self.ticker}'s data couldn't be fetched for these period and intervals.\033[0m"
            )
            exit()

        # And discard everything except Open High Low Close and Volume
        self.df = self.df[self.df.columns[:5]]

        # Set index to datetime
        self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")

        # Add Bollinger Bands
        bbands = BBANDS(self.df.Close).fillna(0)
        self.df = pd.concat([self.df, bbands], axis=1)

        # Normalize the data
        self.df_sc = self.sc.fit_transform(self.df)

    def setup(self, stage: Optional[str]) -> None:
        ticker_full = self.window_series(self.df_sc, self.window, self.steps)

        test_size = int(len(ticker_full) * self.split)

        if stage == "fit" or stage is None:
            train_set = Subset(ticker_full, range(0, test_size))
            train_size = int(0.8 * len(train_set))
            val_size = len(train_set) - train_size
            self.ticker_train, self.ticker_val = random_split(
                train_set, [train_size, val_size]
            )

        if stage == "test":
            self.ticker_test = Subset(ticker_full, range(test_size, len(ticker_full)))

        if stage == "predict":
            self.ticker_pred = ticker_full

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
    def window_series(data: np.array, window: int, steps: int) -> TensorDataset:
        x = torch.tensor(data, dtype=torch.float)
        x = x.unfold(0, window, 1)[:-steps]

        y = torch.tensor(data[..., 3], dtype=torch.float).unsqueeze(1)[window:]

        return TensorDataset(x, y)
