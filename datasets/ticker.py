import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tools.ta import apply_ta
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Union
from tools.utils import get_yfinance, get_from_csv
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split


class TickerDataModule(pl.LightningDataModule):
    """Custom ticker data module class that handles the desired data. It
    can work with CSV files, or fetch data from yfinance. It can transform
    the data to perform classificiation or regression tasks.
    """

    def __init__(
        self,
        mode: str,
        window: int,
        *,
        csv: str = None,
        ticker: str = None,
        interval: str = None,
        period: str = None,
        start: str = None,
        end: str = None,
        target_idx: int = 3,
        split: float = 0.8,
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

        self.start = start
        self.end = end

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
            if self.period is None:
                self.df = get_yfinance(
                    self.ticker, self.interval, start=self.start, end=self.end
                )
            else:
                self.df = get_yfinance(self.ticker, self.interval, self.period)
        else:
            self.df = get_from_csv(self.csv)

        if self.df.empty:
            raise ValueError(
                f"{self.ticker} data couldn't be fetched for these period and intervals."
            )

        # And apply the technical indicators
        self.df = apply_ta(self.df)

        # Normalize the data
        train_size = int(self.split * len(self.df.index))
        self.sc = self.sc.fit(self.df[:train_size])
        self.data = self.sc.transform(self.df)

        # Get the pd.Series that is going to be used as target
        if self.mode.lower() == "clf":
            self.target = self.target_clf(self.df, self.window)
        else:
            self.target = self.data[..., self.target_idx]

    def setup(self, stage: Optional[str]) -> None:
        ticker_data = self.window_series(self.data, self.target, self.window, self.mode)

        test_size = int(len(ticker_data) * self.split)

        if stage == "fit" or stage is None:
            train_set = Subset(ticker_data, range(0, test_size))
            train_size = int(0.9 * len(train_set))
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
        """Transform the data in windows of n steps.

        Args:
            data (np.array): data to trasnform.
            target (np.array): target for training.
            window (int): number of steps for earch window.
            mode (str): clf or reg

        Returns:
            TensorDataset: Tensor Dataset with training and targets data.
        """
        x = torch.tensor(data, dtype=torch.float)
        x = x.unfold(0, window, 1)[:-1]
        y = torch.tensor(target, dtype=torch.float).unsqueeze(1)[window:]

        return TensorDataset(x, y)

    @staticmethod
    def target_clf(data: pd.DataFrame, window: int) -> np.array:
        """This method "creates" the targets for the classification task.
        In the future, it will accept conditions to build the "strategy"

        Args:
            data (pd.DataFrame): dataframe will the ohlc data and the ta.
            window (int): window

        Returns:
            np.array: 1s where the conditions are met.
        """
        half = int(window / 2)
        target = np.where(
            (data.Close < data.EMA50)
            & (data.Close.shift(-half) < data.EMA50.shift(-half)),
            1,
            0,
        )

        return target
