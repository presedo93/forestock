import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

from utils import EMA, SMA, BBANDS


class TickerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        window: int,
        steps: int,
        workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window = window
        self.steps = steps

        self.batch_size = batch_size
        self.workers = workers

        self.sc = MinMaxScaler()

    def prepare_data(self) -> None:
        df = pd.read_csv(self.data_dir)

        # Set index to datetime
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        df = df.drop(
            ["Close_time", "Quote_av", "Trades", "Tb_base_av", "Tb_quote_av", "Ignore"],
            axis=1,
        )

        # Add the percentage change, an exponential ma and Bollinger Bands
        # df["PCT"] = df.Close.pct_change()
        # df["EMA"] = df.Close.ewm(span=12, adjust=False).mean()
        # df = pd.concat([df, BBANDS(df.Close)], axis=1)

        self.df_sc = self.sc.fit_transform(df)

    def setup(self, stage: Optional[str]) -> None:
        ticker_full = self.window_series(self.df_sc)

        test_size = int(len(ticker_full) * 0.8)
        self.ticker_test = Subset(ticker_full, range(test_size, len(ticker_full)))
        train_set = Subset(ticker_full, range(0, test_size))

        train_size = int(0.9 * len(train_set))
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

    def window_series(self, data: np.array) -> TensorDataset:
        x = torch.tensor(data, dtype=torch.float)
        x = x.unfold(0, self.window, self.steps)[:-1]

        y = torch.tensor(data[..., 3], dtype=torch.float)
        y = y.unsqueeze(1)[self.window :]

        return TensorDataset(x, y)
