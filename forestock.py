import torch
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl

from typing import Tuple, Optional
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split

from utils import EMA, SMA, BBANDS


class TickerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.window = 50
        self.steps = 1

        self.batch_size = batch_size
        self.data_dir = data_dir

    def prepare_data(self) -> None:
        df = pd.read_csv(self.data_dir)

        # Set index to datetime
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        df = df.drop(["Close_time", "Quote_av", "Trades", "Tb_base_av", "Tb_quote_av", "Ignore"], axis=1)

        df["PCT"] = df.Close.pct_change()
        df["EMA"] = df.Close.ewm(span=12, adjust=False).mean()
        df = pd.concat([df, BBANDS(df.Close)], axis=1)

        scaler = MinMaxScaler()
        self.df_sc = scaler.fit_transform(df)

    def setup(self, stage: Optional[str] = None) -> None:
        ticker_full = self.window_series(self.df_sc)

        train_size = int(0.8 * len(ticker_full[1]))
        test_size = len(ticker_full) - train_size
        train_dataset, self.ticker_test = random_split(ticker_full, [train_size, test_size])

        train_size = int(0.7 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.ticker_train, self.ticker_val = random_split(train_dataset, [train_size, val_size])

        tmp = 2

    def train_dataloader(self):
        return DataLoader(self.ticker_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ticker_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ticker_test, batch_size=self.batch_size)

    def window_series(self, data: np.array) -> Tuple[torch.tensor, torch.tensor]:
        x = torch.tensor(data)
        x = x.unfold(0, self.window, self.steps).reshape(-1, self.window, x.shape[1])

        y = torch.tensor(data[..., 3])
        y = y.unsqueeze(0).reshape(-1, 1)[self.window - 1:]

        return (x, y)

class LitForestock(pl.LightningModule):
    H_STEPS = 50

    def __init__(self):
        super().__init__()
        self.gru_stock = torch.nn.Sequential(
            torch.nn.Conv1d(50, 32, 7),
            torch.nn.MaxPool1d(3, stride=2),
            torch.nn.GRU(input_size=32, hidden_size=50, num_layers=2, bidirectional=True)
        )

        self.fc1 = torch.nn.Linear(50, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, h = self.gru_stock
        x = F.sigmoid(self.fc1(x))
        x = F.linear(self.fc2(x))
        return x

    def training_step(self, batch):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    # init model
    ticker = TickerDataModule("data/ADAUSDT.csv")
    forestock = LitForestock()
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(forestock, ticker)


if __name__ == '__main__':
    train()