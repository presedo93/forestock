import torch
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl

from typing import Tuple
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import EMA, SMA, BBANDS


class TickerDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.window = 50
        self.steps = 1

    def prepare_data(self) -> None:
        df = pd.read_csv("data/ADAUSDT.csv")

        # Set index to datetime
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        df = df.drop(["Close_time", "Quote_av", "Trades", "Tb_base_av", "Tb_quote_av", "Ignore"], axis=1)

        df["PCT"] = df.Close.pct_change()
        df["EMA"] = df.Close.ewm(span=12, adjust=False).mean()
        self.df = pd.concat([df, BBANDS(df.Close)], axis=1)

    def setup(self, stage=None):
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(self.df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    def series_data(self, data: np.array) -> Tuple[torch.tensor, torch.tensor]:

        x = torch.tensor(data).unfold(1, self.window, self.steps)
        y = torch.tensor(data[0]).unfold(0, 1, 1)[self.window:]

        return x, y

class LitForestock(pl.LightningModule):
    H_STEPS = 50

    def __init__(self):
        super().__init__()
        self.gru_stock = torch.nn.Sequential(
            torch.nn.Conv1d(50, 32, 7),
            torch.nn.MaxPool1d(3, stride=2),
            torch.nn.GRU(hidden_size=50, num_layers=2, bidirectional=True)
        )

        self.fc1 = torch.nn.Linear()
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
    dataset = None
    train_loader = DataLoader(dataset, num_workers=4)

    # init model
    forestock = LitForestock()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(forestock, train_loader)


if __name__ == '__main__':
    train()