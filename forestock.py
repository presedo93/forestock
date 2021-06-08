import torch
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

from utils import EMA, SMA, BBANDS


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

    def prepare_data(self):
        df = pd.read_csv("data/ADAUSDT.csv")

        # Set index to datetime
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        self.df = df.drop(["Close_time", "Quote_av", "Trades", "Tb_base_av", "Tb_quote_av", "Ignore"], axis=1)

        self.df["PCT"] = self.df.Close.pct_change()
        self.df["EMA"] = self.df.Close.ewm(span=12, adjust=False).mean()
        self.df = pd.concat([self.df, BBANDS(self.df.Close)], axis=1)

    def setup(self, stage=None):
        # Scale the all the data
        scaler = MinMaxScaler()
        d_norm = scaler.fit_transform(self.df)
        steps = len(self.df) - self.H_STEPS

        self.stock_h = np.array([d_norm[:, :6][i + self.H_STEPS].copy()
                                        for i in range(steps)])
        # self.ema_h = np.array([d_norm[:, 6:8][i + self.H_STEPS].copy()
        #                                 for i in range(steps)])
        # self.bb_h = np.array([d_norm[8:][i + self.H_STEPS].copy()
        #                                 for i in range(steps)])

        # Target
        tr_h = np.array([d_norm[0][i + self.H_STEPS].copy()
                                        for i in range(steps)])
        self.tr_h = np.expand_dims(tr_h, -1)

    def forward(self, x):
        x, h = self.gru_stock
        x = F.sigmoid(self.fc1(x))
        x = F.linear(self.fc2(x))
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x, h = self.gru_stock
        x = F.sigmoid(self.fc1(x))
        y_hat = F.linear(self.fc2(x))
        loss = F.mse_loss(y_hat, x)
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