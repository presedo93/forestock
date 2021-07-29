import torch
import argparse
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl

from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional, Union
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

from utils import str2bool, EMA, SMA, BBANDS


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

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
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


class LitForestock(pl.LightningModule):
    def __init__(self):
        super().__init__()
        h_steps = 50

        self.ohlc = torch.nn.Sequential(
            torch.nn.Conv1d(5, 128, 3),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.GRU(
                input_size=int(h_steps / 2) - 1,
                hidden_size=h_steps,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.fc1 = torch.nn.Linear(h_steps * 2, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.ohlc(x)
        x = torch.sigmoid(x[:, -1, :])
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/train", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/valid", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss", loss)

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> Any:
        x, y = batch
        y_hat = self(x)

        return y, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduluer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=1
        )
        return {
            "optimizer": optimizer,
            "interval": "epoch",
            "lr_scheduler": {"scheduler": scheduluer, "monitor": "loss/valid"},
        }


def train(args: argparse.Namespace):
    # init model
    ticker = TickerDataModule("data/ADAUSDT.csv", 50, 1)
    forestock = LitForestock()

    tb_logger = pl_loggers.TensorBoardLogger("tb_logs/", name='FST', default_hp_metric=False)
    early_stopping = EarlyStopping("loss/valid")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        gpus=1, logger=tb_logger, callbacks=[early_stopping, lr_monitor]
    )
    trainer.fit(forestock, ticker)

    trainer.test(forestock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--load",
        type=str2bool,
        default="false",
        help="Load weights given by --weights",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to the weights to load",
    )

    train(parser.parse_args())
