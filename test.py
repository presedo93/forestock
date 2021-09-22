import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from models.regression import LitForestockReg
from datasets.ticker import TickerDataModule
from tools.utils import get_checkpoint_hparams


def process_reg_output(
    predicts: list, scaler: MinMaxScaler, plot: bool = True, name: str = "test"
) -> Tuple[np.array, np.array]:
    y = torch.cat(list(map(lambda x: x[0], predicts)))
    y = y.squeeze(1).cpu().numpy()

    y_hat = torch.cat(list(map(lambda x: x[1], predicts)))
    y_hat = y_hat.squeeze(1).cpu().numpy()

    # Unnormalize data
    y = (y - scaler.min_[3]) / scaler.scale_[3]
    y_hat = (y_hat - scaler.min_[3]) / scaler.scale_[3]

    r_mse = np.mean(np.square(y_hat - y))
    s_mse = r_mse / (np.max(y_hat) - np.min(y_hat)) * 100
    print(f"\033[1;32mScaled MSE: {round(s_mse, 2)}\033[0m")

    if plot:
        plt.gcf().set_size_inches(16, 12, forward=True)
        plt.plot(y_hat[:-1], label="real")
        plt.plot(y[:-1], label="predicted")
        plt.legend()

        plt.savefig(f"{name}.png")

    return y, y_hat


def test(args: argparse.Namespace) -> None:
    check_path, hp = get_checkpoint_hparams(args.checkpoint)

    ticker = TickerDataModule(
        hp["ticker"], hp["interval"], hp["period"], hp["window"], hp["steps"]
    )
    forestock = LitForestockReg.load_from_checkpoint(check_path)

    trainer = pl.Trainer.from_argparse_args(args)
    predicts = trainer.predict(forestock, datamodule=ticker)

    process_reg_output(predicts, ticker.sc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    # parser.add_argument("--data", type=str, help="Path to the data")
    parser.add_argument("-t", "--ticker", type=str, help="Ticker name")
    parser.add_argument("-i", "--interval", type=str, help="Interval of time")
    parser.add_argument("-p", "--period", type=str, help="Num of ticks to fetch")

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    test(args)
