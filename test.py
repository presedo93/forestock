import torch
import argparse
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from forestock import LitForestock, TickerDataModule

from utils import str2bool


def test(args: argparse.Namespace):
    # init model
    ticker = TickerDataModule("data/ADAUSDT.csv", 50, 1)
    forestock = LitForestock.load_from_checkpoint("tb_logs/FST/version_0/checkpoints/epoch=10-step=42470.ckpt")

    trainer = pl.Trainer(gpus=1)
    predicts = trainer.predict(forestock, datamodule=ticker)

    # predicts is a list of tuples of tensors...
    y = torch.cat(list(map(lambda x: x[0], predicts)))
    y = y.squeeze(1).cpu().numpy()

    y_hat = torch.cat(list(map(lambda x: x[1], predicts)))
    y_hat = y_hat.squeeze(1).cpu().numpy()

    # Unnormalize data
    y = (y - ticker.sc.min_[3]) / ticker.sc.scale_[3]
    y_hat = (y_hat - ticker.sc.min_[3]) / ticker.sc.scale_[3]

    r_mse = np.mean(np.square(y_hat - y))
    s_mse = r_mse / (np.max(y_hat) - np.min(y_hat)) * 100
    print(f'\033[1;32mScaled MSE: {round(s_mse, 2)}\033[0m')

    plt.gcf().set_size_inches(16, 12, forward=True)
    plt.plot(y_hat[:-1], label='real')
    plt.plot(y[:-1], label='predicted')

    plt.savefig("test.png")


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

    test(parser.parse_args())