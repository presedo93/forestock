import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from models.regression import LitForestockReg
from datasets.ticker import TickerDataModule

# def process_reg()


def test(args: argparse.Namespace) -> None:
    ticker = TickerDataModule(args.data, 50, 1)
    forestock = LitForestockReg.load_from_checkpoint(args.weights, h_steps=50)

    trainer = pl.Trainer.from_argparse_args(args)
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
    print(f"\033[1;32mScaled MSE: {round(s_mse, 2)}\033[0m")

    plt.gcf().set_size_inches(16, 12, forward=True)
    plt.plot(y_hat[:-1], label="real")
    plt.plot(y[:-1], label="predicted")

    plt.savefig("test.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data")
    parser.add_argument("--weights", type=str, help="Path to the weights to load")
    args = parser.parse_args()

    test(args)
