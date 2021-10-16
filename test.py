import argparse
import numpy as np
import matplotlib.figure as fg
import pytorch_lightning as pl

from typing import Tuple
from models import model_picker
from tools.plots import plot_figure
from tools.progress import StProgressBar
from datasets.ticker import TickerDataModule
from tools.utils import get_checkpoint_hparams, process_output


def test(args: argparse.Namespace, is_st: bool = False) -> Tuple[fg.Figure, float]:
    model, check_path, hp = get_checkpoint_hparams(args.checkpoint)

    ticker = TickerDataModule(hp["mode"], hp["window"], **vars(args))
    forestock = model_picker(model).load_from_checkpoint(check_path)

    callbacks = []
    if is_st:
        callbacks += [StProgressBar()]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    predicts = trainer.predict(forestock, datamodule=ticker)

    y_true, y_hat, metric = process_output(predicts, ticker.sc, hp["mode"])

    # Save the image in the ticker_tested folder
    price = ticker.df.Close.to_numpy()
    save_path = "./tickers_test"
    fig = plot_figure(price, y_true, y_hat, save_path, hp["mode"], split=0.0)

    return fig, metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Path to the checkpoint to load"
    )
    parser.add_argument("-c", "--csv", type=str, help="Path to the CSV data")
    parser.add_argument("-t", "--ticker", type=str, help="Ticker name")
    parser.add_argument("-i", "--interval", type=str, help="Interval of time")
    parser.add_argument("-p", "--period", type=str, help="Num of ticks to fetch")

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    test(args)
