import argparse
import numpy as np
import pytorch_lightning as pl

from typing import Tuple
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from models import model_picker
from tools.plots import plot_figure
from tools.progress import StProgressBar
from datasets.ticker import TickerDataModule
from tools.utils import process_output, get_name_from_args


def train(
    args: argparse.Namespace, is_streamlit: bool = False
) -> Tuple[np.array, np.array, np.array, float]:
    args.outs = 1
    ticker = TickerDataModule(**vars(args))
    forestock = model_picker(args.version)(**vars(args))

    name = get_name_from_args(args)
    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/",
        name=name,
        version=f"{args.version}_{args.mode.lower()}",
        default_hp_metric=False,
    )
    early_stopping = EarlyStopping("loss/valid", min_delta=1e-7)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [early_stopping, lr_monitor]
    if is_streamlit:
        progress_bar = StProgressBar()
        callbacks += [progress_bar]

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=callbacks,
    )

    trainer.fit(forestock, ticker)
    trainer.test(forestock)

    # Evaluate and plot the test set
    predicts = trainer.predict(forestock, datamodule=ticker)
    y_true, y_hat, metric = process_output(predicts, ticker.sc, args.mode)

    # Save the image in the tb_logs subfolder
    price = ticker.df.Close.to_numpy()
    save_path = f"tb_logs/{name}/{args.version}_{args.mode.lower()}"
    plot_figure(price, y_true, y_hat, save_path, args.mode, split=0.8)

    return price, y_true, y_hat, metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="CLF or REG")
    parser.add_argument("-c", "--csv", type=str, help="Path to the CSV data")
    parser.add_argument("-t", "--ticker", type=str, help="Ticker name")
    parser.add_argument("-v", "--version", type=str, help="Training model used")
    parser.add_argument("-i", "--interval", type=str, help="Interval of time")
    parser.add_argument("-p", "--period", type=str, help="Num of ticks to fetch")
    parser.add_argument(
        "-w", "--window", type=int, default=50, help="Num. of days to look back"
    )

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
