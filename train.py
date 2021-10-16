import argparse
import matplotlib.figure as fg
import pytorch_lightning as pl

from typing import Tuple
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from models import model_picker
from tools.plots import plot_figure
from tools.progress import StProgressBar
from datasets.ticker import TickerDataModule
from tools.utils import process_output, prepare_args


def train(args: argparse.Namespace, is_st: bool = False) -> Tuple[fg.Figure, float]:
    """Trains a moodel on the dataset. It also performs the test stage and
    the makes predictions in the testset to plot the results.

    Args:
        args (argparse.Namespace): parameters and config used to generate
        the dataset and to create/load the model.
        is_st (bool, optional): checks if the method is called from a
        streamlit dashboard. Defaults to False.

    Returns:
        Tuple[fg.Figure, float]: a figure with the predictions and the metric from it.
    """
    # Prepare the data for the different stages (train/val/test).
    args.outs = 1
    ticker = TickerDataModule(**vars(args))
    hparams = prepare_args(args)

    # Load the model from a checkpoint or create a new one from scratch.
    if "checkpoint" in hparams:
        forestock = model_picker(args.version).load_from_checkpoint(hparams["checkpoint"])
    else:
        forestock = model_picker(args.version)(**hparams)

    # Define the logger used to store the metrics.
    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/",
        name=hparams["ticker"],
        version=f"{args.version}_{args.mode.lower()}",
        default_hp_metric=False,
    )

    # Set the callbacks used during the stages.
    early_stopping = EarlyStopping("loss/valid", min_delta=1e-7)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [early_stopping, lr_monitor]

    # If the method is called from streamlit, a custom progress bar is used.
    if is_st:
        progress_bar = StProgressBar()
        callbacks += [progress_bar]

    # Create the trainer with the params.
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=callbacks,
    )

    # Start the training/validation/test process.
    trainer.fit(forestock, ticker)
    trainer.test(forestock)

    # Evaluate the model with the test set and plot the results.
    predicts = trainer.predict(forestock, datamodule=ticker)
    y_true, y_hat, metric = process_output(predicts, ticker.sc, args.mode)

    # Save the image in the tb_logs subfolder
    price = ticker.df.Close.to_numpy()
    save_path = f"tb_logs/{hparams['ticker']}/{args.version}_{args.mode.lower()}"
    fig = plot_figure(price, y_true, y_hat, save_path, args.mode, split=0.8)

    return fig, metric


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
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Path to the checkpoint to load"
    )

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
