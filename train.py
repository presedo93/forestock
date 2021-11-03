import argparse
import pytorch_lightning as pl
import plotly.graph_objects as go

from typing import Dict, Tuple
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from models import model_picker
from tools.plots import plot_result
from tools.progress import StProgressBar
from datasets.ticker import TickerDataModule
from tools.utils import process_output, get_ticker_args, get_checkpoint_hparams


def train(args: argparse.Namespace, is_st: bool = False) -> Tuple[go.Figure, Dict]:
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
    # Load the model from a checkpoint or create a new one from scratch.
    if "checkpoint" in args:
        model, check_path, hp = get_checkpoint_hparams(args.checkpoint)
        version, mode, name = model, hp["mode"], get_ticker_args(args)
        ticker = TickerDataModule(hp["mode"], hp["window"], **vars(args))
        forestock = model_picker(model).load_from_checkpoint(check_path)
    else:
        version, mode, name = args.version, args.mode, get_ticker_args(args)
        ticker = TickerDataModule(**vars(args))
        forestock = model_picker(args.version)(**vars(args))

    # Define the logger used to store the metrics.
    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/",
        name=name,
        version=f"{version}_{mode.lower()}",
        default_hp_metric=False,
    )

    # Set the callbacks used during the stages.
    early_stopping = EarlyStopping("loss/valid", patience=12)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [early_stopping, lr_monitor]

    # If the method is called from streamlit, a custom progress bar is used.
    if is_st:
        progress_bar = StProgressBar()
        callbacks += [progress_bar]

    # Create the trainer with the params.
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=callbacks)

    # Find the optimal learning rate.
    if args.auto_lr_find:
        trainer.tune(forestock, datamodule=ticker)

    # Start the training/validation/test process.
    trainer.fit(forestock, ticker)
    trainer.test(forestock)

    # Evaluate the model with the test set and plot the results.
    predicts = trainer.predict(forestock, datamodule=ticker)
    y_true, y_hat = process_output(predicts, ticker.sc, args.mode)

    # Save the image in the tb_logs subfolder.
    save_path = f"tb_logs/{name}/{version}_{mode.lower()}"
    fig = plot_result(ticker.df, y_true, y_hat, save_path, args.mode, split=0.8)
    metrics = forestock.get_metrics(["all"])

    if not is_st:
        print(metrics)

    return fig, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--mode", type=str, help="CLF or REG")
    parser.add_argument("--csv", type=str, help="Path to the CSV data")
    parser.add_argument("--ticker", type=str, help="Ticker name")
    parser.add_argument("--version", type=str, help="Training model used")
    parser.add_argument("--interval", type=str, help="Interval of time")
    parser.add_argument("--period", type=str, help="Num of ticks to fetch")
    parser.add_argument("--window", type=int, help="Num. of days to look back")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--metrics", type=str, default="", help="Metrics to use")

    # Training type params
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning Rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--workers", type=int, default=4, help="Num of workers for dataloaders"
    )
    parser.add_argument(
        "--split", type=float, default=0.8, help="Split training & test"
    )
    parser.add_argument(
        "--target_idx", type=int, default=3, help="Column of OHLC to use as target"
    )

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
