import argparse
import pytorch_lightning as pl
import plotly.graph_objects as go

from typing import Dict, Tuple
from models import model_picker
from tools.plots import plot_result
from tools.progress import StProgressBar
from datasets.ticker import TickerDataModule
from tools.utils import get_checkpoint_hparams, process_output, get_ticker_args


def test(args: argparse.Namespace, is_st: bool = False) -> Tuple[go.Figure, Dict]:
    """Run the predict step over the whole dataset. This method is used to test
    a trained model in a different stock.

    Args:
        args (argparse.Namespace): argparse namespace with the flags for testing.
        is_st (bool, optional): checks if the method is called from a strealit
        interface. Defaults to False.

    Returns:
        Tuple[go.Figure, Dict]: a figure with the plot and the metrics.
    """
    model, check_path, hp = get_checkpoint_hparams(args.checkpoint)
    name = get_ticker_args(args)

    ticker = TickerDataModule(hp["mode"], hp["window"], **vars(args))
    forestock = model_picker(model).load_from_checkpoint(check_path)

    callbacks = []
    if is_st:
        callbacks += [StProgressBar()]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    predicts = trainer.predict(forestock, datamodule=ticker)
    y_true, y_hat = process_output(predicts, ticker.sc, hp["mode"])

    # Save the image in the ticker_tested folder
    save_path = f"./tickers_test/{name}"
    fig = plot_result(ticker.df, y_true, y_hat, save_path, hp["mode"], split=0.0)
    metrics = forestock.get_metrics(["pred"])
    if not is_st:
        print(metrics)

    return fig, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--csv", type=str, help="Path to the CSV data")
    parser.add_argument("--ticker", type=str, help="Ticker name")
    parser.add_argument("--interval", type=str, help="Interval of time")
    parser.add_argument("--period", type=str, help="Num of ticks to fetch")
    parser.add_argument("--metrics", action="append", help="Metrics to use")

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    test(args)
