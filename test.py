import argparse
import pytorch_lightning as pl

from models import model_picker
from datasets.ticker import TickerDataModule
from tools.utils import get_checkpoint_hparams, plot_regression, process_output


def test(args: argparse.Namespace) -> None:
    model, check_path, hp = get_checkpoint_hparams(args.checkpoint)

    if args.data is not None:
        ticker = TickerDataModule(hp["mode"], hp["window"], csv_path=args.data)
    else:
        ticker = TickerDataModule(
            hp["mode"],
            hp["window"],
            ticker=args.ticker,
            interval=args.interval,
            period=args.period,
        )

    forestock = model_picker(model).load_from_checkpoint(check_path)

    trainer = pl.Trainer.from_argparse_args(args)
    predicts = trainer.predict(forestock, datamodule=ticker)

    y, y_hat = process_output(predicts, ticker.sc, hp["mode"])
    plot_regression(y, y_hat, path="./ticker_tested", name=args.ticker, split=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Path to the checkpoint to load"
    )
    parser.add_argument("-d", "--data", type=str, help="Path to the CSV data")
    parser.add_argument("-t", "--ticker", type=str, help="Ticker name")
    parser.add_argument("-i", "--interval", type=str, help="Interval of time")
    parser.add_argument("-p", "--period", type=str, help="Num of ticks to fetch")

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    test(args)
