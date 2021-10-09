import argparse
from numpy import arange
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from datasets.ticker import TickerDataModule
from tools.utils import process_output, plot_classification, plot_regression
from models import model_picker


def train(args: argparse.Namespace):
    args.outs = 1
    if args.data is not None:
        ticker = TickerDataModule(args.mode, args.window, csv_path=args.data)
    else:
        ticker = TickerDataModule(
            args.mode,
            args.window,
            ticker=args.ticker,
            interval=args.interval,
            period=args.period,
        )
    forestock = model_picker(args.version)(**vars(args))

    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/",
        name=args.ticker,
        version=f"{args.version}_{args.mode.lower()}",
        default_hp_metric=False,
    )
    early_stopping = EarlyStopping("loss/valid", min_delta=1e-7)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=[early_stopping, lr_monitor],
    )

    trainer.fit(forestock, ticker)
    trainer.test(forestock)

    # Evaluate and plot the test set
    predicts = trainer.predict(forestock, datamodule=ticker)
    y_true, y_hat = process_output(predicts, ticker.sc, args.mode)

    if args.mode == "reg":
        plot_regression(
            y_true,
            y_hat,
            f"tb_logs/{args.ticker}/{args.version}_{args.mode.lower()}",
        )
    else:
        plot_classification(
            ticker.df.Close,
            y_true,
            y_hat,
            f"tb_logs/{args.ticker}/{args.version}_{args.mode.lower()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="CLF or REG")
    parser.add_argument("-d", "--data", type=str, help="Path to the CSV data")
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
