import argparse
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from models.regression import LitForestockReg
from datasets.ticker import TickerDataModule
from test import process_reg_output


def train(args: argparse.Namespace):
    ticker = TickerDataModule(
        args.ticker, args.interval, args.period, args.window, args.steps
    )
    forestock = LitForestockReg(args.window, args.steps)

    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/", name=args.ticker, version=args.version, default_hp_metric=False
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
    process_reg_output(predicts, ticker.sc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", type=str, help="Ticker name")
    parser.add_argument("-v", "--version", type=str, help="Training version")
    parser.add_argument("-i", "--interval", type=str, help="Interval of time")
    parser.add_argument(
        "-p", "--period", type=str, default="max", help="Num of ticks to fetch"
    )
    parser.add_argument(
        "-w", "--window", type=int, default=50, help="Num. of days to look back"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=1, help="Num. of days to look ahead"
    )

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    train(parser.parse_args())
