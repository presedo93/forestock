import argparse
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from forestock import LitForestock
from data.ticker import TickerDataModule


def train(args: argparse.Namespace):
    ticker = TickerDataModule(args.data, 24, 1)
    forestock = LitForestock(24)

    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/", name="FST", default_hp_metric=False
    )
    early_stopping = EarlyStopping("loss/valid", min_delta=1e-7)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        gpus=1, logger=tb_logger, max_epochs=20, callbacks=[early_stopping, lr_monitor]
    )

    trainer.fit(forestock, ticker)
    trainer.test(forestock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, help="Path to the data")
    parser.add_argument("-w", "--weights", type=str, help="Path to the weights to load")

    train(parser.parse_args())
