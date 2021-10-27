import torch
import pytorch_lightning as pl
import torchmetrics as tm

from typing import Any, Dict, List, Optional

CORE_DESC = "Base model of Forestock. It includes the different steps (train/val/test & predict)."


class CoreForestock(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters(ignore="csv")
        self.desc = CORE_DESC
        metrics = {}

        if self.hparams.mode.lower() == "reg":
            self.loss_fn = torch.nn.MSELoss()
            if "r2score" in self.hparams.metrics:
                metrics["r2score"] = tm.R2Score()
            if "mse" in self.hparams.metrics:
                metrics["mse"] = tm.MeanSquaredError(squared=False)

        elif self.hparams.mode.lower() == "clf":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            if "acc" in self.hparams.metrics:
                metrics["acc"] = tm.Accuracy()
            if "recall" in self.hparams.metrics:
                metrics["recall"] = tm.Recall()
        else:
            raise ValueError(f"¨{self.hparams.mode} not supported!")

        basic_metrics = tm.MetricCollection(metrics)
        self.train_metrics = basic_metrics.clone(prefix="train_")
        self.val_metrics = basic_metrics.clone(prefix="val_")
        self.test_metrics = basic_metrics.clone(prefix="test_")
        self.pred_metrics = basic_metrics.clone(prefix="pred_")

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/train", loss, on_step=False, on_epoch=True)

        y = y.int() if self.hparams.mode.lower() == "clf" else y
        met_out = self.train_metrics(y_hat, y)
        self.log_dict(met_out)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/valid", loss, on_step=False, on_epoch=True)

        y = y.int() if self.hparams.mode.lower() == "clf" else y
        met_out = self.val_metrics(y_hat, y)
        self.log_dict(met_out)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/test", loss, on_step=False, on_epoch=True)

        y = y.int() if self.hparams.mode.lower() == "clf" else y
        met_out = self.test_metrics(y_hat, y)
        self.log_dict(met_out)

        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        x, y = batch
        y_hat = self(x)

        y = y.int() if self.hparams.mode.lower() == "clf" else y
        metrics = self.pred_metrics(y_hat, y)

        return y, y_hat, metrics

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduluer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )
        return {
            "optimizer": optimizer,
            "interval": "epoch",
            "lr_scheduler": {"scheduler": scheduluer, "monitor": "loss/valid"},
        }

    def get_metrics(self, mode: List[str]) -> Dict:
        metrics = {}

        if "train" in mode or "all" in mode:
            metrics["Train"] = self.train_metrics.compute()
        if "val" in mode or "all" in mode:
            metrics["Validation"] = self.val_metrics.compute()
        if "test" in mode or "all" in mode:
            metrics["Test"] = self.test_metrics.compute()
        if "pred" in mode or "all" in mode:
            metrics["Predict"] = self.pred_metrics.compute()

        return metrics
