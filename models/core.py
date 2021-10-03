import torch
import pytorch_lightning as pl

from typing import Any, Dict, Optional
from torch.nn import functional as F


class CoreForestock(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.mode.lower() == "reg":
            self.loss_fn = F.mse_loss
        elif self.hparams.mode.lower() == "clf":
            self.loss_fn = F.cross_entropy
        else:
            raise ValueError(f"¨{self.hparams.mode} not supported!")

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/train", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/valid", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/test", loss, on_step=False, on_epoch=True)

        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        x, y = batch
        y_hat = self(x)

        return y, y_hat

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduluer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=1
        )
        return {
            "optimizer": optimizer,
            "interval": "epoch",
            "lr_scheduler": {"scheduler": scheduluer, "monitor": "loss/valid"},
        }
