import torch
import torchmetrics
import pytorch_lightning as pl

from typing import Any, Optional
from torch.nn import functional as F


class LitForestock(pl.LightningModule):
    def __init__(self):
        super().__init__()
        h_steps = 50

        self.ohlc = torch.nn.Sequential(
            torch.nn.Conv1d(5, 128, 3),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.GRU(
                input_size=int(h_steps / 2) - 1,
                hidden_size=h_steps,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.fc1 = torch.nn.Linear(h_steps * 2, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.ohlc(x)
        # x = torch.sigmoid(x[:, -1, :])
        x = x[:, -1, :]
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/train", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/valid", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss", loss)

        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]
    ) -> Any:
        x, y = batch
        y_hat = self(x)

        return y, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduluer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=1
        )
        return {
            "optimizer": optimizer,
            "interval": "epoch",
            "lr_scheduler": {"scheduler": scheduluer, "monitor": "loss/valid"},
        }
