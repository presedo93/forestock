import torch
import pytorch_lightning as pl

from models.core import CoreForestock


class LinearsForestock(CoreForestock):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.fc1 = torch.nn.Linear(11, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, self.hparams.outs)

    def forward(self, x) -> torch.Tensor:
        y = torch.sigmoid(self.fc1(x[..., -1]))
        y = torch.sigmoid(self.fc2(y))
        y = self.fc3(y)

        return y
