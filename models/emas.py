import torch
import pytorch_lightning as pl

from models.core import CoreForestock


class EmasForestock(CoreForestock):
    def __init__(self, **hparams):
        super().__init__(**hparams)

        self.emas = torch.nn.Sequential(
            torch.nn.Conv1d(2, 64, 1),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.GRU(
                input_size=int(self.hparams.window / 2),
                hidden_size=self.hparams.window,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.fc1 = torch.nn.Linear(self.hparams.window * 2, 64)
        self.fc2 = torch.nn.Linear(64, self.hparams.steps)

    def forward(self, x) -> torch.Tensor:
        out_ema, _ = self.emas(x[:, 9:11])
        y = torch.sigmoid(self.fc1(out_ema[:, -1]))
        y = self.fc2(y)

        return y
