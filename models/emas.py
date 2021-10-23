import torch

from models.core import CoreForestock

EMA_DESC = """This model only inputs two EMAS of 50 and 200. It also makes use of
Conv1d, MaxPool1d and bidirectional GRU. Lasts layers are two Linear."""


class EmasForestock(CoreForestock):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.desc = EMA_DESC

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
        self.fc2 = torch.nn.Linear(64, self.hparams.outs)

    def forward(self, x) -> torch.Tensor:
        out_ema, _ = self.emas(x[:, 9:11])
        y = torch.sigmoid(self.fc1(out_ema[:, -1]))
        y = self.fc2(y)

        return y
