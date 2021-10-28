import torch

from models.core import CoreForestock

BBANDS_DESC = """This model consists of two inputs: ohlc and bbands.
It makes use of a **Conv1d**, a **Maxpool1d** and a **bidirectional GRU** for both inputs.
Both GRU outputs are concateneted to finally go through several Linear layes."""


class BBandsForestock(CoreForestock):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.desc = BBANDS_DESC

        self.ohlc = torch.nn.Sequential(
            torch.nn.Conv1d(5, 128, 3),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.GRU(
                input_size=int(self.hparams.window / 2) - 1,
                hidden_size=self.hparams.window,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.bbands = torch.nn.Sequential(
            torch.nn.Conv1d(3, 128, 2),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.GRU(
                input_size=int(self.hparams.window / 2) - 1,
                hidden_size=self.hparams.window,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.fc1 = torch.nn.Linear(self.hparams.window * 4, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x) -> torch.Tensor:
        out_ohlc, _ = self.ohlc(x[:, :5])
        out_bb, _ = self.bbands(x[:, 5:8])
        y = torch.cat([out_ohlc[:, -1], out_bb[:, -1]], dim=1)
        y = torch.sigmoid(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = self.fc3(y)

        return y
