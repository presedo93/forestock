import torch

from models.core import CoreForestock

OHLC_DESC = """Model with OHLC as input. Input goes through a Sequential of
Conv1d, MaxPool1d and bidirectional GRU. Lastly it goes through two Linears."""


class OHLCForestock(CoreForestock):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.desc = OHLC_DESC

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

        self.fc1 = torch.nn.Linear(self.hparams.window * 2, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x) -> torch.Tensor:
        out_ohlc, _ = self.ohlc(x[:, :5])
        y = torch.sigmoid(self.fc1(out_ohlc[:, -1]))
        y = self.fc2(y)

        return y
