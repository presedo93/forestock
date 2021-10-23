import torch

from models.core import CoreForestock

LINEARS_DESC = """Model composed of Linears and BatchNorm1d layers. Its inputs are
the last value of the window. It is used mainly for TESTING PURPOSES."""


class LinearsForestock(CoreForestock):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.fc1 = torch.nn.Linear(11, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fc3 = torch.nn.Linear(64, self.hparams.outs)

    def forward(self, x) -> torch.Tensor:
        y = self.bn1(torch.sigmoid(self.fc1(x[..., -1])))
        y = self.bn2(torch.sigmoid(self.fc2(y)))
        y = self.fc3(y)

        return y
