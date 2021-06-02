import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split


class LitForestock(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.gru_stock = torch.nn.Sequential(
            torch.nn.Conv1d(32, 7),
            torch.nn.MaxPool1d(3, stride=2),
            torch.nn.GRU(hidden_size=50, num_layers=2, bidirectional=True)
        )

        self.fc1 = torch.nn.Linear()
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x, h):
        stock = self.gru_stock(x, h)
        stock = F.sigmoid(self.fc1(stock))
        return F.linear(self.fc2(stock))


def train():
    return


if __name__ == '__main__':
    train()