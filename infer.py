import torch
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from typing import Tuple
from tools.ta import BBANDS
from models.regression import LitForestockReg
from sklearn.preprocessing import MinMaxScaler
from tools.utils import get_checkpoint_hparams, str2bool


def get_50_last(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).set_index("Date")

    # Discard everything except Open High Low Close and Volume
    df = df[df.columns[:5]]

    # Get the last 50 days
    df = df.iloc[-50:]

    # Add Bollinger Bands
    bbands = BBANDS(df.Close).fillna(0)
    df = pd.concat([df, bbands], axis=1)

    return df


def normalize(df: pd.DataFrame) -> Tuple[np.array, MinMaxScaler]:
    sc = MinMaxScaler()
    x = sc.fit_transform(df)
    x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(0)

    return x, sc


def unnormalize(y_hat: torch.tensor, sc: MinMaxScaler) -> np.array:
    if y_hat.device != "cpu":
        y_hat = y_hat.cpu()
    y_hat = y_hat.detach().numpy()
    return (y_hat - sc.min_[3]) / sc.scale_[3]


def inference(args: argparse.Namespace) -> None:
    check_path, hp = get_checkpoint_hparams(args.checkpoint)

    forestock = LitForestockReg.load_from_checkpoint(check_path)
    device = "cuda:0" if torch.cuda.is_available() and args.gpus else "cpu"
    forestock.to(device)

    forestock.eval()
    x, sc = normalize(get_50_last(args.data))

    y_hat = unnormalize(forestock(x.to(device)), sc)

    print(f"\033[1;32mPredicted: {y_hat}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--data", type=str, help="Path to the CSV data")
    parser.add_argument("--gpus", type=str2bool, help="Make use of a GPU")

    args = parser.parse_args()
    inference(args)
