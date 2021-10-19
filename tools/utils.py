import os
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

from typing import Any, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def open_conf(conf_path: str) -> dict:
    with open(os.path.join(os.getcwd(), conf_path), "r") as f:
        conf = json.load(f)

    return conf


def get_yfinance(ticker: str, interval: str,  period: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    if period is not None:
        df = yf.Ticker(ticker).history(period, interval).interpolate()
    else:
        df = yf.Ticker(ticker).history(interval=interval, start=start, end=end).interpolate()
    if df.empty:
        raise ValueError(f"{ticker} data couldn't be fetched for these period and intervals.")

    return df


def get_from_csv(csv: Any) -> pd.DataFrame:
    return pd.read_csv(csv).set_index("Date")


def split_args(args: argparse.Namespace) -> Dict:
    """Split the arguments for the model (the ones to be
    saved as hparams) to the ones for the Trainer or the
    Datamodule. For the moment, replaces the csv for a
    ticker name.

    Args:
        args (argparse.Namespace): all the arguments passed
        to the method.

    Returns:
        Dict: arguments for the LightningModule.
    """
    hparams = vars(args)
    if "csv" in hparams.keys():
        if type(args.csv) == str:
            hparams["ticker"] = args.csv.split("/")[-1]
        else:
            hparams["ticker"] = args.csv.name.split(".")[0]
        hparams.pop("csv")

    return hparams


def get_checkpoint_hparams(
    path: str, checkpoint_idx: int = -1
) -> Tuple[str, str, Dict]:
    all_checks = os.listdir(f"{path}/checkpoints")
    checkpoint = f"{path}/checkpoints/{all_checks[checkpoint_idx]}"
    model_mode = path.split("/")[-1]
    model = model_mode.split("_")[0]

    with open(f"{path}/hparams.yaml", "r") as y_file:
        hparams = yaml.safe_load(y_file)

    return model, checkpoint, hparams


def process_output(
    predicts: list,
    scaler: MinMaxScaler,
    mode: str,
) -> Tuple[np.array, np.array, float]:
    # Get the targets
    y_true = torch.cat(list(map(lambda x: x[0], predicts)))

    # Convert to Numpy Array
    y_true = y_true.cpu() if y_true.device != "cpu" else y_true
    y_true = y_true.numpy()

    # Get the predictions
    y_hat = torch.cat(list(map(lambda x: x[1], predicts)))
    if mode == "clf":
        y_hat = torch.round(torch.sigmoid(y_hat))

    # Convert to Numpy Array
    y_hat = y_hat.cpu() if y_hat.device != "cpu" else y_hat
    y_hat = y_hat.numpy()

    if mode == "reg":
        # Unnormalize data
        y_true = (np.squeeze(y_true, 1) - scaler.min_[3]) / scaler.scale_[3]
        y_hat = (np.squeeze(y_hat, 1) - scaler.min_[3]) / scaler.scale_[3]

        metric = mean_squared_error(y_true, y_hat)
        print(f"\033[1;32mMean Squared Error: {metric:.2f}\033[0m")
    else:
        metric = accuracy_score(y_true, y_hat)
        print(f"\033[1;32mAccuracy: {metric:.2f}\033[0m")

    return y_true, y_hat, metric
