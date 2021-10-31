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


def str2bool(v: str) -> bool:
    """Converts str input to bool

    Args:
        v (str): input as string

    Raises:
        argparse.ArgumentTypeError: if the input is not
        one of the supported ones.

    Returns:
        bool: true or false.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def open_conf(conf_path: str) -> dict:
    """Loads the config JSON.

    Args:
        conf_path (str): config file path.

    Returns:
        dict: config values as dict.
    """
    with open(os.path.join(os.getcwd(), conf_path), "r") as f:
        conf = json.load(f)

    return conf


def get_yfinance(
    ticker: str, interval: str, period: str = None, start: str = None, end: str = None
) -> pd.DataFrame:
    """Fetch data from the yfinance tool.

    Args:
        ticker (str): name of the ticker to fetch
        interval (str): data of the steps desired
        period (str, optional): range of the data to fetch. Defaults to None.
        start (str, optional): start date of the data to fetch. Defaults to None.
        end (str, optional): end date of the data to fetch. Defaults to None.

    Raises:
        ValueError: if the tool can't fetch any data.

    Returns:
        pd.DataFrame: dataframe with all the data.
    """
    if period is not None:
        df = yf.Ticker(ticker).history(period, interval).interpolate()
    else:
        df = (
            yf.Ticker(ticker)
            .history(interval=interval, start=start, end=end)
            .interpolate()
        )
    if df.empty:
        raise ValueError(
            f"{ticker} data couldn't be fetched for these period and intervals."
        )

    return df


def get_from_csv(csv: Any) -> pd.DataFrame:
    """Read a CSV file containing stock data.

    Args:
        csv (Any): str path of file.

    Returns:
        pd.DataFrame: dataframe with the stock data.
    """
    return pd.read_csv(csv).set_index("Date")


def parse_metrics(key: str) -> str:
    """Pair strings from the user input to the accepted ones
    and vice versa.

    Args:
        key (str): str from the user input or from the logic.

    Returns:
        str: the converted value.
    """
    if "r2score" in key:
        return "R2 Score"
    elif "mse" in key:
        return "Root Mean Square Error"
    elif "acc" in key:
        return "Accuracy"
    elif "recall" in key:
        return "Recall"
    # And the other way around
    elif "R2 Score" in key:
        return "r2score"
    elif "Root Mean Square Error" in key:
        return "mse"
    elif "Accuracy" in key:
        return "acc"
    elif "Recall" in key:
        return "recall"

    return "None"


def get_ticker_args(args: argparse.Namespace) -> str:
    """Get the ticker from the args or from the csv
    parameter.

    Args:
        args (argparse.Namespace): all the arguments passed
        to the method.

    Returns:
        str: name of the ticker.
    """
    hparams = vars(args)
    if "csv" in hparams.keys():
        if type(args.csv) == str:
            name = args.csv.split("/")[-1].split(".")[0]
        else:
            name = args.csv.name.split(".")[0]
    else:
        name = args.ticker

    return name


def get_checkpoint_hparams(
    path: str, checkpoint_idx: int = -1
) -> Tuple[str, str, Dict]:
    """Read a YAML file from Pytorch Lightning to get the info of
    the checkpoint desired.

    Args:
        path (str): to the checkpoint
        checkpoint_idx (int, optional): In case of having several
        checkpoints. Defaults to -1.

    Returns:
        Tuple[str, str, Dict]: the model name, the checkpoint and
        the hyperparams.
    """
    path = path[:-1] if path[-1] == "/" else path
    all_checks = os.listdir(f"{path}/checkpoints")
    checkpoint = f"{path}/checkpoints/{all_checks[checkpoint_idx]}"
    model_mode = path.split("/")[-1]
    model = model_mode.split("_")[0]

    with open(f"{path}/hparams.yaml", "r") as y_file:
        hparams = yaml.safe_load(y_file)

    return model, checkpoint, hparams


def process_output(
    predicts: list, scaler: MinMaxScaler, mode: str
) -> Tuple[np.array, np.array]:
    """Transforms the raw predictions of the DNN.

    Args:
        predicts (list): target and predictions.
        scaler (MinMaxScaler): scaler to unnormalize the data.
        mode (str): reg or clf

    Returns:
        Tuple[np.array, np.array]: the targets and the outputs.
    """
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

    # Squeeze dimensions
    y_true = np.squeeze(y_true, 1)
    y_hat = np.squeeze(y_hat, 1)

    # Unnormalize data
    if mode == "reg":
        y_true = (y_true - scaler.min_[3]) / scaler.scale_[3]
        y_hat = (y_hat - scaler.min_[3]) / scaler.scale_[3]

    return y_true, y_hat
