import os
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple
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
) -> Tuple[np.array, np.array]:
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


def plot_regression(
    y_true: np.array,
    y_hat: np.array,
    path: str,
    name: str = "figure",
    split: float = 0.8,
) -> None:
    plt.gcf().set_size_inches(16, 12, forward=True)
    plt.plot(y_hat[:-1], label="predicted")
    plt.plot(y_true[:-1], label="real")
    if split != 0.0:
        x = int(y_hat.shape[0] * split)
        plt.axvline(x, c="r", ls="--")
    plt.title(f"{name}")
    plt.legend()

    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)

    plt.savefig(f"{path}/{name}.png")


def plot_classification(
    p: np.array,
    y_true: np.array,
    y_hat: np.array,
    path: str,
    name: str = "figure",
    split: float = 0.8,
) -> None:
    _, axs = plt.subplots(
        2,
        1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
        sharey=True,
    )
    plt.subplots_adjust(hspace=0)
    x = np.linspace(0, y_hat.shape[0])
    axs[0].plot(p, label="price")
    axs[1].scatter(x, y_hat[:-1], label="predicted")
    axs[1].scatter(x, y_true[:-1], label="real")
    if split != 0.0:
        x = int(y_hat.shape[0] * split)
        plt.axvline(x, c="r", ls="--")
    plt.title(f"{name}")
    plt.legend()

    for ax in axs:
        ax.label_outer()

    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)

    plt.savefig(f"{path}/{name}.png")
