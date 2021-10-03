import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_checkpoint_hparams(path: str, checkpoint_idx: int = -1) -> Tuple[str, str, Dict]:
    all_checks = os.listdir(f"{path}/checkpoints")
    checkpoint = f"{path}/checkpoints/{all_checks[checkpoint_idx]}"
    model = path.split("/")[-1]

    with open(f"{path}/hparams.yaml", "r") as y_file:
        hparams = yaml.safe_load(y_file)

    return model, checkpoint, hparams


def plot_regression(
    y: np.array, y_hat: np.array, path: str, name: str = "figure", split: float = 0.8
) -> None:
    plt.gcf().set_size_inches(16, 12, forward=True)
    plt.plot(y_hat[:-1], label="real")
    plt.plot(y[:-1], label="predicted")
    if split != 0.0:
        x = int(y_hat.shape[0] * split)
        plt.axvline(x, c="r", ls="--")
    plt.title(f"{name}")
    plt.legend()

    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)

    plt.savefig(f"{path}/{name}.png")
