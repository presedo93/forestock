import os
import yaml
import argparse

from typing import Tuple


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_checkpoint_hparams(path: str, checkpoint_idx: int = -1) -> Tuple[str, str]:
    all_checks = os.listdir(f"{path}/checkpoints")
    checkpoint = f"{path}/checkpoints/{all_checks[checkpoint_idx]}"

    with open(f"{path}/hparams.yaml", "r") as y_file:
        hparams = yaml.safe_load(y_file)

    print("Types:", type(checkpoint), type(hparams))
    return checkpoint, hparams


def plot():
    # TODO: Pending to move test.py code here!
    return None
