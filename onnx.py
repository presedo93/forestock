import torch
import argparse
import pytorch_lightning as pl

from models.regression import LitForestockReg


def import_onnx(args: argparse.Namespace) -> None:

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")

    args = parser.parse_args()
    import_onnx(args)
