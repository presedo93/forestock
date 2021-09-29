import torch
import argparse

from models.regression import LitForestockReg
from tools.utils import get_checkpoint_hparams


def import_onnx(args: argparse.Namespace) -> None:
    check_path, hp = get_checkpoint_hparams(args.checkpoint)

    # Save the model
    forestock = LitForestockReg.load_from_checkpoint(check_path)
    sample = torch.randn((1, 8, 50))
    forestock.to_onnx(args.onnx, sample, export_params=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--onnx", type=str, help="Name of the ONNX file")

    args = parser.parse_args()
    import_onnx(args)
