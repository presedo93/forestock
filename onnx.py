import os
import torch
import argparse

from models import model_picker
from tools.utils import get_checkpoint_hparams


def export_onnx(args: argparse.Namespace) -> None:
    if os.path.exists("onnx_models") is False:
        os.makedirs("onnx_models", exist_ok=True)

    model, check_path, _ = get_checkpoint_hparams(args.checkpoint)

    # Save the model
    forestock = model_picker(model).load_from_checkpoint(check_path)
    sample = torch.randn((1, 11, 50))
    forestock.to_onnx(f"onnx_models/{args.onnx}", sample, export_params=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Path to the checkpoint to load"
    )
    parser.add_argument("-o", "--onnx", type=str, help="Name of the ONNX file")

    args = parser.parse_args()
    export_onnx(args)
