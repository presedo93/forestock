import os
import torch
import argparse

from models import model_picker
from tools.utils import get_checkpoint_hparams


def export(args: argparse.Namespace) -> str:
    """Export the model to ONNX or to TorchScript.

    Args:
        args (argparse.Namespace): checkpoint, desired output file, etc...

    Raises:
        ValueError: if type doesn't meet ONNX or TorchScript.
    """
    if os.path.exists(f"exports/{args.type.lower()}") is False:
        os.makedirs(f"exports/{args.type.lower()}", exist_ok=True)

    model, check_path, _ = get_checkpoint_hparams(args.checkpoint)

    # Save the model
    forestock = model_picker(model).load_from_checkpoint(check_path)
    if args.type.lower() == "onnx":
        sample = torch.randn((1, 11, 50))
        forestock.to_onnx(
            f"exports/{args.type.lower()}/{args.name}.onnx", sample, export_params=True
        )
        ext = "onnx"
    elif args.type.lower() == "torchscript":
        script = forestock.to_torchscript()
        torch.jit.save(script, f"exports/{args.type.lower()}/{args.name}.pt")
        ext = "pt"
    else:
        raise ValueError(
            f"Argument type {args.type} not supported! Please use: onnx or torchscript"
        )

    return f"exports/{args.type.lower()}/{args.name}.{ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--type", type=str, help="ONNX / TorchScript export type")
    parser.add_argument("--name", type=str, help="Name of the ONNX / Torchscript file")

    args = parser.parse_args()
    export(args)
