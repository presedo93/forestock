import torch
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort

from tools.ta import BBANDS, EMA
from typing import Tuple
from models import model_picker
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

    df["PCT"] = df["Close"].pct_change(fill_method="ffill")
    df["EMA50"] = EMA(df["Close"], 50, fillna=True)
    df["EMA200"] = EMA(df["Close"], 200, fillna=True)

    return df


def normalize(df: pd.DataFrame) -> Tuple[np.array, MinMaxScaler]:
    sc = MinMaxScaler()
    x = sc.fit_transform(df)

    return x, sc


def unnormalize(y_hat: torch.tensor, sc: MinMaxScaler) -> np.array:
    if type(y_hat) is not list:
        if y_hat.device != "cpu":
            y_hat = y_hat.cpu()
        y_hat = y_hat.detach().numpy()
    return (y_hat - sc.min_[3]) / sc.scale_[3]


def inference(args: argparse.Namespace) -> None:
    x, sc = normalize(get_50_last(args.data))

    if args.type.lower() == "basic":
        model, check_path, _ = get_checkpoint_hparams(args.checkpoint)

        forestock = model_picker(model).load_from_checkpoint(check_path)
        device = "cuda:0" if torch.cuda.is_available() and args.gpus else "cpu"
        forestock.to(device)

        x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(0)
        forestock.eval()
        y_hat = unnormalize(forestock(x.to(device)), sc)[0, 0]
    elif args.type.lower() == "onnx":
        forestock = ort.InferenceSession(args.file, providers=["CUDAExecutionProvider"])
        inputs = forestock.get_inputs()[0].name

        x = np.expand_dims(x.T.astype(np.float32), axis=0)
        f_inputs = {inputs: x}
        y_hat = unnormalize(forestock.run(None, f_inputs), sc)[0, 0, 0]
    elif args.type.lower() == "torchscript":
        forestock = torch.jit.load(args.file)
        device = "cuda:0" if torch.cuda.is_available() and args.gpus else "cpu"
        y_hat = unnormalize(forestock(x.to(device)), sc)[0, 0]
    else:
        raise ValueError(
            f"Argument {args.type} is not correct! Please, choose between ONNX / TorchScript / Basic"
        )

    print(f"\033[1;32mPredicted: {y_hat}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument("--type", type=str, help="Inputs: ONNX / TorchScript / Basic")
    parser.add_argument("--file", type=str, help="ONNX or TorchScript file to load")
    parser.add_argument("--data", type=str, help="Path to the CSV data")
    parser.add_argument("--gpus", type=str2bool, help="Make use of a GPU")

    args = parser.parse_args()
    inference(args)
