import torch
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort

from typing import Tuple
from tools.ta import apply_ta
from models import model_picker
from sklearn.preprocessing import MinMaxScaler
from tools.utils import get_checkpoint_hparams, get_from_csv, get_yfinance


def get_50_last(args: argparse.Namespace) -> pd.DataFrame:
    """Applies the same technical analysis as ticker
    DataModule.

    Args:
        csv_path (str): self explanatory
        window (int, optional): Number of days to
        look back. Defaults to 50.

    Returns:
        pd.DataFrame: the data with the ta indicators.
    """
    if "csv" in args:
        df = get_from_csv(args.csv)
    elif "ticker" in args:
        df = get_yfinance(args.ticker, args.interval, args.period)
    else:
        raise ValueError("Arguments are not correct!")

    # Apply Technical Indicators
    df = apply_ta(df)

    # Get the last n window days
    df = df.iloc[-args.window :]

    return df


def normalize(df: pd.DataFrame) -> Tuple[np.array, MinMaxScaler]:
    """Normalizes the data using the MinMaxScaler.

    Args:
        df (pd.DataFrame): Ticker data.

    Returns:
        Tuple[np.array, MinMaxScaler]: Data normalized between 0 and 1.
    """
    sc = MinMaxScaler()
    x = sc.fit_transform(df)

    return x, sc


def unnormalize(
    y_hat: torch.tensor, sc: MinMaxScaler, mode: str, engine: str
) -> np.array:
    """Process the output of the model.

    Args:
        y_hat (torch.tensor): prediction done.
        sc (MinMaxScaler): scaler in case of regression.
        mode (str): regression or classification.
        engine (str): onnx, torchscript...

    Returns:
        np.array: prediction unnormalized.
    """
    if type(y_hat) is not list:
        if y_hat.device != "cpu":
            y_hat = y_hat.cpu()
        y_hat = y_hat.detach()
    else:
        y_hat = y_hat[0]

    if mode.lower() == "reg":
        y_hat = (y_hat - sc.min_[3]) / sc.scale_[3]
    else:
        y_hat = torch.tensor(y_hat) if engine.lower() == "onnx" else y_hat
        y_hat = torch.round(torch.sigmoid(y_hat))

    return y_hat.numpy()


def inference(args: argparse.Namespace, is_st: bool = False) -> float:
    """Inference an ONNX model, a TorchScript model or a
    checkpoint.

    Args:
        args (argparse.Namespace): file, mode, engine, etc...

    Raises:
        ValueError: if the mode is not supported raises error.
    """
    x, sc = normalize(get_50_last(args))
    mode = "clf" if "_clf" in args.model else "reg"

    if args.type.lower() == "basic":
        model, check_path, hp = get_checkpoint_hparams(args.model)

        forestock = model_picker(model).load_from_checkpoint(check_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        forestock.to(device)

        x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(0)
        forestock.eval()
        y_hat = unnormalize(forestock(x.to(device)), sc, hp["mode"], args.type)[0, 0]
    elif args.type.lower() == "onnx":
        forestock = ort.InferenceSession(
            args.model, providers=["CUDAExecutionProvider"]
        )
        inputs = forestock.get_inputs()[0].name

        x = np.expand_dims(x.T.astype(np.float32), axis=0)
        f_inputs = {inputs: x}
        out = forestock.run(None, f_inputs)
        y_hat = unnormalize(out, sc, mode, args.type)[0, 0]
    elif args.type.lower() == "torchscript":
        forestock = torch.jit.load(args.model)
        x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(0)
        y_hat = unnormalize(forestock(x), sc, mode, args.type)[0, 0]
    else:
        raise ValueError(
            f"Argument {args.type} is not correct! Please, choose between ONNX / TorchScript / Basic"
        )

    if not is_st:
        print(f"\033[1;32mPredicted: {y_hat}\033[0m")

    return y_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--type", type=str, help="Inputs: ONNX / TorchScript / Checkpoint"
    )
    parser.add_argument("--model", type=str, help="Inputs: File or checkpoint")
    parser.add_argument("--csv", type=str, help="Path to the CSV data")
    parser.add_argument("--ticker", type=str, help="Ticker name")
    parser.add_argument("--interval", type=str, help="Interval of time")
    parser.add_argument("--period", type=str, help="Num of ticks to fetch")
    parser.add_argument(
        "--window", type=int, default=50, help="Num. of days to look back"
    )

    args = parser.parse_args()
    inference(args)
