import torch
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort

from tools.ta import BBANDS, EMA
from typing import Tuple
from models import model_picker
from sklearn.preprocessing import MinMaxScaler
from tools.utils import get_checkpoint_hparams


def get_50_last(csv_path: str, window: int = 50) -> pd.DataFrame:
    """Applies the same technical analysis as ticker
    DataModule.

    Args:
        csv_path (str): self explanatory
        window (int, optional): Number of days to
        look back. Defaults to 50.

    Returns:
        pd.DataFrame: the data with the ta indicators.
    """
    df = pd.read_csv(csv_path).set_index("Date")

    # Discard everything except Open High Low Close and Volume
    df = df[df.columns[:5]]

    # Get the last n window days
    df = df.iloc[-window:]

    # Add Bollinger Bands
    bbands = BBANDS(df.Close).fillna(0)
    df = pd.concat([df, bbands], axis=1)

    df["PCT"] = df["Close"].pct_change(fill_method="ffill")
    df["EMA50"] = EMA(df["Close"], 50, fillna=True)
    df["EMA200"] = EMA(df["Close"], 200, fillna=True)

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


def inference(args: argparse.Namespace) -> None:
    """Inference an ONNX model, a TorchScript model or a
    checkpoint.

    Args:
        args (argparse.Namespace): file, mode, engine, etc...

    Raises:
        ValueError: if the mode is not supported raises error.
    """
    x, sc = normalize(get_50_last(args.data))

    if args.type.lower() == "basic":
        model, check_path, hp = get_checkpoint_hparams(args.model)
        assert hp["mode"] == args.mode

        forestock = model_picker(model).load_from_checkpoint(check_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        forestock.to(device)

        x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(0)
        forestock.eval()
        y_hat = unnormalize(forestock(x.to(device)), sc, args.mode, args.type)[0, 0]
    elif args.type.lower() == "onnx":
        forestock = ort.InferenceSession(
            args.model, providers=["CUDAExecutionProvider"]
        )
        inputs = forestock.get_inputs()[0].name

        x = np.expand_dims(x.T.astype(np.float32), axis=0)
        f_inputs = {inputs: x}
        out = forestock.run(None, f_inputs)
        y_hat = unnormalize(out, sc, args.mode, args.type)[0, 0]
    elif args.type.lower() == "torchscript":
        forestock = torch.jit.load(args.model)
        x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(0)
        y_hat = unnormalize(forestock(x), sc, args.mode, args.type)[0, 0]
    else:
        raise ValueError(
            f"Argument {args.type} is not correct! Please, choose between ONNX / TorchScript / Basic"
        )

    print(f"\033[1;32mPredicted: {y_hat}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--type", type=str, help="Inputs: ONNX / TorchScript / Checkpoint"
    )
    parser.add_argument("--model", type=str, help="Inputs: File or checkpoint")
    parser.add_argument("--mode", type=str, help="Inputs: Reg or Clf")
    parser.add_argument("--data", type=str, help="Path to the CSV data")

    args = parser.parse_args()
    inference(args)
