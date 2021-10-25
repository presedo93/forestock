import pytest
import argparse


@pytest.fixture
def basic_args():
    """ Basic args (argparse.Namespace) """
    args = argparse.Namespace()

    # Most common params for the modules
    args.mode = "reg"
    args.ticker = "SPY"
    args.version = "ohlc"
    args.interval = "1h"
    args.period = "1y"
    args.window = 24
    args.metrics = "mse r2score"

    return args
