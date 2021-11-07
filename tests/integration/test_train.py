import pytest
import argparse
import plotly.graph_objects as go

from train import train


@pytest.fixture()
def train_args(basic_args):
    """ Training args (argparse.Namespace) """
    # This flag runs a “unit test” in the model
    basic_args.fast_dev_run = True

    # Training params
    basic_args.learning_rate = 1e-3
    basic_args.batch_size = 16
    basic_args.workers = 4
    basic_args.split = 0.8
    basic_args.target_idx = 3
    basic_args.auto_lr_find = False

    return basic_args


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_training_process(train_args: argparse.Namespace) -> None:

    fig, metrics = train(train_args)

    # Lets assert the returned variables matched the type hint.
    assert isinstance(fig, go.Figure)
    assert isinstance(metrics, dict)
