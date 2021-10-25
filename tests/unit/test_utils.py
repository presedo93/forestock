import torch
import numpy as np

from tools.utils import process_output


def test_process_output() -> None:
    # Generate sample preds.
    y0 = (torch.tensor([1]).unsqueeze_(1), torch.tensor([0.9]).unsqueeze_(1))
    y1 = (torch.tensor([0]).unsqueeze_(1), torch.tensor([0.2]).unsqueeze_(1))
    y2 = (torch.tensor([1]).unsqueeze_(1), torch.tensor([0.7]).unsqueeze_(1))
    preds = [y0, y1, y2]

    # Run the method.
    y_true, y_hat = process_output(preds, None, "clf")

    # Check that target and prediction make sense.
    assert isinstance(y_true, np.ndarray) and y_true.max() == 1
    assert isinstance(y_hat, np.ndarray) and y_hat.max() == 1
