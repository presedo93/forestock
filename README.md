#  Forestock

A forecasting tool for the stock markets using **Deep Neural Networks**! Its main purpose is to be a playground to do some Machine Learning in this unpredictable world. It uses Pytorch-Lightning as the DL framework and covers from the training steps until the inference ones.

## Install
All the needed dependencies are in the `requirements.txt`. First create the virtual environment (recommended):

	python3 -m venv --system-site-packages ./venv
Then install the packages:

	pip install -r requirements.txt

If you want to make changes in the source code, this project has some git hooks for `black`, `mypy`, etc. So please, install them via `pre-commit`:

	pre-commit install

Next time a new commit is done,

-----------------------------------------
**PENDING**

## Train
An example of the command to start the training process:

    python -m train --ticker NVDA --version ohlc --mode reg --interval 1h --period 2y --gpus 1 --max_epochs 36 --window 42

## Test
Command to launch the test on a different ticker:

    python -m test --ticker MSFT --interval 1h --period 2y --checkpoint tb_logs/SPY/ohlc --gpus 1

It also accepts a CSV file as input to test:

    python -m test --data ADAUSDT.csv --checkpoint tb_logs/SPY/ohlc --gpus 1

## Inference (Done)

For a checkpoint:

    python -m infer --csv ADAUSDT.csv --type basic --model tb_logs/ADAUSDT/ohlc_clf

For ONNX:

    python -m infer --csv ADAUSDT.csv --type onnx --model exports/onnx/adausdt_clf.onnx

For TorchScript:

    python -m infer --csv ADAUSDT.csv --type torchscript --model exports/torchscript/adausdt_clf.pt

It also accepts the --ticker, --interval and --period options to fetch data from yfinance!

## Export (Done)

Export models to ONNX:

    python -m export --checkpoint tb_logs/ADAUSDT/ohlc_clf --type torchscript --name adausdt

Or TorchScript:

    python -m export --checkpoint tb_logs/ADAUSDT/ohlc_clf --type onnx --name adausdt

## Docker

    docker run --rm --gpus all -v ${PWD}:/home/scientist/forestock forestock ...

## Tests
This project has some basic tests done with pytest.

    python -m pytest --disable-pytest-warnings

For stdout:

    python -m pytest --disable-pytest-warnings -s

## TODO

Features that are still pending to be implemented.

- [ ] Streamlit section to select strategy for classification.
- [ ] Travis CI/CD.
