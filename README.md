# Forestock

A forecasting tool for the stock markets using Deep Neural Networks. It uses Pytorch-Lightning as the DL framework and covers from the training steps until the "production" one.

## Train
An example of the command to start the training process:

    python -m train --ticker SPY --mode reg --version ohlc --interval 1h --period 2y --gpus 1 --max_epochs 36

## Test
Command to launch the test on a different ticker:

    python -m test --ticker MSFT --interval 1h --period 2y --checkpoint tb_logs/SPY/ohlc --gpus 1

It also accepts a CSV file as input to test:

    python -m test --ticker ADA --data ADAUSDT.csv --checkpoint tb_logs/SPY/ohlc --gpus 1

## Inference (Done)

For a checkpoint:

    python -m infer --data ADAUSDT.csv --type basic --model tb_logs/ADAUSDT/ohlc_clf --mode clf

For ONNX:

    python -m infer --data ADAUSDT.csv --type onnx --model exports/onnx/adausdt.onnx --mode clf

For TorchScript:

    python -m infer --data ADAUSDT.csv --type torchscript --model exports/torchscript/adausdt.pt --mode clf

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

- [x] Docker
- [x] Train fetching from yfinance.
- [x] Train form a CSV file.
- [x] Support a model selector.
- [x] Train starting from a checkpoint.
- [ ] Check how to freeze layers.
- [ ] Support several steps.
