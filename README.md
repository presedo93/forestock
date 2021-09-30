# Forestock

A forecasting tool for the stock markets using Deep Neural Networks. It uses Pytorch-Lightning as the DL framework and covers from the training steps until the "production" one.

## Train
An example of the command to start the training process:

    python -m train --ticker AAPL --version basic --interval 1h --period 2y --gpus 1 --max_epochs 36


## Test
Command to launch the test on a different ticker:

    python -m test --ticker MSFT --interval 1h --period 2y --checkpoint tb_logs/SPY/basic

It also accepts a CSV file as input to test:

    python -m test --ticker ADA --data ADAUSDT.csv --checkpoint tb_logs/SPY/basic --gpus 1

## Inference
There are two possibilities to do the inference. The simplest one based on loading the model from a checkpoint and inferencing it (via PyTorch):

    python -m infer --checkpoint tb_logs/SPY/basic/ --data ADAUSDT.csv --gpus 1

And the second one, which makes use of ONNX. First you need to export the model to an `.onnx` file:

    python -m onnx --checkpoint tb_logs/SPY/basic --onnx forestock.onnx

And once again use **infer**, but with the `--onnx` argument:

    python -m infer --data ADAUSDT.csv --onnx forestock.onnx

## TODO

Features that are still pending to be implemented.

- [x] Train fetching from yfinance.
- [ ] Train form a CSV file.
- [ ] Support a model selector.
- [ ] Train starting from a checkpoint.
- [ ] Check how to freeze layers.
- [ ] Support several steps.
- [ ] Docker