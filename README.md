# Train
An example of the command to start the training process:

    python -m train --ticker AAPL --version basic --interval 1h --period 2y --gpus 1 --max_epochs 36

# Test
Command to launch the :

    python -m test --ticker MSFT --interval 1h --period 2y --checkpoint tb_logs/SPY/basic

It also accepts a CSV file as input to test:

    python -m test --ticker ADA --data ADAUSDT.csv --checkpoint tb_logs/SPY/basic --gpus 1

# TODO

Features that are still pending to be implemented.

- [x] Load CSV data instead of fetching from yfinance.
- [ ] Train starting from a checkpoint.
- [ ] Check how to freeze layers.
- [ ] Support several steps.