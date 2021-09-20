# Train
An example of the command to start the training process:

    python -m train --ticker AAPL --version basic --interval 1h --period 2y --gpus 1 --max_epochs 36

# Test
Command to launch the :

    python -m test -d data/ADAUSDT.csv -w tb_logs/FST/version_0/checkpoints/epoch=10-step=42470.ckp

# TODO

Features that are still pending to be implemented.

- [ ] Support several steps
- [ ] Load CSV data instead of fetching from yfinance.