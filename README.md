# Train
An example of the command to start the training process:

    python -m train --ticker SPY --interval 1h --period 1y --gpus 1 --max_epochs 12

# Test
Command to launch the :

    python -m test -d data/ADAUSDT.csv -w tb_logs/FST/version_0/checkpoints/epoch=10-step=42470.ckp

# TODO

Features that are still pending to be implemented.

- [ ] Support several steps