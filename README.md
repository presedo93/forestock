#  forestock
A forecasting tool for the stock markets using **Deep Neural Networks**! Its main purpose is to be a playground to do some Machine Learning in this unpredictable world. It uses Pytorch-Lightning as the DL framework and covers from the training steps until the inference ones.

## Install
All the needed dependencies are in the `requirements.txt`. First create the virtual environment (recommended):

	python3 -m venv --system-site-packages ./venv
Then install the packages:

	pip install -r requirements.txt

If you want to make changes in the source code, this project has some git hooks for `black`, `mypy`, etc. So please, install them via `pre-commit`:

	pre-commit install

Next time a new commit is done, `pre-commit` will do the work.

## Architecture
The architecture of the project tries to be as modular as possible, allowing users to add new models easily.

### Models
The file `core.py` has the class `CoreForestock` and is the one who has all the training, validation, testing and prediction steps. It also has the optimizers logic and the metrics for each stage (from `torchmetrics`). This class inherits from `pl.LightningModule` and can't be instantiaded, it should work as the common place for the different stages.

The rest of the models must inherit from `CoreForestock` and are based in *LSTM/GRU* layers. Some of them has as inputs **ohlc** and **bollinger bands**, others just work with **exponential movil averages**. All of them are thought to be used with "windows" to take advantage of the *LSTM* layers, making use of *Conv1d* and *MaxPool1d*.

All the models can be used for **regressio** or **classification** tasks.

### Datasets
For the moment, there is only one implementation: `ticker.py`. This implementation accepts ticker name, period and interval to fetch the data from `yfinance` or a csv path to directly load it. It uses `torch.unfold()` to create the windows of n size.

It works with both regression and classification tasks. For regression tasks, it selects a column as target. For classification tasks, it creates the target based in the crossovers of EMA 50 and close prices. In a future, it should be able to accept conditions to create the targets "dinamically" (e.g., to make use of other technical indicators or create more complex rules).

As coming features, I'd like to create other types of `pl.LightningDataModule` to work for different types of architectures and layers.

## Streamlit
Using `streamlit`, the user can train, test, export and inference a model using a web interface. In the sidebar, the parameters for training can be set: batch size, number of workers, metrics, use of GPUs, etc.

The interface is divided in sections that appear depending in the task selected. For **training**, sections as type of training (reg of clf) or window size, or model selection will appear. However, for export task, the interface will ask for the framework or output file (ONNX or TorchScript). The **data** input has its own section and user can select between fetching data from `yfinance` or upload a CSV.

During **training** and **testing**, the interface will show the Pytorch Lightning progress bar thanks to the `stqdm` package, allowing the user to see how things are going!

After training or test tasks, the interface will show the metrics of the training selected by the user in the sidebar. And a plot of the results using `plotly`.

To launch the `streamlit` interface, just run:

    streamlit run dashboard.py

## Train
Training can also be run from the command line, setting the same parameters used in the interface. Here it is an example of launching a training:

    python -m train --ticker NVDA --version ohlc --mode reg --interval 1h --period 2y --gpus 1 --max_epochs 36 --window 42

User has to select input type, window size, version (model to be used) and Pytorch Lightning parameters such as maximum number of epochs. All the training logs and checkpoints will be stored in the `tb_logs` folder. This folder is used by the rest of the tasks.

At the end of the training task, the script runs the prediction stage in all the dataset. Regarding metrics, they are split in training, validation, test and prediction (the whole dataset).

## Test
Taking one of the checkpoints from the training process, models can be tested on other tickers or csv files. In fact, this task runs the `predict` stage of `Pytorch Lightning` as it returns the predictions and the targets to be processed and ploted. Results from this task are stored in the `tickers_test` folder. An example of testing using a CSV file:

    python -m test --data ADAUSDT.csv --checkpoint tb_logs/SPY/ohlc --gpus 1

## Export
Models can be exported to **ONNX** or **TorchScript** for "production" environments! An example of TorchScript export:

    python -m export --checkpoint tb_logs/ADAUSDT/ohlc_clf --type torchscript --name adausdt

And another example of a ONNX export:

    python -m export --checkpoint tb_logs/ADAUSDT/ohlc_clf --type torchscript --name adausdt

## Inference
The `inference.py` file shows the inference of a single value from a model. It supports ONNX, TorchScript or checkpoint "formats". For a checkpoint, the command would be:

    python -m infer --csv ADAUSDT.csv --type basic --model tb_logs/ADAUSDT/ohlc_clf

For an ONNX file (using `onnxruntime`) would be:

    python -m infer --csv ADAUSDT.csv --type onnx --model exports/onnx/adausdt_clf.onnx

And for a TorchScript file:

    python -m infer --csv ADAUSDT.csv --type torchscript --model exports/torchscript/adausdt_clf.pt

Just remember that it also accepts the --ticker, --interval and --period options to fetch data from yfinance. This script can be considered as an easy way to test the models recently exported before taking them to production!

-----------------------------------------

## Docker

    docker run --rm --gpus all -v ${PWD}:/home/scientist/forestock forestock ...
## Unit tests
This project has some basic tests done with pytest.

    python -m pytest --disable-pytest-warnings

For stdout:

    python -m pytest --disable-pytest-warnings -s
## Next steps

Features that are still pending to be implemented.

- [ ] Streamlit section to select strategy for classification.
- [ ] Travis CI/CD.
