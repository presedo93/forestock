import os
import argparse
import streamlit as st

from test import test
from train import train
from infer import inference
from export import export
from datetime import datetime as dt
from tools.plots import ohlc_chart
from typing import Any, Dict, Tuple
from models import available_models, desc_picker
from tools.utils import get_yfinance, get_from_csv, open_conf, parse_metrics


def create_folders() -> None:
    """Check if subfolders already exist."""
    if os.path.exists("tb_logs") is False:
        os.makedirs("tb_logs", exist_ok=True)

    if os.path.exists("exports") is False:
        os.makedirs("exports", exist_ok=True)

    if os.path.exists("tickers_test") is False:
        os.makedirs("tickers_test", exist_ok=True)


def sidebar(args: argparse.Namespace, conf: Dict) -> argparse.Namespace:
    """Sidebar logic is described in this method.

    Args:
        args (argparse.Namespace): argparse namespace with all
        the parameters needed for the tasks.

    Returns:
        argparse.Namespace: Updated namespace with the new parameters.
    """
    st.sidebar.title("Pytorch Lightning ⚡")
    st.sidebar.subheader("GPUs")
    gpus_available = st.sidebar.checkbox("Are GPUs available?", value=True)
    if gpus_available:
        args.gpus = st.sidebar.number_input("Number of GPUs to use", value=1, step=1)

    st.sidebar.subheader("Training parameters")
    args.max_epochs = st.sidebar.number_input("Max num of epochs", value=36, step=1)
    args.auto_lr_find = st.sidebar.checkbox("Find optimal initial Learning Rate?")

    args.learning_rate = st.sidebar.number_input(
        "Learning Rate", value=3e-3, step=1e-5, format="%e"
    )
    args.batch_size = st.sidebar.number_input("Batch size", value=16, step=1)

    args.workers = st.sidebar.number_input("Workers", value=4, step=1)
    args.split = st.sidebar.number_input("Training & test split size", value=0.8)

    st.sidebar.subheader("Targets")
    target_name = st.sidebar.selectbox(
        "Target column for training (only for regression)", conf["targets"], index=3
    )
    args.target_idx = conf["targets"].index(target_name)

    st.sidebar.subheader("Metrics")
    metrics = st.sidebar.multiselect("Metrics to use", conf["metrics"])
    metrics = [parse_metrics(m) for m in metrics]
    args.metrics = " ".join(metrics)

    # TODO: Select loggers
    st.sidebar.subheader("Logger")
    st.sidebar.selectbox(
        "How to log metrics? (Only Tensorboard supported for the moment...",
        conf["loggers"],
    )

    return args


def data_source(args: argparse.Namespace, conf: Dict, n: int = 1) -> argparse.Namespace:
    """All the logic to fetch the data for the next steps.

    Args:
        args (argparse.Namespace): namespace with the parameters
        already defined.
        conf (Dict): JSON with the config parameters.
        n (int, optional): Just the index. Defaults to 1.

    Raises:
        ValueError: When it tries to plot data and the DataFrame
        is empty.

    Returns:
        argparse.Namespace: Updated namespace with the new parameters.
    """
    st.subheader(f"{n}. Data source! 🤺")
    st.markdown("Data can be fetched from Yahoo Finance or uploading a CSV file.")

    use_csv = st.checkbox("Use your own CSV", value=False)
    if use_csv is False:
        # Yahoo finance expander
        with st.expander("Yahoo Finance"):
            use_dates = st.checkbox("Input dates", value=False)
            if not use_dates:
                col1, col2, col3 = st.columns(3)

                # Get the ticker name and interval
                args.ticker = col1.text_input("Ticker")
                args.interval = col2.selectbox("Interval", conf["intervals"], index=8)

                # Limit periods based on interval
                periods_range, periods_idx = len(conf["periods"]), 8
                if conf["intervals"].index(args.interval) < 5:
                    periods_range, periods_idx = 3, 2

                # Get period
                args.period = col3.selectbox(
                    "Period", conf["periods"][:periods_range], index=periods_idx
                )
            else:
                col1, col2, col3, col4 = st.columns(4)
                args.ticker = col1.text_input("Ticker")
                args.interval = col2.selectbox("Interval", conf["intervals"], index=8)
                td = dt.now()
                args.start = col3.date_input(
                    "Start Date", value=td.replace(year=td.year - 1)
                )
                args.end = col4.date_input("End Date", value=td)
    else:
        # CSV read
        with st.expander("CSV Data"):
            st.markdown("CSV file format has to be: Date, Open, High, Low, Close.")
            args.csv = st.file_uploader("OHLC Data")

    plot_ohlc = st.button("Plot data")

    try:
        if plot_ohlc:
            if use_csv is False:
                if "period" in args:
                    df = get_yfinance(args.ticker, args.interval, args.period)
                else:
                    df = get_yfinance(
                        args.ticker, args.interval, start=args.start, end=args.end
                    )
            else:
                df = get_from_csv(args.csv)
            if df.empty:
                raise ValueError("Missing YFinance or CSV data")
            fig = ohlc_chart(df)
            st.plotly_chart(fig, use_container_width=True)
    except ValueError:
        st.error("Fill the **YFinance** data or import a **CSV**!")

    return args


def model_selector(
    task: str, args: argparse.Namespace, n: int = 1
) -> argparse.Namespace:
    """Logic to select the model.

    Args:
        task (str): task selected before.
        args (argparse.Namespace): namespace with the arguments
        already selected.
        n (int, optional): Index number. Defaults to 2.

    Returns:
        argparse.Namespace: updated namespace.
    """
    st.subheader(f"{n}. Model selector! 🏗️")
    st.markdown("There is a list of models that can be selected to give them a try")

    check = True if task.lower() not in ["train"] else False
    use_check = st.checkbox("Load from a checkpoint", value=check)
    if use_check:
        col1, col2 = st.columns(2)
        st.markdown("Select a checkpoint.")
        checkp_ticks = ["-"] + os.listdir("tb_logs/")
        sel_ticker = col1.selectbox("Select ticker", checkp_ticks)

        # Select a checkpoint to start from
        checkp_mods = ["-"]
        if sel_ticker != "-":
            checkp_mods += os.listdir(f"tb_logs/{sel_ticker}")
        sel_model = col2.selectbox("Select model", checkp_mods)

        # Store the variable checkpoint
        if sel_ticker != "-" and sel_model != "-":
            args.checkpoint = os.path.join("tb_logs", sel_ticker, sel_model)
    else:
        args.version = st.selectbox("Models", available_models())
        st.markdown(desc_picker(args.version))

    return args


def model_hyper(args: argparse.Namespace, n: int = 1) -> argparse.Namespace:
    """Selects config parameters for the model.

    Args:
        args (argparse.Namespace): namespace with the arguments
        already selected.
        n (int, optional): Index value. Defaults to 1.

    Returns:
        argparse.Namespace: Updated namespace.
    """
    st.subheader(f"{n}. Model hyperparameters! 💫")
    col1, col2 = st.columns(2)
    mode = col1.selectbox(
        "Classification or Regression", ["Regression", "Classification"]
    )
    args.mode = "reg" if mode == "Regression" else "clf"
    args.window = col2.number_input("Window size", step=1, value=50)

    return args


def pick_task(conf: Dict, n: int = 1) -> str:
    """Select which task to perform.

    Args:
        conf (Dict): JSON with the config parameters.
        n (int, optional): Index value. Defaults to 1.

    Returns:
        str: task selected.
    """
    st.subheader(f"{n}. Task! 📝")
    st.markdown("It is time to select which task to perform.")
    task = st.selectbox("Tasks supported", conf["tasks"])

    return task


def run_task(task: str, args: argparse.Namespace, n: int = 1) -> Any:
    """Run the selected task! It can be: to train a new model, to test
    an already trained one, to export it or to do inference.

    Args:
        task (str): task selected before.
        args (argparse.Namespace): arguments for that task.
        n (int, optional): Index value. Defaults to 1.

    Returns:
        Any: Metrics in case of train and test. Prediction in case
        of inference and the model path in case of export.
    """
    st.subheader(f"{n}. Run! 🧟")
    task_runned = st.button(f"Launch {task.lower()}")
    try:
        if task_runned and task.lower() == "train":
            return train(args, is_st=True)
        elif task_runned and task.lower() == "test":
            return test(args, is_st=True)
        elif task_runned and task.lower() == "inference":
            return inference(args, is_st=True)
        elif task_runned and task.lower() == "export":
            return export(args)
    except ValueError as ve:
        st.error(ve)


def print_metrics(values: Tuple, n: int = 1) -> None:
    """Show the metrics (like R2 Score or Recall) from the task that
    has been run and the plot of the data fetched.

    Args:
        values (Tuple): tuple that has the metrics dict and the figure.
        n (int, optional): Index value. Defaults to 1.
    """
    st.subheader(f"{n}. Metrics! 🗿")

    fig, metrics = values
    st.markdown("Lets see some results!")
    for key, met in metrics.items():
        st.markdown(f"**{key}** metrics:")
        if len(met) > 0:
            cols = st.columns(len(met))
            for idx, (key, val) in enumerate(met.items()):
                cols[idx].metric(parse_metrics(key), round(float(val), 4))

    # Plot some results
    st.markdown("And the resulting figure with the predictions")
    st.plotly_chart(fig, use_container_width=True)


def export_model(
    args: argparse.Namespace, conf: Dict, n: int = 1
) -> argparse.Namespace:
    """Select the checkpoint to export to a ONNX or TorchScript model.

    Args:
        args (argparse.Namespace): arguments selected.
        conf (Dict): config parameters.
        n (int, optional): index values. Defaults to 1.

    Returns:
        argparse.Namespace: updated arguments
    """
    st.subheader(f"{n}. Save model! 💾")
    st.markdown(
        "Export and store the model to ONNX or TorchScript! In the inference task, both exports can be tested."
    )

    col1, col2 = st.columns(2)
    args.type = col1.selectbox("Export", conf["exports"])
    args.name = col2.text_input("File name")

    return args


def download_model(conf: Dict, n: int = 1) -> None:
    """Download a model based on the ones already exported.

    Args:
        conf (Dict): config parameters
        n (int, optional): index value. Defaults to 1.
    """
    st.subheader(f"{n}. Download the model! 📥")
    col1, col2 = st.columns(2)

    st.markdown("Select a exported mode.")
    export_mode = ["-"] + conf["exports"]
    sel_export = col1.selectbox("Select mode", export_mode)

    # Select a model
    export_files = ["-"]
    if sel_export != "-":
        export_files += os.listdir(f"exports/{sel_export.lower()}")
    sel_model = col2.selectbox("Select file", export_files)

    if sel_export != "-" and sel_model != "-":
        type = sel_export.lower()
        path = os.path.join("exports", type, sel_model)
    else:
        path = ""

    if path != "":
        file_name = path.split("/")[-1]
        with open(path, "rb") as file:
            st.download_button(
                "Download",
                data=file,
                file_name=file_name,
                mime="application/octet-stream",
            )


def inference_selector(
    args: argparse.Namespace, conf: Dict, n: int = 1
) -> argparse.Namespace:
    """Selects an exported model to do inference.

    Args:
        args (argparse.Namespace): arguments selected
        conf (Dict): config parameters
        n (int, optional): index value. Defaults to 1.

    Returns:
        argparse.Namespace: updated arguments.
    """
    st.subheader(f"{n}. Exported models! 👷")
    col1, col2, col3 = st.columns(3)

    st.markdown("Select a exported mode.")
    export_mode = ["-"] + conf["exports"]
    sel_export = col1.selectbox("Select mode", export_mode)

    # Select a model
    export_files = ["-"]
    if sel_export != "-":
        export_files += os.listdir(f"exports/{sel_export.lower()}")
    sel_model = col2.selectbox("Select model", export_files)

    if sel_export != "-" and sel_model != "-":
        args.type = sel_export.lower()
        args.model = os.path.join("exports", args.type, sel_model)

    args.window = col3.number_input("Window size", value=50, step=1)

    return args


def print_prediction(value: float, n: int = 1) -> None:
    """Prints the prediction metric.

    Args:
        value (float): prediction value
        n (int, optional): index value. Defaults to 1.
    """
    st.subheader(f"{n}. Inference result! 🤖")
    st.markdown("And the result is...")
    st.metric("Predicted", float(value))


def main():
    """All the Streamlit logic is called from this method."""
    # Argparse Namespace will store all the variables to use the modules.
    args = argparse.Namespace()

    # Get config from JSON.
    conf = open_conf("conf/conf.json")

    # Check if the needed folders exist
    create_folders()

    # Counter for the indexes
    n = 1

    # Set page title and favicon.
    st.set_page_config(
        page_title="Forestock", page_icon="⚗️", initial_sidebar_state="collapsed"
    )

    # Set main title
    st.title("Forestock ⚗️")
    st.markdown("Train your AI forestock predictor easily!")

    # Sidebar
    args = sidebar(args, conf)

    # Select the task
    task = pick_task(conf, n)
    n += 1

    # Data source subheader
    if task.lower() in ["train", "test", "inference"]:
        args = data_source(args, conf, n)
        n += 1

    # Model selector subheader
    if task.lower() != "inference":
        args = model_selector(task, args, n)
    else:
        args = inference_selector(args, conf, n)
    n += 1

    if task.lower() in ["train"] and "checkpoint" not in args:
        # Model parameters subheader
        args = model_hyper(args, n)
        n += 1

    if task.lower() == "export":
        export_model(args, conf, n)
        n += 1

    # Task subheader
    vals = run_task(task, args, n)
    if vals is not None:
        st.success(f"{task} done!")
    n += 1

    if task.lower() in ["train", "test"] and vals is not None:
        print_metrics(vals, n)
        n += 1

    if task.lower() == "inference" and vals is not None:
        print_prediction(vals, n)
        n += 1

    if task.lower() == "export":
        download_model(conf, n)
        n += 1


if __name__ == "__main__":
    main()
