import os
import argparse
import streamlit as st

# from test import test
from train import train
from infer import inference
from onnx import export_onnx
from datetime import datetime as dt
from tools.plots import ohlc_chart
from typing import Any, Dict, Tuple
from models import available_models, desc_picker
from tools.utils import get_yfinance, get_from_csv, open_conf


def create_folders() -> None:
    """Check if subfolders already exist."""
    if os.path.exists("tb_logs") is False:
        os.makedirs("tb_logs", exist_ok=True)

    if os.path.exists("onnx_models") is False:
        os.makedirs("onnx_models", exist_ok=True)

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
        "Target column for training", conf["targets"], index=3
    )
    args.target_idx = conf["targets"].index(target_name)

    # TODO: Select metrics
    st.sidebar.subheader("Metrics")
    st.sidebar.multiselect("Metrics to use", conf["metrics"])

    # TODO: Select loggers
    st.sidebar.subheader("Logger")
    st.sidebar.selectbox("How to log metrics?", conf["loggers"])

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

                # Get the ticker name
                args.ticker = col1.text_input("Ticker")

                # Get interval
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


def model_selector(args: argparse.Namespace, n: int = 1) -> argparse.Namespace:
    """Logic to select the model.

    Args:
        args (argparse.Namespace): namespace with the arguments
        already selected.
        n (int, optional): Index number. Defaults to 2.

    Returns:
        argparse.Namespace: updated namespace.
    """
    st.subheader(f"{n}. Model selector! 🏗️")
    st.markdown("There is a list of models that can be selected to give them a try")

    use_check = st.checkbox("Load from a checkpoint", value=False)
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
    st.subheader(f"{n}. Task! 🧟")
    st.markdown("It is time to select which task to perform.")
    task = st.selectbox("Tasks supported", conf["tasks"])

    return task


def run_task(task: str, args: argparse.Namespace, n: int = 1) -> Any:
    st.subheader(f"{n}. Run! 🧟")
    task_runned = st.button(f"Launch {task.lower()}")
    try:
        if task_runned and task.lower() == "train":
            return train(args, is_st=True)
        # elif task_runned and task.lower() == "test":
        #     return test(args, is_st=True)
        elif task_runned and task.lower() == "inference":
            return inference(args)
        elif task_runned and task.lower() == "export":
            return export_onnx(args)
    except ValueError as ve:
        st.error(ve)


def print_metrics(
    task: str, args: argparse.Namespace, values: Tuple, n: int = 1
) -> None:
    st.subheader(f"{n}. Metrics! 🗿")

    fig, metrics = values
    # if args.mode == "reg":
    metric_name = "Mean Squared Error"
    # else:
    # metric_name = "Accuracy"
    st.markdown(
        f"<span style='color:green; font-weight: bold'>{task} Completed! Printing some metrics...</span>",
        unsafe_allow_html=True,
    )
    st.metric(metric_name, round(float(metrics), 4))

    # Plot some results
    st.pyplot(fig)


def main():
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
    args = model_selector(args, n)
    n += 1

    if task.lower() in ["train"] and "checkpoint" not in args:
        # Model parameters subheader
        args = model_hyper(args, n)
        n += 1

    # Task subheader
    vals = run_task(task, args, n)
    n += 1

    if task.lower() in ["train", "test"] and vals is not None:
        print_metrics(task, args, vals, n)
        n += 1


if __name__ == "__main__":
    main()
