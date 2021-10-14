import argparse
import numpy as np
import pandas as pd
import streamlit as st

from test import test
from train import train
from infer import inference
from onnx import export_onnx
from tools.plots import st_ohlc_chart
from models import available_models, desc_picker
from tools.utils import get_yfinance, get_from_csv

# Yfinance periods and intervals
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
INTERVALS = [
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]

# Supported tasks
TASKS = {
    "Train": {
        "task": train,
        "desc": "This task trains the selected model based on the selected parameters.",
    },
    "Test": {
        "task": test,
        "desc": "This task tests on a model all the data selected previously.",
    },
    "Infer": {
        "task": inference,
        "desc": "Get the prediction based on **X windows** days from a model.",
    },
    "Export": {"task": export_onnx, "desc": "Export a model to ONNX."},
}


def main():
    # Argparse Namespace will store all the variables to use the modules.
    args = argparse.Namespace()

    # Set page title and favicon.
    st.set_page_config(
        page_title="Forestock", page_icon="⚗️", initial_sidebar_state="collapsed"
    )

    # Set main title
    st.title("Forestock ⚗️")
    st.markdown("Train your AI forestock predictor easily!")

    # Sidebar
    st.sidebar.title("Pytorch Lightning ⚡")
    st.sidebar.subheader("GPUs")
    gpus_available = st.sidebar.checkbox("Are GPUs available?", value=True)
    if gpus_available:
        args.gpus = st.sidebar.number_input("Number of GPUs to use", value=1, step=1)

    st.sidebar.subheader("Max num of epochs")
    args.max_epochs = st.sidebar.number_input("Epochs", value=36, step=1)

    # Data source subheader
    st.subheader("1. Data source! 🤺")
    st.markdown("Data can be fetched from Yahoo Finance or uploading a CSV file.")

    use_csv = st.checkbox("Use your own CSV", value=False)
    if use_csv is False:
        # Yahoo finance expander
        with st.expander("Yahoo Finance"):
            col1, col2, col3 = st.columns(3)

            # Get the ticker name
            args.ticker = col1.text_input("Ticker")

            # Get interval and period
            args.interval = col2.selectbox("Interval", INTERVALS, index=8)
            periods_range, periods_idx = len(PERIODS), 8
            if INTERVALS.index(args.interval) < 5:
                periods_range, periods_idx = 3, 2
            args.period = col3.selectbox(
                "Period", PERIODS[:periods_range], index=periods_idx
            )
    else:
        # CSV read
        with st.expander("CSV Data"):
            st.markdown("CSV file format has to be: Date, Open, High, Low, Close.")
            args.csv = st.file_uploader("OHLC Data")

    plot_ohlc = st.button("Plot data")

    try:
        if plot_ohlc:
            if use_csv is False:
                df = get_yfinance(args.ticker, args.period, args.interval)
            else:
                df = get_from_csv(args.csv)
            if df.empty:
                raise ValueError("Missing YFinance or CSV data")
            st_ohlc_chart(df)
    except ValueError:
        st.markdown("Fill the **YFinance** data or import a **CSV**!")

    # Model selector subheader
    st.subheader("2. Model selector! 🏗️")
    st.markdown("There are a list of models that can be selected to give them a try")
    args.version = st.selectbox("Models", available_models())
    st.markdown(desc_picker(args.version))

    # Model parameters subheader
    st.subheader("3. Model hyperparameters! 💫")
    col1, col2 = st.columns(2)
    mode = col1.selectbox(
        "Classification or Regression", ["Regression", "Classification"]
    )
    args.mode = "reg" if mode == "Regression" else "clf"
    args.window = col2.number_input("Window size", step=1, value=50)

    # Task subheader
    st.subheader("4. Task! 🧟")
    st.markdown("It is time to select which task to perform.")
    task_type = st.selectbox("Tasks supported", TASKS.keys())
    st.markdown(TASKS[task_type]["desc"])

    task = TASKS[task_type]["task"]

    task_runned = st.button("Run Task")
    if task_runned:
        price, y_true, y_hat, metric = task(args, is_streamlit=True)
        if args.mode == "reg":
            metric_name = "Mean Squared Error"
        else:
            metric_name = "Accuracy"
        st.markdown(
            f"<span style='color:green; font-weight: bold'>{task_type} Completed! Printing some metrics...</span>",
            unsafe_allow_html=True,
        )
        st.metric(metric_name, round(float(metric), 4))

        # Plot some results
        arr = np.array([price[args.window:], y_true, y_hat])
        df_metrics = pd.DataFrame(arr.T, columns=["price", "real", "predicted"])
        st.line_chart(df_metrics)


if __name__ == "__main__":
    main()
