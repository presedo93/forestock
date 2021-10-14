import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def st_ohlc_chart(df: pd.DataFrame) -> None:
    df = df[df.columns[:5]]
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_figure(
    p: np.array, y_true: np.array, y_hat: np.array, path: str, mode: str, split: float
) -> None:
    if mode.lower() == "reg":
        plot_regression(y_hat, y_true, path)
    elif mode.lower() == "clf":
        plot_classification(p, y_true, y_hat, path)


def plot_regression(
    y_true: np.array,
    y_hat: np.array,
    path: str,
    name: str = "figure",
    split: float = 0.8,
) -> None:
    plt.gcf().set_size_inches(16, 12, forward=True)
    plt.plot(y_hat[:-1], label="predicted")
    plt.plot(y_true[:-1], label="real")
    if split != 0.0:
        x = int(y_hat.shape[0] * split)
        plt.axvline(x, c="r", ls="--")
    plt.title(f"{name}")
    plt.legend()

    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)

    plt.savefig(f"{path}/{name}.png")


def plot_classification(
    p: np.array,
    y_true: np.array,
    y_hat: np.array,
    path: str,
    name: str = "figure",
    split: float = 0.8,
) -> None:
    _, axs = plt.subplots(
        2,
        1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
        sharey=True,
    )
    plt.subplots_adjust(hspace=0)
    x = np.linspace(0, y_hat.shape[0])
    axs[0].plot(p, label="price")
    axs[1].scatter(x, y_hat[:-1], label="predicted")
    axs[1].scatter(x, y_true[:-1], label="real")
    if split != 0.0:
        x = int(y_hat.shape[0] * split)
        plt.axvline(x, c="r", ls="--")
    plt.title(f"{name}")
    plt.legend()

    for ax in axs:
        ax.label_outer()

    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)

    plt.savefig(f"{path}/{name}.png")
