import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def ohlc_chart(df: pd.DataFrame) -> go.Figure:
    df = df[df.columns[:5]]
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("OHLC", "Volume"),
        row_width=[0.2, 0.7],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df.Open,
            high=df.High,
            low=df.Low,
            close=df.Close,
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, showlegend=False), row=2, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)

    return fig


def plot_result(
    df: pd.DataFrame,
    y_true: np.array,
    y_hat: np.array,
    path: str,
    mode: str,
    split: float,
) -> go.Figure:
    fig = ohlc_chart(df)

    window = len(df.index) - len(y_true)
    if mode.lower() == "reg":
        fig.add_trace(go.Scatter(x=df.index[window:], y=y_true, name="Real"))
        fig.add_trace(go.Scatter(x=df.index[window:], y=y_hat, name="Prediction"))
    elif mode.lower() == "clf":
        y = np.where((y_true == 1) & (y_hat == 1), 1, np.nan) * df.Close.mean()
        fig.add_trace(
            go.Scatter(x=df.index[window:], y=y, name="Here", mode="markers"),
            row=1,
            col=1,
        )

    if split > 0.0:
        x = int(len(df.index) * split)
        dt = df.index[x]
        fig.add_vrect(
            x0=dt,
            x1=dt,
            line_dash="dash",
            annotation_text="Test set",
            annotation_position="top left",
        )

    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)

    pio.write_image(fig, f"./{path}/figure.png", width=3840, height=2160)
    return fig


# def plot_classification(
#     p: np.array,
#     y_true: np.array,
#     y_hat: np.array,
#     path: str,
#     name: str = "figure",
#     split: float = 0.8,
# ) -> fg.Figure:
#     fig, axs = plt.subplots(
#         2,
#         1,
#         figsize=(16, 12),
#         gridspec_kw={"height_ratios": [2, 1]},
#         sharex=True,
#         sharey=True,
#     )
#     plt.subplots_adjust(hspace=0)
#     x = np.linspace(0, y_hat.shape[0])
#     axs[0].plot(p, label="price")
#     axs[1].scatter(x, y_hat[:-1], label="predicted")
#     axs[1].scatter(x, y_true[:-1], label="real")
#     if split != 0.0:
#         x = int(y_hat.shape[0] * split)
#         plt.axvline(x, c="r", ls="--")
#     plt.title(f"{name}")
#     plt.legend()

#     for ax in axs:
#         ax.label_outer()

#     if os.path.exists(path) is False:
#         os.makedirs(path, exist_ok=True)

#     plt.savefig(f"{path}/{name}.png")

#     return fig
