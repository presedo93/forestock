import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def ohlc_chart(df: pd.DataFrame, plot_ta: bool = False) -> go.Figure:
    """Create a OHLC candlestick chart from a ticker dataframe.

    Args:
        df (pd.DataFrame): ticker dataframe to plot.

    Returns:
        go.Figure: plotly figure.
    """
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

    if plot_ta:
        for k, ta in df.filter(like="MA").items():
            fig.add_trace(go.Scatter(x=df.index, y=ta, name=k))

        if "BB_MIDDLE" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df.BB_MIDDLE, line_color="black", name="BB_MD"
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.BB_UPPER,
                    line_color="gray",
                    line={"dash": "dash"},
                    name="BB_UP",
                    opacity=0.5,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.BB_LOWER,
                    line_color="gray",
                    line={"dash": "dash"},
                    fill="tonexty",
                    name="BB_LW",
                    opacity=0.5,
                ),
                row=1,
                col=1,
            )

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
    """Plot the results of the classification or the regression tasks.
    It adds the plots in the candlestick chart.

    Args:
        df (pd.DataFrame): data used to do the prediction.
        y_true (np.array): targets.
        y_hat (np.array): predictions.
        path (str): path to store the figure.
        mode (str): clf or reg.
        split (float): plot a vertical line based on the percentage of split
        if it is bigger than 0.

    Returns:
        go.Figure: updated figure.
    """
    fig = ohlc_chart(df, plot_ta=True)

    window = len(df.index) - len(y_true)
    if mode.lower() == "reg":
        fig.add_trace(go.Scatter(x=df.index[window:], y=y_true, name="Real"))
        fig.add_trace(go.Scatter(x=df.index[window:], y=y_hat, name="Prediction"))
    elif mode.lower() == "clf":
        y = np.where((y_true == 1) & (y_hat == 1), 1, np.nan) * df.Close.mean()
        fig.add_trace(
            go.Scatter(x=df.index[window:], y=y, name="Target == Pred", mode="markers"),
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
