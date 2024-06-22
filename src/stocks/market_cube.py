
from typing import Callable

import pandas as pd
import torch

from src.lib import tail_from_row


def use_market_image_from_date(date: str)->Callable[[pd.DataFrame, int], pd.DataFrame | None]:
    def condition(df: pd.DataFrame):
        return df["date"] == date
    return lambda df, t : tail_from_row(df, t, condition)

def csv2Tensor(
        fp: str,
        t: int,
        chart_columns: list[str],
        tail_f: Callable[[pd.DataFrame, int], pd.DataFrame | None] = None,
    ):
    """
        return: (t x 1 x 7 tensor, {min: Tensor, max: Tensor}
    """
    df = pd.read_csv(fp, usecols=chart_columns).sort_values(by=["date"])
    if tail_f is None:
        df_tail = df.tail(t)
    else:
        df_tail = tail_f(df, t)

    # Dataset does not include the given date
    if df_tail is None or len(df_tail) < t:
        return (
            torch.zeros(t, 1, len(df.columns) - 1),
            {
                 "min": torch.zeros(len(df.columns) - 1),
                 "max": torch.zeros(len(df.columns) - 1)
            }
        )
    df_tail = df_tail.set_index("date")
    return (
        torch.Tensor(df_tail.values).unsqueeze(1),
        {
            "min": torch.Tensor(df_tail.min().values.tolist()),
            "max": torch.Tensor(df_tail.max().values.tolist())
        }
    )

def make_market_cube(
        filepaths: list[str],
        t: int,
        chart_columns: list,
        tail_f: Callable[[pd.DataFrame, int], pd.DataFrame | None] = None
    ):
    """
        https://arxiv.org/pdf/1809.03684
        We rotate and stack t market images to construct a 3-D market cube E ∈ R^t×m×n.
        Rows (t) index the temporal dimension, columns (m) index stocks, and channels (n) index indicators.

        return: t x m x 7 tensor
    """
    if tail_f is None:
        tail_f = use_market_image_from_date("2024-04-30")

    indicators_len = len(chart_columns) - 1

    stock_images = list()
    indctr = {
        "min": torch.Tensor([float('inf')] * indicators_len),
        "max": torch.zeros(indicators_len)
    }
    for fp in filepaths:
        stock_image, stock_indctr = csv2Tensor(fp, t, chart_columns, tail_f=tail_f)
        if stock_image is not None:
            stock_images.append(stock_image)
            indctr["min"] = torch.minimum(indctr["min"], stock_indctr["min"])
            indctr["max"] = torch.maximum(indctr["max"], stock_indctr["max"])

    market_cube = torch.cat(stock_images, dim=1)
    # normalize all indicator values relative to
    market_cube = (market_cube - indctr["min"]) / (indctr["max"] - indctr["min"])

    return market_cube
