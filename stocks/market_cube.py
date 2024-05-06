import numpy as np
import pandas as pd
import torch

columns = ["volume", "open", "high", "low", "close", "adj close", "MACD", "RSI"]

from typing import Callable


def use_market_image_from_date(date: str):
    def tail_from_date(df: pd.DataFrame, t: int)->pd.DataFrame | None:
        """
            https://stackoverflow.com/a/71744691/11004209
            get the last t rows of a pandas dataframe that are before the given date

            return:
                - t x n dataframe
				- None if the given date is not in the dataframe
        """
        last_idx: list[int] = np.flatnonzero(df["date"] == date)

        if len(last_idx) > 0:
            return df.iloc[last_idx[0]-t+1: last_idx[0]]
        return None
    return tail_from_date

def csv2Tensor(fp: str, t: int, tail_f: Callable[[pd.DataFrame, int], pd.DataFrame | None] = None):
    """
        read csv file of stock data, ascending sort by dates, and convert it to a tensor of shape t x 1 x 8

        return:
            - t x 1 x 8 tensor
            - None if the dataset does not include the given range of dates
    """
    df = pd.read_csv(fp, usecols=columns).sort_values(by=["date"])
    if tail_f is None:
        df = df.tail(t)
    else:
        df = tail_f(df, t)

    # Dataset does not include the given date
    if df is None or len(df) < t:
        return None
    return torch.Tensor(df.set_index("date").values).unsqueeze(1)

def make_market_cube(filepaths: list[str], t: int = 10, n: int = len(columns), target_date: str = "2020-06-30"):
    """
        https://arxiv.org/pdf/1809.03684
        We rotate and stack t market images to construct a 3-D market cube E ∈ R^t×m×n.
        Rows (t) index the temporal dimension, columns (m) index stocks, and channels (n) index indicators.

        return: t x (m or less) x n tensor
    """
    tail_f = use_market_image_from_date(target_date)
    market_cube = csv2Tensor(filepaths[0], t, tail_f)

    for fp in filepaths[1:]:
        stock_image = csv2Tensor(fp, t, tail_f)
        if stock_image is not None:
            market_cube = torch.cat((market_cube, stock_image), dim=1)

    return market_cube
