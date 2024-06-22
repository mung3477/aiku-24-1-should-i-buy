from typing import Callable

from pandas import DataFrame


def tail_from_row(df: DataFrame, t: int, condition: Callable)->DataFrame | None:
    """
        https://stackoverflow.com/a/71744691/11004209
        df.tail(t) from the row that satisfies the condition
    """
    import numpy as np

    last_idx: list[int] = np.flatnonzero(condition(df))

    if len(last_idx) > 0:
        return df.iloc[last_idx[0]-t+1: last_idx[0]]
    return None
