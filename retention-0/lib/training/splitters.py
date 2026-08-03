from math import floor
from typing import NamedTuple

import numpy as np
import pandas as pd


class SplitDataFrame(NamedTuple):
    """Container for result of two-way dataframe split."""

    left: pd.DataFrame
    right: pd.DataFrame


def multi_slice_split(df: pd.DataFrame, *args) -> tuple[pd.DataFrame, ...]:
    """Split a dataframe into an arbitrary number of parts via the `[]` operator.

    One dataframe is created for each positional argument passing in via *args.
    The positional arguments must be valid inputs for the `[]` operator. Examples
    include slice objects, boolean series, and callables.
    """
    return tuple(df[arg].copy() for arg in args)


def qcut_split(df: pd.DataFrame, split_data: pd.Series | np.ndarray, q: list[float]) -> tuple[pd.DataFrame, ...]:
    """Split a dataframe into two or more parts based on quantiles of some data.

    Use this function for splitting a dataframe based on a list of quantiles.
    The q list must start with 0 and end with 1; this protects against unintended
    dropping of rows via pd.qcut.

    Parameters
    ----------
    df
        The dataframe to split.
    split_data
        The series or array to split on.
    q
        A list of quantiles at which to split.
    """
    assert q[0] == 0, "q must start with 0!"
    assert q[-1] == 1, "q must end with 1!"

    cuts = pd.qcut(split_data, q=q, labels=False)
    ordered_unique_cuts = np.sort(np.unique(cuts))
    return multi_slice_split(df, *(cuts == c for c in ordered_unique_cuts))


def column_quantile_split(df: pd.DataFrame, split_col: str, q: float) -> SplitDataFrame:
    """Split a dataframe based on the quantiles of one of its columns."""
    return SplitDataFrame(*qcut_split(df, df[split_col], [0, q, 1]))


def array_quantile_split(df: pd.DataFrame, arr: np.ndarray, q: float) -> SplitDataFrame:
    """Split a dataframe based on thhe quantiles of an array."""
    return SplitDataFrame(*qcut_split(df, arr, [0, q, 1]))


def row_proportion_split(df: pd.DataFrame, proportion: float) -> SplitDataFrame:
    """Split a dataframe based on its order.

    Parameters
    ----------
    df
        The dataframe to split.
    proportion
        The proportion of the rows (rounding down) in the first split.
    """
    split_row = floor(proportion * df.shape[0])
    return SplitDataFrame(*multi_slice_split(df, slice(None, split_row), slice(split_row, None)))


def column_bool_split(df: pd.DataFrame, split_col: str, split_method: str, *args, **kwargs) -> SplitDataFrame:
    """Split a dataframe based on the results of boolean operation on a series.

    Rows for which the boolean operation return `True` are in the first output,
    and rows for which the boolean operation return `False` are in the second output.

    Parameters
    ----------
    df
        The dataframe to split.
    split_col
        The name of the column on which to perform the splitting operation.
    split_method
        The name of the series method to use for splitting.
    *args
        Positional arguments passed to split_method
    **kwargs
        Keyword arguments passed to split_method.
    """
    split_callable = getattr(df[split_col], split_method)
    bool_series = split_callable(*args, **kwargs)
    return SplitDataFrame(*multi_slice_split(df, bool_series, ~bool_series))
