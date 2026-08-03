from collections.abc import Callable, Iterable
from functools import reduce
from typing import Any

import pandas as pd

MergeArguments = dict[str, Any]
MergeTuple = tuple[pd.DataFrame, MergeArguments]


def reduce_merge(df: pd.DataFrame, mergers: Iterable[MergeTuple]) -> pd.DataFrame:
    """
    Reduce an arbitrary number of dataframes to one dataframe by merging.

    Parameters
    ----------
    df:
        The initial dataframe in the chain of merges.
    mergers:
        Iterables containing the dataframe to merge in and a dictionary of parameters for pd.DataFrame.merge.
    """
    return reduce(lambda _df, _merge_tuple: _df.merge(_merge_tuple[0], **_merge_tuple[1]), mergers, df)


def column_update_factory(update_function: Callable) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Update a DataFrame's column names via a user specified function.

    Implement this function using pipe. Ex: df.pipe(pdx.column_update_factory(update_function)).

    Parameters
    ----------
    update_function:
        A function that specifies how column names will be updated.
    """

    def column_updater(df: pd.DataFrame) -> pd.DataFrame:
        clone = df.copy()
        clone.columns = clone.columns.map(update_function)
        return clone

    return column_updater


flatten_columns = column_update_factory(update_function=lambda tup: "_".join(tup))
