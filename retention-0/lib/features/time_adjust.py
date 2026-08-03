import pandas as pd


def _implement_adjustment(
    df: pd.DataFrame,
    target_col: str,
    suffix: str,
    time_adjustment: pd.DateOffset,
    time_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Return a lagged or led dataframe."""
    if isinstance(group_col, str):
        group_col = [group_col]
    adjusted = df.assign(**{time_col: df[time_col] + time_adjustment}).filter([time_col, *group_col, target_col])
    return df.merge(adjusted, how="left", on=[*group_col, time_col], suffixes=("", suffix)).drop_duplicates()


def lag(
    df: pd.DataFrame,
    target_col: str,
    suffix: str,
    time_adjustment: pd.DateOffset | int,
    time_col: str,
    group_col: list | str = "project_number",
) -> pd.DataFrame:
    """Add a column to a grouped time-series dataframe that lags based on a time column.

    Parameters
    ----------
    df:
        Dataframe that will get a new lag of one of its columns.
    target_col:
        Column name in `df` that we want to lag. This should have a datetime or int type.
    suffix:
        Suffix to add to the newly created lag.
    time_adjustment:
        Amount of time to lag the new column.
    time_col:
        Column name in `df` to use for lagging.
    group_col:
        Column name in `df` identifying group within which we will lag.
    """
    return _implement_adjustment(df, target_col, suffix, time_adjustment, time_col, group_col)


def lead(
    df: pd.DataFrame,
    target_col: str,
    suffix: str,
    time_adjustment: pd.DateOffset | int,
    time_col: str,
    group_col: list | str = "project_number",
) -> pd.DataFrame:
    """Add a column to a grouped time-series dataframe that leads based on a time column.

    Parameters
    ----------
    df:
        Dataframe that will get a new lead of one of its columns.
    target_col:
        Column name in `df` that we want to lead. This should have a datetime type.
    suffix:
        Suffix to add to the newly created lead.
    time_adjustment:
        Amount of time to lead the new column.
    time_col:
        Column name in `df` to use for leading.
    group_col:
        Column name in `df` identifying group within which we will lead.
    """
    time_adjustment *= -1
    return _implement_adjustment(df, target_col, suffix, time_adjustment, time_col, group_col)
