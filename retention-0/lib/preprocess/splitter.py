from collections.abc import Callable

import pandas as pd
from functionals import compose
from training import splitters

TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.5
SORT_COLUMN = "relative_month"


def _split_factory(percentile: float, left_rows: bool) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def percentile_split(df: pd.DataFrame) -> pd.DataFrame:
        """Split a data frame into two at the row equivalent to the provided percentile."""
        left, right = splitters.row_proportion_split(df, percentile)
        return left if left_rows else right

    return percentile_split


# Focus on supers and pms for now; in later modelling, understand these types in comparison to other roles.
def _superintendents_or_pms(df: pd.DataFrame) -> pd.DataFrame:
    """Extract field operations employees and sort."""
    return df.query("super_or_pm == True").sort_values(SORT_COLUMN, ascending=True)


get_train = compose.compose(_superintendents_or_pms, _split_factory(TRAIN_PERCENTAGE, left_rows=True))
get_test = compose.compose(
    _superintendents_or_pms,
    _split_factory(TRAIN_PERCENTAGE, left_rows=False),
    _split_factory(VAL_PERCENTAGE, left_rows=True),
)
get_val = compose.compose(
    _superintendents_or_pms,
    _split_factory(TRAIN_PERCENTAGE, left_rows=False),
    _split_factory(VAL_PERCENTAGE, left_rows=False),
)
