from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd


class PandasFeature(NamedTuple):
    """Required information for features derived from pandas DataFrames."""

    name: str
    na_impute: Callable[[pd.DataFrame], pd.Series]
    encoder: str
    dtype: str | type


@dataclass
class PandasPreprocessor:
    """Provide an interface to extract and prepare features from a pandas dataframe."""

    features: list[PandasFeature]
    preprocess_feature: Callable[[PandasFeature, pd.DataFrame], pd.Series]

    def __iter__(self):
        """Yield a feature when iterating."""
        yield from self.features

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a dataframe based on the provided feature list and single-feature preprocessor."""
        return pd.DataFrame({f.name: self.preprocess_feature(f, df) for f in self})
