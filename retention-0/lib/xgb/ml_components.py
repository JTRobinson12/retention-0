from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple, Protocol

import pandas as pd
from features.preprocessors import PandasFeature
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


class NamedFeature(Protocol):
    """Set minimum requirement for features."""

    name: str


class EncodeableFeature(Protocol):
    """Set minimum standards for encoding."""

    name: str
    encoder: str | None


class FittableModel(Protocol):
    """Set minimum standard for models."""

    def fit(self, *args, **kwargs):
        """Fit a model."""
        ...


SingleFeaturePreprocess = Callable[[PandasFeature, pd.DataFrame], pd.Series]


class Feature(NamedTuple):
    """Features for this model have names, encoders, and dtypes."""

    name: str
    encoder: str
    dtype: str
    label: str | None = None
    plot: bool = False


def preprocess_pandas_feature(feature: PandasFeature, df: pd.DataFrame) -> pd.Series:
    """Return pd.Series data type adjusted feature column."""
    return df[feature.name].astype(feature.dtype)


def pandas_preprocessor(
    features: list[PandasFeature],
    preprocess_feature: SingleFeaturePreprocess = preprocess_pandas_feature,
    **kwargs,
) -> FunctionTransformer:
    """Return a FunctionTransformer to process features."""

    def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f.name: preprocess_feature(f, df) for f in features})

    return FunctionTransformer(preprocess_all, **kwargs)


def assemble_encoders(features: list[EncodeableFeature], encoders: dict, **kwargs) -> ColumnTransformer:
    """Return ColumnTransformer to encode features."""
    encoder_feature_map = defaultdict(list)
    for f in features:
        if f.encoder is not None:
            encoder_feature_map[f.encoder].append(f.name)
    transformer = ColumnTransformer([(k, v, encoder_feature_map[k]) for k, v in encoders.items()], **kwargs)
    return transformer.set_output(transform="pandas")


def imbalanced_binary_class_scaler(s: pd.Series) -> float:
    """Calculate the class imbalance for binary classification target."""
    return sum(s == 0) / sum(s == 1)
