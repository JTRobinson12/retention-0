from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import shap


def _combine_single_one_hot(explanation: shap.Explanation, name: str, mask: np.ndarray) -> None:
    """Combine one-hot encoded features into a single feature in explanation in-place.

    Parameters
    ----------
    explanation:
        The object that contains SHAP explanation information for a specific model.
    name:
        Final name of one_hot_encoded feature.
    mask:
        Boolean numpy array that indexes columns that include the feature to be combined.
    """
    mask_col_names = np.array(explanation.feature_names)[mask]
    data_array, values_array = explanation.data, explanation.values

    non_masked_count = np.sum(~mask)
    new_col_count = non_masked_count + 1
    new_data = np.empty((data_array.shape[0], new_col_count), dtype=object)
    new_values = np.empty((values_array.shape[0], new_col_count))

    new_data[:, :non_masked_count] = data_array[:, ~mask]
    for i in range(data_array.shape[0]):
        category_idx = np.argmax(data_array[i, mask])
        new_data[i, -1] = mask_col_names[category_idx].split("_")[-1]

    new_values[:, :non_masked_count] = values_array[:, ~mask]
    new_values[:, -1] = values_array[:, mask].sum(axis=1)
    explanation.data = new_data
    explanation.values = new_values
    explanation.feature_names = list(np.array(explanation.feature_names)[~mask]) + [name]


def _combine_one_hot_features(
    explanation: shap.Explanation,
    one_hot_list: Iterable[str],
) -> None:
    """Combine one hot encoded features in explanation in-place for list of targets.

    Parameters
    ----------
    explanation:
        The object that contains SHAP explanation information for a specific model.
    one_hot_list:
        Iterable of names of one-hot encoded features.
    """
    if hasattr(explanation.data, "toarray"):
        explanation.data = explanation.data.toarray()

    for feature_name in one_hot_list:
        mask = np.array([feature_name in feature for feature in explanation.feature_names])
        if not any(mask):
            print(f"No columns found for feature: {feature_name}")
            continue
        _combine_single_one_hot(explanation, feature_name, mask)


def split_feature_names(
    split_on: str,
    feature_names: Iterable[str],
) -> list[str]:
    """Return a list of feature names after splitting off a prefix on a given string.

    Parameters
    ----------
    split_on:
        A string to split feature_names on; the second split element is kept.
    feature_names:
        An iterable of feature names retrieved from a fitted encoder.
    """
    return [name.split(split_on, 1)[1] for name in feature_names]


def make_explanation(
    matrix: np.ndarray | pd.DataFrame,
    estimator,
    feature_names: Iterable[str],
    feature_renamer: Callable[[Iterable[str]], Iterable[str]] | None = None,
) -> shap.Explanation:
    """Return a shap.Explanation for an XGBModel.

    Parameters
    ----------
    matrix:
        Numpy array or pandas DataFrame of preprocessed data.
    estimator:
        Tree model (e.g., XGBModel) to use in the shap.TreeExplainer.
    feature_names:
        Iterable of feature names for the data in matrix.
    feature_renamer (optional):
        Function to rename features inside the Explanation.
    """
    explanation = shap.TreeExplainer(estimator, feature_names=feature_names)(matrix)

    if feature_renamer:
        explanation.feature_names = feature_renamer(explanation.feature_names)

    return explanation


def make_combined_explanation(
    matrix: np.ndarray | pd.DataFrame,
    estimator,
    feature_names: Iterable[str],
    features_to_combine: Iterable[str],
    feature_renamer: Callable[[Iterable[str]], Iterable[str]] | None = None,
) -> shap.Explanation:
    """Return an explanation with specified one hot feature categories combined into the original features.

    Parameters
    ----------
    matrix:
        Numpy array or pandas DataFrame of preprocessed data.
    estimator:
        Tree model (e.g., XGBModel) to use in the shap.TreeExplainer.
    feature_names:
        Iterable of feature names for the data in matrix.
    feature_to_combine:
        Iterable of one-hot encoded features to be combined in the final explanation.
    feature_renamer (optional):
        Function to rename all non one-hot encoded features except inside the Explanation.
    """
    explanation = make_explanation(matrix, estimator, feature_names, feature_renamer)
    _combine_one_hot_features(explanation, features_to_combine)
    return explanation


def make_clustering(matrix: np.ndarray | pd.DataFrame, target: pd.Series) -> np.ndarray:
    """Return the clustering object for use as the `clustering` argument of `shap.plots.bar()`.

    Parameters
    ----------
    matrix:
        Numpy array or pandas DataFrame of preprocessed data.
    target
        Pandas Series of target column.
    """
    return shap.utils.hclust(matrix.todense(), target)


def shap_and_value(df: pd.DataFrame, explanation: shap.Explanation, suffix: str) -> pd.DataFrame:
    """Return a wide pandas DataFrame of shap and feature values.

    Use for plotting a feature against its shaps and other feature values.

    Parameters
    ----------
    df:
        Data to be joined with shap.Explanation.
    explanation:
        The object that contains SHAP explanation information for a specific model.
    suffix
        A string to differentiate columns upon a concatenation.
    """
    shaps = (
        pd.DataFrame(explanation.values, columns=explanation.feature_names)
        .rename(columns={col: col + suffix for col in explanation.feature_names})
    )  # fmt:skip
    return pd.concat([df.reset_index(drop=True), shaps], axis=1)
