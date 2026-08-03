from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from training import splitters
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class EarlyStopTrain:
    """Train an XGBoost model with early stopping.

    This is a base class that is not used directly; use ``EarlyStopClassifier``
    or ``EarlyStopRegressor``, which set ``model_class`` accordingly

    Attributes
    ----------
    early_stopping_rounds:
        Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training
    eval_metric:
        The evaluation metric(s) to estimate after each training round.
        If there's more than one metric in eval_metric, the last metric will be used for early stopping.
    holdout_proportion:
        The proportion of the data to use for the evaluation set.
    learning_rate:
        Shrink the new feature weights by learning_rate after each boosting round.
        See XGBoost docs for more information.
    n_estimators:
        The maximum number of training rounds.
    preprocessor:
        An sklearn.pipeline.Pipeline containing steps to get raw data ready for the XGBoost model.
    xgb_params:
        Additional hyperparameters to pass on to the XGBoost model.
    """

    early_stopping_rounds: int
    eval_metric: str | list[str] | Callable
    holdout_proportion: float
    learning_rate: float
    n_estimators: int
    preprocessor: Pipeline
    xgb_params: dict[str, Any]

    model_class: ClassVar[type[XGBClassifier | XGBRegressor]]

    def __post_init__(self):
        """Create the model used for early stopping."""
        self.early_stop_model = self.model_class(
            **self.xgb_params,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
        )

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803 allow capitalized arguments
        y: pd.Series | np.ndarray,
        extra_eval_set: None | tuple[pd.DataFrame, pd.Series | np.ndarray] = None,
        verbose: bool | int = 100,
        **kwargs,
    ) -> Pipeline:
        """Fit the model using early stopping to determine the number of trees."""
        train, holdout = splitters.row_proportion_split(X.assign(y=y), 1 - self.holdout_proportion)

        train_prep = self.preprocessor.fit_transform(train)
        holdout_prep = self.preprocessor.transform(holdout)

        eval_sets = [(train_prep, train.y)]

        if extra_eval_set:
            eval_prep = self.preprocessor.transform(extra_eval_set[0])
            eval_y = extra_eval_set[1]
            eval_sets.append((eval_prep, eval_y))

        eval_sets.append((holdout_prep, holdout.y))  # the set used for early stopping must be last.

        self.early_stop_model.fit(train_prep, train.y, eval_set=eval_sets, verbose=verbose, **kwargs)

        out_model = Pipeline([
            ("preprocessor", self.preprocessor[0]),
            ("encoder", self.preprocessor[1]),
            (
                "model",
                self.model_class(
                    **self.xgb_params,
                    learning_rate=self.learning_rate,
                    n_estimators=self.early_stop_model.best_iteration,
                ),
            ),
        ])

        out_model.fit(X, y, **kwargs)

        return out_model


@dataclass
class EarlyStopClassifier(EarlyStopTrain):
    """Train an XGBoost classifier with early stopping."""

    model_class: type[XGBClassifier] = XGBClassifier


@dataclass
class EarlyStopRegressor(EarlyStopTrain):
    """Train an XGBoost regressor with early stopping."""

    model_class: type[XGBRegressor] = XGBRegressor
