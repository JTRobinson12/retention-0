import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from training import early_stop_train as est
from xgboost import XGBClassifier

import keys
from preprocess import ml_data
from xgb import features, ml_components

df = ml_data.read(keys.ML.TRAIN)
preprocessor = ml_components.pandas_preprocessor(features.features)
encoder = ml_components.assemble_encoders(features.features, features.encoders, remainder="passthrough")

search_params = {
    "model__max_depth": np.arange(2, 4, 1),
    "model__subsample": np.arange(0.85, 0.95, 0.025),
    "model__colsample_bytree": np.arange(0.1, 0.5, 0.05),
}

positive_target_weight = len(df.query("terminated == 0")) / len(df.query("terminated == 1"))

xgb_to_optimize = XGBClassifier(random_state=13, scale_pos_weight=positive_target_weight, enable_categorical=True)
xgb_pipeline = Pipeline([("preprocessor", preprocessor), ("encoder", encoder), ("model", xgb_to_optimize)])

xgb_optimized = BayesSearchCV(
    estimator=xgb_pipeline,
    search_spaces=search_params,
    scoring="neg_log_loss",
    cv=TimeSeriesSplit(3),
    iid=False,
    refit=False,
    verbose=0,
)

xgb_optimized.fit(df, df.terminated)
xgb_params = {k.replace("model__", ""): v for k, v in xgb_optimized.best_params_.items()}

early_stop_params = {
    "early_stopping_rounds": 50,
    "eval_metric": ["auc", "aucpr", "logloss"],
    "holdout_proportion": 0.1,
    "learning_rate": 0.01,
    "n_estimators": 2500,
}

trainer = est.EarlyStopClassifier(
    **early_stop_params,
    preprocessor=Pipeline([("preprocessor", preprocessor), ("encoder", encoder)]),
    xgb_params=xgb_params,
)
