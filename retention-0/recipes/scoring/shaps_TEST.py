from functools import partial

import pandas as pd
from predictions import shap_tools
from recipes.training import terminated_xgb

import assets
import keys
import piping
from preprocess import ml_data
from xgb import features

model_features = [f.name for f in features.features] + ["employee_number", "relative_month", "terminated"]
test = ml_data.read(keys.ML.TEST).filter(model_features)

model = terminated_xgb.model

shap_values_combined = shap_tools.make_combined_explanation(
    matrix=model[:-1].transform(test),
    estimator=model["model"],
    feature_names=model["encoder"].get_feature_names_out(),
    features_to_combine=features.one_hot_list,
    feature_renamer=partial(shap_tools.split_feature_names, "__"),
)

shap_df = pd.DataFrame(shap_values_combined.values, columns=shap_values_combined.feature_names)
feature_df = test.melt(
    id_vars=assets.ID_COLUMNS,
    value_vars=shap_values_combined.feature_names,
    var_name="feature_name",
    value_name="feature_value",
)

test_feature_shaps = (
    pd.concat([test[assets.ID_COLUMNS], shap_df.rename(columns={"relative_month": "__shap_relative_month"})], axis=1)
    .melt(id_vars=assets.ID_COLUMNS, var_name="feature_name", value_name="shap_value")
    .replace({"feature_name": {"__shap_relative_month": "relative_month"}})
    .merge(feature_df, how="left", on=assets.ID_COLUMNS + ["feature_name"])
    .pipe(piping.local_csv_cache.write, key=keys.Scores.TERMINATED_FEATURE_SHAPS)
)
