from recipes.training import terminated_xgb

import keys
import piping
from preprocess import ml_data

model = terminated_xgb.model

(
    ml_data.read(keys.ML.TEST)
    .assign(score=lambda _df: model.predict_proba(_df)[:, 1])
    .pipe(piping.local_csv_cache.write, key=keys.Scores.TERMINATED_SCORED)
)
