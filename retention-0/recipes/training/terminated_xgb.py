import keys
from preprocess import ml_data
from xgb import xgb_components

df = ml_data.read(keys.ML.TRAIN)

model = xgb_components.trainer.fit(df, df.terminated)
