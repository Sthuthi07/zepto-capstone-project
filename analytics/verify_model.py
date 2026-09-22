import os
import pandas as pd
from joblib import load

BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "models", "best_random_forest_pipeline.joblib")
pipeline = load(model_path)

raw_sample = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "C"
}])

prediction = pipeline.predict(raw_sample)
probability = pipeline.predict_proba(raw_sample)[:, 1]
print("Raw input:")
print(raw_sample)
print("Prediction:", prediction.tolist())
print("Probability:", probability.tolist())
print("Reloaded complete preprocessing + estimator pipeline successfully.")
