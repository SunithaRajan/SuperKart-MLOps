
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Sunitha-AIML2026/superkart-sales/resolve/main/SuperKart.csv"
)
TARGET_COLUMN = "Product_Store_Sales_Total"
RANDOM_STATE = 42

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Load the public dataset directly from Hugging Face.
df = pd.read_csv(DATASET_URL)
print("Dataset shape:", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column not found: {TARGET_COLUMN}")

# Remove rows without a target value.
df = df.dropna(subset=[TARGET_COLUMN]).copy()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Identify feature types.
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_features = [
    column for column in X.columns
    if column not in categorical_features
]

# Fit preprocessing only on training data.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="r2",
    cv=3,
    n_jobs=1,
    verbose=1,
)

search.fit(X_train, y_train)
best_pipeline = search.best_estimator_

predictions = best_pipeline.predict(X_test)

metrics = {
    "MAE": float(mean_absolute_error(y_test, predictions)),
    "MSE": float(mean_squared_error(y_test, predictions)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
    "R2": float(r2_score(y_test, predictions)),
    "best_parameters": search.best_params_,
    "best_cross_validation_R2": float(search.best_score_),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}

model_path = ARTIFACT_DIR / "superkart_random_forest_pipeline.joblib"
metrics_path = ARTIFACT_DIR / "metrics.json"

joblib.dump(best_pipeline, model_path)

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

# Verify the saved pipeline can be loaded and used.
loaded_pipeline = joblib.load(model_path)
sample_prediction = float(loaded_pipeline.predict(X_test.iloc[:1])[0])

print("\nBest parameters:", search.best_params_)
print("Evaluation metrics:")
print(json.dumps(metrics, indent=2))
print("Sample prediction:", sample_prediction)
print("Actual value:", float(y_test.iloc[0]))
print("Saved model:", model_path)
print("Saved metrics:", metrics_path)
