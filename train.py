
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

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


# ============================================================
# 1. Configuration
# ============================================================

DATASET_REPO_ID = "Sunitha-AIML2026/superkart-sales"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
    "/resolve/main/SuperKart.csv"
)

MODEL_REPO_ID = "Sunitha-AIML2026/superkart-sales-random-forest"
TARGET_COLUMN = "Product_Store_Sales_Total"
RANDOM_STATE = 42

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load and clean the dataset
# ============================================================

print("Loading dataset from Hugging Face...")

df = pd.read_csv(DATASET_URL)
print("Original dataset shape:", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column not found: {TARGET_COLUMN}")

# Remove duplicate records and completely empty columns.
df = df.drop_duplicates().copy()
df = df.dropna(axis=1, how="all")

# Ensure the target contains numeric values.
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="coerce"
)

# Remove rows where the target is missing or invalid.
df = df.dropna(subset=[TARGET_COLUMN]).copy()

if df.empty:
    raise ValueError("No usable rows remain after cleaning.")

print("Cleaned dataset shape:", df.shape)
print("Missing target values:", df[TARGET_COLUMN].isna().sum())


# ============================================================
# 3. Split and save training/testing datasets
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

# Include the target column in both saved datasets.
train_df = X_train.copy()
train_df[TARGET_COLUMN] = y_train

test_df = X_test.copy()
test_df[TARGET_COLUMN] = y_test

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nTraining dataset saved:", train_path)
print("Training rows:", len(train_df))
print("Testing dataset saved:", test_path)
print("Testing rows:", len(test_df))


# ============================================================
# 4. Upload train/test datasets to Hugging Face
# ============================================================

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("\nUploading train/test datasets to Hugging Face...")

    api = HfApi(token=hf_token)

    api.upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo="train.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Upload cleaned training dataset",
    )

    api.upload_file(
        path_or_fileobj=str(test_path),
        path_in_repo="test.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Upload cleaned testing dataset",
    )

    print("Training and testing datasets uploaded successfully.")
    print(
        f"Dataset repository: "
        f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
    )
else:
    print(
        "\nHF_TOKEN not set. Dataset files were saved locally, "
        "but Hugging Face upload was skipped."
    )


# ============================================================
# 5. Define preprocessing
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


# ============================================================
# 6. Train and tune the Random Forest model
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ]
)

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

print("\nStarting Random Forest hyperparameter tuning...")
search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
predictions = best_pipeline.predict(X_test)


# ============================================================
# 7. Evaluate the best model
# ============================================================

metrics = {
    "MAE": float(mean_absolute_error(y_test, predictions)),
    "MSE": float(mean_squared_error(y_test, predictions)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
    "R2": float(r2_score(y_test, predictions)),
    "best_parameters": search.best_params_,
    "best_cross_validation_R2": float(search.best_score_),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "feature_columns": X.columns.tolist(),
    "target_column": TARGET_COLUMN,
}

print("\nBest parameters:")
print(json.dumps(search.best_params_, indent=2))

print("\nEvaluation metrics:")
print(json.dumps(metrics, indent=2))


# ============================================================
# 8. Save the trained model and metrics
# ============================================================

model_path = ARTIFACT_DIR / "superkart_random_forest_pipeline.joblib"
metrics_path = ARTIFACT_DIR / "metrics.json"

joblib.dump(best_pipeline, model_path)

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nSaved model:", model_path)
print("Saved metrics:", metrics_path)


# ============================================================
# 9. Reload and verify the saved model
# ============================================================

loaded_pipeline = joblib.load(model_path)

sample_prediction = float(
    loaded_pipeline.predict(X_test.iloc[:1])[0]
)

print("\nModel verification successful.")
print("Sample prediction:", sample_prediction)
print("Actual value:", float(y_test.iloc[0]))


# ============================================================
# 10. Summary
# ============================================================

print("\n========== PIPELINE SUMMARY ==========")
print("Cleaned dataset rows:", len(df))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Best test R2:", round(metrics["R2"], 4))
print("Training file:", train_path)
print("Testing file:", test_path)
print("Model file:", model_path)
print("Metrics file:", metrics_path)
print("======================================")
