
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Explicitly load the original Telco dataset
DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Original dataset not found: {DATA_PATH}\n"
        "Check the CSV filename inside your data folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Original dataset loaded:", DATA_PATH.name)
print("Original dataset shape:", df.shape)

# Validate required columns
required_columns = {"customerID", "Churn", "TotalCharges"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Clean the dataset
df = df.drop_duplicates().copy()
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["Churn"]).copy()

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in the Churn column.")

# Identify numeric and categorical features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

# Split into training, reference, and incoming datasets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_reference, X_incoming, y_reference, y_incoming = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# Train the model
model.fit(X_train, y_train)


def evaluate(actual, predicted):
    return {
        "accuracy": round(accuracy_score(actual, predicted), 4),
        "precision": round(
            precision_score(actual, predicted, zero_division=0), 4
        ),
        "recall": round(
            recall_score(actual, predicted, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(actual, predicted, zero_division=0), 4
        ),
    }


# Evaluate reference and incoming datasets
reference_predictions = model.predict(X_reference)
incoming_predictions = model.predict(X_incoming)

reference_output = X_reference.copy()
reference_output["target"] = y_reference
reference_output["prediction"] = reference_predictions

incoming_output = X_incoming.copy()
incoming_output["target"] = y_incoming
incoming_output["prediction"] = incoming_predictions

# Save monitoring datasets
reference_output.to_csv(DATA_DIR / "reference.csv", index=False)
incoming_output.to_csv(DATA_DIR / "incoming.csv", index=False)

joblib.dump(model, MODEL_DIR / "model.joblib")

metrics = {
    "reference": evaluate(y_reference, reference_predictions),
    "incoming": evaluate(y_incoming, incoming_predictions),
}

with open(
    MODEL_DIR / "baseline_metrics.json", "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

print("\nModel trained successfully!")
print("Training rows:", len(X_train))
print("Reference metrics:", metrics["reference"])
print("Incoming metrics:", metrics["incoming"])
print("\nGenerated files:")
print("- data/reference.csv")
print("- data/incoming.csv")
print("- models/model.joblib")
print("- models/baseline_metrics.json")
