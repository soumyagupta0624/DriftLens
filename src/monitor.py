
from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sqlalchemy import create_engine
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "driftlens.db"
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

REFERENCE_PATH = ROOT / "data" / "reference.csv"
INCOMING_PATH = ROOT / "data" / "test_incoming.csv"


def detect_categorical_drift(reference, incoming, feature, threshold=0.05):
    comparison = pd.DataFrame({
        "Reference": reference[feature].value_counts(),
        "Incoming": incoming[feature].value_counts()
    }).fillna(0)

    chi2, p_value, dof, expected = chi2_contingency(comparison)

    return {
        "Feature": feature,
        "Chi-square": round(chi2, 3),
        "P-value": p_value,
        "Drift detected": bool(p_value < threshold)
    }

def detect_numerical_drift(reference, incoming, feature, threshold=0.05):
    reference_values = pd.to_numeric(
        reference[feature], errors="coerce"
    ).dropna()

    incoming_values = pd.to_numeric(
        incoming[feature], errors="coerce"
    ).dropna()

    if len(reference_values) == 0 or len(incoming_values) == 0:
        return None

    statistic, p_value = ks_2samp(
        reference_values, incoming_values
    )

    return {
        "Feature": feature,
        "Chi-square": None,
        "P-value": round(p_value, 6),
        "Drift detected": bool(p_value < threshold)
    }


def calculate_model_metrics(incoming):
    actual = incoming["target"]
    predicted = incoming["prediction"]

    return {
        "Accuracy": round(accuracy_score(actual, predicted), 4),
        "Precision": round(
            precision_score(actual, predicted, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(actual, predicted, zero_division=0), 4
        ),
        "F1-score": round(
            f1_score(actual, predicted, zero_division=0), 4
        )
    }


def main():
    reference = pd.read_csv(REFERENCE_PATH)
    incoming = pd.read_csv(INCOMING_PATH)


  # Detect categorical drift
    categorical_features = reference.select_dtypes(
    include=["object", "str", "category"]
      ).columns.tolist()

    categorical_features = [
      col for col in categorical_features
      if col not in ["target", "prediction"]
      and col in incoming.columns
  ]

    categorical_results = [
      detect_categorical_drift(reference, incoming, feature)
      for feature in categorical_features
  ]

# Detect numerical drift
    numerical_features = reference.select_dtypes(
      include=["number"]
      ).columns.tolist()

    numerical_features = [
      col for col in numerical_features
      if col not in ["target", "prediction"]
      and col in incoming.columns
  ]

    numerical_results = [
      detect_numerical_drift(reference, incoming, feature)
      for feature in numerical_features
  ]

# Combine both results
    all_results = categorical_results + [
      result for result in numerical_results if result is not None
  ]

    results = pd.DataFrame(all_results)
    metrics = calculate_model_metrics(incoming)

    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    output_path = ROOT / "data" / "drift_report.csv"
    results.to_csv(output_path, index=False)
    results["run_time"] = datetime.now().isoformat(timespec="seconds")

    results.to_sql(
    "drift_results",
    engine,
    if_exists="append",
    index=False
    )

    print("Results saved to SQL database!")

    print(results.to_string(index=False))
    print(f"\nDrifted features: {results['Drift detected'].sum()}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
    
