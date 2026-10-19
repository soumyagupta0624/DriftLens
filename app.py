
from pathlib import Path
import sqlite3
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "driftlens.db"
METRICS_PATH = ROOT / "models" / "baseline_metrics.json"

st.set_page_config(
    page_title="DriftLens | Model Monitoring",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 DriftLens")
st.subheader("ML Model Failure Observatory")
st.caption("Monitor feature drift and model performance.")


st.divider()
st.subheader("📂 Upload Incoming Customer Data")

uploaded_file = st.file_uploader(
    "Choose a customer CSV file",
    type=["csv"],
    help="Upload a CSV containing the same feature columns as the reference dataset."
)


if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        reference_data = pd.read_csv(ROOT / "data" / "reference.csv")

        st.write("Uploaded dataset preview")
        st.dataframe(uploaded_data.head(), use_container_width=True)

        st.write(f"Rows uploaded: {len(uploaded_data)}")

        # Check required feature columns
        required_features = [
            col for col in reference_data.columns
            if col not in ["target", "prediction"]
        ]

        missing = [
            col for col in required_features
            if col not in uploaded_data.columns
        ]

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif len(uploaded_data) == 0:
            st.error("The uploaded CSV is empty.")
        else:
            from scipy.stats import chi2_contingency, ks_2samp

            results = []

            for feature in required_features:
                ref = reference_data[feature]
                new = uploaded_data[feature]

                if pd.api.types.is_numeric_dtype(reference_data[feature]):
                    ref_values = pd.to_numeric(ref, errors="coerce").dropna()
                    new_values = pd.to_numeric(new, errors="coerce").dropna()

                    if len(ref_values) and len(new_values):
                        statistic, p_value = ks_2samp(ref_values, new_values)
                        test_name = "KS test"
                    else:
                        continue
                else:
                    comparison = pd.DataFrame({
                        "Reference": ref.value_counts(),
                        "Incoming": new.value_counts()
                    }).fillna(0)

                    chi2, p_value, _, _ = chi2_contingency(comparison)
                    statistic = chi2
                    test_name = "Chi-square test"

                results.append({
                    "Feature": feature,
                    "Test": test_name,
                    "Statistic": round(float(statistic), 4),
                    "P-value": float(p_value),
                    "Drift detected": bool(p_value < 0.05)
                })

            upload_results = pd.DataFrame(results)

            st.subheader("Uploaded Data Drift Report")
            st.dataframe(
                upload_results,
                use_container_width=True,
                hide_index=True
            )

            alert_count = int(upload_results["Drift detected"].sum())
            st.metric("Drift alerts in uploaded data", alert_count)

    except Exception as error:
        st.error(f"Could not process this CSV: {error}")



# Load monitoring history
if not DB_PATH.exists():
    st.error("Database not found. Run: python src/monitor.py")
    st.stop()

with sqlite3.connect(DB_PATH) as connection:
    history = pd.read_sql_query(
        "SELECT * FROM drift_results", connection
    )

if history.empty:
    st.warning("No monitoring results found.")
    st.stop()

# Select the latest monitoring run
history["run_time"] = pd.to_datetime(
    history["run_time"], errors="coerce"
)
latest_time = history["run_time"].max()
latest = history[history["run_time"] == latest_time].copy()

latest["Drift detected"] = (
    latest["Drift detected"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

# Identify feature types
numerical_features = {"tenure", "MonthlyCharges", "TotalCharges"}
latest["Feature type"] = latest["Feature"].apply(
    lambda x: "Numerical" if x in numerical_features else "Categorical"
)

# Monitoring overview
drift_count = int(latest["Drift detected"].sum())
total_features = len(latest)

st.divider()
st.subheader("📊 Monitoring Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Features monitored", total_features)
c2.metric("Drift alerts", drift_count)
c3.metric("No drift detected", total_features - drift_count)

if drift_count:
    st.error(f"⚠️ Drift detected in {drift_count} feature(s).")
else:
    st.success("No statistically significant drift detected.")

# Feature drift report
st.divider()
st.subheader("🧪 Feature Drift Report")

feature_type = st.selectbox(
    "Filter features",
    ["All", "Categorical", "Numerical"]
)

display = latest.copy()

if feature_type != "All":
    display = display[display["Feature type"] == feature_type]

display["P-value"] = display["P-value"].apply(
    lambda x: f"{x:.2e}" if pd.notna(x) else "N/A"
)

st.dataframe(
    display.sort_values("Drift detected", ascending=False),
    use_container_width=True,
    hide_index=True
)

# Drift chart
st.subheader("📈 Drift by Feature")


st.subheader("📈 Drift by Feature")

chart_data = latest.set_index("Feature")["Drift detected"].astype(int)
st.bar_chart(chart_data)

st.divider()
st.subheader("🔍 Why Was Drift Detected?")

alerts = latest[latest["Drift detected"] == True]

if alerts.empty:
    st.success("No features triggered a drift alert in this monitoring run.")
else:
    for _, row in alerts.iterrows():
        st.warning(f"Feature: {row['Feature']}")

        if row["Feature"] in {"tenure", "MonthlyCharges", "TotalCharges"}:
            st.write("**Test used:** Kolmogorov–Smirnov (KS) test")
            st.write(
                "This test compares the numerical value distributions "
                "in the reference and incoming datasets."
            )
        else:
            st.write("**Test used:** Chi-square test")
            st.write(
                "This test checks whether the category proportions "
                "changed between the reference and incoming datasets."
            )

        st.write(f"**P-value:** {row['P-value']:.2e}")
        st.write("**Threshold:** 0.05")
        st.write("A p-value below 0.05 triggers a drift alert.")


# Model performance
st.divider()
st.subheader("🎯 Model Performance")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics_data = json.load(file)

    # Metrics are stored separately for reference and incoming datasets
    reference_metrics = metrics_data.get("reference", {})
    incoming_metrics = metrics_data.get("incoming", {})

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score")
    ]

    st.markdown("**Reference Dataset**")
    cols = st.columns(4)

    for col, (label, key) in zip(cols, metric_names):
        value = reference_metrics.get(key)
        col.metric(
            label,
            f"{float(value):.1%}" if value is not None else "N/A"
        )

    st.markdown("**Incoming Dataset**")
    cols = st.columns(4)

    for col, (label, key) in zip(cols, metric_names):
        value = incoming_metrics.get(key)
        col.metric(
            label,
            f"{float(value):.1%}" if value is not None else "N/A"
        )

    st.caption(
        "These metrics are calculated on held-out datasets. "
        "They are not automatically recalculated on every monitoring run."
    )
else:
    st.warning(
        "Baseline metrics file not found. Check models/baseline_metrics.json."
    )

st.divider()
st.caption(
    "A drift alert indicates a change in feature distribution. "
    "It does not automatically mean model accuracy has decreased."
)
