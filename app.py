
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

chart_data = latest.set_index("Feature")["Drift detected"].astype(int)
st.bar_chart(chart_data)

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
