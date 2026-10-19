# 🔎 DriftLens — ML Model Failure Observatory

**DriftLens is a machine learning monitoring system that detects changes in incoming data and tracks model performance metrics through an interactive dashboard.**

It demonstrates how ML models can be monitored after training to identify potential data distribution changes and help teams investigate model reliability.

## 🚀 Key Features

- **Feature Drift Detection:** Detects changes in categorical feature distributions using the Chi-square test.
- **Numerical Drift Detection:** Uses the Kolmogorov–Smirnov (KS) test to compare numerical feature distributions.
- **Model Performance Monitoring:** Displays accuracy, precision, recall, and F1-score for reference and incoming datasets.
- **SQL-Based Monitoring History:** Stores monitoring results in a SQLite database.
- **Interactive Dashboard:** Visualizes drift alerts, feature-level results, and performance metrics using Streamlit.
- **Statistical Significance:** Uses a configurable p-value threshold to flag potential distribution changes.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas & NumPy | Data processing |
| Scikit-learn | ML pipeline and model training |
| SciPy | Statistical drift detection |
| SQLite & SQLAlchemy | Monitoring result storage |
| Streamlit | Interactive dashboard |
| Matplotlib | Data visualization |
| Git & GitHub | Version control |

## 🏗️ Project Architecture

```text
DriftLens/
├── app.py
├── src/
│   ├── train_model.py
│   └── monitor.py
├── data/
├── models/
│   ├── model.joblib
│   └── baseline_metrics.json
├── explore_data.ipynb
├── README.md
└── .gitignore
```

## ⚙️ How It Works

1. **Data Preparation:** Loads and cleans the Telco Customer Churn dataset.
2. **Model Training:** Trains a Logistic Regression model using a scikit-learn preprocessing pipeline.
3. **Reference and Incoming Data:** Creates separate held-out datasets for comparison.
4. **Drift Detection:** Compares categorical and numerical feature distributions using statistical tests.
5. **Result Storage:** Saves feature-level monitoring results to CSV and SQLite.
6. **Dashboard:** Displays drift alerts, statistical results, and model performance metrics.

## 📊 Model Performance

The current experiment produced the following held-out evaluation results:

| Metric | Reference Dataset | Incoming Dataset |
|---|---:|---:|
| Accuracy | 82.40% | 78.78% |
| Precision | 69.81% | 61.76% |
| Recall | 59.36% | 52.67% |
| F1-score | 64.16% | 56.85% |

These values describe the current experiment and may vary when the model is retrained or the data split changes. A difference in performance between samples does not, by itself, establish that feature drift caused the difference.

## 🔬 Drift Detection

DriftLens uses different statistical tests for different data types:

- **Categorical features:** Chi-square test.
- **Numerical features:** Kolmogorov–Smirnov two-sample test.
- **Alert rule:** A p-value below the configured significance threshold indicates a statistically significant distribution difference.

A drift alert signals a potential change in incoming data. It does not automatically mean that model performance has degraded.

## 💻 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/soumyagupta0624/DriftLens.git
cd DriftLens
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib scipy streamlit sqlalchemy matplotlib
```

### 4. Add the dataset

Download the **Telco Customer Churn** dataset and place the original CSV file in the `data/` directory.

The training script expects the dataset to contain columns such as `customerID`, `Churn`, and `TotalCharges`.

### 5. Train the model

```bash
python src/train_model.py
```

This generates the model, baseline metrics, and reference/incoming datasets.

### 6. Run monitoring

```bash
python src/monitor.py
```

### 7. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal.

## 🔮 Future Improvements

- Automated drift alerts and notifications.
- Time-series tracking of drift across monitoring runs.
- Automated model performance evaluation as new labels become available.
- Model retraining recommendations.
- Power BI integration for business-facing analytics.
- Deployment to a cloud platform.

## 👩‍💻 Author

**Soumya Gupta**

[GitHub Repository](https://github.com/soumyagupta0624/DriftLens)

---

*DriftLens is an educational and portfolio project demonstrating machine learning monitoring concepts. Statistical alerts should be interpreted alongside data quality checks and model evaluation.*
