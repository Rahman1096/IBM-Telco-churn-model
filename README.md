# 📊 IBM Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecom company using the IBM Telco Customer Churn dataset. Includes a full ML pipeline (EDA → Training → Evaluation) and a polished Streamlit dashboard for real-time predictions.

---

## 🚀 Live Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
IBM-Telco-Churn/
│
├── .streamlit/
│   └── config.toml          # Theme & server configuration
│
├── data/
│   ├── raw/                  # Original IBM Telco dataset (7,043 customers)
│   └── processed/            # Train/test splits (X_train, X_test, y_train, y_test)
│
├── models/
│   ├── preprocessor.pkl      # Fitted ColumnTransformer pipeline
│   └── random_forest_churn.pkl  # Tuned Random Forest classifier
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_Model_Finalization.ipynb
│   └── 06_Model_Evaluation.ipynb
│
├── app.py                    # Streamlit dashboard
├── styles.css                # Custom CSS for the dashboard
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🤖 Model Performance

Four algorithms were trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| **Random Forest** | **80.70%** | **67.71%** | **52.14%** | **58.91%** | **0.844** |
| XGBoost | — | — | — | — | — |

**Selected Model:** Tuned Random Forest — best overall balance between predictive performance and generalization.

---

## ✨ Dashboard Features

- **Real-time predictions** — Input customer details and get instant churn probability
- **Plotly gauge chart** — Visual churn probability indicator with color-coded risk zones
- **Feature importance** — Top 10 factors influencing the prediction
- **Risk assessment** — High / Medium / Low risk classification
- **Business recommendations** — Actionable retention strategies based on risk level
- **Model stats** — Pre-prediction display of model accuracy, AUC, and dataset overview

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Scikit-learn** — Model training & preprocessing
- **Streamlit** — Dashboard framework
- **Plotly** — Interactive visualizations
- **Pandas / NumPy** — Data processing
- **Joblib** — Model serialization

---

## 📦 Installation

```bash
# Clone the repository
git clone <repo-url>
cd IBM-Telco-Churn

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn** — 7,043 customers with 19 features covering:
- Demographics (gender, senior citizen, partner, dependents)
- Services (phone, internet, streaming, security, backup)
- Account (tenure, contract type, billing, payment method, charges)

Churn rate: **26.5%** (imbalanced classification)

---

## 👤 Author

**Muhammad Rahman** — 2026