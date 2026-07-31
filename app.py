import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# LOAD CSS
# ==================================================

css_file = Path("styles.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL (CACHED)
# ==================================================

@st.cache_resource
def load_model():
    try:
        preprocessor = joblib.load("models/preprocessor.pkl")
        model = joblib.load("models/random_forest_churn.pkl")
        return preprocessor, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

preprocessor, model = load_model()

# ==================================================
# FEATURE NAMES (for importance chart)
# ==================================================

FEATURE_NAMES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Female", "gender_Male",
    "Partner_No", "Partner_Yes",
    "Dependents_No", "Dependents_Yes",
    "PhoneService_No", "PhoneService_Yes",
    "MultipleLines_No", "MultipleLines_NoPhone", "MultipleLines_Yes",
    "InternetService_DSL", "InternetService_Fiber", "InternetService_No",
    "OnlineSecurity_No", "OnlineSecurity_NoInternet", "OnlineSecurity_Yes",
    "OnlineBackup_No", "OnlineBackup_NoInternet", "OnlineBackup_Yes",
    "DeviceProtection_No", "DeviceProtection_NoInternet", "DeviceProtection_Yes",
    "TechSupport_No", "TechSupport_NoInternet", "TechSupport_Yes",
    "StreamingTV_No", "StreamingTV_NoInternet", "StreamingTV_Yes",
    "StreamingMovies_No", "StreamingMovies_NoInternet", "StreamingMovies_Yes",
    "Contract_Monthly", "Contract_OneYear", "Contract_TwoYear",
    "PaperlessBilling_No", "PaperlessBilling_Yes",
    "Payment_BankTransfer", "Payment_CreditCard", "Payment_ECheck", "Payment_MailedCheck"
]

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="title-banner">
<h1>📊 Customer Churn Prediction</h1>
<p>Predict whether a telecom customer is likely to churn using a trained Random Forest model</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.markdown("## 🧑‍💼 Customer Profile")

# --- Personal Info ---
with st.sidebar.expander("👤 Personal Information", expanded=True):

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
        help="Customer aged 65 or older"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
        help="Whether the customer has a partner"
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
        help="Whether the customer has dependents"
    )

# --- Billing ---
with st.sidebar.expander("💳 Billing Information", expanded=True):

    tenure = st.slider(
        "Tenure (Months)",
        0, 72, 12,
        help="Number of months the customer has been with the company"
    )

    monthly = st.number_input(
        "Monthly Charges ($)",
        0.0, 200.0, 70.0,
        help="Monthly amount charged to the customer"
    )

    total = st.number_input(
        "Total Charges ($)",
        value=float(tenure * monthly),
        disabled=True,
        help="Automatically calculated: Tenure × Monthly Charges"
    )

# --- Phone ---
with st.sidebar.expander("📱 Phone Service", expanded=False):

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

# --- Internet ---
with st.sidebar.expander("🌐 Internet Service", expanded=False):

    internet = st.selectbox(
        "Internet Service",
        ["Fiber optic", "DSL", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

# --- Contract ---
with st.sidebar.expander("📄 Contract & Payment", expanded=False):

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
        help="Whether the customer uses paperless billing"
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ==================================================
# PREDICT BUTTON
# ==================================================

st.markdown("---")

predict = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)

# ==================================================
# PRE-PREDICTION STATE
# ==================================================

if not predict:

    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top:0;">👈 Get Started</h3>
        <p style="opacity:0.7;">Fill in the customer details in the sidebar, then click <strong>Predict Customer Churn</strong> to see the results.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model stats
    st.markdown('<div class="section-header"><h3>🤖 Model Performance</h3></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-value">80.70%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">0.844</div>
                <div class="stat-label">AUC Score</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">67.71%</div>
                <div class="stat-label">Precision</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">58.91%</div>
                <div class="stat-label">F1-Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h3>📋 About the Dataset</h3></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-value">7,043</div>
                <div class="stat-label">Customers</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">19</div>
                <div class="stat-label">Features</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">26.5%</div>
                <div class="stat-label">Churn Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: #22c55e;">Random Forest</div>
                <div class="stat-label">Best Model</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# PREDICTION RESULTS
# ==================================================

if predict:

    # Convert Yes/No to 1/0
    senior_value = 1 if senior == "Yes" else 0

    # Create dataframe
    customer = pd.DataFrame({
        "SeniorCitizen": [senior_value],
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone],
        "MultipleLines": [multiple],
        "InternetService": [internet],
        "OnlineSecurity": [security],
        "OnlineBackup": [backup],
        "DeviceProtection": [protection],
        "TechSupport": [support],
        "StreamingTV": [tv],
        "StreamingMovies": [movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment]
    })

    # Preprocess
    customer_processed = preprocessor.transform(customer)

    # Prediction
    prediction = model.predict(customer_processed)[0]
    probability = model.predict_proba(customer_processed)[0][1]

    # Risk level
    if probability >= 0.70:
        risk = "🔴 High Risk"
        risk_color = "#ef4444"
    elif probability >= 0.40:
        risk = "🟡 Medium Risk"
        risk_color = "#eab308"
    else:
        risk = "🟢 Low Risk"
        risk_color = "#22c55e"

    # ======================================
    # METRICS ROW
    # ======================================

    st.markdown('<div class="section-header"><h3>📊 Prediction Results</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.1f}%"
        )

    with col2:
        st.metric(
            "Prediction",
            "⚠️ Churn" if prediction == 1 else "✅ Stay"
        )

    with col3:
        st.metric(
            "Risk Level",
            risk
        )

    # ======================================
    # GAUGE CHART
    # ======================================

    st.markdown('<div class="section-header"><h3>🎯 Churn Probability Gauge</h3></div>', unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#1e293b"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "rgba(0,0,0,0.2)",
                "tickfont": {"color": "#64748b", "size": 12}
            },
            "bar": {"color": risk_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(34, 197, 94, 0.12)"},
                {"range": [40, 70], "color": "rgba(234, 179, 8, 0.12)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.12)"}
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 3},
                "thickness": 0.8,
                "value": probability * 100
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # RESULT MESSAGE
    # ======================================

    if prediction == 1:
        card_class = "result-card-churn"
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0; color: #ef4444;">⚠️ High Churn Risk Detected</h3>
            <p>This customer has a <strong>{probability*100:.1f}%</strong> probability of churning.
            Immediate retention actions are recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        card_class = "result-card-stay"
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0; color: #22c55e;">✅ Customer Likely to Stay</h3>
            <p>This customer has a <strong>{(1-probability)*100:.1f}%</strong> probability of staying.
            Continue providing excellent service.</p>
        </div>
        """, unsafe_allow_html=True)


    # ======================================
    # BUSINESS RECOMMENDATION
    # ======================================

    st.markdown('<div class="section-header"><h3>💡 Business Recommendation</h3></div>', unsafe_allow_html=True)

    if probability >= 0.70:
        st.warning("""
### 🚨 Immediate Action Required

- **Contact customer immediately** for a retention conversation
- **Offer loyalty discount** or promotional pricing
- **Suggest annual contract** to lock in commitment
- **Assign dedicated support** representative
- **Investigate root causes** of dissatisfaction
""")

    elif probability >= 0.40:
        st.info("""
### ⚡ Proactive Engagement Recommended

- **Monitor customer activity** and usage patterns
- **Offer promotional packages** to increase value
- **Recommend additional services** that fit their profile
- **Encourage contract upgrade** from month-to-month
""")

    else:
        st.success("""
### 🌟 Customer Appears Satisfied

- **Continue delivering** high-quality service
- **Maintain regular engagement** with offers
- **Consider upselling** premium plans or add-ons
- **Use as referral source** for new acquisition
""")

        st.snow()

    # ======================================
    # CUSTOMER SUMMARY
    # ======================================

    with st.expander("📄 Customer Input Summary"):
        st.dataframe(
            customer,
            use_container_width=True
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
<div class="custom-footer">
    📊 Customer Churn Prediction Dashboard &nbsp;|&nbsp; Built with Streamlit & Scikit-learn &nbsp;|&nbsp; Created by Muhammad Rahman &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)