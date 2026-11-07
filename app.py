"""Interactive churn-risk demo.  Run:  streamlit run app.py  (after python -m src.train)"""
import joblib
import pandas as pd
import streamlit as st

from src.data_prep import CATEGORICAL, NUMERIC, ROOT, add_features

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉")
st.title("📉 Customer Churn Risk Predictor")
st.caption("Tuned Logistic Regression trained on the IBM Telco Customer Churn dataset")

bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]
yn = ["No", "Yes"]

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
with c2:
    senior = st.selectbox("Senior citizen", yn)
    partner = st.selectbox("Partner", yn)
    dependents = st.selectbox("Dependents", yn)
    security = st.selectbox("Online security", yn)
    support = st.selectbox("Tech support", yn)
    paperless = st.selectbox("Paperless billing", yn, index=1)

no_net = internet == "No"
net = lambda v: "No internet service" if no_net else v
row = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": internet,
    "OnlineSecurity": net(security), "OnlineBackup": net("No"), "DeviceProtection": net("No"),
    "TechSupport": net(support), "StreamingTV": net("No"), "StreamingMovies": net("No"),
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
}])
row = add_features(row)[NUMERIC + CATEGORICAL]

p = model.predict_proba(row)[0, 1]
st.metric("Churn probability", f"{p:.0%}")
if p >= threshold:
    st.error("High risk: recommend a retention offer (for example, a discount for switching to an annual contract).")
else:
    st.success("Low risk: no action needed.")
