import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Streamlit app title
st.title("Mobile Money Fraud Detection System")

# Sidebar with project explanation
st.sidebar.title("ℹ️ About This Project")
st.sidebar.write("""
Mobile money fraud is a serious issue affecting financial transactions. 
This project aims to detect fraudulent transactions using machine learning. 
Enter transaction details below to check if a transaction is fraudulent.
""")

# Define model and scaler paths
MODEL_PATH = "xgboost_model.joblib"
SCALER_PATH = "scaler.pkl"

# Load the model and scaler with error handling
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Ensure it's saved correctly.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"❌ Scaler file '{SCALER_PATH}' not found. Ensure it's saved correctly.")
    st.stop()

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Ensure model has feature names
if not hasattr(model, "feature_names_in_"):
    st.error("⚠️ Model does not have feature names. Retrain and save it properly!")
    st.stop()

st.sidebar.success("✅ Model loaded successfully!")

# User input fields
st.header("📥 Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=4000.0)

transaction_type = st.selectbox("Transaction Type", ["Bill Payment", "Deposit", "Transfer", "Withdrawal"])

# Create DataFrame for input
df = pd.DataFrame({
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig]
})

# Add transaction type features dynamically
transaction_type_categories = ["Bill Payment", "Deposit", "Transfer", "Withdrawal"]
for t in transaction_type_categories:
    df[f"transaction_type_{t}"] = int(transaction_type == t)  # Convert to int (0 or 1)

# Ensure all features exist
expected_features = [str(feature) for feature in model.feature_names_in_]
df.columns = [str(col) for col in df.columns]
df = df.reindex(columns=expected_features, fill_value=0)
df.columns = df.columns.astype(str)

# Scale input and handle errors
try:
    df_scaled = scaler.transform(df)
except Exception as e:
    st.error(f"⚠️ Scaling error: {e}")
    st.stop()

# Predict fraud
if st.button("🔍 Check for Fraud"):
    try:
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This transaction is potentially fraudulent! (Risk: {probability:.2f})")
        else:
            st.success(f"✅ This transaction appears safe. (Risk: {probability:.2f})")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# Fraud Analysis Visualization
st.header("📊 Fraud Analysis")
if st.checkbox("📌 Show Fraud Distribution"):
    try:
        fraud_data = pd.read_csv("nigerian_mobile_money_transactions.csv")
        fig, ax = plt.subplots()
        fraud_data["is_fraud"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
        ax.set_xticklabels(["Legit", "Fraudulent"], rotation=0)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Error loading fraud data: {e}")

st.sidebar.write("### Created by Oyebamiji Samuel")
