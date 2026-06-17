import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# --- 1. DATA GENERATION & MODEL TRAINING ---
@st.cache_resource
def train_model():
    """Generates synthetic customer data and trains a Random Forest model."""
    np.random.seed(42)
    n_samples = 1000

    # Create dummy customer data
    data = {
        "Tenure_Months": np.random.randint(1, 72, n_samples),
        "Monthly_Charges": np.random.uniform(20.0, 120.0, n_samples),
        "Total_Charges": np.random.uniform(100.0, 5000.0, n_samples),
        "Contract_Type": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n_samples
        ),
        "Tech_Support": np.random.choice(["Yes", "No"], n_samples),
        # Basic logic: High monthly charges + Month-to-month contract = More likely to churn
        "Churn": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    df = pd.DataFrame(data)

    # Simple rule adjustments to make the data realistic for the ML model to learn
    df.loc[
        (df["Contract_Type"] == "Month-to-month") & (df["Monthly_Charges"] > 80),
        "Churn",
    ] = np.random.choice([0, 1], len(df[(df["Contract_Type"] == "Month-to-month") & (df["Monthly_Charges"] > 80)]), p=[0.3, 0.7])

    # Preprocessing text categories to numbers using Label Encoding
    le_contract = LabelEncoder()
    le_tech = LabelEncoder()

    df["Contract_Type"] = le_contract.fit_transform(df["Contract_Type"])
    df["Tech_Support"] = le_tech.fit_transform(df["Tech_Support"])

    # Features and Target
    X = df[
        [
            "Tenure_Months",
            "Monthly_Charges",
            "Total_Charges",
            "Contract_Type",
            "Tech_Support",
        ]
    ]
    y = df["Churn"]

    # Split and Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Return model and the encoders to handle future UI input
    return model, le_contract, le_tech


# Load the trained model assets
model, le_contract, le_tech = train_model()


# --- 2. STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    """
This tool uses a Machine Learning model (**Random Forest Classifier**) to evaluate whether a subscriber 
is likely to cancel their subscription based on their behavior and contract attributes.
"""
)

st.write("---")

# Split layout into input form and results
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("💡 Input Customer Attributes")

    # Sliders for continuous numerical data
    tenure = st.slider(
        "Tenure (How many months they've stayed)",
        min_value=1,
        max_value=72,
        value=12,
    )
    monthly_charges = st.slider(
        "Monthly Charges ($)", min_value=20.0, max_value=120.0, value=70.0
    )
    total_charges = st.slider(
        "Total Charges Accumulated ($)",
        min_value=20.0,
        max_value=8000.0,
        value=840.0,
    )

    # Dropdowns for categorical text data
    contract_input = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    tech_support_input = st.selectbox("Has Premium Tech Support?", ["No", "Yes"])

    # Transform UI text selections to matches the numerical inputs the model expects
    contract_encoded = le_contract.transform([contract_input])[0]
    tech_support_encoded = le_tech.transform([tech_support_input])[0]

with col2:
    st.subheader("🔮 Prediction Analysis")
    st.write("")

    # Construct the query array for the model
    input_data = np.array(
        [
            [
                tenure,
                monthly_charges,
                total_charges,
                contract_encoded,
                tech_support_encoded,
            ]
        ]
    )

    # Run predictions
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    churn_probability = probabilities[1] * 100

    # Display clean visual metrics based on the risk profile
    if prediction == 1:
        st.error(f"⚠️ High Risk: Churn Predicted!")
    else:
        st.success(f"✅ Low Risk: Customer Likely to Stay")

    # Progress bar showing actual risk level
    st.metric(label="Churn Probability Score", value=f"{churn_probability:.1f}%")
    st.progress(int(churn_probability))

    # Contextual insight box
    st.write("")
    with st.expander("ℹ️ How to interpret this result"):
        if churn_probability > 70:
            st.write(
                "**Action Plan Required:** This customer shows heavy signs of churning. Consider offering them a contract upgrade discount or a loyalty promotion to transition them away from a flexible Month-to-month plan."
            )
        elif churn_probability > 40:
            st.write(
                "**Watchlist:** The customer has moderate risk. Keep an eye on their account usage metrics or follow up with an automated customer satisfaction email."
            )
        else:
            st.write(
                "**Healthy Account:** The current metrics align heavily with customers who stay retained long-term."
            )
