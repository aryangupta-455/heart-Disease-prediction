import streamlit as st
import numpy as np
import joblib

# Mapping options
cp_options = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4
}

thal_options = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

# Load model
model = joblib.load('heart_disease_model.pkl')

# App title
st.title("Heart Disease Prediction App")
st.markdown("This app predicts whether a patient is at risk of heart disease based on six clinical inputs.")

# Sidebar inputs
st.sidebar.header("Enter Patient Detail")

cp_label = st.sidebar.selectbox("Chest Pain type", list(cp_options.keys()))
cp = cp_options[cp_label]

max_hr = st.sidebar.slider("Max Heart Rate", min_value=70, max_value=210, value=150)

oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0)

thal_label = st.sidebar.selectbox("Thallium Test Result", list(thal_options.keys()))
thal = thal_options[thal_label]

chol = st.sidebar.number_input("Enter the Cholesterol", min_value=0, max_value=600, step=1)

bp = st.sidebar.number_input("Enter Blood Pressure", min_value=0, max_value=201, step=1)

# Prediction
if st.sidebar.button("Predict"):
    features = np.array([[cp, max_hr, oldpeak, thal, chol, bp]], dtype=float)

    try:
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0][1]

        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.error(f" Presence of Heart Disease (Confidence: {confidence:.2f})")
        else:
            st.success(f"No Heart Disease Detected (Confidence: {confidence:.2f})")

        st.markdown("### Feature Summary")
        st.write({
            "Chest Pain Type": cp_label,
            "Max Heart Rate": max_hr,
            "ST Depression": oldpeak,
            "Thallium": thal_label,
            "Cholesterol": chol,
            "BP": bp
        })

    except Exception as e:
        st.error(f"Prediction failed: {e}")
