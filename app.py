import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

cp_options = { "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4}

thal_options = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}


#model loading
model = joblib.load('heart_disease_model.pkl')
print("Model loaded successfully" , type(model))

#App tittle
st.title("Heart Disease Prediction App")
st.markdown("This app predicts whether a patient is at risk of heart disease based on six clinical inputs.")

st.sidebar.header("Enter Patient Detail")


#variable making
cp_label = st.sidebar.selectbox("Chest Pain type",list(cp_options.keys()))
cp = cp_options[cp_label]
st.write("Your Chestpain is ", cp)


#max heart rate
max_hr = st.sidebar.slider("Max Heart Rate",70,150,210)
st.write("Your Max Heart Rate is ", max_hr)


#ST depression
oldpeak = st.sidebar.slider("ST Depression", min_value=0.0, max_value=6.0, value=1.0)
st.write("Your Entered ST Depression ", oldpeak)


#Thallium
thal_label = st.sidebar.selectbox("Thallium Test Result", list(thal_options.keys()))
thal = thal_options[thal_label]
st.write("Your Thallium ", thal)


#Cholesterol
chol = st.sidebar.slider("Enter the Cholesterol ",min_value=0,max_value=600,value=1)
st.write("Your Cholesterol ", chol)


#Blood Pressure
bp = st.sidebar.slider("Enter Blood pressure ",min_value=0, max_value=201, value=1 )
st.write("Your Blood Pressure is ", bp)


#Age
age = st.sidebar.slider("Enter the Age ",min_value=25,max_value=80,value=1)
st.write("Your Age is ", age)


#Exercise angina
exna = st.sidebar.selectbox("Enter the Exercise angina",[0,1])
st.write("Your Exercise angina is ", exna)


#Number of vessels fluro
vesflu = st.sidebar.selectbox("Choose Number of vessels fluro",[0,1,2,3])
st.write("Your Number of vessels fluro is ", vesflu)

gen = st.sidebar.selectbox("Choose your Gender/Sex",[0,1])

'''________________________________________________________________________________________________________________________'''

if st.sidebar.button('Predict'):
    features = np.array([[cp, max_hr, oldpeak, thal, chol, bp]], dtype=float)
    
    
    try:
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0][1]

        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.error(f"Presence of Heart Disease (Confidence: {confidence:.2f})")
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
