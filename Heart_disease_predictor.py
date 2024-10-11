import streamlit as st
import pickle
import numpy as np

# Load the saved RandomForest model
model = pickle.load(open("random_forest_model_1.pkl", "rb"))

# Set a nice title for the app
st.title("💓 Heart Disease Prediction App")

# Add a description and instructions
st.markdown("""
### Predict Heart Disease Based on Medical Data
This app uses a machine learning model to predict whether a patient is likely to have heart disease based on medical information. Please fill out the following details, and click **Predict** to get the result.
""")

# Create sections for inputs to make the layout more structured
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25, help="Enter the patient's age.")
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Select the patient's sex (0 = Female, 1 = Male).")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"Type {x}", help="0-3 indicate types of chest pain experienced by the patient.")

with col2:
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, help="Patient's resting blood pressure.")
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Serum cholesterol level in mg/dl.")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True", help="1 = True if fasting blood sugar > 120 mg/dl, else 0.")

st.header("Health Metrics")

col3, col4 = st.columns(2)

with col3:
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], help="0-2 represent resting ECG results.")
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150, help="Maximum heart rate achieved during the test.")
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="1 = Yes if exercise induced angina, else 0.")

with col4:
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, help="ST depression induced by exercise relative to rest.")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2], help="0-2 indicate slope of peak exercise ST segment.")
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4], help="0-4 number of major vessels colored by fluoroscopy.")
    thal = st.selectbox("Thalassemia Type", [1, 2, 3], format_func=lambda x: "Normal" if x == 1 else ("Fixed Defect" if x == 2 else "Reversible Defect"), help="1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect.")

# Collect the inputs in an array
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# When the user clicks the 'Predict' button, show prediction
if st.button("🔎Predict"):
    prediction = model.predict(user_input)
    
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease (Positive).")
    else:
        st.success("✅ The patient is unlikely to have heart disease (Negative).")
