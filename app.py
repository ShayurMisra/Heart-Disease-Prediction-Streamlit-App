import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load SVM model and scaler
model = joblib.load('SVM_Model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# App title and description
st.title('Heart Disease Prediction')
st.write('Enter patient details to predict if they have heart disease.')

# Define function to preprocess input data
def preprocess_input_data(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'True' else 0
    exang = 1 if exang == 'Yes' else 0
    
    cp_dict = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    }
    
    restecg_dict = {
        'Normal': 0,
        'ST-T Wave Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }
    
    slope_dict = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }
    
    thal_dict = {
        'Normal': 0,
        'Fixed Defect': 1,
        'Reversible Defect': 2
    }
    
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp_dict[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg_dict[restecg]],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope_dict[slope]],
        'ca': [ca],
        'thal': [thal_dict[thal]]
    })

    input_data_processed = preprocessor.transform(input_data)
    return input_data_processed

# Input fields for patient details
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.radio('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=1)
chol = st.number_input('Cholesterol (mg/dl)', min_value=1)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.selectbox('Resting ECG Result', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.number_input('Max Heart Rate Achieved', min_value=1)
exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Predict button
if st.button('Predict'):
    input_data = preprocess_input_data(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    prediction = model.predict(input_data)

    # Display prediction result with styled message
    if prediction[0] == 1:
        st.success('The patient is likely to have heart disease.')
    else:
        st.success('The patient is unlikely to have heart disease.')
