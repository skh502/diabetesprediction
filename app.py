import streamlit as st
import pandas as pd
import numpy as np
# import joblib
from prediction import predict_svc, predict_knn  

# Define the feature names used in training
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.title('Diabetes Prediction System')
st.markdown('Enter the following details')
col1, col2 = st.columns(2)



with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)


with col2:
    insulin = st.number_input('Insulin', min_value=0, max_value=800, step=1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
    age = st.slider('Age', 0, 120, 25)
     
st.text('')
st.text('')
st.text('')


col1, col2 = st.columns([1, 1])

# Create a DataFrame with feature names
features_df = pd.DataFrame(
    [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
    columns=feature_names)


with col1:
    if st.button('Predict with SVC'): 
        # Make prediction
        prediction = predict_svc(features_df)
    
        # Display the outcome
        if prediction[0] == 0:
            st.write('Chances of diabetes are low')
        else:
            st.write('Chances of diabetes are high')


with col2:
    if st.button('Predict with KNN'):
        # Make prediction
        prediction = predict_knn(features_df)
        
        # Display the outcome
        if prediction[0] == 0:
            st.write('Chances of diabetes are low')
        else:
            st.write('Chances of diabetes are high')


# Pregnancies	Glucose	BloodPressure	
# SkinThickness	Insulin	BMI	DiabetesPedigreeFunction
# Age	Outcome