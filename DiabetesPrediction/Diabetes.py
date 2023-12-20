import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model and other necessary transformations
pipeline = joblib.load('diabetes_pipeline.pkl')

# Display the input form in Streamlit
st.title('Diabetes Prediction App')

# Input form for user
pregnancies = st.number_input('Pregnancies:', value=0, step=1)
glucose = st.number_input('Glucose:', value=0, step=1)
blood_pressure = st.number_input('Blood Pressure:', value=0, step=1)
skin_thickness = st.number_input('Skin Thickness:', value=0, step=1)
insulin = st.number_input('Insulin:', value=0, step=1)
bmi = st.number_input('BMI:', value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function:', value=0.0, step=0.01)
age = st.number_input('Age:', value=0, step=1)

# Button to make predictions
if st.button('Predict Diabetes'):
    # Create a numpy array from the input data
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)

    # Make predictions
    prediction = pipeline.predict(input_data)

    # Display the prediction
    if prediction[0] == 0:
        st.success('The person is not diabetic.')
    else:
        st.error('The person is diabetic.')
