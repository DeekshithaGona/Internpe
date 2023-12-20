import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load the saved model
model_path = 'LinearRegressionModel.pkl'
pipe = pickle.load(open(model_path, 'rb'))

# Display the input form in Streamlit
st.title('Car price Predictor Model')

# Input form for user
name = st.text_input('Car Name:', 'Maruti Suzuki Swift')
company = st.text_input('Company:', 'Maruti')
year = st.number_input('Year:', 2019)
kms_driven = st.number_input('Kilometers Driven:', 100)
fuel_type = st.selectbox('Fuel Type:', ['Petrol', 'Diesel', 'CNG'])


# Button to make predictions
if st.button('Predict'):
    # Transform input data
    input_data = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
      
    })

    prediction = pipe.predict(input_data)

    # Display the prediction
    st.success(f'Predicted Price: {prediction[0]}')
