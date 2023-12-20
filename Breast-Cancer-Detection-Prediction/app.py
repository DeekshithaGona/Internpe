import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
# Load the pipeline
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Display the input form in Streamlit
st.title('Breast Cancer Diagnosis Predictor')

# Input form for user
radius_mean = st.number_input('Radius Mean:', value=11.76)
texture_mean = st.number_input('Texture Mean:', value=21.6)
perimeter_mean = st.number_input('Perimeter Mean:', value=74.72)
area_mean = st.number_input('Area Mean:', value=427.9)
smoothness_mean = st.number_input('Smoothness Mean:', value=0.08637)
compactness_mean = st.number_input('Compactness Mean:', value=0.04966)
concavity_mean = st.number_input('Concavity Mean:', value=0.01657)
concave_points_mean = st.number_input('Concave Points Mean:', value=0.01115)
symmetry_mean = st.number_input('Symmetry Mean:', value=0.1495)
fractal_dimension_mean = st.number_input('Fractal Dimension Mean:', value=0.05888)
radius_se = st.number_input('Radius SE:', value=0.4062)
texture_se = st.number_input('Texture SE:', value=1.21)
perimeter_se = st.number_input('Perimeter SE:', value=2.635)
area_se = st.number_input('Area SE:', value=28.47)
smoothness_se = st.number_input('Smoothness SE:', value=0.005857)
compactness_se = st.number_input('Compactness SE:', value=0.009758)
concavity_se = st.number_input('Concavity SE:', value=0.01168)
concave_points_se = st.number_input('Concave Points SE:', value=0.007445)
symmetry_se = st.number_input('Symmetry SE:', value=0.02406)
fractal_dimension_se = st.number_input('Fractal Dimension SE:', value=0.001769)
radius_worst = st.number_input('Radius Worst:', value=12.98)
texture_worst = st.number_input('Texture Worst:', value=25.72)
perimeter_worst = st.number_input('Perimeter Worst:', value=82.98)
area_worst = st.number_input('Area Worst:', value=516.5)
smoothness_worst = st.number_input('Smoothness Worst:', value=0.1085)
compactness_worst = st.number_input('Compactness Worst:', value=0.08615)
concavity_worst = st.number_input('Concavity Worst:', value=0.05523)
concave_points_worst = st.number_input('Concave Points Worst:', value=0.03715)
symmetry_worst = st.number_input('Symmetry Worst:', value=0.2433)
fractal_dimension_worst = st.number_input('Fractal Dimension Worst:', value=0.06563)
# Add more input fields for other parameters

# Button to make predictions
if st.button('Predict Diagnosis'):
    # Create a numpy array from the input data
    input_data = np.array([
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
        fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
        fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
        area_worst, smoothness_worst, compactness_worst, concavity_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]).reshape(1, -1)
    
    # Make predictions
    prediction = pipeline.predict(input_data)
    prediction_label = [np.argmax(prediction)]
    # Display the prediction
    if prediction_label == 0:
        st.success('The tumor is Malignant.')
    else:
        st.error('The tumor is Benign.')
    

