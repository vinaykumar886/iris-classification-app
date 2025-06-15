import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_rf_model.pkl')

# Title
st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements and predict its species!")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("🌺 Predict Flower Type"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    
    classes = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Flower: **{classes[prediction]}**")
