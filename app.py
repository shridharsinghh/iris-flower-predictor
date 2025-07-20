import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open("iris_model.pkl", "rb"))

st.set_page_config(page_title="🌸 Iris Flower Predictor", page_icon="🌸", layout="centered")

st.markdown("# 🌸 Iris Flower Predictor App")
st.markdown("Enter flower measurements to predict the Iris type.")

# Input sliders in columns
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Predict button
if st.button("🌼 Predict Iris Type"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    st.success(f"✅ **Predicted Iris Type:** `{prediction}`")
