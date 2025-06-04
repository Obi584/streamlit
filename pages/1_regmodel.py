import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from flatbuffers.packer import float32

st.title("Regression Model")

def load_model(path):
    with open(path, 'rb') as file:
        model_ = pickle.load(file)
    return model_

model = load_model('regression_model.pkl')

# Insertar valores para predecir
st.write('Example: 0,0,1,1,195,102,252,72.72,5,2,0,85.73,3,0,115,2022,1')
values = st.text_area("Comma separated values")
values = [float(x.strip()) for x in values.split(',')]
values = np.array(values, dtype = np.float32)

if st.button("Predict"):
    if values.any():
        prediction = model.predict(np.expand_dims(values, axis = 0))  # For text or single row
        st.write("Prediction:", prediction[0])