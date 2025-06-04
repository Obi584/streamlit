import streamlit as st
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Image Classification Model")

def load_model(path):
    with open(path, 'rb') as file:
        model_ = pickle.load(file)
    return model_

model = load_model('image_model.pkl')

def preprocess_image(pil_image):
    img = pil_image.convert("RGB")  # Make sure it's 3-channel
    img = img.resize((224, 224))    # Resize to your model's input
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
    return tf.convert_to_tensor(img_array)

def process_path(file_path, label):
    img = preprocess_image(file_path)
    return img, label

# Subir imagen
uploaded_file = st.file_uploader("Upload an image that includes tom, jerry, both or none", type=["jpg", "png", "jpeg"])
class_names = ['tom', 'jerry', 'none', 'both']

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image", use_column_width = True)

    image, label = process_path(image, 'none')
    image = tf.expand_dims(image, 0)
    # Predict
    if st.button("Classify"):
        prediction = model.predict(image)[0]
        label_prediction = np.argmax(prediction)

        st.success(f"Predicted class: {class_names[label_prediction]}")