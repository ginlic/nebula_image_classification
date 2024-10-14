import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Define the target image size for the model
img_size = 224  

def predict_image(image, model, labels):
    # Preprocess the image
    image = image.resize((img_size, img_size))  # Resize the image
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_label = labels[predicted_class_index]

    return predicted_class_label
    
# Set title
st.title("Nebulae Image Classification")

# Write about the app
st.write("This app allows a user to upload an image of a nebula and predict which of the five categories it belongs to. The primary categories include: emission, reflection, dark, planetary, and supernova.")

# Provide instructions
st.subheader('Please upload an image of a nebula')

# Upload file
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load classifier
model = load_model('EfficientNetB0-84acc_Imagenet_maxpool_tuneddo.keras')


# Load class names
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]  # Remove newline characters

# Display image and classify if file is uploaded
if file is not None:
    # Read the image and preprocess it
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify image
    predicted_label = predict_image(image, model, class_names)

    st.subheader("Predicted Label:")
    st.write(predicted_label)