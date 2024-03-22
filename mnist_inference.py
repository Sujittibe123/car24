import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('mnist_trained_model.joblib')

# Streamlit app title
st.title('MNIST Handwritten Digit Recognition')

# File uploader for user input
st.write("Upload an image of a handwritten digit (0-9):")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize image to 28x28 and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    # Convert image to numpy array and normalize pixel values
    image = np.array(image) / 255.0
    # Flatten image array to 1D
    image = image.flatten()
    # Return preprocessed image
    return image

# Function to make predictions
def predict_digit(image):
    # Reshape image to match model input shape
    image = image.reshape(1, -1)
    # Make prediction
    prediction = model.predict(image)
    # Return predicted digit
    return prediction[0]

# Display the uploaded image and prediction
if uploaded_file is not None:
    # Preprocess and predict digit
    image = preprocess_image(uploaded_file)
    prediction = predict_digit(image)
    
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    # Display predicted digit
    st.write('Predicted Digit:', prediction)