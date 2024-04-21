import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load pre-trained deep learning model for background removal
model = load_model("path_to_your_pretrained_model.h5")  # Replace with the path to your model file

def remove_background(image):
    # Resize image to fit the input size of the model
    image_resized = cv2.resize(image, (224, 224))

    # Normalize pixel values to range [0, 1]
    image_normalized = image_resized / 255.0

    # Predict the mask for the image
    mask = model.predict(np.expand_dims(image_normalized, axis=0))[0]

    # Apply the mask to the original image
    masked_image = image * mask[:, :, np.newaxis]

    return masked_image

def main():
    st.title("Image Background Removal")

    uploaded_image = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Read image file
        image_bytes = uploaded_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image
        processed_image = remove_background(image)

        st.image(processed_image, channels="BGR", caption="Processed Image")

if __name__ == "__main__":
    main()
