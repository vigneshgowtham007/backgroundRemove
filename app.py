import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import urllib.request

# Define a function to download the model from a PyTorch model repository
def download_model():
    model_url = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
    model_path = "deeplabv3_resnet101_coco.pth"
    urllib.request.urlretrieve(model_url, model_path)
    return model_path

# Define a function to load model weights while ignoring auxiliary classifier keys
def load_model_weights(model, model_path):
    state_dict = torch.load(model_path)
    # Remove keys corresponding to auxiliary classifiers
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier')}
    # Load the modified state dictionary
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore missing keys
    return model

# Download the model
model_path = download_model()

# Load pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=False)
model = load_model_weights(model, model_path)
model.eval()

# Define a function to remove background and show only the object
def remove_background(image):
    # Apply transformations to the image
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize image to match model input size
        transforms.ToTensor(),           # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # Create a mask where background is black and foreground is white
    background_mask = output_predictions != 0  # Background class index is 0
    background_mask = background_mask.cpu().numpy().astype(np.uint8) * 255  # Convert mask to 0s and 255s

    # Resize the background mask to match the shape of the image
    background_mask = np.array(Image.fromarray(background_mask).resize(image.size))

    # Invert the background mask
    foreground_mask = 255 - background_mask

    # Convert the foreground mask to 3 channels to match the image
    foreground_mask = np.stack([foreground_mask] * 3, axis=2)

    # Apply the mask to the original image to remove the background
    object_image = np.array(image) * (foreground_mask / 255)

    # Ensure the output array is of type np.uint8
    object_image = np.clip(object_image, 0, 255).astype(np.uint8)

    return object_image

def main():
    st.title("Background Removal")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)

        # Remove background
        object_image = remove_background(image)

        # Display only the object
        st.image(object_image, caption='Object without Background', use_column_width=True)

if __name__ == "__main__":
    main()
