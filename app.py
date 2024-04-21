import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st
import urllib.request

# Define a function to download the model from a GitHub repository
def download_model():
    model_url = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
    model_path = "deeplabv3_resnet101_coco.pth"
    urllib.request.urlretrieve(model_url, model_path)
    return model_path

# Download the model
model_path = download_model()

# Load pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=False)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define a function to perform background removal
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

    # Apply the mask to the original image
    background_removed_image = np.array(image) * (1 - background_mask[:, :, np.newaxis] / 255)

    return background_removed_image

# Main function for Streamlit app
def main():
    st.title("Image Background Removal")

    uploaded_image = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Read and preprocess image
        image = Image.open(uploaded_image)

        # Remove background
        background_removed_image = remove_background(image)

        # Display processed image
        st.image(background_removed_image, caption="Processed Image", channels="RGB")

if __name__ == "__main__":
    main()
