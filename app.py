import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

# Function to remove background using PyTorch model
def remove_background(image, model):
    # Ensure image has 3 channels (RGB)
    image = image.convert("RGB")
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Run inference on the model
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # Convert output to binary mask
    mask = (output.argmax(0) == 1).float()

    # Resize mask to match original image size
    mask = TF.resize(mask, image.size, interpolation=Image.NEAREST)

    # Apply mask to original image
    result = np.array(image) * mask.unsqueeze(2)

    return Image.fromarray(result.astype(np.uint8))

# Main function to run the Streamlit app
def main():
    st.title("Background Removal App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Remove background
        result = remove_background(image, model)

        # Display background-removed image
        st.image(result, caption="Background Removed", use_column_width=True)

if __name__ == "__main__":
    # Load your PyTorch model here
    # Example: model = torch.load("path_to_your_model.pth")
    model = None  # Placeholder for your model
    st.set_option('deprecation.showfileUploaderEncoding', False)  # This line avoids a warning message
    main()
