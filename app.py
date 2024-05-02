import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Function to remove background using PyTorch model
def remove_background(image):
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
    
    # Post-process the output to get binary mask
    mask = (output.argmax(0) == 1).float()
    
    # Apply mask to original image
    result = image * mask.unsqueeze(2)

    return result

# Main function to run the Streamlit app
def main():
    st.title("Background Removal App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Remove background
        result = remove_background(image)

        # Display background-removed image
        st.image(result, caption="Background Removed", use_column_width=True)

if __name__ == "__main__":
    main()
