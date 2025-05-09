import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from briarmbg import BriaRMBG
from PIL import Image
import streamlit as st
from io import BytesIO
import requests
from urllib.parse import urlparse

# Load model
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

def process(image):
    orig_image = image.convert("RGB")
    w, h = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = im_tensor / 255.0
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()

    with torch.no_grad():
        result = net(im_tensor)

    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
    result = (result - result.min()) / (result.max() - result.min())

    result_array = (result * 255).cpu().numpy().astype(np.uint8)
    pil_mask = Image.fromarray(np.squeeze(result_array))

    new_im = orig_image.copy()
    new_im.putalpha(pil_mask)
    
   
    img_byte_arr = BytesIO()
    new_im.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  
    
    return img_byte_arr

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="Abhinav Adarsh", layout="centered")

# Add custom CSS for the gradient animation color on title text
st.markdown("""
    <style>
        @keyframes gradientAnimation {
            0% { color: #3F5EFB; }
            50% { color: #FC466B; }
            100% { color: #3F5EFB; }
        }

        .gradient-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            animation: gradientAnimation 3s linear infinite;
        }

        .gradient-container {
            background: linear-gradient(45deg, #3F5EFB, #FC466B);
            background-size: 400% 400%;
            color: transparent;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            padding: 50px;
            border-radius: 10px;
            animation: gradientAnimation 3s linear infinite;
            background-clip: text;
        }

        .gradient-container span {
            background: linear-gradient(45deg, #3F5EFB, #FC466B);
            -webkit-background-clip: text;
            color: transparent;
        }

        .tab-container {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title with gradient animation on text color
st.markdown('<div class="gradient-title">RMBG</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Updated description
st.markdown("""
<div style='text-align: right'>
Developed by <a href='https://www.linkedin.com/in/abhinavgautam08' target='_blank'>Abhinav Adarsh</a>
</div>
""", unsafe_allow_html=True)

# Create tabs for upload methods
tab1, tab2 = st.tabs(["UP Img", "Img URL"])

# Tab 1: File Upload
with tab1:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image (Original)", use_container_width=True)
        st.write("Processing...")
        
        # Get the processed image in PNG format
        img_bytes = process(image)
        
        # Show the processed image
        st.image(img_bytes, caption="Image (RMBG)", use_container_width=True)
        
        # Optional: Download button
        st.download_button("Download", img_bytes, file_name="abhinavgautam08.(RMBG).png", mime="image/png")

# Tab 2: URL Input
with tab2:
    url = st.text_input("Enter the URL of an image")
    if url:
        if is_valid_url(url):
            st.write("Downloading image...")
            image = load_image_from_url(url)
            if image:
                st.image(image, caption="Image (Original)", use_container_width=True)
                st.write("Processing...")
                
                # Get the processed image in PNG format
                img_bytes = process(image)
                
                # Show the processed image
                st.image(img_bytes, caption="Image (RMBG)", use_container_width=True)
                
                # Optional: Download button
                st.download_button("Download", img_bytes, file_name="abhinavgautam08.(RMBG).png", mime="image/png")
        else:
            st.error("Please enter a valid URL")