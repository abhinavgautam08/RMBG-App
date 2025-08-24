import numpy as np # (abhinavgautam08)
import torch # (abhinavgautam08)
import torch.nn.functional as F # (abhinavgautam08)
from torchvision.transforms.functional import normalize # (abhinavgautam08)
from rmbg import BriaRMBG # (abhinavgautam08)
from PIL import Image # (abhinavgautam08)
import streamlit as st # (abhinavgautam08)
from io import BytesIO # (abhinavgautam08)
import requests # (abhinavgautam08)
from urllib.parse import urlparse # (abhinavgautam08)

# Load model with caching to avoid multiple copies on reruns (abhinavgautam08)
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    net.to(device)
    net.eval()
    # Use half precision on CUDA to reduce memory
    if device.type == "cuda":
        net.half()
    return net, device

net, device = load_model()

def resize_image(image, size=(1024, 1024)):
    image = image.convert('RGB')
    image = image.resize(size, Image.BILINEAR)
    return image

def process(image, size=(1024, 1024)):
    orig_image = image.convert("RGB")
    w, h = orig_image.size
    image_resized = resize_image(orig_image, size=size)
    im_np = np.array(image_resized)

    # Prepare tensor on the right device/dtype
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    im_tensor = torch.from_numpy(im_np).to(device=device, dtype=dtype).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = im_tensor / 255.0
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.inference_mode():
        outputs, _ = net(im_tensor)

    result = torch.squeeze(F.interpolate(outputs[0], size=(h, w), mode='bilinear'), 0)
    result = (result - result.min()) / (result.max() - result.min() + 1e-6)

    result_array = (result * 255).detach().to("cpu").numpy().astype(np.uint8)
    pil_mask = Image.fromarray(np.squeeze(result_array))

    new_im = orig_image.copy()
    new_im.putalpha(pil_mask)

    # Free GPU memory promptly
    del im_tensor, outputs, result
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Return PIL Image; create bytes only when needed for download
    return new_im

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

# Streamlit UI(abhinavgautam08)
st.set_page_config(page_title="Abhinav Adarsh", layout="centered")

# Add custom CSS for the gradient animation color on title text(abhinavgautam08)
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

# Title with gradient animation on text color(abhinavgautam08)
st.markdown('<div class="gradient-title">RMBG</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Updated description(abhinavgautam08)
st.markdown("""
<div style='text-align: right'>
Developed by <a href='https://www.linkedin.com/in/abhinavgautam08' target='_blank'>Abhinav Adarsh</a>
</div>
""", unsafe_allow_html=True)

# Create tabs for upload methods(abhinavgautam08)
tab1, tab2 = st.tabs(["UP Img", "Img URL"])

# Controls were moved from sidebar into tab-specific sections
# Tab 1: File Upload(abhinavgautam08)
with tab1:
    # Quality/Memory control placed above the uploader
    quality = st.select_slider(
        "Quality/Memory",
        options=["Low (512)", "Medium (768)", "High (1024)"],
        value="High (1024)"
    )
    if "512" in quality:
        target_size = (512, 512)
    elif "768" in quality:
        target_size = (768, 768)
    else:
        target_size = (1024, 1024)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image (Original)", use_container_width=True)
        st.write("Processing...")

        # Process image to PIL with alpha (abhinavgautam08)
        out_pil = process(image, size=target_size)

        # Show the processed image(abhinavgautam08)
        st.image(out_pil, caption="Image (RMBG)", use_container_width=True)

        # Optional: Download button (create bytes on demand)
        buf = BytesIO()
        out_pil.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download", buf, file_name="abhinavgautam08.(RMBG).png", mime="image/png")

# Tab 2: URL Input(abhinavgautam08)
with tab2:
    url = st.text_input("Enter the URL of an image")
    # Default to High quality if using URL flow (no sidebar control)
    target_size = (1024, 1024)
    if url:
        if is_valid_url(url):
            st.write("Downloading image...")
            image = load_image_from_url(url)
            if image:
                st.image(image, caption="Image (Original)", use_container_width=True)
                st.write("Processing...")

                # Process image to PIL with alpha (abhinavgautam08)
                out_pil = process(image, size=target_size)

                # Show the processed image(abhinavgautam08)
                st.image(out_pil, caption="Image (RMBG)", use_container_width=True)

                # Optional: Download button (create bytes on demand)
                buf = BytesIO()
                out_pil.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download", buf, file_name="abhinavgautam08.(RMBG).png", mime="image/png")
        else:
            st.error("Please enter a valid URL")
