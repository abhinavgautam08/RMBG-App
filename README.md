# RMBG - Image Background Removal Tool

![RMBG AI Demo](https://img.shields.io/badge/Demo-Live-brightgreen)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://github.com/abhinavgautam08/RMBG-App/blob/main/app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/abhinavgautam08/RMBG-App/blob/main/LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://abhinavgautam08-rmbg.streamlit.app/)

A powerful, user-friendly web application 

## ✨ Features

- 🖼️ Removes background from any image with AI precision
- 🚀 Fast processing with GPU acceleration (when available)
- 📱 Responsive web interface built with Streamlit
- 📤 Upload images directly or via URL
- 💾 Download processed images with transparent backgrounds
- 🔄 High-quality output with alpha channel preservation

## 🔧 Installation

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/abhinavgautam08/RMBG-App.git
cd RMBG-App
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will start and automatically open in your default web browser at `http://localhost:8501`.

## 📖 How to Use

1. **Upload an Image**: You can either:
   - Upload an image from your device (JPG, PNG, JPEG)
   - Provide a URL to an online image

2. **Processing**: The application will automatically process the image to remove the background.

3. **Download**: Download the processed image with a transparent background in PNG format.

## 💻 Tech Stack

- **Frontend & Backend**: Streamlit
- **Model**: RMBG 1.4
- **Image Processing**: PyTorch, NumPy, PIL

## 🔮 How It Works

The application uses the BriaRMBG model, which employs deep learning to identify and separate foreground objects from backgrounds. The process includes:

1. Image preprocessing and resizing
2. Passing the image through the BriaRMBG neural network
3. Post-processing the model output to create a smooth alpha mask
4. Applying the alpha mask to create a transparent background
5. Serving the processed image to the user

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abhinav Adarsh**
- LinkedIn: [abhinavgautam08](https://www.linkedin.com/in/abhinavgautam08)


Created ❤️ by [Abhinav Adarsh](https://www.linkedin.com/in/abhinavgautam08)
# RMBG-App
