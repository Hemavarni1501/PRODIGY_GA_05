import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="Prodigy GA Task 5", layout="wide")
st.title("🎨 Professional Neural Style Transfer")
st.caption("PRODIGY INFOTECH | GA TRACK | TASK 05")

# 2. Load the High-Resolution Stylization Model
@st.cache_resource
def get_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

hub_model = get_model()

# 3. Image Engine
def load_img(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# 4. Layout
col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Upload Content (Group Photo)", type=['jpg', 'png', 'jpeg'])
with col2:
    style_file = st.file_uploader("Upload Style (Paintings work best!)", type=['jpg', 'png', 'jpeg'])

# Settings to fix the "Bad Output"
st.sidebar.header("Optimization Tuning")
style_prediction_size = st.sidebar.slider("Style Sharpness", 128, 512, 256)
content_strength = st.sidebar.slider("Face Preservation", 0.0, 1.0, 0.2)

if content_file and style_file:
    if st.button("GENERATE MASTERPIECE"):
        with st.spinner("Executing Neural Synthesis..."):
            # Load images
            content_image = load_img(content_file)
            style_image = load_img(style_file)
            
            # Resize style image to the prediction size for better texture extraction
            style_image_resized = tf.image.resize(style_image, (style_prediction_size, style_prediction_size))

            # Run Style Transfer
            outputs = hub_model(tf.constant(content_image), tf.constant(style_image_resized))
            stylized_image = outputs[0]

            # Blend back some content to keep faces looking good (Alpha Blending)
            if content_strength > 0:
                stylized_image = (content_strength * content_image) + ((1 - content_strength) * stylized_image)

            final_img = tensor_to_image(stylized_image)

            # Show Result
            st.markdown("---")
            st.image(final_img, caption="Professional Stylized Output", use_container_width=True)
            
            # Download
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            st.download_button("Export PNG", buf.getvalue(), "task05_output.png")