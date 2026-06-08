import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import time

# Set page layout and title
st.set_page_config(
    page_title="Pneumonia AI Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="🫁"
)

# Custom CSS for a beautiful, clean, modern medical UI
st.markdown("""
<style>
    /* Clean background and fonts */
    .main {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header Container */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* Diagnostic Alert Banner */
    .result-card {
        border-radius: 16px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .result-status {
        font-size: 1.75rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
        letter-spacing: 0.05em;
    }
    
    .result-confidence {
        font-size: 1.1rem;
        font-weight: 500;
        color: #475569;
    }
    
    /* Card design for images */
    .img-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        text-align: center;
    }
    
    /* Custom divider */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load the keras model (cached)
@st.cache_resource
def load_pneumonia_model():
    model_path = "pneumonia_vgg16_model.h5"
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

# Grad-CAM Heatmap generation function
def generate_gradcam(img_array, model, last_conv_layer_name="block5_conv3", threshold=0.5):
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
        
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Cast to tensor explicitly — Keras 3 / TF 2.20 can return a list inside GradientTape
        predictions = tf.cast(predictions, tf.float32)
        prob = float(predictions[0][0])
        
        # Visualize features contributing to the predicted class
        if prob >= threshold:
            target_output = predictions[:, 0]
        else:
            target_output = 1.0 - predictions[:, 0]
            
    grads = tape.gradient(target_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-10
    heatmap = heatmap / max_val
    return heatmap.numpy()

# Load the model
model = load_pneumonia_model()

# Header Section
st.markdown("""
<div class="header-container">
    <div class="main-title">🫁 Pneumonia AI Detector</div>
    <div class="subtitle">Upload a chest X-ray image to run automated VGG-16 neural network classification with Grad-CAM explainability.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar - keeps options neatly tucked away
with st.sidebar:
    st.markdown("### ⚙️ Diagnostic Settings")
    threshold = st.slider(
        "Decision Boundary (Sensitivity)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust the threshold. Lower values increase pneumonia sensitivity."
    )
    st.markdown("---")
    st.markdown("### 🔬 Model Details")
    st.caption("Model: VGG-16 Transfer Learning\nResolution: 224x224x3")

if model is None:
    st.error("Model file 'pneumonia_vgg16_model.h5' was not found in the current folder.")
else:
    # File upload
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display analysis spinner
        with st.spinner("Analyzing Chest X-Ray..."):
            # Resize and scale
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            processed_img = img_array / 255.0  # Normalized input
            
            # Predict
            input_tensor = np.expand_dims(processed_img, axis=0)
            prediction = model.predict(input_tensor)[0][0]
            
            # Determine class
            is_pneumonia = prediction >= threshold
            confidence = prediction if is_pneumonia else (1.0 - prediction)
            confidence_percent = confidence * 100
            
            # Generate Grad-CAM overlay
            heatmap = generate_gradcam(processed_img, model, threshold=threshold)
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            superimposed_img = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_color_rgb, 0.4, 0)

        # 1. Output Banner
        banner_bg = "#fef2f2" if is_pneumonia else "#f0fdf4"
        banner_border = "#fecaca" if is_pneumonia else "#bbf7d0"
        text_color = "#991b1b" if is_pneumonia else "#166534"
        finding_text = "Pneumonia Detected" if is_pneumonia else "Normal / No Pneumonia"
        
        st.markdown(f"""
        <div class="result-card" style="background-color: {banner_bg}; border-color: {banner_border};">
            <div class="result-status" style="color: {text_color};">{finding_text}</div>
            <div class="result-confidence">AI Confidence: {confidence_percent:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Side-by-Side Images
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(image, caption="Original X-Ray", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(superimposed_img, caption="Explainability Map (Grad-CAM)", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # 3. Model Architecture info (Collapsible)
        with st.expander("Technical Model Details"):
            col_spec1, col_spec2 = st.columns(2)
            with col_spec1:
                st.write("**Model Architecture**: Fine-tuned VGG-16 backbone")
                st.write("**Input Size**: 224 x 224 pixels (3 channels)")
            with col_spec2:
                st.write("**Output Activation**: Sigmoid (Binary probability)")
                st.write("**Classification Threshold**: " + str(threshold))
                
        # Clinical disclaimer
        st.caption(
            "⚠️ Disclaimer: This is an educational demonstration project. It is not intended for diagnostic use. "
            "All findings should be verified by clinical professionals."
        )
    else:
        st.info("Upload a frontal chest X-ray image (PNG, JPG, or JPEG) to start analysis.")
