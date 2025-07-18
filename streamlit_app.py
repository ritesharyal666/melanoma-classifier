import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
import io
import base64

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("app/models/final.keras")

model = load_model()

# Set up styling
st.set_page_config(page_title="Melanoma Classifier", layout="wide")

# Custom CSS for dark theme and styles
with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.markdown(
    "<h1 class='title'>Melanoma Skin Lesion Classifier</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='subtitle'>Upload an image of a skin lesion to classify it as benign or malignant with our AI-powered diagnostic tool.</p>",
    unsafe_allow_html=True,
)

# Upload and display section
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("## 📌 Image Upload Guidelines")
    st.markdown(
        """
        **For best results, use images that:**
        - Are clear, focused close-ups of the skin lesion
        - Have good lighting without heavy shadows or reflections
        - Show the lesion centered, filling most of the frame
        - Are taken with a neutral background (skin only)

        **Avoid images that:**
        - Are blurry, low-res, or contain multiple lesions
        - Include unrelated objects or filters
        - Have shadows or poor lighting
        """
    )

    st.markdown("## 🖼️ Example Images")
    st.image("app/static/images/goodexample.jpeg", caption="✅ Good example: Clear close-up with proper lighting")
    st.image("app/static/images/badexample.jpg", caption="❌ Bad example: Zoomed out and unfocused")
    st.image("app/static/images/benign.jpg", caption="🟢 Benign sample")
    st.image("app/static/images/melanoma.jpg", caption="🔴 Malignant sample")


with col2:
    st.markdown("## 📤 Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    image=None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview", use_column_width=True)

    def preprocess(img: Image.Image, target_size=(224, 224)):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    if image:
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    input_tensor = preprocess(image)
                    pred = model.predict(input_tensor)[0][0]
                    is_malignant = pred > 0.5
                    confidence = pred if is_malignant else 1 - pred

                    if is_malignant:
                        st.error(
                            f"🔴 **Result: Malignant (Possible Melanoma)**\n\n"
                            f"**Confidence:** {confidence*100:.2f}%\n\n"
                            f"**Recommendation:** Please consult a dermatologist immediately. This result suggests signs of melanoma."
                        )
                    else:
                        st.success(
                            f"🟢 **Result: Benign**\n\n"
                            f"**Confidence:** {confidence*100:.2f}%\n\n"
                            f"**Recommendation:** No signs of malignancy detected. Routine skin checks still advised."
                        )

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
