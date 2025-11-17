import streamlit as st
from transformers import pipeline
from PIL import Image

# Title
st.title("🤖 AI vs Real Image Detector")

# Load Hugging Face model
@st.cache_resource  # ensures model loads only once
def load_model():
    return pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")

classifier = load_model()

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    st.write("🔎 Analyzing...")
    results = classifier(image)

    # Show results
    st.subheader("Prediction Results:")
    for result in results:
        st.write(f"**{result['label']}** → {result['score']:.4f}")
