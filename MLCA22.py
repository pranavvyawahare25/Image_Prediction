import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

@st.cache_data
def load_model():
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    return model, feature_extractor

def classify_image(image, model, feature_extractor):
    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Run the image through the model
    outputs = model(**inputs)
    
    # Get the predicted class index
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Get the class label
    return model.config.id2label[predicted_class_idx]

# Load the model and feature extractor
model, feature_extractor = load_model()

# Streamlit interface
st.title("Image Prediction  Using TensorFlow Hub Dataset")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Classify the image
    label = classify_image(image, model, feature_extractor)
    
    # Display the classification result
    st.write(f"Prediction: {label}")
