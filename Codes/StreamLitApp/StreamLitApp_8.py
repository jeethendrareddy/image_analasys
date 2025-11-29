import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os

# ----------------------------
# Load trained models
# ----------------------------

# Use os.getcwd() for Streamlit compatibility
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "Models", "Test-8")

# Load models
svm_model = joblib.load(os.path.join(MODELS_DIR, "SVM_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "RandomForest_model.pkl"))
dt_model = joblib.load(os.path.join(MODELS_DIR, "DecisionTree_model.pkl"))

# ----------------------------
# Load feature extractor (ResNet50 without top layer)
# ----------------------------
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    """Extract 2048-d feature vector from image"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_model.predict(x, verbose=0)
    return features.flatten().reshape(1, -1)  # shape (1,2048)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌾 Khesari % Detection in Arhar Dal")
st.write("Upload an image of dal mixture to detect % of Khesari dal using 3 models.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Save temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = extract_features(temp_path)

    # Predictions
    svm_pred = svm_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    dt_pred = dt_model.predict(features)[0]

    # Display results
    st.subheader("📊 Predictions")
    st.write(f"*SVM Prediction:* {svm_pred:.2f}% Khesari")
    st.write(f"*Random Forest Prediction:* {rf_pred:.2f}% Khesari")
    st.write(f"*Decision Tree Prediction:* {dt_pred:.2f}% Khesari")