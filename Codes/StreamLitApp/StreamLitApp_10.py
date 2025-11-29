import streamlit as st
import sys
import os
import numpy as np
import joblib
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import MODELS_TEST_10_DIR

# Load DenseNet121 feature extractor
base_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten().reshape(1, -1)

# Load trained models
svm_model = joblib.load(os.path.join(MODELS_TEST_10_DIR, "SVM_DenseNet.pkl"))
rf_model = joblib.load(os.path.join(MODELS_TEST_10_DIR, "RandomForest_DenseNet.pkl"))
dt_model = joblib.load(os.path.join(MODELS_TEST_10_DIR, "DecisionTree_DenseNet.pkl"))

# Streamlit UI
st.title("Khesari % Detection in Arhar Dal (DenseNet)")

uploaded_file = st.file_uploader("Upload an image of dal", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Extract features
    st.write("Extracting features using DenseNet...")
    feats = extract_features(img)

    # Predictions
    svm_pred = svm_model.predict(feats)[0]
    rf_pred = rf_model.predict(feats)[0]
    dt_pred = dt_model.predict(feats)[0]

    # Show results
    st.subheader("Predicted Khesari %")
    st.write(f"**SVM:** {svm_pred:.2f} %")
    st.write(f"**Random Forest:** {rf_pred:.2f} %")
    st.write(f"**Decision Tree:** {dt_pred:.2f} %")

    # Compare in table
    st.table({
        "Model": ["SVM", "Random Forest", "Decision Tree"],
        "Predicted %": [round(svm_pred, 2), round(rf_pred, 2), round(dt_pred, 2)]
    })
