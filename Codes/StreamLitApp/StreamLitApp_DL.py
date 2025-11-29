import os
import sys
import cv2
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DL_MODELS_DIR

# CONFIG
MODEL_PATH = os.path.join(DL_MODELS_DIR, "dal_cnn.pth")
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, checkpoint["classes"]

model, class_names = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def preprocess_image(img_bytes):
    img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor_img = transform(img_rgb).unsqueeze(0).to(DEVICE)
    return img_rgb, tensor_img

# Prediction
def predict_image(img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        return class_names[pred_idx], probs.cpu().numpy()

# Streamlit UI
st.title("Deep Learning: Arhar vs Khesari Dal Detection")
st.write("Upload a dal mixture image and the CNN will classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png","bmp"])

if uploaded_file is not None:
    # preprocess
    img_rgb, img_tensor = preprocess_image(uploaded_file.read())

    # show uploaded image
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # predict
    label, probs = predict_image(img_tensor)

    st.subheader("Prediction")
    st.success(f"Predicted Class: **{label}**")

    # probability table
    prob_dict = {cls: f"{probs[i]*100:.2f}%" for i, cls in enumerate(class_names)}
    st.write("Class probabilities:")
    st.table(prob_dict)
