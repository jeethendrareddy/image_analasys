"""
Configuration file for Arhar-Khesari Dal Image Analysis Project
Centralized path management for all scripts
"""
import os

# Base directory - root of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "Data")
FEATURES_DIR = os.path.join(BASE_DIR, "Features")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
CODES_DIR = os.path.join(BASE_DIR, "Codes")

# Create directories if they don't exist
for directory in [DATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, CODES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model-specific directories
MODELS_TEST_8_DIR = os.path.join(MODELS_DIR, "Test-8")
MODELS_TEST_9_DIR = os.path.join(MODELS_DIR, "Test-9")
MODELS_TEST_10_DIR = os.path.join(MODELS_DIR, "Test-10")
MODELS_TEST_12_DIR = os.path.join(MODELS_DIR, "Test-12")
DL_MODELS_DIR = os.path.join(BASE_DIR, "DL_Models", "Test-1")

# Create model directories
for model_dir in [MODELS_TEST_8_DIR, MODELS_TEST_9_DIR, MODELS_TEST_10_DIR,
                  MODELS_TEST_12_DIR, DL_MODELS_DIR]:
    os.makedirs(model_dir, exist_ok=True)

# Feature file paths
FEATURES_RESNET_7 = {
    'features': os.path.join(FEATURES_DIR, "dal_features_7.npy"),
    'labels': os.path.join(FEATURES_DIR, "dal_labels_7.npy")
}

FEATURES_DENSENET_8 = {
    'features': os.path.join(FEATURES_DIR, "DenseNet_features_8.npy"),
    'labels': os.path.join(FEATURES_DIR, "DenseNet_labels_8.npy"),
    'csv': os.path.join(FEATURES_DIR, "DenseNet_features_8.csv")
}

FEATURES_DENSENET_9 = {
    'features': os.path.join(FEATURES_DIR, "dal_features_9.npy"),
    'labels': os.path.join(FEATURES_DIR, "dal_labels_9.npy"),
    'csv': os.path.join(FEATURES_DIR, "dal_features_9.csv")
}

FEATURES_DENSENET_10 = {
    'features': os.path.join(FEATURES_DIR, "DenseNet_features_10.npy"),
    'labels': os.path.join(FEATURES_DIR, "DenseNet_labels_10.npy")
}

# Model hyperparameters (can be adjusted)
RANDOM_STATE = 42
IMG_SIZE = (224, 224)
AUG_PER_IMAGE = 5

# Target class folders for augmentation
TARGET_FOLDERS = ["20_Percent", "30_Percent", "40_Percent", "50_Percent",
                  "60_Percent", "70_Percent", "80_Percent", "90_Percent"]
