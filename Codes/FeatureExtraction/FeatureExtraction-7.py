import re
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
DATASET_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Data"  # folder containing your class folders (0%,1%,2%...100%)
FEATURES_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Features"  # folder to save extracted features
os.makedirs(FEATURES_DIR, exist_ok=True)

# ----------------------------
# Load Pretrained Model (ResNet50 without top layer)
# ----------------------------
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)


# ----------------------------
# Function to extract features from one image
# ----------------------------
def extract_features(img_path_of_function):
    img = image.load_img(img_path_of_function, target_size=(224, 224))  # resize
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # preprocess for ResNet
    features = model.predict(x, verbose=0)
    return features.flatten()  # 1D feature vector


# ----------------------------
# Loop over dataset
# ----------------------------
all_features = []
all_labels = []

for label in os.listdir(DATASET_DIR):  # each folder = percentage label
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    # Extract numeric percentage (e.g., "3_Percent" -> 3.0)
    match = re.search(r"(\d+)", label)
    if match:
        numeric_label = float(match.group(1))
    else:
        print(f"⚠️ Skipping folder {label}, no number found")
        continue

    print(f"Extracting features for class {label} (numeric {numeric_label})... \n")
    for img_file in tqdm(os.listdir(class_dir)):
        img_path = os.path.join(class_dir, img_file)
        try:
            feat = extract_features(img_path)
            all_features.append(feat)
            all_labels.append(numeric_label)  # use numeric value
        except Exception as e:
            print(f"Error with {img_path}: {e}")

# for label in os.listdir(DATASET_DIR):  # each folder = percentage label
#     class_dir = os.path.join(DATASET_DIR, label)
#     if not os.path.isdir(class_dir):
#         continue
#
#     print(f"Extracting features for class {label}...")
#     for img_file in tqdm(os.listdir(class_dir)):
#         img_path = os.path.join(class_dir, img_file)
#         try:
#             feat = extract_features(img_path)
#             all_features.append(feat)
#             all_labels.append(float(label))  # label as percentage
#         except Exception as e:
#             print(f"Error with {img_path}: {e}")

# ----------------------------
# Save features and labels
# ----------------------------
all_features = np.array(all_features)
all_labels = np.array(all_labels)

np.save(os.path.join(FEATURES_DIR, "dal_features_7.npy"), all_features)
np.save(os.path.join(FEATURES_DIR, "dal_labels_7.npy"), all_labels)

print("✅ Features saved in", FEATURES_DIR)
print("Shape of features:", all_features.shape)
print("Shape of labels:", all_labels.shape)
