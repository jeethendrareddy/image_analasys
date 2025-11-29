import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# 📂 Paths
DATA_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Data"
FEATURES_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Features"

os.makedirs(FEATURES_DIR, exist_ok=True)

# 🔹 Load DenseNet121 (without top classifier)
base_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from a single image
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to DenseNet input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        return None

# 🔹 Loop over dataset
all_features = []
all_labels = []

for class_folder in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue

    # Extract numeric label (e.g., "10_Percent" -> 10)
    label = float(class_folder.replace("_Percent", ""))

    print(f"\nExtracting features for class {class_folder} (numeric {label})...\n")

    for img_file in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_file)
        feats = extract_features(img_path)
        if feats is not None:
            all_features.append(feats)
            all_labels.append(label)

# 🔹 Convert to arrays
all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(f"✅ Features shape: {all_features.shape}")
print(f"✅ Labels shape: {all_labels.shape}")

# Save features and labels
np.save(os.path.join(FEATURES_DIR, "DenseNet_features_10.npy"), all_features)
np.save(os.path.join(FEATURES_DIR, "DenseNet_labels_10.npy"), all_labels)

# Also save CSV for quick inspection
# df = pd.DataFrame(all_features)
# df["label"] = all_labels
# df.to_csv(os.path.join(FEATURES_DIR, "DenseNet_features_8.csv"), index=False)

print(f"\n🎉 DenseNet features saved in {FEATURES_DIR}")
