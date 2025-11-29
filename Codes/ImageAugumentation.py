# augment.py
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

DATA_DIR = "../Data"   # relative path (adjust if needed)
AUG_PER_IMAGE = 5      # how many augmented versions per original image
IMG_SIZE = (224, 224)  # resize all images to same size

# ✅ Image augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

def augment_class_folder(class_path):
    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    for file in tqdm(files, desc=f"Augmenting {os.path.basename(class_path)}"):
        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=0)

        # Generate augmentations
        aug_iter = datagen.flow(img, batch_size=1)
        for i in range(AUG_PER_IMAGE):
            aug_img = next(aug_iter)[0].astype(np.uint8)
            save_name = f"{os.path.splitext(file)[0]}_aug{i+1}.jpg"
            save_path = os.path.join(class_path, save_name)
            cv2.imwrite(save_path, aug_img)

if __name__ == "__main__":
    # ✅ Only these target folders
    target_folders = ["20_Percent", "30_Percent", "40_Percent", "50_Percent",
                      "60_Percent", "70_Percent", "80_Percent", "90_Percent"]

    for cls in target_folders:
        class_path = os.path.join(DATA_DIR, cls)
        if os.path.isdir(class_path):
            augment_class_folder(class_path)
        else:
            print(f"⚠️ Skipping missing folder: {cls}")

    print("✅ Augmentation complete! All augmented images saved in their respective folders.")
