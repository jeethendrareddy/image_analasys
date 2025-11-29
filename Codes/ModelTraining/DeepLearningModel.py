import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DATA_DIR, DL_MODELS_DIR, RANDOM_STATE

# ==============================
# CONFIG
# ==============================
DATASET_DIR = DATA_DIR
MODEL_DIR = DL_MODELS_DIR
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# Dataset Class
# ==============================
class DalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

# ==============================
# Data Preparation
# ==============================
# Collect paths & labels
classes = os.listdir(DATASET_DIR)  # each folder = class
all_paths, all_labels = [], []

for c in classes:
    folder = os.path.join(DATASET_DIR, c)
    for f in os.listdir(folder):
        all_paths.append(os.path.join(folder, f))
        all_labels.append(c)

# Encode labels (Pure Arhar -> 0, Pure Khesari -> 1, etc.)
le = LabelEncoder()
labels_encoded = le.fit_transform(all_labels)

# Train-val split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, labels_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_encoded
)

# Transforms
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_dataset = DalDataset(train_paths, train_labels, transform=train_tf)
val_dataset = DalDataset(val_paths, val_labels, transform=val_tf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# Model (Transfer Learning: ResNet18)
# ==============================
num_classes = len(le.classes_)
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# Training Loop
# ==============================
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_dataset)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} Train Acc: {acc:.3f} Val Acc: {val_acc:.3f}")

# ==============================
# Save Model
# ==============================
torch.save({
    "model_state": model.state_dict(),
    "classes": le.classes_
}, os.path.join(MODEL_DIR, "dal_cnn.pth"))

print("✅ Model saved to dal_cnn.pth")
