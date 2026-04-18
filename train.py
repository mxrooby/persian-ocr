"""
train.py
────────
Trains the Persian CNN on dataset.
Dataset structure:
    dataset/train/  ← flat folder, files like 08_خ_0106.png + 08_خ_0106.txt
    dataset/test/   ← same flat structure
    dataset/label_map.json  ← Persian char → int index
"""
import os
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ── Config ────────────────────────────────────────────────
TRAIN_DIR       = "dataset/train"
TEST_DIR        = "dataset/test"
LABEL_MAP_PATH  = "dataset/label_map.json"
MODEL_SAVE_PATH = "model/persian_cnn.pth"
OUT_LABEL_MAP   = "model/label_map.json"
IMG_SIZE        = 64
BATCH_SIZE      = 32
EPOCHS          = 25
LEARNING_RATE   = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load label map ────────────────────────────────────────
# The dataset ships with a label_map.json — use it directly
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    print(f"Loaded label map: {len(label_map)} classes")
else:
    # Build from txt files if label_map.json not found
    label_map = {}
    print("label_map.json not found — will build from .txt files")

# ── CNN Architecture ──────────────────────────────────────
class PersianCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64→32
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2: 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3: 16→8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── Dataset ───────────────────────────────────────────────
class PersianDataset(Dataset):

    def __init__(self, data_dir, label_map, augment=False):
        self.samples   = []
        self.label_map = label_map
        self.augment   = augment
        skipped = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith('.png'):
                continue

            img_path = os.path.join(data_dir, fname)
            stem     = os.path.splitext(fname)[0]

            # Get label: try .txt file first, then parse filename
            persian_char = None
            txt_path = os.path.join(data_dir, stem + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    persian_char = content[0]

            if not persian_char:
                # Fallback: filename is like "08_خ_0106.png" → middle part
                parts = stem.split('_')
                if len(parts) >= 2:
                    persian_char = parts[1]

            if not persian_char or persian_char not in self.label_map:
                skipped += 1
                continue

            self.samples.append((img_path, self.label_map[persian_char]))

        if skipped > 0:
            print(f"  Skipped {skipped} files (no valid label)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # OpenCV preprocessing: grayscale → resize (per project specification)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        pil = Image.fromarray(img)

        if self.augment:
            t = transforms.Compose([
                transforms.RandomRotation(12),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        return t(pil), label

# ── Load data ─────────────────────────────────────────────
print("\nLoading datasets...")
train_ds = PersianDataset(TRAIN_DIR, label_map, augment=True)
test_ds  = PersianDataset(TEST_DIR,  label_map, augment=False)
num_classes = len(label_map)

print(f"  Train : {len(train_ds)} images")
print(f"  Test  : {len(test_ds)} images")
print(f"  Classes: {num_classes}")

if len(train_ds) == 0:
    print("\nERROR: No training images loaded.")
    print(f"Make sure {TRAIN_DIR} contains .png files with matching .txt label files.")
    exit(1)

if len(test_ds) == 0:
    print("\nERROR: No test images loaded.")
    print(f"Make sure {TEST_DIR} exists and has images.")
    exit(1)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model, optimizer, scheduler ──────────────────────────
model     = PersianCNN(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

# ── Training ──────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
os.makedirs("model", exist_ok=True)
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = out.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = 100. * correct / total

    model.eval()
    val_c = val_t = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, pred = model(images).max(1)
            val_t  += labels.size(0)
            val_c  += pred.eq(labels).sum().item()

    val_acc = 100. * val_c / val_t
    scheduler.step()

    tag = " ★ saved" if val_acc > best_acc else ""
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Loss: {running_loss/len(train_loader):.4f}  "
          f"Train: {train_acc:.1f}%  "
          f"Val: {val_acc:.1f}%{tag}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# Save label map to model folder for app.py
with open(OUT_LABEL_MAP, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f"\nDone! Best validation accuracy: {best_acc:.2f}%")
print(f"Model:     {MODEL_SAVE_PATH}")
print(f"Label map: {OUT_LABEL_MAP}")