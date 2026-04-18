"""
test.py
────────
Evaluates the trained CNN on the test set.
Run after training: python3 test.py
"""
import os
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

MODEL_PATH     = "model/persian_cnn.pth"
LABEL_MAP_PATH = "model/label_map.json"
TEST_DIR       = "dataset/test"
IMG_SIZE       = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PERSIAN_NAMES = {
    'ا':'aleph', 'ب':'beh',  'پ':'peh',  'ت':'teh',  'ث':'theh',
    'ج':'jim',   'چ':'che',  'ح':'he jimi','خ':'khe', 'د':'daal',
    'ذ':'zaal',  'ر':'re',   'ز':'ze',   'ژ':'zhe',  'س':'sin',
    'ش':'shin',  'ص':'sad',  'ض':'zad',  'ط':'taa',  'ظ':'zaa',
    'ع':'ayn',   'غ':'ghayn','ف':'feh',  'ق':'qaf',  'ک':'kaf',
    'گ':'gaf',   'ل':'lam',  'م':'mim',  'ن':'nun',  'و':'vav',
    'ه':'he',    'ی':'yaa',
}

PERSIAN_TO_LATIN = {
    'ا':'-', 'ب':'b',  'پ':'p',  'ت':'t',  'ث':'s',
    'ج':'j', 'چ':'ch', 'ح':'h',  'خ':'kh', 'د':'d',
    'ذ':'z', 'ر':'r',  'ز':'z',  'ژ':'zh', 'س':'s',
    'ش':'sh','ص':'s',  'ض':'z',  'ط':'t',  'ظ':'z',
    'ع':"'",'غ':'gh', 'ف':'f',  'ق':'q',  'ک':'k',
    'گ':'g', 'ل':'l',  'م':'m',  'ن':'n',  'و':'w',
    'ه':'h', 'ی':'y',
}

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found. Run python3 train.py first.")
    exit(1)

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
idx_to_char = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# Same CNN as train.py
class PersianCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.MaxPool2d(2,2),nn.Dropout2d(0.25),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),
            nn.MaxPool2d(2,2),nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.MaxPool2d(2,2),nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8,512),nn.ReLU(True),nn.Dropout(0.5),
            nn.Linear(512,256),nn.ReLU(True),nn.Dropout(0.5),
            nn.Linear(256,num_classes),
        )
    def forward(self,x): return self.classifier(self.features(x))

model = PersianCNN(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

char_correct = {}
char_total   = {}

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.lower().endswith('.png'):
        continue

    stem = os.path.splitext(fname)[0]

    # Get true label from .txt
    true_char = None
    txt_path = os.path.join(TEST_DIR, stem + '.txt')
    if os.path.exists(txt_path):
        content = open(txt_path,'r',encoding='utf-8').read().strip()
        if content: true_char = content[0]
    if not true_char:
        parts = stem.split('_')
        if len(parts) >= 2:
            true_char = parts[1]

    if not true_char or true_char not in label_map:
        continue

    img = cv2.imread(os.path.join(TEST_DIR, fname))
    if img is None: continue
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor  = transform(Image.fromarray(resized)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out        = model(tensor)
        _, pred_idx = out.max(1)
        pred_char  = idx_to_char.get(pred_idx.item(), '?')

    char_total[true_char]   = char_total.get(true_char, 0) + 1
    char_correct[true_char] = char_correct.get(true_char, 0) + (1 if pred_char == true_char else 0)

print(f"\n{'─'*62}")
print(f"{'Char':<6} {'Name':<12} {'Latin':<8} {'Correct':>8} {'Total':>7} {'Acc':>8}  Status")
print(f"{'─'*62}")

overall_c = overall_t = 0
for char in sorted(char_total.keys(), key=lambda c: label_map.get(c, 99)):
    c     = char_correct[char]
    t     = char_total[char]
    acc   = 100. * c / t
    name  = PERSIAN_NAMES.get(char, '?')
    latin = PERSIAN_TO_LATIN.get(char, '?')
    status = "ACCEPTABLE" if acc >= 70 else "LOW ACCURACY"
    print(f"{char:<6} {name:<12} {latin:<8} {c:>8} {t:>7} {acc:>7.1f}%  {status}")
    overall_c += c
    overall_t += t

overall_acc = 100. * overall_c / overall_t if overall_t else 0
print(f"{'─'*62}")
print(f"{'OVERALL':<6} {'':<12} {'':<8} {overall_c:>8} {overall_t:>7} {overall_acc:>7.2f}%")