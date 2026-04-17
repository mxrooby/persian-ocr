import gradio as gr
import easyocr
import numpy as np
import cv2
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from database import init_db, save_recognition, get_history, clear_history
from PIL import Image

# ── Initialize EasyOCR (fallback only) ──────────────────
reader = easyocr.Reader(['fa'])
init_db()

# ── Persian character map ────────────────────────────────
# Maps Persian character → (Latin equivalent, character name)
PERSIAN_CHARS = {
    'ا': ('alef',   '-'),
    'ب': ('be',     'b'),
    'پ': ('pe',     'p'),
    'ت': ('te',     't'),
    'ث': ('se',     's'),
    'ج': ('jim',    'j'),
    'چ': ('che',    'ch'),
    'ح': ('he',     'h'),
    'خ': ('khe',    'kh'),
    'د': ('dal',    'd'),
    'ذ': ('zal',    'z'),
    'ر': ('re',     'r'),
    'ز': ('ze',     'z'),
    'ژ': ('zhe',    'zh'),
    'س': ('sin',    's'),
    'ش': ('shin',   'sh'),
    'ص': ('sad',    's'),
    'ض': ('zad',    'z'),
    'ط': ('ta',     't'),
    'ظ': ('za',     'z'),
    'ع': ('ain',    "'"),
    'غ': ('ghain',  'gh'),
    'ف': ('fe',     'f'),
    'ق': ('ghaf',   'q'),
    'ک': ('kaf',    'k'),
    'گ': ('gaf',    'g'),
    'ل': ('lam',    'l'),
    'م': ('mim',    'm'),
    'ن': ('nun',    'n'),
    'و': ('vav',    'w'),
    'ه': ('he',     'h'),
    'ی': ('ye',     'y'),
}

PERSIAN_TO_LATIN = {k: v[1] for k, v in PERSIAN_CHARS.items()}
PERSIAN_TO_NAME  = {k: v[0] for k, v in PERSIAN_CHARS.items()}

# ── CNN Model Architecture ───────────────────────────────
# Custom CNN as specified in the project document:
# Conv layers → Pooling → Fully Connected → Classification
class PersianCNN(nn.Module):
    def __init__(self, num_classes):
        super(PersianCNN, self).__init__()

        # Feature extraction: convolutional + pooling layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # grayscale input (1 channel)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 64x64 → 32x32
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 32x32 → 16x16
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 16x16 → 8x8
            nn.Dropout2d(0.25),
        )

        # Classification: fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── Load CNN Model ────────────────────────────────────────
CUSTOM_MODEL_PATH = "model/persian_cnn.pth"
LABEL_MAP_PATH    = "model/label_map.json"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

cnn_model   = None
idx_to_char = None

def load_cnn_model():
    global cnn_model, idx_to_char
    if not os.path.exists(CUSTOM_MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("CNN model not found. Using EasyOCR fallback.")
        return False
    try:
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        idx_to_char = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)
        cnn_model = PersianCNN(num_classes=num_classes)
        cnn_model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE))
        cnn_model.to(DEVICE)
        cnn_model.eval()
        print(f"CNN model loaded! ({num_classes} classes)")
        return True
    except Exception as e:
        print(f"Could not load CNN model: {e}")
        return False

load_cnn_model()

# ── OpenCV Preprocessing ──────────────────────────────────
# As specified in the document:
# Step 1: Convert to grayscale
# Step 2: Resize to 64x64
# Step 3: Normalize
def preprocess_for_cnn(image_np):
    """
    OpenCV-based preprocessing pipeline per project specification:
    grayscale → resize → normalize → tensor
    """
    # Step 1: Grayscale (OpenCV)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Step 2: Resize to 64x64 (OpenCV)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    # Step 3: Normalize and convert to tensor
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])(Image.fromarray(resized))

    return tensor

def classify_with_cnn(image_np):
    """Run image through the CNN model and return predicted character."""
    if cnn_model is None:
        return None, 0.0
    try:
        tensor = preprocess_for_cnn(image_np)
        tensor = tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output     = cnn_model(tensor)
            probs      = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)
            char = idx_to_char.get(pred_idx.item(), None)
            return char, confidence.item()
    except Exception as e:
        print(f"CNN error: {e}")
        return None, 0.0

# ── EasyOCR Fallback ──────────────────────────────────────
def classify_with_easyocr(image_np):
    """Fallback to EasyOCR if CNN model is not available."""
    try:
        # Preprocess with OpenCV first (grayscale + resize)
        gray    = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        results = reader.readtext(
            resized, detail=0, paragraph=True,
            contrast_ths=0.1, adjust_contrast=0.5,
            text_threshold=0.5, low_text=0.3
        )
        if not results:
            results = reader.readtext(
                image_np, detail=0, paragraph=True,
                contrast_ths=0.1, adjust_contrast=0.5,
                text_threshold=0.5, low_text=0.3
            )
        if results:
            return ' '.join(results)
    except Exception as e:
        print(f"EasyOCR error: {e}")
    return None

# ── Format History ────────────────────────────────────────
def format_history():
    rows = get_history()
    if not rows:
        return "<div style='text-align:center;padding:28px 0;color:rgba(165,130,60,0.5);font-style:italic;font-size:0.86em;font-family:Inter,Segoe UI,sans-serif;'>No recognition history yet.</div>"
    table = "<div style='width:100%;overflow-x:hidden;'><table style='width:100%;border-collapse:collapse;font-family:Inter,Segoe UI,sans-serif;font-size:0.9em;table-layout:fixed;'><colgroup><col style='width:18%;'><col style='width:24%;'><col style='width:18%;'><col style='width:40%;'></colgroup><thead><tr style='background:#1A1206;'><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Letter</th><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Name</th><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Latin</th><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Timestamp</th></tr></thead><tbody>"
    for i, (persian, latin, timestamp) in enumerate(rows):
        row_bg   = "rgba(192,139,31,0.04)" if i % 2 == 0 else "transparent"
        char_name = PERSIAN_TO_NAME.get(persian[0], '-') if persian else '-'
        table += f"<tr style='background:{row_bg};'><td style='padding:10px 12px;color:#F0D898;border-bottom:1px solid rgba(192,139,31,0.07);direction:rtl;font-size:1.3em;font-weight:600;vertical-align:middle;'>{persian}</td><td style='padding:10px 12px;color:#F0D898;border-bottom:1px solid rgba(192,139,31,0.07);font-weight:600;vertical-align:middle;'>{char_name}</td><td style='padding:10px 12px;color:#D4A843;border-bottom:1px solid rgba(192,139,31,0.07);font-weight:500;vertical-align:middle;'>{latin}</td><td style='padding:10px 12px;color:#8A6A30;border-bottom:1px solid rgba(192,139,31,0.07);font-size:0.85em;vertical-align:middle;'>{timestamp}</td></tr>"
    table += "</tbody></table></div>"
    return table

# ── Main Recognition Function ─────────────────────────────
# Pipeline as per document:
# Upload → OpenCV grayscale & resize → CNN → Output Latin equivalent
def recognize_persian(image_data):
    if image_data is None:
        return (
            "No image uploaded.",
            "—",
            "—",
            format_history()
        )

    # Accept numpy array from Gradio
    if isinstance(image_data, np.ndarray):
        image = image_data
    else:
        return "Could not read image.", "—", "—", format_history()

    # Strip alpha channel if present
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    persian_char = None
    confidence   = 0.0

    # ── Step 1: Try CNN model (primary) ──
    if cnn_model is not None:
        persian_char, confidence = classify_with_cnn(image)

    # ── Step 2: Fallback to EasyOCR ──
    if not persian_char:
        print("CNN not available or low confidence — falling back to EasyOCR.")
        ocr_text = classify_with_easyocr(image)
        if ocr_text:
            persian_char = ocr_text[0]  # take first character

    # ── Step 3: Build output ──
    if persian_char and persian_char in PERSIAN_CHARS:
        char_name  = PERSIAN_TO_NAME[persian_char]
        latin_eq   = PERSIAN_TO_LATIN[persian_char]
        conf_str   = f"{confidence*100:.1f}%" if confidence > 0 else "—"
        save_recognition(persian_char, latin_eq)
        return persian_char, char_name, latin_eq, format_history()

    return (
        "No character detected.",
        "—",
        "—",
        format_history()
    )

def clear_all():
    return None, "—", "—", "—", format_history()

def clear_db():
    clear_history()
    return format_history()

# ── CSS ───────────────────────────────────────────────────
CSS = """
footer, .footer, div[class*="footer"],
a[href*="gradio.app"], p:has(a[href*="gradio.app"]) { display: none !important; }
*, *::before, *::after { box-sizing: border-box; }
html, body {
    background: #1A1206 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #E6D8A7 !important;
    margin: 0 !important; padding: 0 !important;
}
.gradio-container {
    background: #1A1206 !important;
    max-width: 1020px !important;
    margin: 0 auto !important;
    padding: 40px 24px 60px !important;
}
.gr-group, [data-testid="group"], .gradio-group {
    border: none !important; background: transparent !important;
    box-shadow: none !important; padding: 0 !important; border-radius: 0 !important;
}
.page-title { text-align: center; margin-bottom: 36px; padding-top: 8px; }
.page-title h1 { color: #F0D898; font-size: 2.2em; font-weight: 900; margin-bottom: 10px; line-height: 1.2; }
.page-title p  { color: #7A5E28; font-size: 1.02em; line-height: 1.6; max-width: 560px; margin: 0 auto; }
.pipeline-note {
    text-align: center;
    color: rgba(192,139,31,0.55);
    font-size: 0.78em;
    margin-bottom: 28px;
    font-family: Inter, sans-serif;
    letter-spacing: 0.3px;
}
.pipeline-note span { color: #C08B1F; font-weight: 600; }
label > span:first-child, .label-wrap > span {
    color: #8A6A30 !important; font-size: 0.76em !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important;
}
textarea {
    background: #1A1206 !important; border: none !important; border-radius: 10px !important;
    color: #F0D898 !important; font-size: 1.08em !important; padding: 14px 16px !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important; resize: none !important;
}
textarea::placeholder { color: rgba(122,94,40,0.5) !important; font-style: italic !important; }
button.primary, button[variant="primary"] {
    background: #C08B1F !important; border: none !important; border-radius: 10px !important;
    color: #fff !important; font-weight: 700 !important; font-size: 1.04em !important;
    padding: 12px 22px !important; min-width: 140px !important; cursor: pointer !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important; min-height: 48px !important;
    transition: background 0.18s !important;
}
button.primary:hover, button[variant="primary"]:hover { background: #A87A1A !important; }
button.secondary, button[variant="secondary"] {
    background: transparent !important; border: 1.5px solid rgba(192,139,31,0.3) !important;
    border-radius: 10px !important; color: #8A6A30 !important; font-weight: 600 !important;
    font-size: 1.04em !important; padding: 12px 22px !important; min-width: 140px !important;
    cursor: pointer !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; min-height: 48px !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    background: rgba(192,139,31,0.07) !important; border-color: #C08B1F !important; color: #E6D8A7 !important;
}
.upload-card {
    background: #221809 !important; border: 2px dashed rgba(192,139,31,0.35) !important;
    border-radius: 16px !important; padding: 10px !important; margin-bottom: 14px !important;
}
.results-card, .history-card {
    background: #221809 !important; border: 1.5px solid rgba(192,139,31,0.2) !important;
    border-radius: 16px !important; padding: 20px 22px !important;
}
.history-card { margin-top: 24px !important; }
.card-title {
    color: #F0D898; font-size: 0.9em; font-weight: 800; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: 18px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(192,139,31,0.12); font-family: 'Inter', 'Segoe UI', sans-serif;
}
/* Big Persian character output */
.persian-char-box textarea {
    font-size: 3em !important;
    text-align: center !important;
    line-height: 1.4 !important;
    min-height: 80px !important;
    color: #F0D898 !important;
    direction: rtl !important;
}
.result-name-box textarea {
    font-size: 1.4em !important;
    font-weight: 700 !important;
    color: #D4A843 !important;
    text-align: center !important;
}
.result-latin-box textarea {
    font-size: 1.4em !important;
    color: #C08B1F !important;
    text-align: center !important;
    font-weight: 700 !important;
}
@media (max-width: 640px) {
    .gradio-container { padding: 20px 14px 48px !important; }
    .page-title h1 { font-size: 1.6em; }
}
"""

# ── UI ────────────────────────────────────────────────────
with gr.Blocks(
    css=CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:

    gr.HTML("""
    <div class='page-title'>
        <h1>Persian Alphabet Character Recognition</h1>
        <p>Upload an image of a Persian character — the system will identify it and display its name and Latin equivalent.</p>
    </div>
    <div class='pipeline-note'>
        <span>Image Upload</span> → OpenCV Grayscale &amp; Resize →
        <span>CNN Classification</span> → <span>Output</span>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: Upload ──────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["upload-card"]):
                image_input = gr.Image(
                    sources=["upload"],
                    type="numpy",
                    label="Upload Persian Character Image",
                    height=280,
                    image_mode="RGB",
                )

            with gr.Row():
                clear_btn  = gr.Button("Clear",     variant="secondary")
                submit_btn = gr.Button("Recognize", variant="primary")

        # ── RIGHT: Results ────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["results-card"]):
                gr.HTML("<div class='card-title'>Recognition Results</div>")

                detected_char = gr.Textbox(
                    label="Detected Persian Character",
                    placeholder="—",
                    lines=2,
                    interactive=False,
                    rtl=True,
                    elem_classes=["persian-char-box"]
                )
                char_name = gr.Textbox(
                    label="Character Name",
                    placeholder="—",
                    lines=1,
                    interactive=False,
                    elem_classes=["result-name-box"]
                )
                latin_equiv = gr.Textbox(
                    label="Latin Equivalent",
                    placeholder="—",
                    lines=1,
                    interactive=False,
                    elem_classes=["result-latin-box"]
                )

    # ── History ───────────────────────────────────────────
    with gr.Group(elem_classes=["history-card"]):
        gr.HTML("<div class='card-title'>Recognition History</div>")
        history_output = gr.HTML(value=format_history())

    gr.HTML("<div style='height:10px'></div>")
    clear_history_btn = gr.Button("Clear History", variant="secondary", size="sm")

    # ── Events ───────────────────────────────────────────
    submit_btn.click(
        fn=recognize_persian,
        inputs=image_input,
        outputs=[detected_char, char_name, latin_equiv, history_output]
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, detected_char, char_name, latin_equiv, history_output]
    )
    clear_history_btn.click(fn=clear_db, outputs=history_output)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)