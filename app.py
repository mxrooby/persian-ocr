import gradio as gr
import numpy as np
import cv2
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from preprocessing import preprocess_image
from database import init_db, save_recognition, get_history, clear_history
from PIL import Image

init_db()

PERSIAN_CHARS = {
    'ا': ('aleph',   '-'),  'ب': ('beh',     'b'),
    'پ': ('peh',     'p'),  'ت': ('teh',     't'),
    'ث': ('theh',    's'),  'ج': ('jim',     'j'),
    'چ': ('che',     'ch'), 'ح': ('he jimi', 'h'),
    'خ': ('khe',     'kh'), 'د': ('daal',    'd'),
    'ذ': ('zaal',    'z'),  'ر': ('re',      'r'),
    'ز': ('ze',      'z'),  'ژ': ('zhe',     'zh'),
    'س': ('sin',     's'),  'ش': ('shin',    'sh'),
    'ص': ('sad',     's'),  'ض': ('zad',     'z'),
    'ط': ('taa',     't'),  'ظ': ('zaa',     'z'),
    'ع': ('ayn',     "'"),  'غ': ('ghayn',   'gh'),
    'ف': ('feh',     'f'),  'ق': ('qaf',     'q'),
    'ک': ('kaf',     'k'),  'گ': ('gaf',     'g'),
    'ل': ('lam',     'l'),  'م': ('mim',     'm'),
    'ن': ('nun',     'n'),  'و': ('vav',     'w'),
    'ه': ('he',      'h'),  'ی': ('yaa',     'y'),
}
PERSIAN_TO_NAME  = {k: v[0] for k, v in PERSIAN_CHARS.items()}
PERSIAN_TO_LATIN = {k: v[1] for k, v in PERSIAN_CHARS.items()}

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
    def forward(self, x): return self.classifier(self.features(x))

DEVICE         = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH     = "model/persian_cnn.pth"
LABEL_MAP_PATH = "model/label_map.json"
cnn_model      = None
idx_to_char    = None

def load_model():
    global cnn_model, idx_to_char
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("Model not found — run python3 train.py first.")
        return False
    try:
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        idx_to_char = {v: k for k, v in label_map.items()}
        cnn_model   = PersianCNN(num_classes=len(label_map))
        cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        cnn_model.to(DEVICE)
        cnn_model.eval()
        print(f"Model loaded — {len(label_map)} classes")
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False

load_model()

# ── Global retry state ────────────────────────────────────
# Stores the ranked candidate list from the last prediction.
# On retry, we return the next candidate instead of re-running the model.
_last_candidates = []   # list of (char, name, latin, confidence_str)
_retry_index     = 0    # which candidate to show next

def get_all_candidates(image_np):
    """
    Run 3 preprocessing strategies, average probabilities,
    return ALL 32 characters ranked by confidence.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def run(img_gray):
        t = transform(Image.fromarray(img_gray)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(cnn_model(t), dim=1)[0]

    probs_list = []

    # Strategy 1: full adaptive preprocessing
    try:
        p1 = preprocess_image(image_np, target_size=64)
        probs_list.append(run(p1))
    except Exception as e:
        print(f"S1 error: {e}")

    # Strategy 2: simple grayscale + resize (matches training data)
    try:
        g2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        r2 = cv2.resize(g2, (64, 64), interpolation=cv2.INTER_AREA)
        probs_list.append(run(r2))
    except Exception as e:
        print(f"S2 error: {e}")

    # Strategy 3: OTSU threshold
    try:
        g3 = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, b3 = cv2.threshold(g3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(b3) < 128:
            b3 = cv2.bitwise_not(b3)
        r3 = cv2.resize(b3, (64, 64), interpolation=cv2.INTER_AREA)
        probs_list.append(run(r3))
    except Exception as e:
        print(f"S3 error: {e}")

    if not probs_list:
        return []

    avg = torch.stack(probs_list).mean(dim=0)
    top_vals, top_idx = torch.sort(avg, descending=True)

    candidates = []
    for val, idx in zip(top_vals, top_idx):
        char = idx_to_char.get(idx.item())
        if char and char in PERSIAN_CHARS:
            name  = PERSIAN_TO_NAME[char]
            latin = PERSIAN_TO_LATIN[char]
            conf  = f"{val.item()*100:.1f}%"
            candidates.append((char, name, latin, conf))

    return candidates

def recognize_persian(image_data, retry_state):
    """
    retry_state: int — how many times Recognize has been clicked for this image.
    On first click (0): run model, show top-1.
    On second click (1): show top-2 candidate.
    On third click (2): show top-3 candidate.
    On Clear: reset to 0.
    """
    global _last_candidates, _retry_index

    if image_data is None:
        return "—", "—", "—", "Upload an image first.", format_history(), 0

    if not isinstance(image_data, np.ndarray):
        return "—", "—", "—", "Could not read image.", format_history(), 0

    if cnn_model is None:
        return "—", "—", "—", "Model not loaded. Run python3 train.py first.", format_history(), 0

    image = image_data
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    current_retry = retry_state if retry_state else 0

    # First attempt — run model and cache all candidates
    if current_retry == 0:
        _last_candidates = get_all_candidates(image)
        _retry_index = 0

    if not _last_candidates:
        return "—", "—", "—", "Recognition failed. Try a clearer image.", format_history(), 0

    # Pick which candidate to show
    idx_to_show = min(current_retry, len(_last_candidates) - 1)
    char, name, latin, conf = _last_candidates[idx_to_show]

    # Build top-3 hint string
    top3_parts = []
    for i, (c, n, l, cv_) in enumerate(_last_candidates[:3]):
        marker = " ◀ current" if i == idx_to_show else ""
        top3_parts.append(f"#{i+1} {c} ({n}) {cv_}{marker}")
    top3_str = "\n".join(top3_parts)

    attempt_label = ["1st", "2nd", "3rd"][min(current_retry, 2)]
    detail = f"Attempt {current_retry+1} — showing {attempt_label} best match\n{top3_str}"

    # Save to history
    save_recognition(char, latin)

    next_retry = current_retry + 1

    return char, name, latin, detail, format_history(), next_retry

def clear_all():
    global _last_candidates, _retry_index
    _last_candidates = []
    _retry_index = 0
    return None, "—", "—", "—", "", format_history(), 0

def clear_db():
    clear_history()
    return format_history()

def format_history():
    rows = get_history()
    if not rows:
        return "<div style='text-align:center;padding:28px 0;color:rgba(165,130,60,0.5);font-style:italic;font-size:0.86em;font-family:Inter,Segoe UI,sans-serif;'>No recognition history yet.</div>"
    table = "<div style='width:100%;overflow-x:hidden;'><table style='width:100%;border-collapse:collapse;font-family:Inter,Segoe UI,sans-serif;font-size:0.9em;table-layout:fixed;'><colgroup><col style='width:15%;'><col style='width:22%;'><col style='width:15%;'><col style='width:48%;'></colgroup><thead><tr style='background:#1A1206;'><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Letter</th><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Name</th><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Latin</th><th style='color:#A07830;padding:10px 12px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Timestamp</th></tr></thead><tbody>"
    for i, (persian, latin, timestamp) in enumerate(rows):
        row_bg    = "rgba(192,139,31,0.04)" if i % 2 == 0 else "transparent"
        char_name = PERSIAN_TO_NAME.get(persian[0], '-') if persian else '-'
        table += f"<tr style='background:{row_bg};'><td style='padding:10px 12px;color:#F0D898;border-bottom:1px solid rgba(192,139,31,0.07);direction:rtl;font-size:1.3em;font-weight:600;vertical-align:middle;'>{persian}</td><td style='padding:10px 12px;color:#F0D898;border-bottom:1px solid rgba(192,139,31,0.07);font-weight:600;vertical-align:middle;'>{char_name}</td><td style='padding:10px 12px;color:#D4A843;border-bottom:1px solid rgba(192,139,31,0.07);font-weight:500;vertical-align:middle;'>{latin}</td><td style='padding:10px 12px;color:#8A6A30;border-bottom:1px solid rgba(192,139,31,0.07);font-size:0.85em;vertical-align:middle;'>{timestamp}</td></tr>"
    table += "</tbody></table></div>"
    return table

CSS = """
footer,.footer,div[class*="footer"],a[href*="gradio.app"],p:has(a[href*="gradio.app"]){display:none!important}
*,*::before,*::after{box-sizing:border-box}
html,body{background:#1A1206!important;font-family:'Inter','Segoe UI',sans-serif!important;color:#E6D8A7!important;margin:0!important;padding:0!important}
.gradio-container{background:#1A1206!important;max-width:1020px!important;margin:0 auto!important;padding:40px 24px 60px!important}
.gr-group,[data-testid="group"],.gradio-group{border:none!important;background:transparent!important;box-shadow:none!important;padding:0!important;border-radius:0!important}
.page-title{text-align:center;margin-bottom:12px;padding-top:8px}
.page-title h1{color:#F0D898;font-size:2.2em;font-weight:900;margin-bottom:10px;line-height:1.2}
.page-title p{color:#7A5E28;font-size:1.02em;line-height:1.6;max-width:560px;margin:0 auto}
.pipeline-note{text-align:center;color:rgba(192,139,31,0.55);font-size:0.78em;margin-bottom:28px;font-family:Inter,sans-serif;letter-spacing:0.3px}
.pipeline-note span{color:#C08B1F;font-weight:600}
label>span:first-child,.label-wrap>span{color:#8A6A30!important;font-size:0.76em!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:0.8px!important}
textarea{background:#1A1206!important;border:none!important;border-radius:10px!important;color:#F0D898!important;font-size:1.08em!important;padding:14px 16px!important;font-family:'Inter','Segoe UI',sans-serif!important;resize:none!important}
textarea::placeholder{color:rgba(122,94,40,0.5)!important;font-style:italic!important}
button.primary,button[variant="primary"]{background:#C08B1F!important;border:none!important;border-radius:10px!important;color:#fff!important;font-weight:700!important;font-size:1.04em!important;padding:12px 22px!important;min-width:140px!important;cursor:pointer!important;font-family:'Inter','Segoe UI',sans-serif!important;min-height:48px!important;transition:background 0.18s!important}
button.primary:hover,button[variant="primary"]:hover{background:#A87A1A!important}
button.secondary,button[variant="secondary"]{background:transparent!important;border:1.5px solid rgba(192,139,31,0.3)!important;border-radius:10px!important;color:#8A6A30!important;font-weight:600!important;font-size:1.04em!important;padding:12px 22px!important;min-width:140px!important;cursor:pointer!important;font-family:'Inter','Segoe UI',sans-serif!important;min-height:48px!important}
button.secondary:hover,button[variant="secondary"]:hover{background:rgba(192,139,31,0.07)!important;border-color:#C08B1F!important;color:#E6D8A7!important}
.upload-card{background:#221809!important;border:2px dashed rgba(192,139,31,0.35)!important;border-radius:16px!important;padding:10px!important;margin-bottom:14px!important}
.results-card,.history-card{background:#221809!important;border:1.5px solid rgba(192,139,31,0.2)!important;border-radius:16px!important;padding:20px 22px!important}
.history-card{margin-top:24px!important}
.card-title{color:#F0D898;font-size:0.9em;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid rgba(192,139,31,0.12);font-family:'Inter','Segoe UI',sans-serif}
.persian-char-box textarea{font-size:3.5em!important;text-align:center!important;line-height:1.4!important;min-height:90px!important;color:#F0D898!important;direction:rtl!important}
.result-name-box textarea,.result-latin-box textarea{font-size:1.4em!important;font-weight:700!important;text-align:center!important}
.result-name-box textarea{color:#D4A843!important}
.result-latin-box textarea{color:#C08B1F!important}
.confidence-box textarea{font-size:0.82em!important;color:rgba(192,139,31,0.7)!important;line-height:1.7!important}
.retry-hint{text-align:center;color:rgba(192,139,31,0.5);font-size:0.78em;margin-top:6px;font-family:Inter,sans-serif}
@media(max-width:640px){.gradio-container{padding:20px 14px 48px!important}.page-title h1{font-size:1.6em}}
"""

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

    # Hidden state to track retry count per image
    retry_state = gr.State(0)

    with gr.Row(equal_height=False):

        # LEFT — upload
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["upload-card"]):
                image_input = gr.Image(
                    sources=["upload"],
                    type="numpy",
                    label="Upload Persian Character Image",
                    height=280,
                    image_mode="RGB",
                )
            gr.HTML("<p class='retry-hint'>Upload image → click <b>Recognize</b> up to 3 times for different candidates</p>")
            with gr.Row():
                clear_btn  = gr.Button("Clear",     variant="secondary")
                submit_btn = gr.Button("Recognize", variant="primary")

        # RIGHT — results
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["results-card"]):
                gr.HTML("<div class='card-title'>Recognition Results</div>")
                detected_char = gr.Textbox(
                    label="Detected Persian Character",
                    placeholder="—", lines=2,
                    interactive=False, rtl=True,
                    elem_classes=["persian-char-box"]
                )
                with gr.Row():
                    char_name = gr.Textbox(
                        label="Character Name",
                        placeholder="—", lines=1,
                        interactive=False,
                        elem_classes=["result-name-box"]
                    )
                    latin_equiv = gr.Textbox(
                        label="Latin Equivalent",
                        placeholder="—", lines=1,
                        interactive=False,
                        elem_classes=["result-latin-box"]
                    )
                confidence_out = gr.Textbox(
                    label="Top 3 Candidates",
                    placeholder="Details will appear here after recognition...",
                    lines=3, interactive=False,
                    elem_classes=["confidence-box"]
                )

    # History
    with gr.Group(elem_classes=["history-card"]):
        gr.HTML("<div class='card-title'>Recognition History</div>")
        history_output = gr.HTML(value=format_history())

    gr.HTML("<div style='height:10px'></div>")
    clear_history_btn = gr.Button("Clear History", variant="secondary", size="sm")

    # Events
    submit_btn.click(
        fn=recognize_persian,
        inputs=[image_input, retry_state],
        outputs=[detected_char, char_name, latin_equiv, confidence_out, history_output, retry_state]
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, detected_char, char_name, latin_equiv, confidence_out, history_output, retry_state]
    )
    clear_history_btn.click(fn=clear_db, outputs=history_output)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)