import gradio as gr
import easyocr
import numpy as np
from preprocessing import preprocess_image
from database import init_db, save_recognition, get_history, clear_history

# Initialize
reader = easyocr.Reader(['fa'])
init_db()

PERSIAN_TO_LATIN = {
    'ا': '-',  'ب': 'b',  'پ': 'p',  'ت': 't',
    'ث': 's',  'ج': 'j',  'چ': 'ch', 'ح': 'h',
    'خ': 'kh', 'د': 'd',  'ذ': 'z',  'ر': 'r',
    'ز': 'z',  'ژ': 'zh', 'س': 's',  'ش': 'sh',
    'ص': 's',  'ض': 'z',  'ط': 't',  'ظ': 'z',
    'ع': "'",  'غ': 'gh', 'ف': 'f',  'ق': 'q',
    'ک': 'k',  'گ': 'g',  'ل': 'l',  'م': 'm',
    'ن': 'n',  'و': 'w',  'ه': 'h',  'ی': 'y',
}

def transliterate(text):
    result = []
    for char in text:
        if char in PERSIAN_TO_LATIN:
            result.append(PERSIAN_TO_LATIN[char])
        elif char.strip():
            result.append(char)
    return ' '.join(result) if result else '-'

def format_history():
    rows = get_history()
    if not rows:
        return """
        <div style='
            text-align:center;
            padding:28px 0;
            color:rgba(165,130,60,0.5);
            font-style:italic;
            font-size:0.86em;
            font-family:Inter,Segoe UI,sans-serif;
        '>No recognition history yet.</div>
        """
    table = """
    <div style='width:100%;overflow-x:hidden;'>
    <table style='
        width:100%;
        border-collapse:collapse;
        font-family:Inter,Segoe UI,sans-serif;
        font-size:0.9em;
        table-layout:fixed;
    '>
        <colgroup>
            <col style='width:25%;'>
            <col style='width:25%;'>
            <col style='width:50%;'>
        </colgroup>
        <thead>
            <tr style='background:#1A1206;'>
                <th style='color:#A07830;padding:10px 14px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Persian</th>
                <th style='color:#A07830;padding:10px 14px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Romanized</th>
                <th style='color:#A07830;padding:10px 14px;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;font-size:0.72em;border-bottom:1px solid rgba(192,139,31,0.2);'>Timestamp</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, (persian, latin, timestamp) in enumerate(rows):
        row_bg = "rgba(192,139,31,0.04)" if i % 2 == 0 else "transparent"
        table += f"""
        <tr style='background:{row_bg};'>
            <td style='padding:11px 14px;color:#F0D898;border-bottom:1px solid rgba(192,139,31,0.07);direction:rtl;font-size:1.15em;font-weight:600;vertical-align:middle;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{persian}</td>
            <td style='padding:11px 14px;color:#D4A843;border-bottom:1px solid rgba(192,139,31,0.07);font-weight:500;vertical-align:middle;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{latin}</td>
            <td style='padding:11px 14px;color:#8A6A30;border-bottom:1px solid rgba(192,139,31,0.07);font-size:0.85em;vertical-align:middle;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{timestamp}</td>
        </tr>
        """
    table += "</tbody></table></div>"
    return table

def recognize_persian(image_data):
    if image_data is None:
        return "Please upload an image first.", "-", format_history()

    image = None
    if isinstance(image_data, dict):
        for key in ["composite", "background", "image"]:
            val = image_data.get(key)
            if val is not None and isinstance(val, np.ndarray) and val.size > 0:
                image = val
                break
    elif isinstance(image_data, np.ndarray):
        image = image_data

    if image is None:
        return "No image detected. Please try again.", "-", format_history()

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    try:
        processed = preprocess_image(image)
    except Exception as e:
        return f"Preprocessing error: {str(e)}", "-", format_history()

    try:
        results = reader.readtext(
            processed, detail=0, paragraph=True,
            contrast_ths=0.1, adjust_contrast=0.5,
            text_threshold=0.5, low_text=0.3
        )
        if not results:
            results = reader.readtext(
                image, detail=0, paragraph=True,
                contrast_ths=0.1, adjust_contrast=0.5,
                text_threshold=0.5, low_text=0.3
            )
    except Exception as e:
        return f"Recognition error: {str(e)}", "-", format_history()

    if results:
        persian_text = ' '.join(results)
        latin_text = transliterate(persian_text)
        save_recognition(persian_text, latin_text)
        return persian_text, latin_text, format_history()

    return "No character detected. Try a clearer image.", "-", format_history()

def clear_all():
    return None, "", "-", format_history()

def clear_db():
    clear_history()
    return format_history()

CSS = """
/* ── Hide Gradio footer ── */
footer,
.footer,
div[class*="footer"],
a[href*="gradio.app"],
p:has(a[href*="gradio.app"]) {
    display: none !important;
}

/* ── Reset ── */
*, *::before, *::after {
    box-sizing: border-box;
}

/* ── Base ── */
html, body {
    background: #1A1206 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #E6D8A7 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ── Container ── */
.gradio-container {
    background: #1A1206 !important;
    max-width: 1020px !important;
    margin: 0 auto !important;
    padding: 40px 24px 60px !important;
}

/* ── NEUTRALIZE gr.Group default styling ── */
/* This is the key fix — removes Gradio's built-in border
   so only our custom class border shows */
.gr-group,
[data-testid="group"],
.gradio-group {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
}

/* ── Page title ── */
.page-title {
    text-align: center;
    margin-bottom: 36px;
    padding-top: 8px;
}

.page-title h1 {
    color: #F0D898;
    font-size: 2.2em;
    font-weight: 900;
    margin-bottom: 10px;
    line-height: 1.2;
}

.page-title p {
    color: #7A5E28;
    font-size: 1.02em;
    line-height: 1.6;
    max-width: 500px;
    margin: 0 auto;
}

/* ── Upload zone ── */
.upload-zone {
    background: #221809 !important;
    border: 2px dashed rgba(192,139,31,0.45) !important;
    border-radius: 16px !important;
    padding: 10px !important;
    margin-bottom: 12px !important;
    transition: border-color 0.2s ease !important;
}

.upload-zone:hover {
    border-color: rgba(192,139,31,0.75) !important;
}

/* ── Format hint ── */
.format-hint {
    text-align: center;
    color: rgba(122,94,40,0.6);
    font-size: 0.76em;
    margin-bottom: 14px;
    letter-spacing: 0.2px;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Labels ── */
label > span:first-child,
.label-wrap > span {
    color: #8A6A30 !important;
    font-size: 0.76em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* ── Textboxes ── */
textarea {
    background: #1A1206 !important;
    border: none !important; /* remove inner border to avoid double-border with card */
    border-radius: 10px !important;
    color: #F0D898 !important;
    font-size: 1.08em !important;
    padding: 14px 16px !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    resize: none !important;
    transition: box-shadow 0.18s !important;
}

textarea:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(192,139,31,0.08) !important;
}

textarea::placeholder {
    color: rgba(122,94,40,0.5) !important;
    font-style: italic !important;
    font-size: 0.95em !important;
}

/* ── Buttons ── */
/* Buttons: make sizes consistent and avoid forced full-width to allow equal sizing in rows */
button.primary, button[variant="primary"] {
    background: #C08B1F !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1.04em !important;
    padding: 12px 22px !important;
    min-width: 140px !important;
    cursor: pointer !important;
    transition: background 0.18s, transform 0.12s !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    min-height: 48px !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: #A87A1A !important;
    transform: translateY(-1px) !important;
}

button.secondary, button[variant="secondary"] {
    background: transparent !important;
    border: 1.5px solid rgba(192,139,31,0.3) !important;
    border-radius: 10px !important;
    color: #8A6A30 !important;
    font-weight: 600 !important;
    font-size: 1.04em !important;
    padding: 12px 22px !important;
    min-width: 140px !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    min-height: 48px !important;
}

button.secondary:hover, button[variant="secondary"]:hover {
    background: rgba(192,139,31,0.07) !important;
    border-color: #C08B1F !important;
    color: #E6D8A7 !important;
}

/* ── Results card ── */
.results-card {
    background: #221809 !important;
    border: 1.5px solid rgba(192,139,31,0.2) !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
}

/* ── History card ── */
.history-card {
    background: #221809 !important;
    border: 1.5px solid rgba(192,139,31,0.2) !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    margin-top: 24px !important;
}

/* ── Card title ── */
.card-title {
    color: #F0D898;
    font-size: 0.9em;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(192,139,31,0.12);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Mobile ── */
@media (max-width: 640px) {
    .gradio-container { padding: 20px 14px 48px !important; }
    .page-title h1 { font-size: 1.35em; }
}
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
        <p>Upload or capture an image of a Persian character — the system will recognize and transliterate it.</p>
    </div>
    """)

    with gr.Row(equal_height=True):

        # Left — input
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["upload-zone"]):
                image_input = gr.Image(
                    sources=["upload", "webcam", "clipboard"],
                    type="numpy",
                    label=None,
                    show_label=False,
                    height=270,
                )

            gr.HTML("<p class='format-hint'>JPG · PNG · WEBP · BMP &nbsp;|&nbsp; Take a Photo or Paste Image</p>")

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Recognize", variant="primary")

        # Right — results
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["results-card"]):
                gr.HTML("<div class='card-title'>Recognition Results</div>")
                persian_output = gr.Textbox(
                    label="Recognized Persian Character",
                    placeholder="Persian character will appear here...",
                    rtl=True,
                    lines=3,
                    interactive=False
                )
                latin_output = gr.Textbox(
                    label="Romanized / Latin Equivalent",
                    placeholder="Transliteration will appear here...",
                    lines=3,
                    interactive=False
                )

    # History — full width
    with gr.Group(elem_classes=["history-card"]):
        gr.HTML("<div class='card-title'>Recognition History</div>")
        history_output = gr.HTML(value=format_history())

    gr.HTML("<div style='height:10px'></div>")
    clear_history_btn = gr.Button(
        "Clear History",
        variant="secondary",
        size="sm"
    )

    # Events
    submit_btn.click(
        fn=recognize_persian,
        inputs=image_input,
        outputs=[persian_output, latin_output, history_output]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, persian_output, latin_output, history_output]
    )

    clear_history_btn.click(
        fn=clear_db,
        outputs=history_output
    )

demo.launch(
    server_name="127.0.0.1",
    server_port=7860
)