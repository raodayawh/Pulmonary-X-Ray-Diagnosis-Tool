"""
Pulmonary X-Ray Diagnosis Tool
CNN-based chest X-ray classifier: Healthy / Pneumonia / Tuberculosis / COVID-19
"""

import os
import json
import numpy as np
import gradio as gr
from PIL import Image
import model_setup

# ── Load class names ──────────────────────────────────────────────────────────
with open(model_setup.paths["class_names.json"]) as f:
    CLASS_NAMES = json.load(f)  # ["healthy", "pneumonia", "tuberculosis", "covid"]

# ── Load model (lazy, shown in loading state) ─────────────────────────────────
import tensorflow as tf

print("Loading model…")
MODEL = tf.keras.models.load_model(model_setup.paths["cnn_model_lung_detection.keras"])
INPUT_SHAPE = MODEL.input_shape  # e.g. (None, H, W, C)
IMG_H = INPUT_SHAPE[1] or 224
IMG_W = INPUT_SHAPE[2] or 224
print(f"Model loaded. Input shape: {INPUT_SHAPE}")

# ── Fun facts per class ───────────────────────────────────────────────────────
FUN_FACTS = {
    "healthy": [
        "🫁 Healthy lungs move roughly 11,000 litres of air every day.",
        "🌬️ The surface area of both lungs is about the size of a tennis court.",
        "💪 Regular aerobic exercise can increase lung capacity by up to 15%.",
        "🧬 Lung tissue is the only organ that can float on water.",
        "🌿 Deep breathing activates the parasympathetic nervous system, reducing stress.",
    ],
    "pneumonia": [
        "🦠 Pneumonia is caused by bacteria, viruses, or fungi infecting the air sacs.",
        "📊 It affects around 450 million people globally each year.",
        "💊 Bacterial pneumonia is typically treated with antibiotics within 5–7 days.",
        "🌡️ High fever, chills, and a productive cough are hallmark symptoms.",
        "👶 Children under 5 and adults over 65 are at the highest risk.",
    ],
    "tuberculosis": [
        "🧫 TB is caused by *Mycobacterium tuberculosis*, discovered by Robert Koch in 1882.",
        "🌍 TB remains one of the top 10 causes of death worldwide.",
        "⏳ TB bacteria can lie dormant (latent TB) for years before becoming active.",
        "💉 The BCG vaccine provides protection, especially against severe forms in children.",
        "📉 Global TB deaths have fallen by 58% since 2000 due to improved treatments.",
    ],
    "covid": [
        "🔬 COVID-19 is caused by SARS-CoV-2, first identified in late 2019.",
        "🫁 The virus can cause bilateral ground-glass opacities visible on X-ray.",
        "🧠 'Long COVID' symptoms can persist for months after initial recovery.",
        "💨 High-flow oxygen therapy is a key treatment for severe lung involvement.",
        "🌐 The COVID-19 pandemic led to the fastest vaccine development in history.",
    ],
}

# ── Prediction function ───────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return None, "Please upload or select an X-ray image."

    # Pre-process
    img = image.convert("RGB").resize((IMG_W, IMG_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)  # (1, H, W, 3)

    # Inference
    probs = MODEL.predict(arr, verbose=0)[0]  # shape: (num_classes,)

    # Build confidence dict for Gradio Label
    confidence = {CLASS_NAMES[i].capitalize(): float(probs[i]) for i in range(len(CLASS_NAMES))}

    # Top prediction
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]

    # Pick a fun fact
    import random
    fact = random.choice(FUN_FACTS[top_class])

    # Result markdown
    icons = {"healthy": "✅", "pneumonia": "⚠️", "tuberculosis": "🔴", "covid": "🟠"}
    icon = icons.get(top_class, "🔵")
    confidence_pct = probs[top_idx] * 100
    result_md = f"""
### {icon} {top_class.upper()} — {confidence_pct:.1f}% confidence

**💡 Did you know?**  
{fact}

> *This tool is for educational purposes only and does not constitute medical advice.*
"""
    return confidence, result_md


# ── Gather example images ─────────────────────────────────────────────────────
EXAMPLE_DIR = "sample_images"
EXAMPLES = []
seen_classes = set()
# Pick one example per class for the examples row
for cls in CLASS_NAMES:
    for fname in sorted(os.listdir(EXAMPLE_DIR)):
        if fname.startswith(cls) and fname.endswith(".png") and cls not in seen_classes:
            EXAMPLES.append([os.path.join(EXAMPLE_DIR, fname)])
            seen_classes.add(cls)
            break

# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
/* ─── Global ─── */
body { background: #f0f4f8 !important; }
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ─── Header ─── */
#header {
    background: linear-gradient(135deg, #0f2b4e 0%, #1a4a7a 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(15, 43, 78, 0.3);
}
#header h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0 !important;
}
#header p { color: #a8c4e0 !important; font-size: 1rem; margin: 0; }

/* ─── Upload zone ─── */
.upload-zone {
    border: 2px dashed #90aac4 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #1a6fbf !important; }

/* ─── Buttons ─── */
#analyze-btn {
    background: linear-gradient(135deg, #1a6fbf, #0f4c8a) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    box-shadow: 0 3px 12px rgba(26, 111, 191, 0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s;
    width: 100% !important;
}
#analyze-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(26, 111, 191, 0.45) !important;
}

/* ─── Results ─── */
#result-box {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
    border: 1px solid #e2e8f0 !important;
}
#fact-box {
    background: #eef5ff !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    border-left: 4px solid #1a6fbf !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* ─── Examples ─── */
.examples-table td img { border-radius: 8px !important; }
.label-selected { font-weight: 700 !important; }

/* ─── Footer ─── */
#footer {
    text-align: center;
    color: #8097b0;
    font-size: 0.82rem;
    margin-top: 24px;
    padding-bottom: 16px;
}
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Pulmonary X-Ray Diagnosis Tool") as demo:

    # Header
    gr.HTML("""
    <div id="header">
        <h1>🫁 Pulmonary X-Ray Diagnosis Tool</h1>
        <p>AI-powered chest X-ray analysis · Powered by a convolutional neural network</p>
    </div>
    """)

    with gr.Row():
        # ── Left column: upload + examples ──
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload X-Ray Image")
            image_input = gr.Image(
                type="pil",
                label="Chest X-Ray",
                elem_classes=["upload-zone"],
                show_label=False,
                height=320,
            )
            analyze_btn = gr.Button("🔍 Analyze X-Ray", elem_id="analyze-btn")

            gr.Markdown("#### 🖼️ Try a sample image")
            gr.Examples(
                examples=EXAMPLES,
                inputs=image_input,
                label="",
                examples_per_page=4,
            )

        # ── Right column: results ──
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Diagnosis Results")
            label_output = gr.Label(
                num_top_classes=4,
                label="Confidence Scores",
                elem_id="result-box",
            )
            fact_output = gr.Markdown(
                value="*Upload an X-ray and click **Analyze** to see results.*",
                elem_id="fact-box",
            )

    # ── Event ──
    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, fact_output],
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, fact_output],
    )

    # Footer
    gr.HTML("""
    <div id="footer">
        ⚕️ For educational and research use only &nbsp;·&nbsp;
        Not a substitute for professional medical diagnosis &nbsp;·&nbsp;
        Classes: Healthy · Pneumonia · Tuberculosis · COVID-19
    </div>
    """)

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True, css=CSS)
