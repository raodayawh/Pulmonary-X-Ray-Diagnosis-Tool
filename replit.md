# Pulmonary X-Ray Diagnosis Tool

A CNN-based chest X-ray classifier that detects four lung conditions:
**Healthy · Pneumonia · Tuberculosis · COVID-19**

## Stack

- **Frontend/UI**: Gradio (Python)
- **Model**: TensorFlow/Keras CNN (`cnn_model_lung_detection.keras`)
- **Classes**: `["healthy", "pneumonia", "tuberculosis", "covid"]`
- **Sample data**: `sample_images/` (20 images per class)

## How to run

```
python app.py
```

The app binds to `0.0.0.0` on port `5000` (or the `PORT` env var if set).

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio application |
| `model_setup.py` | Auto-generated helper — maps filenames to local paths |
| `cnn_model_lung_detection.keras` | Trained CNN model |
| `class_names.json` | Class label list |
| `model_config.json` | HuggingFace Hub config (hub_repo_id is null → local model) |
| `sample_images/` | 20 sample X-ray images per class for built-in examples |
| `requirements.txt` | Python dependencies |

## Deployment

Run command: `python app.py`  
No build step required.

## User preferences

- Medical, clean, minimal visual style
- Confidence bars for all 4 classes
- Fun facts panel per prediction
- Preloaded examples from `sample_images/`
