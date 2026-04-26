from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image
import io
import base64
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE   = 224
HF_REPO_ID = "SanthoshBR-30/disaster-cnn-ensemble" 
HF_TOKEN   = os.environ.get("HF_TOKEN")

MODEL_FILES = {
    "efficientnet": "EfficientNetV2_final.keras",
    "convnext":     "ConvNeXt_final.keras",
    "densenet":     "DenseNet201_final.keras",
}
DENSE_CLASSIFIER_FILE = "dense_classifier_final.keras"
CLASS_NAMES_PATH      = os.path.join(BASE_DIR, "class_names.json")

# ── Download models from Hugging Face ────────────────────────────────────────
print("Downloading models from Hugging Face...")
MODEL_PATHS = {}
for name, filename in MODEL_FILES.items():
    print(f"  Downloading {filename}...")
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, token=HF_TOKEN)
    MODEL_PATHS[name] = path

print("  Downloading dense classifier...")
DENSE_CLASSIFIER_PATH = hf_hub_download(
    repo_id=HF_REPO_ID, filename=DENSE_CLASSIFIER_FILE, token=HF_TOKEN
)
print("All downloads complete ✅")

# ── Load class names ──────────────────────────────────────────────────────────
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
print(f"Classes: {CLASS_NAMES}")

# ── Load models & build feature extractors ────────────────────────────────────
print("Loading models into memory...")
feature_extractors = {}
for name, path in MODEL_PATHS.items():
    print(f"  Loading {name}...")
    full_model = tf.keras.models.load_model(path)
    feature_layer = full_model.get_layer("predictions").input
    feature_extractors[name] = tf.keras.Model(
        inputs  = full_model.input,
        outputs = feature_layer,
        name    = f"{name}_extractor"
    )

print("  Loading dense classifier...")
dense_classifier = tf.keras.models.load_model(DENSE_CLASSIFIER_PATH)
print("All models loaded ✅")

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def run_ensemble(img_array):
    features = []
    for name, extractor in feature_extractors.items():
        feat = extractor.predict(img_array, verbose=0)
        features.append(feat)
    combined = np.concatenate(features, axis=-1)
    probs = dense_classifier.predict(combined, verbose=0)[0]
    return probs

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "running", "classes": CLASS_NAMES})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        elif request.is_json and "image" in request.json:
            image_bytes = base64.b64decode(request.json["image"])
        else:
            return jsonify({"error": "No image provided"}), 400

        img_array = preprocess_image(image_bytes)
        probs     = run_ensemble(img_array)
        top_idx   = int(np.argmax(probs))
        result = {
            "predicted_class": CLASS_NAMES[top_idx],
            "confidence":      float(probs[top_idx]),
            "all_probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/app")
def serve_frontend():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)