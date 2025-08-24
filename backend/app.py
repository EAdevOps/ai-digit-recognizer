# app.py (top section)
import base64, io, os, sys, traceback
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf

MODEL_PATH = os.environ.get("MODEL_PATH", "model/mnist_cnn.keras")

app = Flask(__name__)
# Global CORS for simple GETs (/, /health)
CORS(app, resources={
    r"/": {"origins": "*"},
    r"/health": {"origins": "*"},
})

ALLOWED_ORIGINS = [
    "https://aidigitrecognizer.netlify.app",
    "http://localhost:3000",
]

print(f"[BOOT] Python: {sys.version}", flush=True)
print(f"[BOOT] CWD: {os.getcwd()}", flush=True)
print(f"[BOOT] Looking for model at: {MODEL_PATH}", flush=True)

model = tf.keras.models.load_model(MODEL_PATH)
print("[BOOT] Model loaded ✅", flush=True)
# ---- Preprocessing ---------------------------------------------------------
def preprocess_from_base64(b64_png: str) -> np.ndarray:
    # Strip data URL prefix if present
    if b64_png.startswith("data:image"):
        b64_png = b64_png.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale

    # Resize large first for stable steps
    img = img.resize((280, 280), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Auto-invert if background is bright
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Light threshold to boost contrast
    arr_bin = (arr > 0.2).astype(np.float32)

    # If nothing drawn, return zeros (1,28,28,1)
    if arr_bin.sum() == 0:
        arr28 = np.zeros((28, 28), dtype=np.float32)
        return arr28[None, ..., None]

    # Crop to bounding box
    ys, xs = np.where(arr_bin > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = arr[y0:y1, x0:x1]

    # Pad to square, resize to 20x20
    h, w = crop.shape
    m = max(h, w)
    sq = np.zeros((m, m), dtype=np.float32)
    y_start = (m - h) // 2
    x_start = (m - w) // 2
    sq[y_start:y_start + h, x_start:x_start + w] = crop

    pil20 = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    d20 = np.array(pil20, dtype=np.float32) / 255.0

    # Center 20x20 into 28x28 (MNIST style)
    arr28 = np.zeros((28, 28), dtype=np.float32)
    arr28[4:24, 4:24] = d20

    # Recenter via integer CoM shift
    ys2, xs2 = np.nonzero(arr28 > 0.01)
    if len(ys2) > 0:
        weights = arr28[ys2, xs2]
        cy = int(round(np.average(ys2, weights=weights)))
        cx = int(round(np.average(xs2, weights=weights)))
        dy = 14 - cy
        dx = 14 - cx
        arr28 = np.roll(arr28, shift=dy, axis=0)
        arr28 = np.roll(arr28, shift=dx, axis=1)

    return arr28[None, ..., None]

def preprocess_from_image(img: Image.Image) -> np.ndarray:
    """Same pipeline as base64, starting from a PIL Image (grayscale)."""
    img = img.convert("L").resize((280, 280), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr_bin = (arr > 0.2).astype(np.float32)
    if arr_bin.sum() == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)
    ys, xs = np.where(arr_bin > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = arr[y0:y1, x0:x1]
    h, w = crop.shape
    m = max(h, w)
    sq = np.zeros((m, m), dtype=np.float32)
    y_start = (m - h) // 2
    x_start = (m - w) // 2
    sq[y_start:y_start + h, x_start:x_start + w] = crop
    pil20 = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    d20 = np.array(pil20, dtype=np.float32) / 255.0
    arr28 = np.zeros((28, 28), dtype=np.float32)
    arr28[4:24, 4:24] = d20
    ys2, xs2 = np.nonzero(arr28 > 0.01)
    if len(ys2) > 0:
        weights = arr28[ys2, xs2]
        cy = int(round(np.average(ys2, weights=weights)))
        cx = int(round(np.average(xs2, weights=weights)))
        dy = 14 - cy
        dx = 14 - cx
        arr28 = np.roll(arr28, shift=dy, axis=0)
        arr28 = np.roll(arr28, shift=dx, axis=1)
    return arr28[None, ..., None].astype(np.float32)

# ---- Routes ----------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "health": "/health", "predict": "/predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
@cross_origin(origins=ALLOWED_ORIGINS, methods=["POST"], allow_headers=["Content-Type"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 503

    # ---- accept either form-data (file) OR JSON { image: "dataURL" }
    if "file" in request.files:
        f = request.files["file"]
        img = Image.open(f.stream).convert("L")
        # ... (keep your same preprocessing steps)
        # build x = (1,28,28,1) np.float32
    else:
        data = request.get_json(silent=True) or {}
        data_url = data.get("image", "")
        if not data_url:
            return jsonify({"error": "Provide form-data file or JSON { image }"}), 400
        x = preprocess_from_base64(data_url)

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return jsonify({"prediction": pred, "confidence": conf, "probs": probs.tolist()})

# ---- Entrypoint ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
