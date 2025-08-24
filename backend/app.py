# app.py
import os, io, sys, base64, traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFile
import tensorflow as tf

# Tolerate short/odd PNG streams
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = os.environ.get("MODEL_PATH", "model/mnist_cnn.keras")

app = Flask(__name__)

# Simple global CORS (open during debug; tighten later by listing your Netlify origin)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
)

# Force CORS headers on EVERY response, including errors
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"  # or "https://aidigitrecognizer.netlify.app"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Max-Age"] = "600"
    resp.headers["Vary"] = "Origin"
    return resp

print(f"[BOOT] Python: {sys.version}", flush=True)
print(f"[BOOT] CWD: {os.getcwd()}", flush=True)
print(f"[BOOT] Looking for model at: {MODEL_PATH}", flush=True)

# Load model once at boot
model = tf.keras.models.load_model(MODEL_PATH)
print("[BOOT] Model loaded ✅", flush=True)

# Pick a safe resample constant across Pillow versions
try:
    from PIL import Image as _PILImage
    RESAMPLE = _PILImage.Resampling.LANCZOS  # Pillow ≥ 10
except Exception:
    RESAMPLE = Image.LANCZOS  # older Pillow fallback

def robust_png_to_gray(png_bytes: bytes) -> Image.Image:
    """Load PNG -> grayscale, tolerating truncated streams."""
    try:
        return Image.open(io.BytesIO(png_bytes)).convert("L")
    except Exception:
        parser = ImageFile.Parser()
        parser.feed(png_bytes)
        img = parser.close()
        return img.convert("L")

def preprocess_from_base64(data_url: str) -> np.ndarray:
    # Strip data URL prefix if present
    if data_url.startswith("data:image"):
        data_url = data_url.split(",", 1)[1]
    data_url = data_url.strip()

    img_bytes = base64.b64decode(data_url)
    img = robust_png_to_gray(img_bytes)

    # === Preprocess to MNIST format ===
    img = img.resize((280, 280), RESAMPLE)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Auto-invert if background is white
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Light threshold → bounding box
    arr_bin = (arr > 0.2).astype(np.float32)
    if arr_bin.sum() == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    ys, xs = np.where(arr_bin > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = arr[y0:y1, x0:x1]

    # Pad to square, then resize to 20x20
    h, w = crop.shape
    m = max(h, w)
    sq = np.zeros((m, m), dtype=np.float32)
    y_start = (m - h) // 2
    x_start = (m - w) // 2
    sq[y_start:y_start + h, x_start:x_start + w] = crop

    pil20 = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), RESAMPLE)
    d20 = np.array(pil20, dtype=np.float32) / 255.0

    # Place 20x20 into center of 28x28
    arr28 = np.zeros((28, 28), dtype=np.float32)
    arr28[4:24, 4:24] = d20

    # Center-of-mass recenter (integer shift)
    yy, xx = np.nonzero(arr28 > 0.01)
    if len(yy) > 0:
        weights = arr28[yy, xx]
        cy = int(round(np.average(yy, weights=weights)))
        cx = int(round(np.average(xx, weights=weights)))
        dy = 14 - cy
        dx = 14 - cx
        arr28 = np.roll(arr28, dy, axis=0)
        arr28 = np.roll(arr28, dx, axis=1)

    # Final model shape: (1, 28, 28, 1)
    return arr28[None, ..., None]

@app.get("/")
def root():
    return {"ok": True, "health": "/health", "predict": "/predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Explicitly handle preflight quickly
@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return ("", 204)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Case 1: JSON { image: "data:image/png;base64,..." }
        if request.is_json:
            data = request.get_json(silent=True) or {}
            img_data = data.get("image")
            if not img_data:
                return jsonify({"error": "Provide JSON { image: <dataURL> }"}), 400
            x = preprocess_from_base64(img_data)

        # Case 2: multipart/form-data with "file"
        elif "file" in request.files:
            fbytes = request.files["file"].read()
            gray = robust_png_to_gray(fbytes)
            # Reuse the same pipeline by re-encoding to base64 PNG once
            buf = io.BytesIO()
            gray.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            x = preprocess_from_base64(b64)

        else:
            return jsonify({"error": "Provide JSON {image} or form-data 'file'"}), 400

        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return jsonify({"prediction": pred, "confidence": conf, "probs": probs.tolist()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
