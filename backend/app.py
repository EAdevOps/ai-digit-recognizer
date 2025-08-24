# app.py
import base64, io, os, sys, traceback
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

MODEL_PATH = os.environ.get("MODEL_PATH", "model/mnist_cnn.keras")

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {
        "origins": [
            "https://aidigitrecognizer.netlify.app",  # your site
            "http://localhost:3000"                   # local dev
        ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
# ---- Model load with clear logs ----
print(f"[BOOT] Python: {sys.version}", flush=True)
print(f"[BOOT] CWD: {os.getcwd()}", flush=True)
print(f"[BOOT] Looking for model at: {MODEL_PATH}", flush=True)

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[BOOT] Model loaded", flush=True)
except Exception as e:
    print("[BOOT] Model load FAILED", flush=True)
    traceback.print_exc()
    # Re-raise so you notice the failure during dev
    raise

def preprocess_from_base64(b64_png: str) -> np.ndarray:
    # Strip data URL prefix if present
    if b64_png.startswith("data:image"):
        b64_png = b64_png.split(",", 1)[1]

    # Decode -> grayscale PIL image
    img_bytes = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # 0..255, white high

    # Resize once to a larger working size to stabilize steps
    img = img.resize((280, 280), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # 0..1

    # Auto invert if background is bright (white)
    # White background → mean high → invert to black background
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Light threshold to boost contrast (tweak 0.2..0.4 as needed)
    arr_bin = (arr > 0.2).astype(np.float32)

    # If nothing drawn, return zeros
    if arr_bin.sum() == 0:
        arr28 = np.zeros((28, 28), dtype=np.float32)
        return arr28[None, ..., None]

    # Crop to bounding box of the digit
    ys, xs = np.where(arr_bin > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = arr[y0:y1, x0:x1]

    # Pad to square (keep aspect ratio), then resize to 20x20
    h, w = crop.shape
    m = max(h, w)
    sq = np.zeros((m, m), dtype=np.float32)
    y_start = (m - h) // 2
    x_start = (m - w) // 2
    sq[y_start:y_start + h, x_start:x_start + w] = crop

    pil20 = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    d20 = np.array(pil20, dtype=np.float32) / 255.0

    # Put 20x20 in the center of a 28x28 image (MNIST style padding)
    arr28 = np.zeros((28, 28), dtype=np.float32)
    arr28[4:24, 4:24] = d20

    # Optional: recentre via (integer) center-of-mass shift
    # Use brightness as weights; add tiny epsilon to avoid div by zero
    ys, xs = np.nonzero(arr28 > 0.01)
    if len(ys) > 0:
        weights = arr28[ys, xs]
        cy = int(round(np.average(ys, weights=weights)))
        cx = int(round(np.average(xs, weights=weights)))
        # Target center is (14,14)
        dy = 14 - cy
        dx = 14 - cx
        arr28 = np.roll(arr28, shift=dy, axis=0)
        arr28 = np.roll(arr28, shift=dx, axis=1)

    # Final shape for the model: (1, 28, 28, 1)
    return arr28[None, ..., None]


@app.get("/")
def root():
    return {"ok": True, "health": "/health", "predict": "/predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 503

    try:
        # --- Case 1: multipart/form-data (file upload) ---
        if "file" in request.files:
            f = request.files["file"]
            img = Image.open(f.stream).convert("L")
            # reuse your exact preprocessing steps (no base64 needed):
            img = img.resize((280, 280), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            if arr.mean() > 0.5:
                arr = 1.0 - arr
            arr_bin = (arr > 0.2).astype(np.float32)
            if arr_bin.sum() == 0:
                x = np.zeros((1, 28, 28, 1), dtype=np.float32)
            else:
                ys, xs = np.where(arr_bin > 0)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                crop = arr[y0:y1, x0:x1]
                h, w = crop.shape
                m = max(h, w)
                sq = np.zeros((m, m), dtype=np.float32)
                y_start = (m - h) // 2
                x_start = (m - w) // 2
                sq[y_start:y_start+h, x_start:x_start+w] = crop
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
                x = arr28[None, ..., None].astype(np.float32)

        # --- Case 2: JSON { image: "data:image/png;base64,..." } ---
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            data_url = data.get("image", "")
            if not data_url:
                return jsonify({"error": "Provide JSON { image: <dataURL> }"}), 400
            x = preprocess_from_base64(data_url)

        else:
            return jsonify({"error": "Provide form-data file or JSON {image}"}), 400

        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return jsonify({"prediction": pred, "confidence": conf, "probs": probs.tolist()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---- IMPORTANT: start the server ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
