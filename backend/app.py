# app.py
import base64, io, os, sys, traceback
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

# ---- Pillow resample compat ----
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = getattr(Image, "LANCZOS", Image.BICUBIC)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/mnist_cnn.keras")

app = Flask(__name__)

# 🔓 TEMP: allow any origin while we debug
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Force the headers onto every response, including errors and 204s
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"   # no credentials used, so * is ok
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Vary"] = "Origin"
    return resp

print(f"[BOOT] Python: {sys.version}", flush=True)
print(f"[BOOT] CWD: {os.getcwd()}", flush=True)
print(f"[BOOT] Looking for model at: {MODEL_PATH}", flush=True)

try:
    mdl = tf.keras.models.load_model(MODEL_PATH)
    app.config["MODEL"] = mdl
    print("[BOOT] Model loaded ✅", flush=True)
except Exception:
    print("[BOOT] Model load FAILED ❌", flush=True)
    traceback.print_exc()
    app.config["MODEL"] = None

def get_model():
    return current_app.config.get("MODEL")

def preprocess_from_base64(b64_png: str) -> np.ndarray:
    if b64_png.startswith("data:image"):
        b64_png = b64_png.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize((280, 280), RESAMPLE)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr_bin = (arr > 0.2).astype(np.float32)
    if arr_bin.sum() == 0:
        arr28 = np.zeros((28, 28), dtype=np.float32)
        return arr28[None, ..., None]
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
    pil20 = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), RESAMPLE)
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
        print("[/predict] preflight", flush=True)
        return ("", 204)

    print("[/predict] POST hit", flush=True)
    model = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 503

    try:
        if "file" in request.files:
            print("[/predict] multipart file", flush=True)
            f = request.files["file"]
            img = Image.open(f.stream).convert("L")
            img = img.resize((280, 280), RESAMPLE)
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
                pil20 = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), RESAMPLE)
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

        elif request.is_json:
            print("[/predict] JSON body", flush=True)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
