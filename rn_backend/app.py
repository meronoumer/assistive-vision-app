from flask import Flask, request, jsonify
from inference_pipeline import ImagePreprocessor, ObjectDetector, ResultProcessor
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Initialize model once
MODEL_PATH = "./models/mobilenet_transfer.tflite"
LABELS_PATH = None
pre = ImagePreprocessor(image_size=224)
detector = ObjectDetector(MODEL_PATH, labels=None, use_gpu_delegate=False)
processor = ResultProcessor(threshold=0.6, important=["person","car","truck"])


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    # Accept JSON with base64-encoded image or multipart file upload
    img = None
    if "image_base64" in request.json:
        b64 = request.json["image_base64"]
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
    elif "file" in request.files:
        img = Image.open(request.files["file"].stream).convert("RGB")
    else:
        return jsonify({"error": "no image provided"}), 400

    arr = np.array(img)
    preprocessed = pre.preprocess_array(arr)
    out = detector.predict(preprocessed, top_k=5)
    results = out.get("results", [])
    filtered = processor.filter_by_confidence(results)
    prioritized = processor.prioritize(filtered)

    return jsonify({"results": prioritized, "latency_ms": out.get("latency_ms")})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
