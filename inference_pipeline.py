"""inference_pipeline.py

Modular inference pipeline with three independent classes:
- ImagePreprocessor: resize, normalize, handle file/camera inputs
- ObjectDetector: load TFLite model, optional GPU delegate, run inference, measure latency
- ResultProcessor: filter by confidence, format human-readable text, prioritize important objects

Example usage at bottom shows how to run a simple file-based inference loop.
"""
from typing import List, Optional, Sequence, Tuple, Dict, Any
import time
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


class ImagePreprocessor:
    """Preprocess images for MobileNetV2 inference.

    Methods accept numpy arrays or read from files/camera. Outputs float32 arrays
    shaped (H, W, 3) or batched (1, H, W, 3) depending on caller.
    """

    def __init__(self, image_size: int = 224, use_mobilenet_preprocess: bool = True):
        self.image_size = image_size
        self.use_mobilenet_preprocess = use_mobilenet_preprocess

    def preprocess_array(self, img: np.ndarray) -> np.ndarray:
        """Resize and normalize a single image numpy array.

        Args:
            img: uint8/float image HxWx3
        Returns:
            float32 image HxWx3 ready for model input
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a numpy array")
        # Convert to float32
        img = img.astype(np.float32)
        # Resize using TF to benefit from high-quality resizing
        img_tf = tf.image.resize(img, [self.image_size, self.image_size]).numpy()
        img_tf = img_tf / 255.0
        if self.use_mobilenet_preprocess:
            img_tf = tf.keras.applications.mobilenet_v2.preprocess_input(img_tf * 255.0).numpy()
        return img_tf.astype(np.float32)

    def preprocess_batch(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        arrs = [self.preprocess_array(im) for im in imgs]
        batch = np.stack(arrs, axis=0)
        return batch

    def load_from_file(self, path: str) -> np.ndarray:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        return self.preprocess_array(img_np)

    def camera_generator(self, index: int = 0):
        """Yield frames from a connected camera if `cv2` is available.

        Yields raw numpy arrays (H,W,3) uint8; caller should call `preprocess_array`.
        """
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("OpenCV is required for camera support (install opencv-python)") from e

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV gives BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        finally:
            cap.release()


class ObjectDetector:
    """Generic TFLite-based object detector/classifier.

    Designed to be model-agnostic: for a classification model it returns top-k
    labels and confidences; for a detection model that outputs boxes/scores/labels
    the user can adapt `postprocess_output` accordingly.
    """

    def __init__(
        self,
        model_path: str,
        labels: Optional[Sequence[str]] = None,
        use_gpu_delegate: bool = True,
        num_threads: int = 1,
    ):
        self.model_path = str(model_path)
        self.labels = list(labels) if labels is not None else None
        self.use_gpu_delegate = use_gpu_delegate
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()

    def _try_load_gpu_delegate(self):
        # Try common delegate library names across platforms
        delegate_names = [
            "libtensorflowlite_gpu_delegate.so",  # Linux
            "libtensorflowlite_gpu.dll",         # Windows
            "libtensorflowlite_gpu.dylib",       # macOS
        ]
        for name in delegate_names:
            try:
                delegate = tf.lite.experimental.load_delegate(name)
                return delegate
            except Exception:
                continue
        # Fallback to TF builtin GPU delegate API if available
        try:
            from tensorflow.lite.experimental import load_delegate

            for name in delegate_names:
                try:
                    return load_delegate(name)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _load_model(self):
        use_delegate = None
        if self.use_gpu_delegate:
            try:
                use_delegate = self._try_load_gpu_delegate()
            except Exception:
                use_delegate = None

        if use_delegate is not None:
            print("Using GPU delegate for TFLite interpreter")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path, experimental_delegates=[use_delegate], num_threads=self.num_threads)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=self.num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, preprocessed_image: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run inference on a single preprocessed image (H,W,3 float32).

        Returns list of dicts: {"label": str, "score": float}
        """
        if preprocessed_image.ndim == 3:
            inp = np.expand_dims(preprocessed_image, axis=0)
        else:
            inp = preprocessed_image

        # Respect interpreter input dtype
        input_dtype = self.input_details[0]["dtype"]
        if input_dtype == np.uint8:
            inp = (np.clip(inp, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            inp = inp.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        start = time.perf_counter()
        self.interpreter.invoke()
        latency = (time.perf_counter() - start) * 1000.0

        out = self.interpreter.get_tensor(self.output_details[0]["index"])  # model-dependent

        # If classification output shape is (1, num_classes)
        results: List[Dict[str, Any]] = []
        if out.ndim == 2:
            probs = out[0]
            # If probs are logits, apply softmax
            if probs.max() > 1.0 or probs.min() < 0.0:
                probs = tf.nn.softmax(probs).numpy()
            top_idx = np.argsort(probs)[::-1][:top_k]
            for i in top_idx:
                label = str(i) if self.labels is None else self.labels[i]
                results.append({"label": label, "score": float(probs[i])})
        else:
            # For generic detection models, just return raw output
            results.append({"raw_output": out.tolist()})

        return {"results": results, "latency_ms": latency}

    def measure_latency(self, preprocessed_image: np.ndarray, runs: int = 50) -> float:
        # Warm-up
        _ = self.predict(preprocessed_image)
        total = 0.0
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = self.predict(preprocessed_image)
            total += (time.perf_counter() - t0)
        avg_ms = (total / runs) * 1000.0
        return avg_ms


class ResultProcessor:
    """Process raw model outputs into filtered, prioritized, human-readable results."""

    def __init__(self, labels: Optional[Sequence[str]] = None, threshold: float = 0.7, important: Optional[Sequence[str]] = None):
        self.labels = list(labels) if labels is not None else None
        self.threshold = threshold
        # labels considered higher priority (e.g., person, car, obstacle)
        self.important = set(important) if important is not None else set()

    def filter_by_confidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in results if r.get("score", 0.0) >= self.threshold]

    def prioritize(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def score_key(r: Dict[str, Any]):
            label = r.get("label", "")
            base = r.get("score", 0.0)
            boost = 1.0 if label in self.important else 0.0
            return (boost, base)

        return sorted(results, key=score_key, reverse=True)

    def format_results(self, results: List[Dict[str, Any]], max_items: int = 5) -> str:
        if not results:
            return "No objects detected above confidence threshold."
        lines = []
        for r in results[:max_items]:
            label = r.get("label", "unknown")
            score = r.get("score", 0.0)
            lines.append(f"{label}: {score*100:.1f}%")
        return "; ".join(lines)


if __name__ == "__main__":
    # Example usage: load a TFLite model, preprocess an image, run detection, print results
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to TFLite model")
    parser.add_argument("--labels", help="Path to labels txt file (one label per line)")
    parser.add_argument("--image", help="Path to an input image file")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    labels = None
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]

    pre = ImagePreprocessor(image_size=224, use_mobilenet_preprocess=True)
    det = ObjectDetector(args.model, labels=labels, use_gpu_delegate=True)
    rp = ResultProcessor(labels=labels, threshold=args.threshold, important=["person", "car", "truck"]) 

    if not args.image:
        print("Please pass --image to test the pipeline.")
        raise SystemExit(1)

    img = pre.load_from_file(args.image)
    out = det.predict(img, top_k=5)
    results = out.get("results", [])
    latency = out.get("latency_ms", None)

    filtered = rp.filter_by_confidence(results)
    prioritized = rp.prioritize(filtered)
    text = rp.format_results(prioritized)

    print("Detection results:", text)
    if latency is not None:
        print(f"Inference latency: {latency:.2f} ms")
