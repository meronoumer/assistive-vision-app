"""realtime_app.py

Real-time camera app using OpenCV, processing frames every 500ms.

Features:
- Capture at ~30 FPS, buffer frames to avoid processing backlog
- Process latest frame every 500ms using TFLite ObjectDetector
- Display annotated frames with labels/confidence, FPS, and latency
- Keyboard controls: 'q' quit, 's' toggle sound, 'p' pause/resume, 'c' capture frame

Usage:
  python realtime_app.py --model path/to/model.tflite --labels path/to/labels.txt

Note: For classifiers (e.g., CIFAR/MobileNet) bounding boxes are not available; the script
draws label text on the frame. If your model outputs detection boxes/scores/labels,
it will draw them when the output keys are present.
"""
import argparse
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import os

import cv2
import numpy as np

from inference_pipeline import ImagePreprocessor, ObjectDetector, ResultProcessor
from tts_engine import TTSEngine


class RealtimeApp:
    def __init__(self, model_path: str, labels_path: str = None, camera_index: int = 0,
                 process_interval: float = 0.5, target_fps: int = 30, output_dir: str = "captures"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.camera_index = camera_index
        self.process_interval = process_interval
        self.target_fps = target_fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        labels = None
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = [l.strip() for l in f.readlines() if l.strip()]

        self.prep = ImagePreprocessor(image_size=224)
        self.detector = ObjectDetector(model_path, labels=labels, use_gpu_delegate=True)
        self.processor = ResultProcessor(labels=labels, threshold=0.5, important=["person", "car", "truck"])
        self.tts = TTSEngine()

        # Shared state
        self.frame_buffer = deque(maxlen=5)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.paused = False
        self.sound_on = True
        self.last_results = None
        self.last_latency = None
        self.last_processed_time = 0.0

    def start(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        self._display_loop()

    def stop(self):
        self.running = False
        try:
            self.capture_thread.join(timeout=1.0)
            self.process_thread.join(timeout=1.0)
        except Exception:
            pass
        self.tts.shutdown()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        # Try to set FPS
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                # store BGR frame; our pipeline expects RGB
                self.latest_frame = frame.copy()
                self.frame_buffer.append(frame.copy())
            # sleep a little to respect camera capture speed
            time.sleep(0.001)

        cap.release()

    def _process_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - self.last_processed_time < self.process_interval:
                time.sleep(0.01)
                continue

            # Get latest frame from buffer
            with self.lock:
                if len(self.frame_buffer) == 0:
                    frame = None
                else:
                    frame = self.frame_buffer[-1]

            if frame is None:
                time.sleep(0.01)
                continue

            # Preprocess (convert BGR->RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pre = self.prep.preprocess_array(rgb)

            # Run detector
            t0 = time.time()
            try:
                out = self.detector.predict(pre, top_k=5)
            except Exception as e:
                print("Detection error:", e)
                self.tts.audio_error()
                time.sleep(self.process_interval)
                continue
            latency = out.get("latency_ms", None)
            results = out.get("results", [])
            self.last_latency = latency
            self.last_results = results
            self.last_processed_time = time.time()

            # Post-process and announce
            filtered = self.processor.filter_by_confidence(results)
            prioritized = self.processor.prioritize(filtered)

            if self.sound_on:
                # interrupt previous speech and announce
                self.tts.announce_detection(out, result_processor=self.processor, interrupt=True)

            # small sleep to allow TTS queueing without blocking processing loop
            time.sleep(0.01)

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        # Draw results: if detection model provides boxes
        if self.last_results:
            for r in self.last_results:
                # expected keys: label, score, optionally box [ymin,xmin,ymax,xmax]
                label = r.get("label", "obj")
                score = r.get("score", 0.0)
                box = r.get("box") or r.get("bbox")
                text = f"{label}: {score*100:.1f}%"
                if box is not None and isinstance(box, (list, tuple)) and len(box) == 4:
                    ymin, xmin, ymax, xmax = box
                    # box may be normalized [0,1]
                    if 0.0 <= ymin <= 1.0 and 0.0 <= xmin <= 1.0:
                        x1 = int(xmin * w)
                        y1 = int(ymin * h)
                        x2 = int(xmax * w)
                        y2 = int(ymax * h)
                    else:
                        x1 = int(xmin)
                        y1 = int(ymin)
                        x2 = int(xmax)
                        y2 = int(ymax)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(out, text, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Draw label at top-left area, stacking if multiple
                    base_y = 20
                    idx = self.last_results.index(r)
                    cv2.putText(out, text, (10, base_y + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS and latency overlay
        now = time.time()
        fps_text = f"Target FPS: {self.target_fps}"
        latency_text = f"Latency: {self.last_latency:.1f} ms" if self.last_latency is not None else "Latency: --"
        cv2.putText(out, fps_text, (10, out.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(out, latency_text, (10, out.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    def _display_loop(self):
        window_name = "Assistive Vision - Realtime"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        last_frame_time = time.time()
        frame_count = 0
        fps = 0.0

        while self.running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            with self.lock:
                frame = self.latest_frame.copy()

            if self.paused:
                display = frame
            else:
                display = self._draw_annotations(frame)

            cv2.imshow(window_name, display)

            frame_count += 1
            now = time.time()
            if now - last_frame_time >= 1.0:
                fps = frame_count / (now - last_frame_time)
                frame_count = 0
                last_frame_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
                break
            elif key == ord("s"):
                self.sound_on = not self.sound_on
                print("Sound:", self.sound_on)
            elif key == ord("p"):
                self.paused = not self.paused
                print("Paused:" , self.paused)
            elif key == ord("c"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = self.output_dir / f"capture_{ts}.jpg"
                cv2.imwrite(str(fname), frame)
                print("Saved capture to", fname)

        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to TFLite model")
    p.add_argument("--labels", help="Path to labels file (one per line)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.5, help="Processing interval in seconds")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = RealtimeApp(model_path=args.model, labels_path=args.labels, camera_index=args.camera,
                      process_interval=args.interval, target_fps=args.fps)
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()
    except Exception as e:
        print("Error running app:", e)
        app.stop()
