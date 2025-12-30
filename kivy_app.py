from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
import threading
import cv2
import numpy as np
from pathlib import Path

KV = Builder.load_file("kivy.kv")


class CameraScreen(BoxLayout):
    running = BooleanProperty(False)
    sound_on = BooleanProperty(True)
    interval = NumericProperty(0.5)
    threshold = NumericProperty(0.7)
    tts_voice = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.cap_thread = None
        self.frame = None
        self.lock = threading.Lock()

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
        if not self.running:
            self.running = True
            self.cap_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.cap_thread.start()
            Clock.schedule_interval(self.update_preview, 1/30.)

    def stop_camera(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _camera_loop(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def update_preview(self, dt):
        # This method should draw `self.frame` into a Kivy texture.
        # Minimal implementation: no-op placeholder.
        pass

    def on_start_stop(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def on_capture(self):
        if self.frame is None:
            return
        Path("captures").mkdir(exist_ok=True)
        fname = Path("captures") / f"capture_{int(time.time())}.jpg"
        cv2.imwrite(str(fname), self.frame)


class KivyApp(App):
    def build(self):
        return CameraScreen()


if __name__ == '__main__':
    KivyApp().run()
