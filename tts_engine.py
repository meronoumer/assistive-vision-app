"""tts_engine.py

TTSEngine: primary offline TTS with `pyttsx3`, fallback to `gTTS` + `playsound`.

Features:
- Speak detection results with configurable speed and voice
- Queue management for multiple detections
- Interrupt/cancel speaking when new detections arrive
- Audio cues: processing, error, ready
- Integration helper to announce ObjectDetector results

Notes:
- `gTTS` requires network but is used only if `pyttsx3` fails.
- `playsound` is a lightweight playback option for saved mp3 files.
"""
import threading
import queue
import tempfile
import os
import time
from typing import Optional, Sequence, List

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from playsound import playsound
except Exception:
    playsound = None


class TTSEngine:
    def __init__(self, rate: int = 150, voice: Optional[str] = None, use_pyttsx3: bool = True):
        self.rate = rate
        self.voice = voice
        self.use_pyttsx3 = use_pyttsx3 and (pyttsx3 is not None)

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._engine = None
        self._current_playback = None
        self._lock = threading.Lock()

        if self.use_pyttsx3:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                if self.voice is not None:
                    self._engine.setProperty("voice", self.voice)
            except Exception:
                # disable pyttsx3 if init fails
                self._engine = None
                self.use_pyttsx3 = False

        self._worker.start()

    def _speak_pyttsx3(self, text: str):
        if self._engine is None:
            raise RuntimeError("pyttsx3 engine not initialized")
        # pyttsx3 runAndWait blocks until completion; we call it in worker thread
        self._engine.say(text)
        self._engine.runAndWait()

    def _speak_gtts(self, text: str):
        if gTTS is None or playsound is None:
            raise RuntimeError("gTTS or playsound not available")
        tts = gTTS(text=text, lang="en")
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            tts.save(path)
            playsound(path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def speak(self, text: str, interrupt: bool = False):
        """Queue text to speak. If interrupt is True, cancel current playback and speak immediately."""
        if interrupt:
            self.stop()
        self._queue.put(text)

    def stop(self):
        """Interrupt current playback and clear queue."""
        with self._lock:
            # Clear the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            # Stop pyttsx3 speaking
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass
            # playsound does not support programmatic stop; rely on short clips or use pydub for advanced control

    def set_rate(self, rate: int):
        self.rate = rate
        if self._engine is not None:
            try:
                self._engine.setProperty("rate", self.rate)
            except Exception:
                pass

    def set_voice(self, voice: str):
        self.voice = voice
        if self._engine is not None:
            try:
                self._engine.setProperty("voice", self.voice)
            except Exception:
                pass

    def _worker_loop(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            # short-circuit: if stop requested, skip
            if self._stop_event.is_set():
                self._stop_event.clear()
                continue
            try:
                if self.use_pyttsx3 and self._engine is not None:
                    self._speak_pyttsx3(text)
                else:
                    # fallback to gTTS
                    self._speak_gtts(text)
            except Exception:
                # If TTS fails, ignore and continue
                continue

    # Audio cue convenience methods
    def audio_processing(self, interrupt: bool = True):
        self.speak("Processing.", interrupt=interrupt)

    def audio_ready(self, interrupt: bool = False):
        self.speak("Ready.", interrupt=interrupt)

    def audio_error(self, interrupt: bool = True):
        self.speak("An error occurred.", interrupt=interrupt)

    def announce_detection(self, detector_output: dict, result_processor=None, interrupt: bool = True):
        """Integrate with detector output.

        `detector_output` is expected to be the dict returned by `ObjectDetector.predict()`
        which contains `results` list of {label, score} and `latency_ms`.
        If `result_processor` is provided, it will be used to format and filter results.
        """
        results = detector_output.get("results", [])
        if result_processor is not None:
            filtered = result_processor.filter_by_confidence(results)
            prioritized = result_processor.prioritize(filtered)
            text = result_processor.format_results(prioritized)
        else:
            if not results:
                text = "No objects detected."
            else:
                parts = []
                for r in results[:5]:
                    parts.append(f"{r.get('label', 'object')} {int(r.get('score',0)*100)} percent")
                text = ", ".join(parts)

        # Compose audio cue sequence
        seq = ["Processing.", f"Detected: {text}", "Ready."]
        # Queue them; interrupt ensures new detection replaces old announcements
        for i, t in enumerate(seq):
            self.speak(t, interrupt=(i == 0 and interrupt))

    def shutdown(self):
        # signal worker to exit
        try:
            self._queue.put(None)
            self._worker.join(timeout=2.0)
        except Exception:
            pass


if __name__ == "__main__":
    # Quick interactive demo
    t = TTSEngine()
    t.audio_processing()
    t.speak("Detected: person 92 percent", interrupt=True)
    t.audio_ready()
    time.sleep(2)
    t.shutdown()
