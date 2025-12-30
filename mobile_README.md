Mobile app options (Kivy and React Native)

Option A — Kivy (Python-native)

Files included:
- `kivy_app.py` — Kivy application skeleton with camera preview, controls, settings, and overlay.
- `kivy.kv` — UI layout file used by Kivy.
- `requirements_kivy.txt` — Python packages for Kivy build and runtime.

Quick setup (desktop / testing):

1. Create virtualenv and install:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements_kivy.txt
```

2. Run app:

```powershell
python kivy_app.py
```

Notes:
- Packaging for Android/iOS requires Buildozer or Kivy‑iOS; follow Kivy docs.
- Use the existing `inference_pipeline.py` and `tts_engine.py` to hook into detection and audio.

Option B — React Native frontend + Python backend

Files included:
- `rn_backend/app.py` — Flask REST API that exposes `/predict` and `/health` endpoints and integrates `inference_pipeline`.
- `rn_backend/requirements.txt` — Python backend dependencies.
- `README_RN.md` — instructions for creating the React Native frontend and connecting to the backend.

Quick setup (backend):

```powershell
python -m venv venv_backend
venv_backend\Scripts\activate
pip install -r rn_backend/requirements.txt
python rn_backend/app.py
```

Frontend (React Native):
- Create a React Native app using `npx react-native init MyApp`.
- Implement camera preview with `react-native-camera` or `expo-camera` and POST frames to `http://<backend-ip>:5000/predict`.
- Display results, implement accessibility features (large buttons, high contrast).

Security/Performance notes:
- Send lower-resolution frames (e.g., 320x240) to backend for faster inference.
- Use HTTPS and authentication for production.
- Consider running model inference on-device with TensorFlow Lite Mobile for lower latency.
