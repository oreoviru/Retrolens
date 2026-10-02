# RetroLens 🖐️✨

A real-time hand gesture-controlled camera filter portal built with Python, OpenCV, and MediaPipe.

Spread both hands apart to open a dimensional portal window in your webcam feed. The area inside the portal is transformed in real-time with vibrant visual effects (dual-tone, thermal, sketch, glitch, and more). Cycle through filters on the fly using a simple thumb-to-pinky pinch gesture!

---

## 🚀 Installation

Ensure you have **Python 3.8 – 3.11** installed:

```bash
pip install -r requirements.txt
```

> ⚠️ **Apple Silicon Users:** Do not upgrade `mediapipe` beyond the pinned version (`0.10.9`) — newer versions have known issues on ARM Macs.

---

## ▶️ Running RetroLens

```bash
python3 Retrolens.py   # macOS / Linux
python Retrolens.py    # Windows
```

> **macOS Note:** Make sure camera permissions are enabled in:  
> *System Settings → Privacy & Security → Camera*.

---

## 🎮 Controls & Gestures

| Gesture / Key | Action |
| :--- | :--- |
| **Spread both hands** | Open and scale the portal |
| **Pinch thumb + pinky** | Switch to the next filter |
| **Fist both hands** or `C` | Toggle 2D / 3D wireframe mesh mode |
| `N` / `P` | Next / previous filter |
| `S` | Take screenshot (`cap_<timestamp>.png`) |
| `Q` | Quit application |

---

## 🎨 Available Filters

- **Dual Tone**: Stylized high-contrast two-tone threshold
- **Thermal**: Jet colormap heatmap effect
- **Sketch**: Hand-drawn pencil sketch
- **Pixelate**: 8-bit retro pixelation
- **Glitch**: RGB chromatic shift with scanline artifacts
- **Invert**: Color negative inversion
- **Red Channel**: High-intensity red monochrome channel
- **Edge**: Neon Canny edge detection
- **Blur**: Heavy Gaussian depth blur
- **Cartoon**: Bilateral smoothing with ink outlines
- **Rainbow Wave**: Dynamic oscillating HSV rainbow wave

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/) — Real-time computer vision & image processing
- [MediaPipe](https://developers.google.com/mediapipe) — Hand tracking & landmark detection
- [NumPy](https://numpy.org/) — Array manipulation & coordinate math

---

## 📄 License

Distributed under the [MIT License](LICENSE).
