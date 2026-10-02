"""
Retro Lens - Real-time Hand Gesture Filter Pipeline
"""

from dataclasses import dataclass
import os
import random
import time
from typing import Dict, List, Tuple, Callable
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PipelineConfig:
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    pinch_threshold_px: float = 45.0
    filter_cooldown_sec: float = 0.15
    mode_cooldown_sec: float = 1.2
    fist_dist_threshold_px: float = 80.0


class FilterBank:
    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def glitch(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            y = random.randint(0, h - 1)
            out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(roi)
        zeros = np.zeros_like(b)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def edge(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(roi, (25, 25), 0)

    @staticmethod
    def cartoon(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        t = time.time() * 5.0
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_fist_closed(landmarks, w: int, h: int, threshold: float) -> bool:
        wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
        tips = [8, 12, 16, 20]
        distances = [np.linalg.norm(np.array([landmarks[t].x * w, landmarks[t].y * h]) - wrist) for t in tips]
        return float(np.mean(distances)) < threshold

    @staticmethod
    def is_hand_rotated(thumb: Tuple[int, int], index: Tuple[int, int]) -> bool:
        dx, dy = index[0] - thumb[0], index[1] - thumb[1]
        return (dy > 25) or (abs(dx) > abs(dy) * 1.1)

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[0], rightmost[1], leftmost[1]], dtype=np.int32)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[1], rightmost[0], leftmost[1]], dtype=np.int32)


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "dual-tone": FilterBank.dual_tone,
            "thermal": FilterBank.thermal,
            "sketch": FilterBank.sketch,
            "pixelate": FilterBank.pixelate,
            "glitch": FilterBank.glitch,
            "invert": FilterBank.invert,
            "red-channel": FilterBank.red_channel,
            "edge": FilterBank.edge,
            "blur": FilterBank.blur,
            "cartoon": FilterBank.cartoon,
            "rainbow-wave": FilterBank.rainbow_wave,
        }
        self.filter_keys = list(self.filters.keys())
        self.active_filter_idx = 0
        self.is_3d_mode = False
        
        self.last_switch_time = 0.0
        self.last_mode_toggle = 0.0

        if hasattr(mp, "solutions"):
            self.use_tasks_api = False
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8,
            )
        else:
            self.use_tasks_api = True
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            self.mp_vision = mp_vision

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
            if not os.path.exists(model_path):
                print("[INFO] Mengunduh model hand_landmarker.task MediaPipe...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] Unduhan selesai.")

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    @property
    def secondary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 1) % len(self.filter_keys)]

    def cycle_filter(self, step: int = 1) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)

    def render_portal(self, frame: np.ndarray, pts: List[Tuple[int, int]], filter_key: str) -> np.ndarray:
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        if w <= 10 or h <= 10:
            return frame

        roi = frame[y : y + h, x : x + w].copy()
        processed_roi = self.filters[filter_key](roi)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        all_hand_tips = []
        fist_count = 0
        is_bowtie = False

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks:
                for hand_lm in results.hand_landmarks:
                    self.mp_vision.drawing_utils.draw_landmarks(
                        frame, hand_lm, self.mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
                    )
                    tips = [
                        (int(hand_lm[i].x * self.cfg.frame_width), int(hand_lm[i].y * self.cfg.frame_height))
                        for i in [4, 8, 12, 16, 20]
                    ]
                    all_hand_tips.append(tips)

                    # Fast Pinch
                    if GeometryUtils.euclidean_dist(tips[0], tips[4]) < self.cfg.pinch_threshold_px:
                        if now - self.last_switch_time > self.cfg.filter_cooldown_sec:
                            self.cycle_filter(1)
                            self.last_switch_time = now

                    if GeometryUtils.is_fist_closed(hand_lm, self.cfg.frame_width, self.cfg.frame_height, self.cfg.fist_dist_threshold_px):
                        fist_count += 1
        else:
            results = self.detector.process(rgb)
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
                    lm = hand_lm.landmark
                    tips = [
                        (int(lm[i].x * self.cfg.frame_width), int(lm[i].y * self.cfg.frame_height))
                        for i in [4, 8, 12, 16, 20]
                    ]
                    all_hand_tips.append(tips)

                    # Fast Pinch
                    if GeometryUtils.euclidean_dist(tips[0], tips[4]) < self.cfg.pinch_threshold_px:
                        if now - self.last_switch_time > self.cfg.filter_cooldown_sec:
                            self.cycle_filter(1)
                            self.last_switch_time = now

                    if GeometryUtils.is_fist_closed(lm, self.cfg.frame_width, self.cfg.frame_height, self.cfg.fist_dist_threshold_px):
                        fist_count += 1

        # Dual Fist Mode Switch
        if fist_count == 2 and (now - self.last_mode_toggle > self.cfg.mode_cooldown_sec):
            self.is_3d_mode = not self.is_3d_mode
            self.last_mode_toggle = now

        if self.is_3d_mode:
            if len(all_hand_tips) == 2:
                t1, t2 = all_hand_tips[0], all_hand_tips[1]
                frame = self.render_portal(frame, [t1[0], t1[1], t1[2], t2[2], t2[1], t2[0]], self.current_filter_name)
                frame = self.render_portal(frame, [t1[2], t1[3], t1[4], t2[4], t2[3], t2[2]], self.secondary_filter_name)
            elif len(all_hand_tips) == 1:
                frame = self.render_portal(frame, all_hand_tips[0], self.current_filter_name)
        else:
            if len(all_hand_tips) == 2:
                corners = [all_hand_tips[0][0], all_hand_tips[0][1], all_hand_tips[1][0], all_hand_tips[1][1]]
                if GeometryUtils.is_hand_rotated(corners[0], corners[1]) or GeometryUtils.is_hand_rotated(corners[2], corners[3]):
                    quad = GeometryUtils.sort_quad_bowtie(corners)
                    is_bowtie = True
                else:
                    quad = GeometryUtils.sort_quad_clean(corners)
                frame = self.render_portal(frame, quad, self.current_filter_name)
            elif len(all_hand_tips) == 1:
                t = all_hand_tips[0]
                frame = self.render_portal(frame, [t[0], t[1], t[2], t[4]], self.current_filter_name)

        self._draw_hud(frame, is_bowtie)
        return frame

    def _draw_hud(self, frame: np.ndarray, is_bowtie: bool) -> None:
        mode_str = "3D Mesh" if self.is_3d_mode else ("2D Bowtie" if is_bowtie else "2D Quad")
        cv2.putText(frame, f"MODE: {mode_str} [Key 'C' / Dual Fist]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FILTER: {self.current_filter_name.upper()} [Pinch / Key 'N'/'P']", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main() -> None:
    cfg = PipelineConfig()
    print("[INFO] Menginisialisasi RetroLens Engine...")
    processor = PortalProcessor(cfg)

    cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened() and cfg.cam_index == 0:
        alt_index = 1
        cap = cv2.VideoCapture(alt_index)
        if cap.isOpened():
            cfg.cam_index = alt_index
            print(f"[INFO] Terhubung ke kamera alternatif indeks {alt_index}")

    if not cap.isOpened():
        print(f"[ERROR] Kamera (indeks {cfg.cam_index}) tidak terdeteksi atau izin belum diberikan!")
        print("[TIPS] Di macOS, pastikan izin kamera diaktifkan:")
        print("       System Settings -> Privacy & Security -> Camera")
        return

    print(f"[INFO] Kamera aktif (indeks {cfg.cam_index}).")
    print("[KONTROL]")
    print("  'q'       : Keluar")
    print("  'c'       : Toggle mode 2D / 3D Mesh (atau kepal kedua tangan)")
    print("  'n' / 'p' : Filter berikutnya / sebelumnya (atau pinch jempol-kelingking)")
    print("  's'       : Ambil screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        out_frame = processor.process_frame(frame)
        cv2.imshow("RetroLens Engine", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            processor.is_3d_mode = not processor.is_3d_mode
        elif key == ord("n"):
            processor.cycle_filter(1)
        elif key == ord("p"):
            processor.cycle_filter(-1)
        elif key == ord("s"):
            fn = f"cap_{int(time.time())}.png"
            cv2.imwrite(fn, out_frame)
            print(f"[INFO] Screenshot disimpan: {fn}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()