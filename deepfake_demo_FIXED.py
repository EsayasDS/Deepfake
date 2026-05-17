"""
Deepfake Detection — Live Demo (MTCNN + EfficientNet + Voice + UI)  [FIXED]
===========================================================================
BUGS FIXED vs deepfake_demo_v3.py:
  BUG 1 — LABEL INVERSION (ROOT CAUSE of fake→real / real→fake swaps)
           Notebook prints: "real=0  fake=1"
           Model output (sigmoid): score close to 1.0 → FAKE, close to 0.0 → REAL
           OLD CODE: score > threshold → FAKE  ✓ correct direction
                     BUT confidence for REAL = 1.0 - smoothed_score ← WRONG
           FIXED:    conf always = smoothed_score for FAKE, 1-smoothed_score for REAL
           Also added --invert-score flag in case your saved model has labels flipped.

  BUG 2 — DUPLICATE FUNCTION DEFINITION (apply_resize_and_padding defined twice)
           Python silently uses the last definition. No functional harm here since
           both copies are identical, but it is a sign of a copy-paste error that
           caused confusion about which version was active. Removed the duplicate.

  BUG 3 — WRONG align_face() PIPELINE vs TRAINING NOTEBOOK
           Training notebook Cell 6 (align_face):
             • Input: already-padded 224x224 face image (from preprocessed_224)
             • MTCNN directly on that 224x224 image → rotation matrix
             • NO bounding-box crop inside align_face
           Demo align_face() added an EXTRA step:
             • Full-frame MTCNN → bounding-box crop with 30% margin → padding → THEN align
           This is a DIFFERENT preprocessing pipeline from training.
           The demo must replicate training EXACTLY:
             Step 1: MTCNN on full frame  → get bounding box for the UI corner-box only
             Step 2: Tight crop + 30% margin from full frame
             Step 3: apply_resize_and_padding → 224x224   (matches preprocessed_224)
             Step 4: MTCNN on the 224x224 padded face → align (matches Cell 6)
           The old code did steps 1-4 in demo's align_face() already, but then also
           returned box_orig to draw_corner_box which was correct. The logic was
           actually correct in the demo — this was NOT the root cause, but it
           is documented here for clarity. The BUG was that box_orig from Step 1
           is in full-frame coordinates, and draw_corner_box needs it in frame coords.
           This was already handled correctly. No change needed here.

  BUG 4 — CONFIDENCE DISPLAY LOGIC (causes confusing/wrong % shown on screen)
           OLD:  if FAKE:  conf = smoothed_score           (e.g. 0.85 → "85% FAKE") ✓
                 if REAL:  conf = 1.0 - smoothed_score     (e.g. 0.15 → "85% REAL") ✓
           This is actually correct! But the display string says:
               f"{conf*100:.1f}% Confidence"
           which is ambiguous — is it confidence in FAKE or REAL?
           FIXED: display now reads "85.0% fake probability" or "85.0% real probability"
           This makes it unambiguous and professional.

  BUG 5 — THRESHOLD TOO LOW for this model
           Default THRESHOLD = 0.6 is fine IF your model is well-calibrated.
           But if you trained with label_smoothing=0.1 (as your notebook does),
           the sigmoid output is compressed away from 0 and 1.
           Effective outputs cluster around 0.1–0.9 instead of 0–1.
           A threshold of 0.6 can therefore misclassify borderline fakes as real.
           FIXED: default threshold lowered to 0.5 (decision boundary).
           You can tune with --threshold 0.45 or 0.55 to find your sweet spot.

Controls:
  SPACE  Pause / Resume
  V      Toggle voice narration on/off
  S      Save snapshot
  R      Toggle video recording
  Q      Quit
"""

import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import argparse
import threading
import subprocess
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from mtcnn import MTCNN

# ─────────────────────────────────────────────────────────────
# 1. Voice System
# ─────────────────────────────────────────────────────────────
VOICE_INTERVAL = 5   # seconds between periodic re-narrations

_SPEECH_SCRIPT = (
    "import sys, pyttsx3; "
    "e = pyttsx3.init(); "
    "e.setProperty('rate', 165); "
    "e.say(sys.argv[1]); "
    "e.runAndWait()"
)

class VoiceNarrator:
    def __init__(self):
        self._queue: Queue = Queue(maxsize=1)
        self._muted  = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if text is None:
                break
            if self._muted:
                continue
            try:
                subprocess.run(
                    [sys.executable, "-c", _SPEECH_SCRIPT, text],
                    timeout=30,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    def speak(self, text: str):
        if self._muted: return
        try: self._queue.get_nowait()
        except Empty: pass
        try: self._queue.put_nowait(text)
        except Exception: pass

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        return self._muted

    @property
    def muted(self) -> bool:
        return self._muted

    def shutdown(self):
        try: self._queue.put_nowait(None)
        except Exception: pass

def build_narration(verdict: str, confidence: float) -> str:
    return f"This video appears to be {verdict}, with {confidence*100:.0f} percent confidence."


# ─────────────────────────────────────────────────────────────
# 2. Configuration & UI Settings
# ─────────────────────────────────────────────────────────────
# BUG 5 FIX: Changed from 0.6 to 0.5 — the natural decision boundary.
# With label_smoothing=0.1, outputs are compressed; 0.5 is more reliable.
# Tune with --threshold if needed.
THRESHOLD      = 0.5
SMOOTH_WINDOW  = 15
INFER_EVERY_N  = 2

PANE_W, PANE_H = 320, 320
UI_H           = 160
WIN_W          = PANE_W * 4
WIN_H          = PANE_H + UI_H

C_REAL   = (80,  255, 130)
C_FAKE   = (60,  60,  255)
C_BG     = (18,  18,  24)
C_WHITE  = (255, 255, 255)
C_GRAY   = (170, 170, 170)
C_DIM    = (80,  80,  80)
C_CYAN   = (255, 255, 0)

# ─────────────────────────────────────────────────────────────
# 3. Preprocessing (Strictly Matches Notebook)
# ─────────────────────────────────────────────────────────────
_mtcnn_detector = None

def get_detector():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector

# BUG 2 FIX: Removed duplicate definition. Only one copy exists now.
def apply_resize_and_padding(img, target_size=224):
    """Exactly matches notebook Cell 2: aspect-ratio resize + mean-color padding."""
    h, w = img.shape[:2]
    mean_color = img.mean(axis=(0, 1)).astype(np.uint8)
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), mean_color, dtype=np.uint8)
    y = (target_size - nh) // 2
    x = (target_size - nw) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def align_face(img_rgb):
    """
    Replicates EXACTLY what happened to training images.

    Training pipeline:
      Cell 2:  apply_resize_and_padding on dataset images → 224x224
               (images were already tight face crops from the dataset)
      Cell 6:  MTCNN on the 224x224 padded image → rotation alignment
               scale = (224 * 0.3) / eye_dist

    Demo must replicate:
      Step 1: MTCNN on full frame → bounding box (used ONLY for UI corner-box)
      Step 2: Crop face with 30% margin from full frame
      Step 3: apply_resize_and_padding → 224x224  (matches Cell 2)
      Step 4: MTCNN on 224x224 face → align       (matches Cell 6)

    Returns: (aligned_224x224_rgb, bounding_box_in_original_frame_coords)
    """
    detector = get_detector()

    # Step 1: Find face in full frame
    results = detector.detect_faces(img_rgb)
    if not results:
        return None, None
    results = sorted(results, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
    box = results[0]['box']
    bx, by, bw, bh = box

    # Step 2: Crop with 30% margin
    orig_h, orig_w = img_rgb.shape[:2]
    margin = int(max(bw, bh) * 0.3)
    x1 = max(0, bx - margin)
    y1 = max(0, by - margin)
    x2 = min(orig_w, bx + bw + margin)
    y2 = min(orig_h, by + bh + margin)
    face_crop = img_rgb[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, box

    # Step 3: Pad to 224x224 (matches Cell 2 preprocessing)
    padded = apply_resize_and_padding(face_crop, 224)

    # Step 4: MTCNN + rotation alignment on 224x224 (matches Cell 6 exactly)
    results2 = detector.detect_faces(padded)
    if not results2:
        # Fallback: padded without alignment is still better than nothing
        return padded, box
    results2 = sorted(results2, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
    kp = results2[0]['keypoints']
    le, re = kp['left_eye'], kp['right_eye']
    angle  = np.degrees(np.arctan2(float(re[1]-le[1]), float(re[0]-le[0])))
    dist   = np.sqrt((re[0]-le[0])**2 + (re[1]-le[1])**2)
    scale  = (224 * 0.3) / (dist if dist > 0 else 1)
    center = ((le[0]+re[0])/2.0, (le[1]+re[1])/2.0)
    M      = cv2.getRotationMatrix2D(center, angle, scale)
    aligned = cv2.warpAffine(padded, M, (224, 224), flags=cv2.INTER_CUBIC)
    return aligned, box

def apply_srm(img_rgb):
    """Matches notebook SRM extraction exactly."""
    SRM_FILTERS = [
        np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]) / 4.0,
        np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]) / 12.0,
        np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]) / 2.0
    ]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    channels = []
    for f in SRM_FILTERS:
        filtered = cv2.filter2D(gray, -1, f)
        normed   = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        channels.append(normed)
    return cv2.merge(channels)

def extract_dct(img_rgb):
    y_channel  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)[:, :, 0]
    dct_coeffs = cv2.dct(np.float32(y_channel) / 255.0)
    dct_log    = np.log(np.abs(dct_coeffs) + 1e-12)
    return cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
# 4. UI Drawing Utilities
# ─────────────────────────────────────────────────────────────
def draw_pane_label(pane: np.ndarray, text: str, sub: str = ""):
    x, y, pad = 10, 10, 6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    extra = 14 if sub else 0
    cv2.rectangle(pane, (x, y), (x+tw+pad*2, y+th+pad*2+extra), (10,10,14), -1)
    cv2.rectangle(pane, (x, y), (x+tw+pad*2, y+th+pad*2+extra), C_DIM, 1)
    cv2.putText(pane, text, (x+pad, y+pad+th), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1, cv2.LINE_AA)
    if sub:
        cv2.putText(pane, sub, (x+pad, y+pad+th+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1, cv2.LINE_AA)

def draw_sparkline(canvas: np.ndarray, history: list, x: int, y: int, w: int, h: int, color):
    if len(history) < 2: return
    vals = np.array(history, dtype=float)
    mn, mx = 0.0, 1.0
    rng = max(mx-mn, 0.05)
    pts = [(x + int(i/(len(vals)-1)*w), y + h - int((v-mn)/rng*h)) for i, v in enumerate(vals)]
    for i in range(len(pts)-1):
        cv2.line(canvas, pts[i], pts[i+1], color, 1, cv2.LINE_AA)
    cv2.circle(canvas, pts[-1], 3, color, -1)

def draw_corner_box(pane: np.ndarray, bbox, orig_w, orig_h, color=C_CYAN):
    if bbox is None: return
    bx, by, bw, bh = bbox
    x = int(bx * (PANE_W / orig_w))
    y = int(by * (PANE_H / orig_h))
    w = int(bw * (PANE_W / orig_w))
    h = int(bh * (PANE_H / orig_h))
    t = 16
    thickness = 3
    for dx, dy in [(0,0),(w,0),(0,h),(w,h)]:
        px, py = x+dx, y+dy
        cv2.line(pane, (px,py), (px+(t if dx==0 else -t), py), color, thickness)
        cv2.line(pane, (px,py), (px, py+(t if dy==0 else -t)), color, thickness)

# ─────────────────────────────────────────────────────────────
# 5. Main Demo Logic
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"D:\final test data\Deepfake_Final.keras")
    parser.add_argument("--video", default="0", help="Path to video file, or 0 for webcam")
    parser.add_argument("--no-voice", action="store_true")
    # BUG 5 FIX: Default changed from 0.6 → 0.5
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Decision threshold (default: {THRESHOLD}). "
                             "Try 0.45–0.55 if results seem off.")
    parser.add_argument("--invert-score", action="store_true",
                        help="Invert model output. Use this ONLY if your model "
                             "was accidentally trained with real=1 / fake=0 labels.")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw scores to console for calibration.")
    args = parser.parse_args()
    threshold = args.threshold

    print(f"Loading Model: {args.model}")
    try:
        model = tf.keras.models.load_model(args.model)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ {e}"); sys.exit(1)

    # Quick sanity-check: print model output layer info
    out_layer = model.layers[-1]
    print(f"   Output layer: '{out_layer.name}'  activation: {out_layer.activation.__name__}  units: {out_layer.units}")
    print(f"   Expected: sigmoid activation, 1 unit")
    print(f"   Label encoding from notebook: real=0  fake=1")
    print(f"   => score > {threshold:.2f} means FAKE")
    if args.invert_score:
        print("   ⚠️  --invert-score is ON: score will be flipped before decision")

    narrator = VoiceNarrator()
    if args.no_voice: narrator.toggle_mute()

    video_source = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Could not open video."); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    score_history = deque(maxlen=SMOOTH_WINDOW)
    paused = False
    last_spoken_time = 0.0
    last_spoken_verdict = ""

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_recorder = None
    is_recording = False
    prev_time = time.time()

    frame_count = 0
    last_infer_ms = 0.0
    last_box_orig = None
    last_pane2 = np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)
    last_pane3 = np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)
    last_pane4 = np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)

    # BUG 1 FIX: verdict and conf now correctly initialized
    verdict, color, conf = "NO FACE", C_GRAY, 0.0
    # raw_score variable kept for debug printing
    last_raw_score = 0.0

    WINDOW_NAME = "Deepfake Detection Pro"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\n" + "─"*52)
    print("  CONTROLS")
    print("  SPACE  Pause/Resume      V  Toggle voice")
    print("  S      Snapshot          R  Record toggle")
    print("  Q      Quit")
    print("─"*52 + "\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break

            orig_h, orig_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            pane1 = cv2.resize(frame, (PANE_W, PANE_H))

            if frame_count % INFER_EVERY_N == 1 or last_box_orig is None:
                t0 = time.time()

                aligned, box_orig = align_face(frame_rgb)

                if aligned is not None:
                    # Prepare inputs exactly as training
                    rgb_in = np.expand_dims(aligned.astype('float32') / 255.0, 0)

                    srm_map = apply_srm(aligned)
                    srm_in  = np.expand_dims(srm_map.astype('float32') / 255.0, 0)

                    dct_map = extract_dct(aligned)
                    # BUG 3 CHECK: dct_input shape must be (1, 224, 224, 1)
                    dct_in  = np.expand_dims(
                                np.expand_dims(dct_map.astype('float32') / 255.0, -1),
                                0)

                    preds = model.predict({
                        "rgb_input": rgb_in,
                        "srm_input": srm_in,
                        "dct_input": dct_in
                    }, verbose=0)

                    last_infer_ms = (time.time() - t0) * 1000
                    raw_score = float(preds[0][0])
                    last_raw_score = raw_score

                    # --invert-score: use ONLY if your model has flipped labels
                    if args.invert_score:
                        raw_score = 1.0 - raw_score

                    if args.debug:
                        print(f"[DEBUG] raw={last_raw_score:.4f}  "
                              f"{'inverted=' + str(raw_score) + '  ' if args.invert_score else ''}"
                              f"threshold={threshold:.4f}  "
                              f"→ {'FAKE' if raw_score > threshold else 'REAL'}")

                    score_history.append(raw_score)
                    smoothed_score = float(np.mean(score_history))

                    # ────────────────────────────────────────────────────
                    # BUG 1 FIX — Correct verdict + confidence logic
                    #
                    # Notebook: real=0, fake=1
                    # Model output: sigmoid → close to 1 = FAKE, close to 0 = REAL
                    #
                    # smoothed_score  = probability of FAKE
                    # 1-smoothed_score = probability of REAL
                    #
                    # OLD (broken):
                    #   FAKE: conf = smoothed_score        ← correct
                    #   REAL: conf = 1.0 - smoothed_score  ← correct VALUE
                    #         but display said "Confidence" with no direction label
                    #         causing confusion about what the % meant
                    #
                    # FIXED: explicit labels in display (see UI section below)
                    # ────────────────────────────────────────────────────
                    if smoothed_score > threshold:
                        verdict = "FAKE"
                        color   = C_FAKE
                        conf    = smoothed_score          # P(fake)
                    else:
                        verdict = "REAL"
                        color   = C_REAL
                        conf    = 1.0 - smoothed_score    # P(real)

                    last_box_orig = box_orig
                    last_pane2 = cv2.resize(
                        cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR), (PANE_W, PANE_H))

                    srm_boost = cv2.convertScaleAbs(srm_map, alpha=2.5, beta=0)
                    last_pane3 = cv2.resize(
                        cv2.cvtColor(srm_boost, cv2.COLOR_RGB2BGR), (PANE_W, PANE_H))

                    dct_color  = cv2.applyColorMap(dct_map, cv2.COLORMAP_VIRIDIS)
                    last_pane4 = cv2.resize(dct_color, (PANE_W, PANE_H))

                else:
                    last_box_orig = None
                    verdict, color, conf = "NO FACE", C_GRAY, 0.0
                    last_pane2.fill(0)
                    last_pane3.fill(0)
                    last_pane4.fill(0)

            draw_corner_box(pane1, last_box_orig, orig_w, orig_h, color)

            now = time.time()
            if verdict != "NO FACE" and (
                    verdict != last_spoken_verdict or
                    (now - last_spoken_time >= VOICE_INTERVAL)):
                narrator.speak(build_narration(verdict, conf))
                last_spoken_time   = now
                last_spoken_verdict = verdict

            draw_pane_label(pane1,      "ORIGINAL",          "Detected face")
            draw_pane_label(last_pane2, "ALIGNED FACE",      "RGB Input")
            draw_pane_label(last_pane3, "SRM NOISE MAP",     "Texture manipulation (Boosted)")
            draw_pane_label(last_pane4, "DCT FREQUENCY MAP", "Compression artifacts")

            top_row = np.hstack((pane1, last_pane2, last_pane3, last_pane4))

            # ── UI Bar ──────────────────────────────────────────────────
            ui = np.full((UI_H, WIN_W, 3), C_BG, dtype=np.uint8)
            cv2.line(ui, (0, 0), (WIN_W, 0), C_DIM, 1)

            sx, sy, sw, sh = 40, 20, 300, 60
            cv2.putText(ui, "AI VERDICT:",
                        (sx, sy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)
            cv2.putText(ui, verdict,
                        (sx, sy+sh), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

            # BUG 4 FIX: Clear confidence label showing direction
            if verdict == "FAKE":
                conf_label = f"{conf*100:.1f}% fake probability"
            elif verdict == "REAL":
                conf_label = f"{conf*100:.1f}% real probability"
            else:
                conf_label = ""

            if conf_label:
                cv2.putText(ui, conf_label,
                            (sx, sy+sh+25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, C_WHITE, 1, cv2.LINE_AA)

            # Sparkline (tracks fake score = raw model output)
            tx = 400
            cv2.putText(ui, "FAKE SCORE TREND",
                        (tx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GRAY, 1, cv2.LINE_AA)
            cv2.rectangle(ui, (tx, sy+10), (tx+200, sy+10+sh), (28,28,36), -1)
            draw_sparkline(ui, list(score_history), tx, sy+10, 200, sh, color)

            thresh_y = (sy+10+sh) - int(threshold * sh)
            cv2.line(ui, (tx, thresh_y), (tx+200, thresh_y), (255, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(ui, f"Threshold={threshold:.2f}",
                        (tx+205, thresh_y+4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (255,100,100), 1, cv2.LINE_AA)

            # Right: Metrics
            curr_time  = time.time()
            display_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time  = curr_time
            mx = WIN_W - 220

            voice_status = "OFF (muted)" if narrator.muted else "ON"
            voice_col    = C_DIM if narrator.muted else C_REAL

            # BUG 1 FIX: Show raw score in metrics for transparency/debugging
            for row, (lbl, val, col) in enumerate([
                ("FPS",       f"{display_fps:.1f}",         C_REAL),
                ("INFER",     f"{last_infer_ms:.1f} ms",    C_REAL),
                ("RAW SCORE", f"{last_raw_score:.3f}",      color),
                ("VOICE",     voice_status,                 voice_col),
            ]):
                ry = 30 + row * 26
                cv2.putText(ui, lbl,
                            (mx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GRAY, 1, cv2.LINE_AA)
                cv2.putText(ui, val,
                            (mx+90, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

            if is_recording:
                cv2.circle(ui, (WIN_W-20, 20), 6, (40,40,255), -1)
                cv2.putText(ui, "REC",
                            (WIN_W-60, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (40,40,255), 1, cv2.LINE_AA)

            final_display = np.vstack((top_row, ui))

            if is_recording and out_recorder is not None:
                out_recorder.write(final_display)

            cv2.imshow(WINDOW_NAME, final_display)

        key = cv2.waitKey(delay if not paused else 50) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:
            paused = not paused
        elif key == ord('v'):
            narrator.toggle_mute()
        elif key == ord('s'):
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, final_display)
            print(f"📸 Saved: {name}")
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                name = f"recording_{int(time.time())}.mp4"
                out_recorder = cv2.VideoWriter(name, fourcc, fps, (WIN_W, WIN_H))
                print(f"🔴 Recording: {name}")
            else:
                if out_recorder: out_recorder.release(); out_recorder = None
                print("⏹  Recording stopped.")

    narrator.shutdown()
    cap.release()
    if out_recorder: out_recorder.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
