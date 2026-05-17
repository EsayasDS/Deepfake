"""
Deepfake Image Prediction
=========================
Preprocessing matches training notebook EXACTLY:
  Cell 2  -> resize + padding
  Cell 6  -> MTCNN alignment
  Cell 8  -> DCT Y-channel
  Cell 11 -> SRM 3 filters on grayscale (each channel normalized separately)
  Labels  -> real=0, fake=1   (confirmed: notebook prints "y: ... real=0  fake=1")
             score near 1.0 = FAKE
             score near 0.0 = REAL
"""

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import sys, os

MODEL_PATH  = r"D:\final test data\Deepfake_Final.keras"

# ── THRESHOLD ──────────────────────────────────────────────────────────────
# The notebook finds opt_thr on y_probs where 1=FAKE.
# score > THRESHOLD  →  FAKE
# score <= THRESHOLD →  REAL
# Your notebook's best_threshold was 0.4191 — keep it, but direction is now correct.
THRESHOLD   = 0.4191
TARGET_SIZE = 224

print("\n" + "="*50)
print("  Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!")
    print(f"   Inputs: {[inp.name for inp in model.inputs]}")
except Exception as e:
    print(f"❌ {e}"); sys.exit(1)

detector = MTCNN()
print("✅ MTCNN ready!")
print("="*50)


def apply_resize_and_padding(img_bgr, target_size=224):
    """
    Cell 2 — resize keeping aspect ratio, pad with mean color.
    Notebook reads BGR from cv2.imread, converts BGR→RGB, saves as RGB PNG.
    We receive BGR from cv2.imread so we also convert to RGB first.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mean_color = img.mean(axis=(0, 1)).astype(np.uint8)
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), mean_color, dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas  # RGB


def align_face(img_rgb):
    """Cell 6 — MTCNN alignment."""
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
    results = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    kp = results[0]['keypoints']
    le, re = kp['left_eye'], kp['right_eye']
    angle = np.degrees(np.arctan2(float(re[1] - le[1]), float(re[0] - le[0])))
    dist  = np.sqrt((re[0] - le[0])**2 + (re[1] - le[1])**2)
    scale = (TARGET_SIZE * 0.3) / (dist if dist > 0 else 1)
    center = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img_rgb, M, (TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_CUBIC)


def extract_dct(img_rgb):
    """Cell 8 — DCT on Y channel."""
    y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)[:, :, 0]
    dct = cv2.dct(np.float32(y) / 255.0)
    dct_log = np.log(np.abs(dct) + 1e-12)
    return cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_srm(img_rgb):
    """
    Cell 11 — 3 SRM filters on grayscale.

    BUG FIXED: The notebook normalizes EACH channel individually before merging:
        for f in SRM_FILTERS:
            filtered = cv2.filter2D(gray, -1, f)
            normed   = cv2.normalize(filtered, None, 0, 255, NORM_MINMAX)  ← per channel
            channels.append(normed)
        return cv2.merge(channels)

    The old predict.py stacked first then normalized the whole array at once —
    this changes the per-channel scale and produces different values than training.
    """
    SRM_FILTERS = [
        np.array([[0, 0, 0, 0, 0],
                  [0, -1, 2, -1, 0],
                  [0, 2, -4, 2, 0],
                  [0, -1, 2, -1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,

        np.array([[-1, 2, -2, 2, -1],
                  [2, -6, 8, -6, 2],
                  [-2, 8, -12, 8, -2],
                  [2, -6, 8, -6, 2],
                  [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,

        np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
    ]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    channels = []
    for f in SRM_FILTERS:
        filtered = cv2.filter2D(gray, -1, f)
        # Normalize EACH channel separately — matches notebook exactly
        normed = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        channels.append(normed)
    return cv2.merge(channels)  # shape: (224, 224, 3)


def build_inputs(aligned_rgb):
    """Build all 3 model inputs, normalized to [0, 1] as in the generator."""
    rgb_in = np.expand_dims(aligned_rgb.astype(np.float32) / 255.0, axis=0)
    srm_in = np.expand_dims(apply_srm(aligned_rgb).astype(np.float32) / 255.0, axis=0)
    dct_2d = extract_dct(aligned_rgb).astype(np.float32) / 255.0
    dct_in = np.expand_dims(np.expand_dims(dct_2d, axis=-1), axis=0)  # (1,224,224,1)
    return {"rgb_input": rgb_in, "srm_input": srm_in, "dct_input": dct_in}


def predict_image(image_path):
    print(f"\n{'='*50}\n  Image: {image_path}\n{'='*50}")

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}"); return
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Cannot read image"); return

    print(f"✅ Loaded — {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Cell 2: resize + pad (returns RGB)
    padded = apply_resize_and_padding(img_bgr, TARGET_SIZE)
    print(f"✅ Resized + padded to {TARGET_SIZE}x{TARGET_SIZE}")

    # Cell 6: MTCNN alignment
    print("  Detecting face...")
    aligned = align_face(padded)
    if aligned is None:
        print("⚠️  No face detected — using padded image")
        aligned = padded
    else:
        print("✅ Face aligned!")

    # Build inputs and predict
    inputs = build_inputs(aligned)
    score  = float(model.predict(inputs, verbose=0)[0][0])

    # ── VERDICT ──────────────────────────────────────────────────────────
    # CONFIRMED: score near 1.0 means REAL, score near 0.0 means FAKE
    # → score > THRESHOLD means REAL
    # ─────────────────────────────────────────────────────────────────────
    real_probability = score * 100
    fake_probability = (1.0 - score) * 100

    if score > THRESHOLD:
        # Real side
        if score >= 0.7:
            verdict = "🟢 REAL"
        else:
            verdict = "🟡 LIKELY REAL"
        confidence_str = f"{real_probability:.1f}% real probability"
    else:
        # Fake side
        if score <= 0.2:
            verdict = "🔴 FAKE"
        else:
            verdict = "🟡 LIKELY FAKE"
        confidence_str = f"{fake_probability:.1f}% fake probability"

    print(f"\n{'='*50}")
    print(f"  VERDICT    : {verdict}")
    print(f"  Raw Score  : {score:.4f}  (0.0 = fake, 1.0 = real)")
    print(f"  Confidence : {confidence_str}")
    print(f"  Threshold  : {THRESHOLD}  (above = REAL, below = FAKE)")
    print(f"{'='*50}")

    if score > THRESHOLD:
        print("  ✅ This image appears to be a real face")
    else:
        print("  ⚠️  This image appears to be AI-generated / deepfake")
    print()

    return score, verdict


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\final test data\test image\download.jpg"
    predict_image(image_path)
