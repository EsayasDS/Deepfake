"""
model_utils.py — Backend for DeepShield web app.

CRITICAL: All preprocessing functions are copied EXACTLY from predict.py
which matches the training notebook. DO NOT modify these functions.

predict.py pipeline:
  Cell 2  -> resize + padding (BGR->RGB conversion happens inside)
  Cell 6  -> MTCNN alignment (simple: on the 224x224 padded image)
  Cell 8  -> DCT Y-channel
  Cell 11 -> SRM 3 filters on grayscale (normalize ALL channels together)
  Cell 13 -> score near 0 = FAKE, score near 1 = REAL
"""

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import os

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "Deepfake_Final.keras")
TARGET_SIZE = 224
THRESHOLD   = 0.4191  # From predict.py — optimal Youden index

detector = None
model = None

def load_models():
    global detector, model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("[SUCCESS] Model loaded!")
            print(f"   Inputs: {[inp.name for inp in model.inputs]}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise e
    if detector is None:
        detector = MTCNN()
        print("[SUCCESS] MTCNN ready!")


# ─────────────────────────────────────────────────────────────
# EXACT COPIES FROM predict.py — DO NOT MODIFY
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# EXACT COPIES FROM predict.py (For Image Demo)
# ─────────────────────────────────────────────────────────────

def apply_resize_and_padding(img_bgr, target_size=224):
    """Cell 2 — resize keeping aspect ratio, pad with mean color."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mean_color = img.mean(axis=(0,1)).astype(np.uint8)
    scale = target_size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), mean_color, dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas  # Returns RGB

def align_face_simple(img_rgb):
    """Cell 6 — MTCNN alignment. Used in predict.py for pre-cropped images."""
    global detector
    if detector is None:
        load_models()
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
    results = sorted(results, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
    kp = results[0]['keypoints']
    le, re = kp['left_eye'], kp['right_eye']
    angle = np.degrees(np.arctan2(float(re[1]-le[1]), float(re[0]-le[0])))
    dist  = np.sqrt((re[0]-le[0])**2 + (re[1]-le[1])**2)
    scale = (TARGET_SIZE * 0.3) / (dist if dist > 0 else 1)
    center = ((le[0]+re[0])/2.0, (le[1]+re[1])/2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img_rgb, M, (TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_CUBIC)

# ─────────────────────────────────────────────────────────────
# EXACT COPIES FROM deepfake_demo_v3.py (For Video Demo)
# ─────────────────────────────────────────────────────────────

def extract_and_align_face_video(img_bgr):
    """
    Two-stage extraction from deepfake_demo_v3.py for full video frames.
    """
    global detector
    if detector is None:
        load_models()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Step 1: Detect face in full frame
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
        
    results = sorted(results, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
    bx, by, bw, bh = results[0]['box']
    
    # Step 2: Crop face with 30% margin
    orig_h, orig_w = img_rgb.shape[:2]
    margin = int(max(bw, bh) * 0.3)
    x1, y1 = max(0, bx - margin), max(0, by - margin)
    x2, y2 = min(orig_w, bx + bw + margin), min(orig_h, by + bh + margin)
    face_crop = img_rgb[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
        
    # Step 3: Pad face crop to 224x224 (like predict.py Cell 2)
    h, w = face_crop.shape[:2]
    mean_color = face_crop.mean(axis=(0,1)).astype(np.uint8)
    scale = 224 / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(face_crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((224, 224, 3), mean_color, dtype=np.uint8)
    yo, xo = (224 - nh)//2, (224 - nw)//2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    
    # Step 4: MTCNN alignment on 224x224 (like predict.py Cell 6)
    res2 = detector.detect_faces(canvas)
    if not res2:
        return canvas  # Fallback to padded crop
        
    res2 = sorted(res2, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
    kp = res2[0]['keypoints']
    le, re = kp['left_eye'], kp['right_eye']
    angle = np.degrees(np.arctan2(float(re[1]-le[1]), float(re[0]-le[0])))
    dist  = np.sqrt((re[0]-le[0])**2 + (re[1]-le[1])**2)
    scale2 = (224 * 0.3) / (dist if dist > 0 else 1)
    center = ((le[0]+re[0])/2.0, (le[1]+re[1])/2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale2)
    return cv2.warpAffine(canvas, M, (224, 224), flags=cv2.INTER_CUBIC)

# ─────────────────────────────────────────────────────────────
# SHARED FEATURE EXTRACTION (Used by both)
# ─────────────────────────────────────────────────────────────

def extract_dct(img_rgb):
    y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)[:,:,0]
    dct = cv2.dct(np.float32(y) / 255.0)
    dct_log = np.log(np.abs(dct) + 1e-12)
    return cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_srm(img_rgb):
    """
    Cell 11 — 3 SRM filters on grayscale.
    Matches the FIXED notebook logic: each channel is normalized individually.
    """
    SRM_FILTERS = [
        np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],
                  [0,-1,2,-1,0],[0,0,0,0,0]], dtype=np.float32) / 4.0,
        np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
                  [2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32) / 12.0,
        np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
                  [0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 2.0
    ]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    channels = []
    for f in SRM_FILTERS:
        filtered = cv2.filter2D(gray, -1, f)
        normed = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        channels.append(normed)
    return cv2.merge(channels)

def build_inputs(aligned_rgb):
    rgb_in = np.expand_dims(aligned_rgb.astype(np.float32)/255.0, axis=0)
    srm_in = np.expand_dims(apply_srm(aligned_rgb).astype(np.float32)/255.0, axis=0)
    dct_in = np.expand_dims(
        np.expand_dims(extract_dct(aligned_rgb).astype(np.float32)/255.0, axis=-1), axis=0)
    return {"rgb_input": rgb_in, "srm_input": srm_in, "dct_input": dct_in}

# ─────────────────────────────────────────────────────────────
# PIPELINE 1: IMAGE DEMO API (predict.py)
# ─────────────────────────────────────────────────────────────

def process_and_predict_image_strict(img_bgr):
    """Strictly follows predict.py for cropped images."""
    if model is None:
        load_models()

    padded = apply_resize_and_padding(img_bgr, TARGET_SIZE)
    aligned = align_face_simple(padded)
    if aligned is None:
        aligned = padded

    inputs = build_inputs(aligned)
    score = float(model.predict(inputs, verbose=0)[0][0])

    # FIXED LOGIC: score > THRESHOLD means FAKE
    if score > THRESHOLD:
        verdict = "FAKE"
        confidence = score * 100
    else:
        verdict = "REAL"
        confidence = (1.0 - score) * 100

    return {"score": score, "verdict": verdict, "confidence": confidence, "threshold": THRESHOLD}

# ─────────────────────────────────────────────────────────────
# PIPELINE 2: VIDEO DEMO API (deepfake_demo_v3.py)
# ─────────────────────────────────────────────────────────────

def process_and_predict_video_frame(img_bgr):
    """Strictly follows deepfake_demo_v3.py for full video frames."""
    if model is None:
        load_models()

    aligned = extract_and_align_face_video(img_bgr)
    if aligned is None:
        raise Exception("No face detected in the video frame.")

    inputs = build_inputs(aligned)
    score = float(model.predict(inputs, verbose=0)[0][0])

    # FIXED LOGIC: score > THRESHOLD means FAKE
    if score > THRESHOLD:
        verdict = "FAKE"
        confidence = score * 100
    else:
        verdict = "REAL"
        confidence = (1.0 - score) * 100

    return {"score": score, "verdict": verdict, "confidence": confidence, "threshold": THRESHOLD}
