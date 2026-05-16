import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model_utils import (
    load_models,
    process_and_predict_image_strict,
    process_and_predict_video_frame
)

app = Flask(__name__)
# Enable CORS for all routes so the Vercel frontend can call this Render backend
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max-limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Pre-load the models when the server starts
print("Initializing models. This may take a moment...")
load_models()
print("Server ready!")

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "Deepfake Detection API is running.",
        "endpoints": ["/api/analyze/image", "/api/analyze/video"]
    })

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'webp'}):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read image
            img_bgr = cv2.imread(filepath)
            if img_bgr is None:
                return jsonify({'error': 'Failed to read image'}), 400
                
            # Process and predict using exact predict.py logic
            result = process_and_predict_image_strict(img_bgr)
            
            return jsonify({
                'success': True,
                'type': 'image',
                'result': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename, {'mp4', 'avi', 'mov', 'webm'}):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        cap = None
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video'}), 400
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample up to 5 frames evenly distributed across the video
            num_samples = min(5, max(1, total_frames))
            frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
            scores = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    try:
                        # Process using exact deepfake_demo_v3.py logic
                        res = process_and_predict_video_frame(frame)
                        scores.append(res['score'])
                    except Exception as e:
                        # Skip frames without faces
                        pass
                        
            if not scores:
                return jsonify({'error': 'Failed to extract frames or no faces were detected in the video.'}), 400
                
            avg_score = sum(scores) / len(scores)
            
            from model_utils import THRESHOLD
            # FIXED LOGIC: score > THRESHOLD means FAKE
            if avg_score > THRESHOLD:
                verdict = "FAKE"
                confidence = avg_score * 100
            else:
                verdict = "REAL"
                confidence = (1.0 - avg_score) * 100
                
            return jsonify({
                'success': True,
                'type': 'video',
                'result': {
                    'score': avg_score,
                    'verdict': verdict,
                    'confidence': confidence,
                    'threshold': THRESHOLD,
                    'frames_analyzed': len(scores)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if cap is not None:
                cap.release()
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
