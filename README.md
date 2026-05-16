# Deepfake Detection System

A Computer Vision research project developed by **Debre Birhan University, Department of Data Science**.

This project provides an AI-powered forensic media scanner capable of accurately classifying images and videos as real or AI-generated (deepfakes). The system is built using a **Multi-Branch CNN architecture**, combining spatial appearance features and frequency domain artifacts.

## Overview

Deepfake generation techniques have advanced rapidly, creating serious concerns regarding the authenticity of digital media. As manipulated images and videos become increasingly realistic, conventional visual inspection is no longer sufficient for reliable detection.

Our multi-branch deep learning framework combines three complementary facial representations:
1. **RGB Branch (EfficientNetB3)**: Extracts high-level semantic facial features including skin texture, lighting inconsistencies, and blending artifacts.
2. **SRM Noise Residuals (4-Layer CNN)**: Three fixed 5x5 high-pass filters from the Spatial Rich Model suppress semantic content and expose subtle noise fingerprints (e.g., GAN generation traces).
3. **DCT Frequency (3-Layer CNN)**: The Discrete Cosine Transform applied on the Y-channel luminance reveals unnatural frequency distributions that violate the natural 1/f² decay pattern found in real photographs.

These branches are fused through dense layers to make a highly accurate final prediction.

## Performance Metrics

Evaluated on 3,776 held-out test samples with a stratified 80/20 split, the model achieves state-of-the-art performance:

* **AUC-ROC:** 92.12%
* **Accuracy:** 82.94%

The training dataset comprised **19,000 images** compiled from 4 distinct sources (FFHQ, Celeb-DF v2, FaceForensics++, SFHQ) perfectly balanced 50/50 between real and fake samples.

## Repository Structure

The repository is structured to separate the presentation layer (frontend) from the heavy inference engine (backend) to allow for independent cloud deployment (e.g., Vercel + Render).

```text
Deepfake/
├── frontend/             # Vercel-ready static UI
│   ├── index.html
│   └── static/           # CSS, JS, and image assets
├── backend/              # Render-ready Flask API
│   ├── app.py            # Main API routing (CORS enabled)
│   ├── model_utils.py    # Deepfake inference pipelines
│   └── requirements.txt  # Python dependencies
├── models/
│   └── Deepfake_Final.keras # Final trained multi-branch model
├── notebooks/
│   └── deepfake-detection-final-model.ipynb # Complete training pipeline
└── docs/
    └── deepfake_documentation.pdf # Academic documentation
```

## Running Locally

To run the application on your local machine, you will need to start the Flask backend and open the static frontend.

### 1. Start the Backend API
The backend requires Python 3.9+ and TensorFlow.
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*Note: The API runs on `http://127.0.0.1:5000` by default. The first run may take a moment to initialize the 88MB Keras model and MTCNN detector.*

### 2. Open the Frontend
Since the frontend is purely static, you can simply open `frontend/index.html` in your web browser. Or, for a better experience, serve it using Python's built-in HTTP server:
```bash
cd frontend
python -m http.server 3000
```
Then navigate to `http://localhost:3000` in your browser.

## Cloud Deployment Guide

* **Frontend (Vercel)**: Simply connect your GitHub repository to Vercel, set the Root Directory to `frontend`, and deploy. Before deploying, ensure you update the `BACKEND_URL` variable in `frontend/static/script.js` to point to your deployed backend.
* **Backend (Render)**: Connect your repository to Render, create a new "Web Service", set the Root Directory to `backend`, set the build command to `pip install -r requirements.txt`, and the start command to `gunicorn app:app`.

## Research Team

**Debre Birhan University**  
Department of Data Science  
Computer Vision Project — May 2026  
Advisor: **Abey Bekele**

* Esayas Melaku
* Haymanot Tadese
* Melis Melakie
* Yosef Getaneh
* Genet Minda
* Kenean Bisrat
* WudasieMaryam Muluken
* Demirewu Manidefro
