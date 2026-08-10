 Deepfake Detection System

A computer vision research project developed by Esayas Melaku at Debre Birhan University.

This project investigates deepfake detection using a multi-branch CNN architecture that combines spatial appearance features with frequency-domain and noise-residual information. The system is designed to classify facial images as real or AI-generated (deepfake).

> Important: The reported performance reflects evaluation on the project's held-out test set and should not be interpreted as universal real-world deepfake detection performance. Deepfake detection models can generalize differently to manipulation techniques, datasets, image qualities, and generation methods that were not represented during training.

 Overview
Deepfake generation techniques have advanced rapidly, creating serious concerns about the authenticity of digital media. As manipulated images become increasingly realistic, conventional visual inspection becomes less reliable.

This project explores a multi-branch deep learning framework that analyzes three complementary facial representations:

 1. RGB Branch — EfficientNetB3

Extracts high-level semantic and visual features from RGB facial images, including:

* skin texture
* lighting inconsistencies
* facial blending artifacts
* other spatial appearance patterns

 2. SRM Noise Residual Branch — 4-Layer CNN

Uses three fixed 5×5 high-pass filters derived from the Spatial Rich Model (SRM).

The filters suppress much of the semantic image content and emphasize subtle noise residuals that may contain forensic fingerprints associated with image manipulation.

 3. DCT Frequency Branch — 3-Layer CNN

Applies the Discrete Cosine Transform (DCT) to the Y-channel luminance information to analyze frequency-domain characteristics.

This branch is designed to capture abnormal frequency patterns that may differ from those commonly observed in natural photographs.

 Feature Fusion

The three branches are combined through dense layers to produce the final binary classification prediction:

                 Input Face Image
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
     RGB Branch     SRM Branch    DCT Branch
    EfficientNetB3   CNN + SRM     CNN + DCT
          │             │             │
          └─────────────┼─────────────┘
                        ↓
                 Feature Fusion
                        ↓
                Final Classifier
                        ↓
                 REAL / FAKE

Performance

The model was evaluated on 3,776 held-out test samples using a stratified train/test split.

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | 82.94% |
| AUC-ROC  | 92.12% |

The training dataset contained approximately 19,000 images, compiled from four sources:

* FFHQ
* Celeb-DF v2
* FaceForensics++
* SFHQ

The dataset was balanced approximately 50/50 between real and fake samples.

These results demonstrate promising performance on the project's evaluation data, while the model's ability to generalize to unseen deepfake generation methods and real-world media remains an important limitation and area for further research.

 Repository Structure

The repository separates the presentation layer from the inference backend, allowing the components to be developed and deployed independently.

Deepfake-Detection/
├── frontend/
│   ├── index.html
│   └── static/
│  
├── backend/
│   ├── app.py
│   ├── model_utils.py
│   └── requirements.txt
├── models/
│   └── Deepfake_Final.keras
├── notebooks/
│   └── deepfake-detection-final-model.ipynb
└── docs/
    └── deepfake_documentation.pdf

 Running Locally

 1. Start the Backend API

The backend requires Python 3.9+ and TensorFlow.

bash
cd backend
pip install -r requirements.txt
python app.py
```

The API runs locally at:

```text
http://127.0.0.1:5000
```

The first startup may take some time because the application needs to load the trained Keras model and the MTCNN face detector.

 2. Open the Frontend

The frontend is a static web application.

You can open:

```text
frontend/index.html
```

directly in a browser.

Alternatively, use Python's built-in HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Then open:

```text
http://localhost:3000
```

## Limitations

This project should be considered a **research and portfolio project**, not a universal forensic detection system.

The main limitations include:

* The model was trained on approximately 19,000 images.
* The training data came from four specific datasets.
* Performance can vary on unseen datasets and newer deepfake generation techniques.
* The reported 82.94% accuracy and 92.12% AUC-ROC are specific to the held-out evaluation set.
* Real-world media can differ substantially in resolution, compression, lighting, face pose, and manipulation technique.
* Further evaluation on larger and more diverse datasets would be required before making claims about broad real-world generalization.

## Documentation

For a detailed explanation of the architecture, dataset, training methodology, experiments, evaluation, and project development, see:

**[Deepfake Project Documentation](docs/deepfake_documentation.pdf)**

## Academic Context

**Debre Birhan University**
Department of Data Science
Computer Vision Project — May 2026

**Developer:** Esayas Melaku
