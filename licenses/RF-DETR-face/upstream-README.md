---
license: apache-2.0
tags:
- computer-vision
- object-detection
- rf-detr
- face-detection
datasets:
- fareselmenshawii/face-detection-dataset
metrics:
- mAP
---

# Model Card for RFDETR-Medium - Face Detection Finetune

This model is a fine-tuned version of **RFDETR-Medium** (Real-time Detection Transformer) specifically optimized for the detection of human faces. It leverages a DINOv2 backbone for high-quality feature extraction and a transformer-based head for NMS-free, end-to-end detection.

## Model Details

### Model Description

- **Model type:** Object Detection
- **Task:** Human Face Detection
- **Finetuned from model:** RFDETR-Medium
- **Language(s):** N/A (Computer Vision)
- **License:** Apache 2.0

---

## Training Details

### Training Hyperparameters
The following hyperparameters were used during the fine-tuning process:

* **Epochs:** 20
* **Batch Size:** 16
* **Learning Rate:** $1 \times 10^{-4}$
* **Input Image Size:** $576 \times 576$

### Evalutation Results

* **mAP@50:** 0.9
* **mAP@5095:** 0.6

### Training Data
The model was trained on the Face Detection Dataset (Kaggle), which contains approximately 16,700 images with bounding box annotations for human faces.

### Model Sources

- **Repository:** [Roboflow RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- **Dataset:** [Face Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset/data)

## Uses

### Direct Use
This model is designed for **real-time human face detection** in images and video streams.

## How to Get Started with the Model

```python
from rfdetr import RFDETRMedium

# Load your fine-tuned weights
model = RFDETRMedium(device="cuda", pretrain_weights="rfdetr_medium_face.pth")
model.optimize_for_inference()

# Run inference
results = model.predict("input_image.jpg", threshold=0.5)