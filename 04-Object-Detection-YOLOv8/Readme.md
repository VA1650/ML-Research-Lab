# 🎭 Face Mask Detection: Custom YOLOv8 Pipeline

This project demonstrates a complete end-to-end Computer Vision workflow: from manual data annotation to training a high-precision real-time object detector using **YOLOv8**.

## 🎯 Objective
The goal was to build a robust system capable of identifying individuals with and without face masks. The project focuses on data quality and the impact of **Data Augmentation** on model generalization.

---

## 🛠 Methodology & Training

### 1. Data Engineering
* **Custom Annotation:** Manually labeled a dataset using classes: `with_mask` and `without_mask`.
* **Augmentation Strategy:** Implemented a pipeline to increase model robustness against varied lighting, rotations, and noise.
* **Dataset Split:** 100% custom-built and verified data.

### 2. Model Configuration
* **Architecture:** YOLOv8 Nano (optimized for real-time edge inference).
* **Hardware:** Trained on **NVIDIA GeForce RTX 4060 Laptop GPU** (8GB VRAM).
* **Environment:** CUDA 12.8, PyTorch 2.9.0.
* **Training Depth:** 100 Epochs.

---

## 📊 Performance Analysis

The model achieved exceptional results, demonstrating the effectiveness of transfer learning and custom data curation.

| Metric | Baseline Training | With Augmentation |
| :--- | :--- | :--- |
| **mAP50** | 0.963 | **0.975** |
| **Recall (R)** | 0.952 | **0.967** |
| **Precision (P)** | 0.961 | 0.950 |
| **mAP50-95** | 0.671 | 0.641 |

### Class-Specific Metrics (Augmented Model):
* **`with_mask`**: 0.955 mAP50
* **`without_mask`**: 0.995 mAP50 (near-perfect recognition)

### ⚡ Inference Speed (Real-time Benchmarks)
* **Inference:** 8.4ms
* **Total Pipeline (End-to-End):** ~12ms per frame
* **Estimated FPS:** ~80-100 FPS (Production-ready for real-time streams).

---

## 🚀 Key Insights
* **Augmentation Impact:** Applying data augmentation increased the **Recall** and **mAP50**, making the model more reliable in diverse environments.
* **Efficiency:** The model was successfully compressed (stripped weights ~6.3MB), making it ideal for deployment on mobile or IoT devices.

---

## 📂 Project Structure
* `weights/` - Contains `best.pt` and `last.pt` trained weights.
* `data/` - Dataset configuration and sample images.
* `YOLOv8_Training.ipynb` - Full training logs and validation code.

## 🚀 How to Reproduce Training

The model was trained using the Ultralytics CLI. To replicate the results, ensure you have the environment set up and run the following commands:

### 1. Baseline Model (No Augmentation)
```bash
yolo train model=yolov8n.pt \
  data="dataset/dataset.yaml" \
  epochs=100 imgsz=640 batch=16 \
  name=my_custom_yolov8_model \
  device=0
  ```
 
### 2. Augmented Model (High Robustness)

I used a custom augmentation policy to improve model stability against varied conditions:
```bash

yolo train model=yolov8n.pt \
  data="dataset/dataset.yaml" \
  epochs=100 imgsz=640 batch=16 \
  name=my_custom_yolov8_model_aug \
  hsv_h=0.05 hsv_s=1.0 hsv_v=0.7 \
  degrees=15.0 translate=0.2 scale=0.5 \
  shear=5.0 fliplr=0.5 mosaic=1.0 mixup=0.1 \
  device=0 ```