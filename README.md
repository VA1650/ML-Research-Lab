# 🧪 ML & Deep Learning Research Lab

A comprehensive collection of machine learning experiments, ranging from core statistical modeling to advanced generative networks and real-time object detection.

## 📂 Repository Structure

### 01-Classification-Visualisation
* **Titanic: Survival Analysis** — Binary classification with deep Exploratory Data Analysis (EDA) and feature engineering.
* **Digits (8x8):** Traditional multi-class classification using SVM and Random Forest.

### 02-ML-Core-MNIST-Regression
* **MNIST CNN Study:** Comparative analysis of optimizers (Adam vs. SGD) and architectural scalability.
* **Auto MPG Prediction:** Multivariate regression analysis focused on fuel efficiency modeling.

### 03-Computer-Vision-Basics
* Fundamental image processing experiments and basic spatial transformations.

### 04-Object-Detection-YOLOv8
* **Face Mask Detector:** Full-cycle pipeline: manual annotation -> augmentation -> training on RTX 4060.
* *Metrics:* **0.975 mAP50**, ~100 FPS inference.

### 05-RNN-Legacy-Research
* Sequence modeling with LSTM/GRU (Weather forecasting, Eq-solvers).
* **Hall of Fame:** A collection of surrealist AI-generated text artifacts.
* ![Stoybe Artifact](./05-RNN-Legacy-Research/rnn_chaos.jpg)
* *Internal Note: "Critical Error stoybe deteted"*

### 06-Generative-Adversarial-Networks
* **GAN Lab:** Synthesis of medical masks on faces and automated **Minecraft Skin generation** (64x64 pixel art).

---

## 🛠 Tech Stack
* **Frameworks:** PyTorch, TensorFlow/Keras, Ultralytics (YOLOv8).
* **Libraries:** Scikit-learn, OpenCV, Pandas, NumPy, Matplotlib.
* **Hardware:** Optimized for CUDA-enabled NVIDIA GPUs (RTX 40-series).

---

## 🚀 Key Research Insights
* **Augmentation Impact:** Custom spatial and HSV augmentations increased YOLO recall by ~1.5%.
* **Optimizer Efficiency:** Adam demonstrated 30% faster convergence in MNIST tasks compared to momentum-based SGD.
* **Generative Domain Adaptation:** Successfully adapted DCGAN architectures for low-resolution pixel-art asset generation.

---
*Developed as part of a deep dive into AI and neural architectures. All stoybes were handled with care.*