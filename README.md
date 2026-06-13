# 🧪 ML & Deep Learning Research Lab

A comprehensive collection of machine learning experiments, ranging from core statistical modeling to advanced generative networks and real-time object detection.

## 📂 Repository Structure

### 01-Classification-Visualisation
* **Titanic: Survival Analysis** — Binary classification with deep Exploratory Data Analysis (EDA) and feature engineering.
* **Digits (8x8):** Traditional multi-class classification using SVM and Random Forest.
* **Student Risk Simulation Pipeline** — Assessing the robustness of nonlinear classification models (Decision Tree/ID3) on controlled synthetic data. The pipeline simulates students' multivariate academic risks, injects stochastic noise into the target variable, and verifies the decision tree's ability to extract deterministic rules (Decision Rules) through entropy/information gain estimation.
<details>
<summary>📊 View pipeline and Decision Rules logs</summary>

```text
[INFO] Class distribution:
is_at_risk
1 0.543333
0 0.456667
Name: proportion, dtype: float64

=== Classification quality report ===
precision recall f1-score support

Stable 0.93 0.93 0.93 110
At risk 0.94 0.94 0.94 130

accuracy 0.93 240
macro avg 0.93 0.93 0.93 240
weighted avg 0.93 0.93 0.93 240

=== Feature Importance ===
attendance_pct: 0.4300
failed_credits: 0.3003
gpa: 0.2698
is_active_community: 0.0000

=== Extracted Decision Rules ===
|--- attendance_pct <= 39.50
| |--- gpa <= 4.71
| | |--- attendance_pct <= 34.50
| | | |--- gpa <= 2.14
| | | | |--- class: 1
| | | |--- gpa > 2.14
| | | | | |--- class: 1
| | |--- attendance_pct > 34.50
| | | |--- class: 1
| |--- gpa > 4.71
| | |--- attendance_pct <= 25.50
| | | |--- class: 1
| | |--- attendance_pct > 25.50
| | | |--- gpa <= 4.81
| | | | |--- class: 0
| | | |--- gpa > 4.81
| | | | |--- class: 1
|--- attendance_pct > 39.50
| |--- gpa <= 3.49
| | |--- failed_credits <= 1.50
| | | |--- attendance_pct <= 80.00
| | | | |--- class: 0
| | | |--- attendance_pct > 80.00
| | | | |--- class: 0
| | |--- failed_credits > 1.50
| | | |--- failed_credits <= 2.50
| | | | |--- class: 1
| | | |--- failed_credits > 2.50
| | | | |--- class: 1
| |--- gpa > 3.49
| | |--- attendance_pct <= 85.50
| | | |--- attendance_pct <= 71.50
| | | | |--- class: 0
| | | |--- attendance_pct > 71.50
| | | | |--- class: 0
| | |--- attendance_pct > 85.50
| | | |--- gpa <= 4.58
| | | | |--- class: 0
| | | |--- gpa > 4.58
| | | | |--- class: 0
```
</details>
### 02-ML-Core-MNIST-Regression
* **MNIST CNN Study:** Comparative analysis of optimizers (Adam vs. SGD) and architectural scalability.
* **MNIST CNN & Real-Time** Inference Study: Comparative analysis of optimizers and a real-time digit recognition pipeline via OpenCV webcam stream.
* **Auto MPG Prediction:** Multivariate regression analysis focused on fuel efficiency modeling.
* **Heart Disease Classification Pipeline** — Medical screening of cardiovascular risks. Implemented using `Sklearn Pipeline` and `ColumnTransformer`. The architecture completely isolates the training context from the test one (`train/test leakage prevention`), performs dynamic encoding of categorical features via `OneHotEncoder` with protection against unknown values ​​(`handle_unknown='ignore'`), and implements regularized logistic inference.

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