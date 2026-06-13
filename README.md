# 🧪 ML & Deep Learning Research Lab

A comprehensive collection of machine learning experiments, ranging from core statistical modeling to advanced generative networks and real-time object detection.

## 📂 Repository Structure

### 01-ML-Core-Data-Analysis-Classification-Regression
* **Titanic: Survival Analysis** — Binary classification with deep Exploratory Data Analysis (EDA) and feature engineering.
* **Digits (8x8):** Traditional multi-class classification using SVM and Random Forest.
* **University Rankings EDA Engine (CWUR)** — A research pipeline for macro-analysis of global education quality. The script aggregates institutional metrics by country, filters time slices, and constructs inverted multivariate scatter plots, where the spatial scale of the points is tied to the final ranking score. The pipeline includes automatic calculation of the nonlinear Spearman's rank correlation to verify latent relationships between the influence of the academic environment and teaching quality.
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

### 02-ML-Core-Pipelines
* **MNIST CNN Study:** Comparative analysis of optimizers (Adam vs. SGD) and architectural scalability.
* **MNIST CNN & Real-Time** Inference Study: Comparative analysis of optimizers and a real-time digit recognition pipeline via OpenCV webcam stream.
* **Auto MPG Prediction:** Multivariate regression analysis focused on fuel efficiency modeling.
* **Heart Disease Classification Pipeline** — Medical screening of cardiovascular risks. Implemented using `Sklearn Pipeline` and `ColumnTransformer`. The architecture completely isolates the training context from the test one (`train/test leakage prevention`), performs dynamic encoding of categorical features via `OneHotEncoder` with protection against unknown values ​​(`handle_unknown='ignore'`), and implements regularized logistic inference.
* **Student Performance Multi-Output Pipeline** — Forecasting the vector of student academic performance across three disciplines in parallel. The pipeline is built on the MultiOutputRegressor and linear models with L2 regularization (`Ridge`). Strict protection against the `Dummy Variable Trap` is implemented at the one-hot coding stage, feature scaling is performed, and a synchronous multivariate inference mechanism is deployed.
* **Manifold Learning & Non-linear Dimension Reduction Study** — An experimental study of manifold topology based on the Isomap algorithm and the KNN classifier. The pipeline demonstrates the difference between metrics in the full 4D feature space and on an isolated 2D manifold. Strict separation of transformation phases (`fit_transform` on the train, `transform` on the test) is implemented to prevent data leakage through graph geodesic distances.
* **YouTube Factor Analysis Pipeline (PCA)** — Factor analysis and dimensionality reduction of multidimensional media platform metrics. The script isolates highly correlated economic and social channel metrics (views, revenue, subscriptions), eliminates scale imbalances using StandardScaler, and projects data onto orthogonal principal component axes. Includes calculation of a loadings matrix for the mathematical interpretation of latent factors.
* **Financial Time Series & Risk Analysis (VaR)**: A Basic Implementation of Market Risk Assessment for Cryptoassets. Critical Insight: Applying a normal (Gaussian) distribution to volatile assets (BTC) has been proven ineffective due to the presence of "fat tails." Historical simulation is recognized as the optimal method.

### 03-Computer-Vision-Basics
* Fundamental image processing experiments and basic spatial transformations.

### 04-Object-Detection-YOLOv8
* **Face Mask Detector:** Full-cycle pipeline: manual annotation -> augmentation -> training on RTX 4060.
* *Metrics:* **0.975 mAP50**, ~100 FPS inference.

### 05-RNN-Legacy-Research
* Sequence modeling with LSTM/GRU (Weather forecasting, Eq-solvers).
* **Hall of Fame:** A collection of surrealist AI-generated text artifacts.
 ![Stoybe Artifact](./05-RNN-Legacy-Research/rnn-chaos.jpg)
* *Internal Note: "Critical Error stoybe deteted"*

### 06-Generative-Adversarial-Networks
* **GAN Lab:** Synthesis of medical masks on faces and automated **Minecraft Skin generation** (64x64 pixel art).

### 07-Nonlinear-Dynamics-Optimization
* **Nonlinear Oscillator Numerical Integration & Gradient Optimization Study** — A study of the convergence of optimization methods on signals of nonlinear dynamic systems.
* *Process Physics:* The script models the trajectory $x(t)$ of forced oscillations of a nonlinear oscillator (van der Pol / Duffing scheme with cubic damping nonlinearity $\sim L(1-x^2)$) using a second-order discrete difference scheme (Verlet method). The system is subjected to an external harmonic force with a frequency $\omega$.
* *Optimization Mathematics:* A multiparameter regression problem is implemented to approximate a chaotic signal with a trigonometric polynomial. Two custom optimization engines, written in pure NumPy without using Autograd (PyTorch/TensorFlow), were deployed:
1. Full-Batch Gradient Descent (GD) — calculating the exact analytical gradient along the entire trajectory (500 epochs per iteration).
2. Stochastic Gradient Descent (SGD) — stochastic descent with gradient vector estimation at a single random time point per step, minimizing algorithmic complexity to O(N) \to O(1).
* *Research Insight:* Benchmarking of the convergence time and stability of gradient steps was performed under conditions of a highly non-convex loss function landscape caused by the trigonometric dependence on the optimized frequencies $\omega_1, \omega_2$.

### 08-Stat-Hypothesis-Testing
* **Binomial Hypothesis & Power Analysis Pipeline** — A toolkit for product analytics and experimental validation.
* *Features:* Implements the exact binomial test (replacing deprecated methods via scipy.stats.binomtest) and calculates true statistical power ($1-\beta$).
* *Practical Application:* Automatically estimates the "breakthrough power" of tests for a given sample size. Avoids "underfitting" errors when the sample size is too small to detect a real change in the metric (effect).
* *Research Insight:* The module visualizes the gap between statistical significance ($p$-value) and the power of a test. Proves that for a fixed $N=100$, to detect small effects (skill $1/5 \to 1/7$) one must either increase the sample or accept a high risk of type II error.

### 09-Optimization-Lab
* **Gradient Descent Step-Size Strategies Study** — A comparative analysis of the effectiveness of step size selection methods ($\alpha$) in convex optimization problems.
* *Methods:* Three convergence strategies are implemented:
1. **Constant Step:** A basic method requiring knowledge of the Lipschitz constant ($L$).
2. **Polyak Step Size:** An adaptive method using knowledge of the optimal value of $f^*$, ensuring the fastest convergence.
3. **Armijo Line Search:** An iterative step size search method that guarantees a "sufficiently reduced" function condition without requiring knowledge of the function's curvature parameters.
* *Visualization:* The pipeline includes the generation of contour plots of optimization trajectories and the plotting of convergence curves on a logarithmic scale to monitor the rate of gradient decay.

### 10-Ridge-Regression-Benchmark
* **Ridge Regression: Analytical vs. Iterative Optimization** — A comparative study of methods for solving a linear regression problem with L2 regularization.
* *Implemented methods:*
1. **Analytical (Normal Equation):** Direct matrix solution via the pseudoinverse matrix.
2. **Standard Gradient Descent:** Iterative minimization with direct calculation of the penalty gradient.
3. **Proximal Gradient Descent:** Using the proximal operator (soft-thresholding for Ridge) for more efficient weight updates.
* *Research Insight:* The code demonstrates the equivalence of iterative methods to the analytical solution and visualizes the convergence of the loss function. The implementation allows us to see how the regularization parameters $\lambda$ affect weight shrinkage and overfitting prevention.

### 11-Sparsity-Optimization-Lab
* **Sparse Optimization & LASSO Benchmarking** — Study of optimization methods for functions with nonsmooth terms ($L_1$-regularization).
* *Implemented approaches:*
1. **Subgradient Method:** Direct use of the subgradient for a nonsmooth function (slow convergence).
2. **Proximal Gradient Descent (ISTA):** Using the proximal operator (Soft-Thresholding) to efficiently achieve sparsity.
3. **Frank-Wolfe (Conditional Gradient):** Optimization on an $L_1$-ball via linear minimization, ideal for problems with constraints.
* *Research Insight:* Visual comparison of how each method recovers the original sparse coefficient vector. It has been demonstrated that the Proximal method outperforms the Subgradient method in terms of convergence rate and quality of feature structure recovery.

### 12-Advanced-Optimization-Benchmarking
* **Comparative Optimizer Suite** — A benchmark for testing first- and second-order algorithms.
* *Implemented methods:*
1. **GD with Wolfe Line Search:** Gradient descent with an adaptive step size that guarantees the Wolfe conditions.
2. **Newton's Method:** Second-order optimization using the direct Hessian inverse.
3. **BFGS (Manual Implementation):** Implementation of the quasi-Newton method with inverse Hessian updating.
4. **SciPy Benchmarking:** Comparison of manual implementations with optimized solvers (`BFGS`, `L-BFGS-B`).
* *Research Insight:* The benchmark clearly demonstrates that in high-dimensional problems (Log-Barrier, N=100), second-order and quasi-Newton methods show exponentially greater efficiency compared to first-order methods, converging in a significantly smaller number of iterations.

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
