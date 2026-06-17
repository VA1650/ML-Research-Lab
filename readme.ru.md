# 🧪 Лаборатория исследований ML и Deep Learning

Комплексная коллекция экспериментов в области машинного обучения, от фундаментального статистического моделирования до передовых генеративных сетей и детектирования объектов.

## 📂 Структура репозитория

### 01-ML-Core-Data-Analysis-Classification-Regression
* **Titanic: Survival Analysis** — Бинарная классификация с глубоким EDA и генерацией признаков.
* **Digits (8x8):** Традиционная классификация (SVM, Random Forest).
* **University Rankings EDA Engine (CWUR)** — Аналитическая система для макро-анализа качества образования. Исследование латентных связей через корреляцию Спирмена.
* **Student Risk Simulation Pipeline** — Моделирование академических рисков. Извлечение детерминированных правил из зашумленных данных с помощью деревьев решений.

### 02-ML-Core-Pipelines
* **MNIST CNN Study:** Сравнение оптимизаторов (Adam vs. SGD) и масштабируемости архитектур.
* **MNIST Real-Time:** Система распознавания цифр через веб-камеру.
* **Auto MPG Prediction:** Регрессионное моделирование топливной эффективности.
* **Heart Disease Classification:** Медицинский скрининг сердечно-сосудистых рисков с защитой от утечек данных.
* **Student Performance Pipeline:** Прогнозирование успеваемости по нескольким дисциплинам с использованием `Ridge` (L2) регуляризации.
* **Manifold Learning:** Изучение нелинейного снижения размерности (Isomap).
* **YouTube Factor Analysis (PCA):** Интерпретация скрытых факторов многомерных метрик каналов.
* **Financial Time Series (VaR):** Оценка рыночного риска для криптоактивов (историческое моделирование).

### 03-Computer-Vision-Basics
* Фундаментальные эксперименты по обработке изображений и пространственным трансформациям.

### 04-Object-Detection-YOLOv8
* **Face Mask Detector:** Полный цикл обучения (аннотация -> аугментация -> обучение).
* *Метрики:* 0.975 mAP50, ~100 FPS на RTX 4060.

### 05-RNN-Legacy-Research
* Моделирование последовательностей (LSTM/GRU).
* **Hall of Fame:** Коллекция артефактов, сгенерированных ИИ.

### 06-Generative-Adversarial-Networks
* **GAN Lab:** Синтез медицинских масок и генерация Minecraft-скинов (64x64).

### 07-Nonlinear-Dynamics-Optimization
* **Nonlinear Oscillator Study:** Численное интегрирование нелинейных систем методом Верле. Собственные движки оптимизации (GD и SGD).

### 08-Stat-Hypothesis-Testing
* **Binomial Hypothesis & Power Analysis:** Инструментарий для продуктовой аналитики (оценка статистической мощности $1-\beta$).

### 09-Optimization-Lab
* **Step-Size Strategies:** Анализ методов выбора шага ($\alpha$): постоянный шаг, правило Поляка и поиск Армихо.

### 10-Ridge-Regression-Benchmark
* **Ridge Regression:** Сравнение аналитического решения и итеративных методов (GD, Proximal GD).

### 11-Sparsity-Optimization-Lab
* **Sparse Optimization (LASSO):** Изучение методов оптимизации функций с $L_1$-регуляризацией.

### 12-Advanced-Optimization-Benchmarking
* **Comparative Optimizer Suite:** Сравнение алгоритмов 1-го и 2-го порядка (Newton, BFGS, SciPy) на задачах высокой размерности.

---

## 🛠 Технологический стек
* **Frameworks:** PyTorch, TensorFlow/Keras, Ultralytics (YOLOv8).
* **Libraries:** Scikit-learn, OpenCV, Pandas, NumPy, Matplotlib.
* **Hardware:** Оптимизировано под NVIDIA RTX 40-series (CUDA).

---

## 🚀 Основные выводы
* **Augmentation:** Кастомные аугментации повысили recall YOLO на ~1.5%.
* **Optimizers:** Adam показал на 30% более быструю сходимость в задачах MNIST.
* **Optimization:** Методы второго порядка (Newton/BFGS) значительно эффективнее при работе с высокой размерностью ($N=100$).

---
*Разработано в процессе глубокого погружения в нейронные архитектуры.*
