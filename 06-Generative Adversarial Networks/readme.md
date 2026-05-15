# 🎨 Generative AI Lab: From Face Masks to Minecraft Skins

A collection of Generative Adversarial Networks (GAN) experiments focusing on image synthesis and domain-specific asset generation.

---

## 🎭 Experiment 1: Face Mask Synthesis
* **Objective:** Generating realistic human faces with and without medical masks to understand latent space representation.
* **Architecture:** DCGAN (Deep Convolutional GAN).
* **Application:** Synthetic data generation for enhancing object detection datasets (YOLO).

## ⛏️ Experiment 2: Minecraft Skin Generator
* **Objective:** Automating the creation of 64x64 pixel-art characters.
* **The Challenge:** Handling strict pixel constraints and maintaining anatomical consistency (arms, legs, head positions) in the generated layout.
* **Results:** The model learned to distribute color clusters corresponding to typical Minecraft skin patterns.

---

## 🛠 Tech Stack
* **Frameworks:** PyTorch / TensorFlow
* **Models:** DCGAN, GAN 
* **Processing:** PIL / OpenCV for pixel-art handling