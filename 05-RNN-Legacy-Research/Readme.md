# 🌀 RNN Legacy: Sequence Modeling & The Hall of Architecture Fails

This repository is a deep dive into the era of Recurrent Neural Networks (RNN, LSTM, GRU). It documents my experiments with temporal dependencies, vanishing gradients, and the surrealist outputs of character-level generation.

---

## 🏗 Theoretical Background
Before the era of Transformers, we fought for every hidden state. This archive demonstrates a fundamental understanding of:
* **Vanishing/Exploding Gradients:** Managed via LSTM cells and Gradient Clipping.
* **Many-to-One / Many-to-Many:** Different architectural patterns for diverse data types.
* **Stateless vs. Stateful:** Proper handling of long-term sequences.

---

## 🌩 1. Time-Series Forecasting: Weather Prediction
A classic "Many-to-One" regression task.
* **Goal:** Predicting temperature and atmospheric pressure based on historical sensor data.
* **Implementation:** Stacked LSTM layers with `MinMaxScaler` normalization.
* **Insight:** The model successfully captures seasonal cycles but struggles with rapid, non-linear weather anomalies.

## 🔢 2. Mathematical Logic: Eq-Solver Seq2Seq
A symbolic regression experiment where equations are treated as a language.
* **Goal:** Solving simple algebraic equations using an Encoder-Decoder architecture.
* **Implementation:** Character-level mapping to internal vector space.
* **Verdict:** A fascinating look at how neural networks attempt to learn deterministic logic through probabilistic patterns.

## 🎭 3. The Hall of Fame: RNN "Masterpieces" (Artifacts)
Generating text with Char-RNN often leads to digital hallucinations. These are preserved here as a testament to the challenges of NLP before the GPT era.

> **Visual Representation of Model State during a "Critical Error":**
> ![RNN Chaos](./rnn_chaos.jpg)
> *Actual depiction of a hidden state struggling with "stoybe" objects.*

### Featured Hallucinations:
1. **The Cyber-Manifesto:** *"в банков интеллекта интеллекта ставит вебстраницах в конфессии..."* — A surreal blend of fintech terminology and digital existentialism.
2. **The "QML Fever Dream":** *"от «запревся → объект payload стоёбе объекта..."* — A chaotic mix of C++, QML syntax, and pure linguistic entropy.

---

## 🛠 Tech Stack
* **Frameworks:** TensorFlow / Keras (Legacy & Modern implementations).
* **Architectures:** SimpleRNN, LSTM, GRU.
* **Preprocessing:** Tokenization, Sliding Window generators, Feature Scaling.

---

## 🧪 Research Conclusions
While modern Attention-based models (Transformers) outperform these architectures in long-term memory, RNNs remain a crucial study in **sequential logic**. These experiments highlight the importance of proper data normalization and the inherent unpredictability of character-level modeling without pre-trained embeddings.

---
*Disclaimer: No AI models were permanently harmed during these experiments, though some (like Gemma) experienced significant emotional distress during testing.*