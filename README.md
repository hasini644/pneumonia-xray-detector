# 🫁 Pneumonia Detection with Explainable AI (Grad-CAM)

A deep learning web application that classifies chest X-ray images as **Normal** or **Pneumonia** using a fine-tuned **VGG-16** convolutional neural network, with **Grad-CAM** heatmaps for visual explainability.



---

## 📸 Demo

Upload a frontal chest X-ray → get an instant AI diagnosis + a Grad-CAM heatmap showing *where* the model looked.

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Base Model | VGG-16 (pretrained on ImageNet) |
| Strategy | Transfer Learning — frozen convolutional blocks |
| Custom Head | GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid) |
| Input Size | 224 × 224 × 3 |
| Output | Binary probability: P(Pneumonia \| Image) |
| Total Parameters | ~14.78M (65K trainable) |

**Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes the last convolutional layer (`block5_conv3`) to show which lung regions influenced the prediction.

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/pneumonia-xray-detector.git
cd pneumonia-xray-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model
The `.h5` model file is too large for GitHub. Download it and place it in the project root:

> **[⬇️ Download pneumonia_vgg16_model.h5](#)** ← *(add your Google Drive / HuggingFace link here)*

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
pneumonia-xray-detector/
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── pneumonia_vgg16_model.h5 # Trained model (download separately)
└── README.md
```

---

## 🔬 Explainability: Grad-CAM

Grad-CAM computes gradients of the output prediction with respect to the feature maps of the final convolutional block. This highlights which spatial regions of the X-ray contributed most to the classification — an essential requirement for **trustworthy AI in medical applications**.

---

## 📦 Tech Stack

- **TensorFlow / Keras** — Model loading & Grad-CAM
- **Streamlit** — Interactive web UI
- **OpenCV** — Heatmap rendering & overlay
- **Pillow / NumPy** — Image preprocessing

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is not a certified medical device and must not be used for clinical diagnosis.
