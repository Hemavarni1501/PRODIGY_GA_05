# GA_05: High-Resolution Neural Style Transfer System

## 🔗 Project Links
* **Live Application:** [Neural Style Transfer Engine](https://appigyga05-78appbt6xwhachzeevuwdhw.streamlit.app/)

---

## 📌 Project Overview
Developed as part of the **Prodigy Infotech** Generative AI internship, this project implements an advanced **Neural Style Transfer (NST)** system. The application utilizes Deep Learning to blend the semantic content of a user-provided image with the artistic textures and styles of a second source image.

Unlike standard image filters, this system performs a neural synthesis that identifies complex features—such as brushstrokes, lighting patterns, and geometric textures—to create a unique artistic asset that maintains the structural integrity of the original photo.

## 🛠️ Technical Architecture
* **Neural Engine:** Optimized VGG-19 based Feature Mapping.
* **Model Source:** Arbitrary Image Stylization (Magenta Architecture via TensorFlow Hub).
* **Frameworks:** TensorFlow 2.x, NumPy, and PIL (Pillow).
* **Interface:** Streamlit with dynamic parameter injection for real-time synthesis.



## 🚀 Key Engineering Features
1. **Dynamic Feature Extraction:** Leveraged pre-trained neural layers to separate "content" from "style," allowing for high-fidelity reconstruction of subjects while applying heavy artistic textures.
2. **Multi-Scale Optimization:** Implemented a resolution-aware pipeline that allows for synthesis at various scales (256px to 1024px) to balance performance and visual quality.
3. **Face Preservation Logic:** Integrated alpha-blending and intensity controls to ensure human subjects remain recognizable even when heavy stylistic weights are applied.
4. **Resilient Memory Management:** Optimized for cloud deployment on Streamlit Cloud by managing tensor shapes and caching the model to prevent GPU/RAM overflow.

## 📁 Installation & Local Setup
To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Hemavarni1501/PRODIGY_GA_05.git](https://github.com/Hemavarni1501/PRODIGY_GA_05.git)
   cd PRODIGY_GA_05
### **2. Install dependencies:**
```bash
pip install -r requirements.txt
```
### **3. Launch the application:**
```bash
streamlit run app.py
```
## 📊 Results Summary
The system successfully achieves the "Required Internship Output" by maintaining high structural integrity of the content image (e.g., facial features in group photos) while accurately mapping the color palette and textures of the style source (e.g., Van Gogh's Starry Night).

### Internship: Prodigy Infotech

#### Track: Generative AI

#### Task: 05

#### Developer: Hemavarni.S
