# Multi-Label Retinal Disease Classification using Deep Learning (ODIR-5K)

## 1. Problem Definition
This project addresses the problem of **multi-label retinal disease classification** from fundus images.
Unlike multi-class classification, a single retinal image may contain **multiple co-existing diseases**, therefore the task is formulated as a **multi-label classification problem**.

The model outputs independent probabilities for each disease using a sigmoid activation function, enabling the prediction of more than one condition per image.

**Disease Labels (8):**
- N: Normal
- D: Diabetic Retinopathy
- G: Glaucoma
- C: Cataract
- A: Age-related Macular Degeneration
- H: Hypertension
- M: Myopia
- O: Other abnormalities

---

## 2. Dataset and Preprocessing
- **Dataset:** ODIR-5K (Ocular Disease Intelligent Recognition)
- **Source:** Kaggle
- **Split:** Train / Validation = 80 / 20
- **Input Size:** 224 × 224 RGB images

### Preprocessing Steps
- Resize images to 224×224
- Normalize pixel values to [0,1]
- Data augmentation is applied **only to the training set**
- Validation data is kept unchanged for unbiased evaluation

> Note: The dataset is not included in this repository due to size and licensing constraints.  
> Users should download the dataset separately and update the data paths accordingly.

---

## 3. Model Architectures and Rationale
The following models are evaluated:

- **Baseline CNN (from scratch)**  
  A simple convolutional neural network trained without pre-trained weights, used as a reference floor model.

- **VGG16 (Transfer Learning)**  
  Used as a baseline among pre-trained CNN architectures due to its simple and well-known structure.

- **ResNet50 (Transfer Learning)**  
  Incorporates residual connections to improve gradient flow in deeper networks.

- **DenseNet121 (Improved Model)**  
  Employs dense connections that enable feature reuse and improved representation learning, particularly beneficial for medical images.

### Improved Training Strategy (DenseNet121)
- Focal Loss to address class imbalance
- PR-AUC as a monitoring metric
- Decision threshold optimization for multi-label prediction

---

## 4. How to Run the Project

This project is designed to be executed using **Google Colab**.

### A. Environment Setup
Most required libraries (TensorFlow, NumPy, Pandas, Scikit-learn) are pre-installed in Google Colab.
If needed, additional dependencies can be installed using:
```bash
pip install -r requirements.txt

