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
If additional dependencies are needed, they can be installed using:

```bash
pip install -r requirements.txt
```

### B. Running the Experiments
To run the project:
1. Open the notebook deep_learning_retinal_disease.ipynb in Google Colab.
2. Upload or mount your dataset (e.g., via Google Drive).
3. Ensure that the file paths match the dataset structure.
4. Run cells in order:
  - Data loading and preprocessing
  - Baseline and transfer learning model training
  - Evaluation and metric computation
  - Threshold optimization and final comparisons
The notebook contains all steps from data preparation to evaluation.

---

## 5. Model Outputs
```markdown
### Performance Comparison

The table below summarizes the evaluation results of all models on the validation dataset:

| Model                | Micro Precision | Micro Recall | Micro F1 |
|----------------------|-----------------|--------------|----------|
| Baseline CNN         | 0.5261          | 0.0854       | 0.1470   |
| VGG16                | 0.7015          | 0.1139       | 0.1960   |
| ResNet50             | 0.5340          | 0.1000       | 0.1684   |
| DenseNet121          | 0.6377          | 0.2091       | 0.3149   |
| DenseNet121 Improve  | 0.3546          | 0.7073       | 0.4724   |

*Note: Micro metrics provide an overall measure across all labels.*

### Per-Label Performance (DenseNet121 Improve)

                precision    recall  f1-score   support

           N       0.33      0.92      0.49       438
           D       0.35      0.92      0.50       442
           G       1.00      0.04      0.07        85
           C       0.74      0.37      0.50        91
           A       0.00      0.00      0.00        61
           H       0.43      0.07      0.11        46
           M       0.69      0.58      0.63        69
           O       0.35      0.66      0.45       418

   micro avg       0.35      0.71      0.47      1650
   macro avg       0.49      0.44      0.34      1650
weighted avg       0.40      0.71      0.44      1650
 samples avg       0.37      0.72      0.47      1650
```

---

## 6. Evaluation Summary

```markdown
## 6. Evaluation Summary

The DenseNet121 with improved pipeline achieved the best overall Micro F1 score among the evaluated models, indicating a good balance between precision and recall on the multi-label validation dataset.

Although certain labels (e.g., AMD (A) and Glaucoma (G)) exhibited lower per-label F1 scores due to class imbalance and visual similarity with other conditions, the improved DenseNet121 model demonstrated robustness in recognizing common disease patterns.

This section provides a concise overview of the model comparison results and highlights the performance trade-offs observed during evaluation. More detailed evaluation charts and per-class analyses are available within the project notebook (`deep_learning_retinal_disease.ipynb`).



