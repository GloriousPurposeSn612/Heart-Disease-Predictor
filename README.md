# ❤️ Heart Disease Predictor

A machine-learning based academic project that compares classification algorithms and provides an interactive Streamlit interface for exploring heart-disease predictions.

> **Purpose:** Academic / educational demonstration  
> **Medical disclaimer:** This project is **not a medical diagnostic system** and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Contents

- [❤️ Heart Disease Predictor](#️-heart-disease-predictor)
  - [Contents](#contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Problem Statement](#2-problem-statement)
  - [3. Dataset](#3-dataset)
    - [Dataset provenance](#dataset-provenance)
  - [4. Features](#4-features)
  - [5. Machine Learning Workflow](#5-machine-learning-workflow)
  - [6. Models](#6-models)
    - [Logistic Regression](#logistic-regression)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Random Forest Classifier](#random-forest-classifier)
  - [7. Model Evaluation](#7-model-evaluation)
  - [8. Data Visualization](#8-data-visualization)
    - [Dataset-level](#dataset-level)
    - [Model comparison](#model-comparison)
    - [Model interpretation](#model-interpretation)
  - [9. Final Model Bundles](#9-final-model-bundles)
    - [Required bundle information](#required-bundle-information)
    - [Directory](#directory)
  - [10. Streamlit Application](#10-streamlit-application)
    - [Main sections](#main-sections)
    - [Prediction flow](#prediction-flow)
  - [11. Project Structure](#11-project-structure)
  - [12. Running the Project](#12-running-the-project)
  - [13. Input Validation \& Safety](#13-input-validation--safety)
    - [Observed training ranges](#observed-training-ranges)
    - [Feature-order protection](#feature-order-protection)
  - [14. Interpretation of Results](#14-interpretation-of-results)
    - [Prediction](#prediction)
    - [Probability](#probability)
    - [Performance metrics](#performance-metrics)
  - [15. Limitations](#15-limitations)
  - [16. Future Scope](#16-future-scope)
  - [17. Visual / Media Assets](#17-visual--media-assets)
  - [18. Disclaimer](#18-disclaimer)
  - [19. Resources](#19-resources)
    - [Dataset](#dataset)
    - [Medical reference](#medical-reference)
    - [Machine learning](#machine-learning)
    - [Application](#application)

---

## 1. Project Overview

This project demonstrates an end-to-end supervised machine-learning workflow for binary classification of heart-disease records.

The workflow covers:

1. Dataset inspection and preparation
2. Feature/target separation
3. Train-test splitting
4. Training multiple classification models
5. Evaluation using standard classification metrics
6. Data and model-performance visualization
7. Selection of final models
8. Saving trained model bundles with `joblib`
9. Integration of the final models into a Streamlit application
10. Interactive prediction from user-provided feature values

The implementation is intentionally kept understandable and suitable for an academic project.

---

## 2. Problem Statement

The objective is to investigate whether machine-learning classifiers can learn patterns in the available heart-disease dataset and distinguish between two target classes:

- `0` — No Heart Disease
- `1` — Heart Disease

Multiple algorithms are compared rather than assuming that one algorithm is automatically the best.

---

## 3. Dataset

The project uses a heart-disease dataset containing **303 instances and 13 input features** for the classification workflow.

The [UCI Machine Learning Repository Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease) identifies the dataset as a classification dataset with categorical, integer, and real-valued features. Its published experiments commonly use the Cleveland subset and distinguish absence (`0`) from presence (`1–4`) for binary classification.

### Dataset provenance

The UCI repository is the project's official dataset reference. If the working CSV is a downloaded or mirrored copy, the exact file provenance should be retained in the project records for reproducibility.

---

## 4. Features

The final model expects these features **in exactly this order**:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | Dataset sex encoding |
| `cp` | Chest-pain category |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar category |
| `restecg` | Resting ECG category |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression measurement |
| `slope` | Peak-exercise ST-segment slope |
| `ca` | Number of major vessels in the dataset |
| `thal` | Thal-related encoded category |

See **[Medical_Terms.md](Medical_Terms.md)** for simple explanations and the distinction between clinical reference information and dataset encodings.

---

## 5. Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Inspection & Preparation
   │
   ▼
Feature / Target Separation
   │
   ▼
Train-Test Split
   │
   ├───────────────┐
   ▼               ▼
Training Set     Test Set
   │               │
   ▼               │
Multiple ML Models │
   │               │
   ▼               ▼
Predictions & Evaluation
   │
   ▼
Metric Comparison
   │
   ▼
Visualization & Analysis
   │
   ▼
Final Model Selection
   │
   ▼
Joblib Model Bundles
   │
   ▼
Streamlit Application
```

---

## 6. Models

The notebook evaluates multiple classification approaches. The Streamlit application currently exposes three selected final models.

### Logistic Regression

A linear classification algorithm that models class probabilities using a logistic function. Scikit-learn applies regularization by default.

Official documentation: https://scikit-learn.org/stable/modules/linear_model.html

**Bundle:** `Final Models/Logistic_Regression.pkl`

### K-Nearest Neighbors (KNN)

An instance-based classifier that predicts using nearby training samples. Scikit-learn's `KNeighborsClassifier` uses a configurable number of neighbors.

Official documentation: https://scikit-learn.org/stable/modules/neighbors.html

**Bundle:** `Final Models/KNN.pkl`

### Random Forest Classifier

An ensemble classifier made from multiple decision trees whose predictions are combined. It can improve predictive performance and help control over-fitting compared with a single unrestricted tree.

Official documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

**Bundle:** `Final Models/Random_Forest_Classifier.pkl`

---

## 7. Model Evaluation

The notebook evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

It also examines:

- Confusion matrices
- ROC curves
- Model-performance comparisons
- Feature relationships
- Model-derived feature importance for appropriate tree-based models

The project does not rely on accuracy alone when comparing models.

> **Interpretation:** These metrics describe performance on the project's held-out test set. They do not represent a user's personal medical risk.

---

## 8. Data Visualization

The completed visualization stage includes:

### Dataset-level

- Target Distribution — bar chart
- Feature Correlation — heatmap

### Model comparison

- Model Performance Comparison — grouped bar chart
- ROC-AUC Scores Comparison — bar chart
- ROC Curves — combined graph
- Confusion Matrices — subplot

### Model interpretation

- Model-Derived Feature Importance — subplot for appropriate tree models
- Feature Relationship — subplot across evaluated models/features

> Feature importance describes how a trained model uses features. It does **not** establish medical causation or clinical importance.

---

## 9. Final Model Bundles

Final models are stored as bundles rather than only as estimator objects.

### Required bundle information

```text
model_name
model
accuracy
precision
recall
f1_score
roc_auc
y_test
y_test_pred
y_test_prob
feature_names
```

This allows the Streamlit application to display model metrics and visualizations without retraining.

### Directory

```text
Final Models/
├── Logistic_Regression.pkl
├── KNN.pkl
└── Random_Forest_Classifier.pkl
```

The application validates required bundle keys and feature order before using a model.

---

## 10. Streamlit Application

The application provides an interactive front end for the final model bundles.

### Main sections

1. Project resources / documentation
2. Model selection
3. Patient parameters
4. Prediction
5. Model-estimated probability
6. Input summary
7. Selected-model performance
8. Confusion matrix
9. ROC curve
10. Model-specific interpretation
11. Medical disclaimer

### Prediction flow

```text
Select model
     │
     ▼
Enter feature values
     │
     ▼
Convert UI labels to dataset encodings
     │
     ▼
Create feature DataFrame
     │
     ▼
Selected trained model
     │
     ├──────────────► Class prediction
     │
     └──────────────► Class-1 probability
     │
     ▼
Prediction + metrics + visualizations
```

The app uses the same feature order stored with the final model bundles.

---

## 11. Project Structure

```text
Heart Disease Predictor/
│
├── HDP_Model.ipynb
├── App.py
├── Medical_Terms.md
├── README.md
│
├── Final Models/
│   ├── Logistic_Regression.pkl
│   ├── KNN.pkl
│   └── Random_Forest_Classifier.pkl
│
└── Assets/
    └── [future screenshots / diagrams / visuals]
```

`App1.py`, if retained, is a UI-development copy and is not required by the core ML architecture.

---

## 12. Running the Project

Install the required Python packages:

```text
streamlit
joblib
pandas
matplotlib
seaborn
scikit-learn
```

From the project root, run:

```bash
streamlit run App.py
```

Before launching, verify that `Final Models/` exists beside `App.py` and contains all three final `.pkl` files.

---

## 13. Input Validation & Safety

The application uses Streamlit controls such as sliders and select boxes to prevent common invalid entries such as negative ages or arbitrary text in numeric fields.

### Observed training ranges

| Feature | Training-data range |
|---|---:|
| Age | 29–77 |
| Resting BP | 94–200 mm Hg |
| Cholesterol | 126–564 mg/dL |
| Maximum heart rate | 71–202 bpm |
| Oldpeak | 0.0–6.2 |

These are **dataset ranges, not medical reference ranges**.

If an input falls outside an observed training range, the application warns that the prediction involves extrapolation beyond the values represented in training data.

### Feature-order protection

The app checks the stored feature names against:

```text
age → sex → cp → trestbps → chol → fbs → restecg
→ thalach → exang → oldpeak → slope → ca → thal
```

---

## 14. Interpretation of Results

### Prediction

The classifier returns:

- `0` → No Heart Disease
- `1` → Heart Disease

This is a machine-learning class prediction, not a medical diagnosis.

### Probability

For models supporting `predict_proba`, the app displays the model-estimated probability of Class 1.

This should **not** be described as a clinically validated personal risk percentage.

### Performance metrics

The app displays test-set:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

These describe model performance on the project's test data, not the user's personal medical risk.

---

## 15. Limitations

- The dataset is small: 303 instances.
- Results depend on the project's dataset split and implementation.
- The models have not been clinically validated.
- The 13 features do not represent every factor relevant to cardiovascular health.
- Model probabilities are not automatically calibrated clinical risk estimates.
- Tree-based feature importance does not demonstrate causation.
- The underlying dataset is a historical machine-learning benchmark rather than a modern clinical registry.

---

## 16. Future Scope

Possible future improvements include:

- Cross-validation and more robust model comparison
- Targeted hyperparameter tuning
- Probability calibration analysis
- External validation on an independent dataset
- Downloadable prediction reports
- QR code linking to the deployed application
- Additional explainability methods
- Experiment tracking and reproducibility improvements
- Containerized deployment

These are future enhancements, not capabilities claimed for the current version.

---

## 17. Visual / Media Assets

Future screenshots and diagrams can be stored in an `Assets/` directory.

Suggested assets:

```text
Assets/
├── app-home.png
├── app-prediction.png
├── model-comparison.png
├── confusion-matrices.png
├── roc-curves.png
└── architecture.png
```

Example Markdown usage:

```markdown
![Application prediction screen](Assets/app-prediction.png)
```

> Visual assets are intentionally represented as placeholders in this documentation version.

---

## 18. Disclaimer

> ### ⚠️ Medical Disclaimer
>
> This is an academic machine-learning project created for educational and demonstration purposes. It is **not a medical diagnostic tool** and must not be used as a substitute for professional medical advice, diagnosis, treatment, or emergency care.
>
> A model prediction does not confirm or rule out heart disease. Model-estimated probabilities are not clinically validated personal risk estimates.
>
> If you have concerning or potentially serious symptoms, seek appropriate medical care rather than relying on this application.

For medical background, see the [WHO cardiovascular-disease resource](https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29).

---

## 19. Resources

### Dataset

- [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease)

### Medical reference

- [World Health Organization — Cardiovascular diseases](https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29)
- [Centers for Disease Control and Prevention — Heart Disease Risk Factors](https://www.cdc.gov/heart-disease/risk-factors/)
- [CDC — Risk Factors for High Cholesterol](https://www.cdc.gov/cholesterol/risk-factors/index.html)
- [American Heart Association — Blood Pressure Explained](https://www.heart.org/en/health-topics/high-blood-pressure/blood-pressure-explained)

### Machine learning

- [scikit-learn — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn — Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn — Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### Application

- [Streamlit Documentation](https://docs.streamlit.io/)

> Official resources are included for educational and technical reference. None of these organizations endorse or clinically validate this academic application.
