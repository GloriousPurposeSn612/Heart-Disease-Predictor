# Heart Disease Predictor — Medical Terms & Feature Guide

> **Educational reference only.** This document explains the input features used by the Heart Disease Predictor project in simple language.
>
> **Medical disclaimer:** The ranges and explanations below are educational reference information, not diagnostic thresholds. A value outside a reference range does not by itself mean that a person has heart disease. Clinical interpretation depends on the individual, measurement conditions, age, sex, medical history, medications, and other factors. The machine-learning model also does **not** establish medical causation.

## Contents

- [Important: Dataset Values vs Clinical Reference Values](#important-dataset-values-vs-clinical-reference-values)
- [Feature Guide](#feature-guide)
- [How to Read the Ranges](#how-to-read-the-ranges)
- [Target Variable](#target-variable)
- [Safety Note](#safety-note)
- [Official Resources](#official-resources)

---

## Important: Dataset Values vs Clinical Reference Values

The application uses **13 features from the project dataset**. Some are measurements with established clinical reference categories; others are categorical or test-result encodings.

Do **not** treat every input as a medical “low / normal / high” scale.

This guide distinguishes:

1. **Clinical reference information** — where an official medical source provides a meaningful reference.
2. **Dataset encoding** — the numeric code used by the project for a categorical feature.
3. **Observed training-data range** — the minimum and maximum values observed in the final project dataset. This is **not** a clinical normal range.

---

# Feature Guide

## 1. Age

**Feature:** `age`  
**Meaning:** Age in years.

- Application range: **10–100 years**
- Observed project training range: **29–77 years**

There is no medical “low / normal / high” range for age. Age is a demographic characteristic and cardiovascular risk generally increases with age, but age alone cannot determine whether someone has heart disease.

**Official reference:** CDC identifies age and family history among factors that can affect heart-disease risk.

---

## 2. Sex

**Feature:** `sex`

| Code | Meaning |
|---:|---|
| `0` | Female |
| `1` | Male |

This is a demographic feature used by the dataset. It has no low/normal/high range.

---

## 3. Chest Pain Type (`cp`)

| Code | Dataset category | Simple description |
|---:|---|---|
| `0` | Typical Angina | Chest discomfort with a pattern classified as typical angina |
| `1` | Atypical Angina | Chest discomfort that does not meet all typical angina characteristics |
| `2` | Non-anginal Pain | Chest pain classified as non-anginal |
| `3` | Asymptomatic | No reported chest-pain symptom in the dataset |

These are **dataset categories**, not a self-diagnosis tool.

**Official reference:** WHO lists chest discomfort among possible heart-attack symptoms and advises immediate medical care for relevant symptoms.

---

## 4. Resting Blood Pressure (`trestbps`)

**Unit:** mm Hg  
**Meaning:** Resting systolic blood pressure recorded in the dataset.

- Application range: **70–220 mm Hg**
- Observed project training range: **94–200 mm Hg**

### Adult blood-pressure reference categories

The American Heart Association classifies adult blood pressure using both systolic and diastolic pressure:

| Category | Systolic | Diastolic |
|---|---:|---:|
| Normal | `<120` | `<80` |
| Elevated | `120–129` | `<80` |
| Stage 1 hypertension | `130–139` | `80–89` |
| Stage 2 hypertension | `≥140` | `≥90` |

The project feature contains systolic/resting blood pressure only, so the complete AHA category cannot be determined from this feature alone.

**Official reference:** [American Heart Association — Blood Pressure Explained](https://www.heart.org/en/health-topics/high-blood-pressure/blood-pressure-explained)

---

## 5. Serum Cholesterol (`chol`)

**Unit:** mg/dL  
**Meaning:** Serum cholesterol measurement recorded in the dataset.

- Application range: **100–600 mg/dL**
- Observed project training range: **126–564 mg/dL**

Total cholesterol is only one part of a lipid assessment. Clinical interpretation commonly considers a lipid profile and other risk factors. A single total-cholesterol value should not be treated as a diagnosis.

**Official reference:** [CDC — Heart Disease Risk Factors](https://www.cdc.gov/heart-disease/risk-factors/)

---

## 6. Fasting Blood Sugar (`fbs`)

| Code | Dataset definition |
|---:|---|
| `0` | Fasting blood sugar ≤ 120 mg/dL |
| `1` | Fasting blood sugar > 120 mg/dL |

This is a **dataset-defined binary encoding**, not the complete clinical criteria for diagnosing diabetes.

The CDC identifies diabetes/high blood glucose as a factor associated with increased heart-disease risk.

---

## 7. Resting ECG (`restecg`)

| Code | Dataset category |
|---:|---|
| `0` | Normal |
| `1` | ST-T Wave Abnormality |
| `2` | Left Ventricular Hypertrophy |

These are categorical test-result encodings, not a low/normal/high scale. Their clinical significance requires professional interpretation.

---

## 8. Maximum Heart Rate (`thalach`)

**Unit:** beats per minute (bpm)  
**Meaning:** Maximum heart rate achieved during the relevant exercise test in the dataset.

- Application range: **50–220 bpm**
- Observed project training range: **71–202 bpm**

Maximum heart rate varies with age and individual characteristics. Common formulas provide estimates, but an estimated maximum heart rate is not a clinical normal range.

---

## 9. Exercise-Induced Angina (`exang`)

| Code | Meaning |
|---:|---|
| `0` | No |
| `1` | Yes |

This records whether exercise-induced angina was reported during testing. It is a categorical test/history feature, not a numeric low/normal/high measurement.

---

## 10. ST Depression (`oldpeak`)

**Meaning:** ST-segment depression recorded during the exercise test and represented numerically in the dataset.

- Application range: **0.0–7.0**
- Observed project training range: **0.0–6.2**

This is a test-derived dataset feature. It should not be assigned a universal “normal/abnormal” label from this project alone.

---

## 11. Slope of Peak Exercise ST Segment (`slope`)

| Code | Dataset category |
|---:|---|
| `0` | Upsloping |
| `1` | Flat |
| `2` | Downsloping |

This describes the direction/slope of the ST segment during peak exercise in the dataset. It is not a low/normal/high scale.

---

## 12. Number of Major Vessels (`ca`)

The application accepts:

`0`, `1`, `2`, `3`, `4`

This is a dataset feature representing the number of major vessels associated with the original test data. The numeric value should not be interpreted as a general disease-severity score.

---

## 13. Thal Encoded Category (`thal`)

| Code | Application label |
|---:|---|
| `0` | Code 0 |
| `1` | Normal |
| `2` | Fixed Defect |
| `3` | Reversible Defect |

These are the categorical codes used by the project. The feature originates from a historical medical-test variable in the dataset and should not be interpreted without the underlying clinical context.

---

# How to Read the Ranges

### Application range

The range allowed by the Streamlit control. It prevents obviously unsuitable entries such as negative age while allowing broader exploration.

### Observed training range

| Feature | Observed training range |
|---|---:|
| Age | 29–77 years |
| Resting BP | 94–200 mm Hg |
| Cholesterol | 126–564 mg/dL |
| Maximum heart rate | 71–202 bpm |
| Oldpeak | 0.0–6.2 |

These are **dataset ranges, not clinical reference ranges**.

If an input is outside an observed training range, the application warns that the model is extrapolating beyond the values represented in its training data.

---

# Target Variable

The project performs binary classification:

| Target | Meaning |
|---:|---|
| `0` | No heart disease |
| `1` | Heart disease |

The UCI Heart Disease dataset describes the original goal as distinguishing absence (`0`) from presence (original values `1–4`) of heart disease.

The model output is a **machine-learning prediction**, not a medical diagnosis.

---

# Safety Note

A prediction of Class 0 does not prove that a person does not have heart disease, and a prediction of Class 1 does not prove that the person has heart disease.

Similarly, a model-estimated probability is **not a clinically validated personal risk percentage**.

If someone has concerning symptoms such as chest discomfort, shortness of breath, faintness, or other potentially serious symptoms, they should seek appropriate medical care rather than relying on this application.

---

# Official Resources

- [World Health Organization — Cardiovascular diseases](https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29)
- [Centers for Disease Control and Prevention — Heart Disease Risk Factors](https://www.cdc.gov/heart-disease/risk-factors/)
- [CDC — Risk Factors for High Cholesterol](https://www.cdc.gov/cholesterol/risk-factors/index.html)
- [American Heart Association — Blood Pressure Explained](https://www.heart.org/en/health-topics/high-blood-pressure/blood-pressure-explained)
- [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease)

> Official resources are provided for educational reference. They do not endorse, certify, or clinically validate this academic application.
