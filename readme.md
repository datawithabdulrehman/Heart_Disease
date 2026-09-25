# 🫀 Heart Disease Risk Assessment System

A professional, single-page desktop GUI application built with **CustomTkinter** that wraps a pre-trained **Scikit-Learn K-Nearest Neighbors (KNN)** machine learning pipeline to estimate patient heart disease risk from clinical vitals in real time.

---

## 📌 Key Features

* ⚡ **Single-Page Modern UI:** High-contrast single-screen layout designed with a Black, White, Electric Cyan, and Crimson Red theme.
* 🫀 **Real-Time ML Diagnostics:** Instant risk classification (**HIGH RISK** / **LOW RISK**) based on patient vitals.
* 🎨 **Rich Icon Integration:** Visual icons across all demographics, clinical metrics, cardiac tests, and action elements.
* 🔒 **Validation & Error Handling:** Input validation for physiological bounds (Blood Pressure, Cholesterol) and missing model file protection.
* 📊 **Complete Pipeline:** Integrated feature scaling (`StandardScaler`), strict one-hot feature order alignment, and KNN prediction.

---

## 📁 Repository Structure

```text
Heart_Disease/
├── .vscode/                 # Editor configurations
├── app.py                   # Main CustomTkinter GUI application
├── HeartdiseaseFinal.ipynb  # Jupyter notebook for EDA, Data Preprocessing & Model Training
├── heart_columns.pkl        # Expected one-hot encoded feature column order
├── heart_scaler.pkl         # Fitted StandardScaler artifact
└── knn_heart_model.pkl      # Pre-trained KNN Classifier model artifact
```

---

## 🩺 Patient Diagnostic Inputs

The single-page form accepts patient parameters across three distinct sections:

### 👤 Patient Demographics

* 🎂 **Age:** Age in years (18–100).
* ⚧ **Sex:** Male (`M`) or Female (`F`).
* 💔 **Chest Pain Type:** `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `TA` (Typical Angina), `ASY` (Asymptomatic).

### 🩺 Clinical Measurements

* 🩸 **Resting Blood Pressure:** Resting BP in mm Hg (80–200).
* 🧪 **Cholesterol:** Serum cholesterol in mg/dL (100–600).
* 🍬 **Fasting Blood Sugar:** Sugar > 120 mg/dL (`1` = True, `0` = False).
* ⚡ **Max Heart Rate:** Maximum heart rate achieved (60–220 bpm).

### 📈 Cardiac Test Results

* 📊 **Resting ECG:** `Normal`, `ST`, or `LVH` (Left Ventricular Hypertrophy).
* 🏃 **Exercise-Induced Angina:** Yes (`Y`) or No (`N`).
* 📉 **Oldpeak:** ST depression induced by exercise relative to rest (0.0–6.0).
* 📐 **ST Slope:** Slope of peak exercise ST segment (`Up`, `Flat`, `Down`).

---

## ⚙️ Machine Learning Pipeline Workflow

1. **Raw Feature Preprocessing:** User inputs are gathered via GUI entry fields and sliders.
2. **One-Hot Encoding Alignment:** Inputs are mapped against `heart_columns.pkl` to preserve exact column matrix positions.
3. **Standardization:** Input vector is scaled via `heart_scaler.pkl` (`StandardScaler`).
4. **Classification:** KNN Classifier (`knn_heart_model.pkl`) predicts the probability class for risk assessment.

---

## ⚠️ Disclaimer

This tool is created solely for **educational and portfolio demonstration purposes**. It is not intended for official medical diagnostic use or to replace professional healthcare advice.

---

## 👨‍💻 Developer & Portfolio Links

**Built with ❤️ by ABXREHMAN**

* 💼 **LinkedIn:** [datawithabdulrehman](https://www.linkedin.com/in/datawithabdulrehman)
* 📊 **Kaggle:** [datawithabxrehman](https://www.kaggle.com/datawithabxrehman)
* 💻 **GitHub:** [datawithabdulrehman](https://github.com/datawithabdulrehman)
