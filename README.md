# Heart Disease Prediction 🫀💓

This project demonstrates a complete machine learning workflow to **predict the presence of heart disease** using clinical data. It covers everything from data loading and exploratory analysis to model training, evaluation, and future improvements. This is designed for giving readers and potential collaborators a clear, step-by-step overview of the process.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Environment Setup](#environment-setup)
4. [Data Loading](#data-loading)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Data Preprocessing](#data-preprocessing)
7. [Feature Engineering](#feature-engineering)
8. [Model Building](#model-building)
9. [Model Evaluation](#model-evaluation)
10. [Results & Insights](#results--insights)
11. [Future Work](#future-work)
12. [Usage](#usage)
13. [References](#references)

---

## Introduction

Heart disease is one of the leading causes of mortality worldwide. Early detection using machine learning can help in timely intervention and treatment. In this project, we use a classical dataset of patient records to build classifiers that predict whether a patient has heart disease.

**Key objectives:**

* Demonstrate end-to-end ML workflow
* Compare multiple classification algorithms
* Highlight best-performing model and metrics

---

## Dataset

* **Source:** Kaggle (UCI Heart Disease Dataset)
* **Format:** CSV (`.csv`)
* **Size:** \~303 rows, 14 columns

| Feature    | Description                                                       |
| ---------- | ----------------------------------------------------------------- |
| `age`      | Age in years                                                      |
| `sex`      | 1 = male, 0 = female                                              |
| `cp`       | Chest pain type (4 values)                                        |
| `trestbps` | Resting blood pressure (mm Hg)                                    |
| `chol`     | Serum cholesterol (mg/dl)                                         |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)             |
| `restecg`  | Resting electrocardiographic results (values 0,1,2)               |
| `thalach`  | Maximum heart rate achieved                                       |
| `exang`    | Exercise-induced angina (1 = yes; 0 = no)                         |
| `oldpeak`  | ST depression induced by exercise relative to rest                |
| `slope`    | Slope of the peak exercise ST segment                             |
| `ca`       | Number of major vessels (0-3) colored by fluoroscopy              |
| `thal`     | Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) |
| **Target** | `target` (1 = presence of heart disease; 0 = absence)             |

---

## Environment Setup

Ensure you have Python 3.8+ installed.

```bash
# Clone the repo
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Key libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Data Loading

Read the CSV and display basic information.

```python
import pandas as pd

df = pd.read_csv('data/heart.csv')
print(df.shape)        # (303, 14)
df.head()             # First five rows
df.info()             # Data types & non-null counts
```

---

## Exploratory Data Analysis (EDA)

1. **Summary statistics**

   ```python
   df.describe()
   ```
2. **Missing values**

   ```python
   df.isnull().sum()
   ```
3. **Target distribution**

   ```python
   import seaborn as sns
   sns.countplot(x='target', data=df)
   ```
4. **Feature correlations**

   ```python
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12,10))
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   ```

> **Insights:**
>
> * The classes are relatively balanced (\~55% no disease vs 45% disease).
> * Features like `thalach`, `cp`, and `oldpeak` show strong correlations with the target.

---

## Data Preprocessing

1. **Encoding categorical variables**

   ```python
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   df['thal'] = le.fit_transform(df['thal'])
   df['cp']   = le.fit_transform(df['cp'])
   ```
2. **Train-test split**

   ```python
   from sklearn.model_selection import train_test_split
   X = df.drop('target', axis=1)
   y = df['target']
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```
3. **Feature scaling** (for distance-based models)

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled  = scaler.transform(X_test)
   ```

---

## Feature Engineering

* No additional features were generated in this version.
* Consider polynomial features or interaction terms in future iterations.

---

## Model Building

We train and compare the following algorithms:

| Model                        | Import Path                               |
| ---------------------------- | ----------------------------------------- |
| Logistic Regression          | `sklearn.linear_model.LogisticRegression` |
| Random Forest Classifier     | `sklearn.ensemble.RandomForestClassifier` |
| Support Vector Machine (SVM) | `sklearn.svm.SVC`                         |
| K-Nearest Neighbors (KNN)    | `sklearn.neighbors.KNeighborsClassifier`  |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{name} trained.")
```

---

## Model Evaluation

Evaluate each model on the test set using metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC AUC**

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(confusion_matrix(y_test, y_pred))
```

**Sample Output:**

```
--- Random Forest ---
Precision: 0.89, Recall: 0.85, F1-Score: 0.87
ROC AUC: 0.92
Confusion Matrix:
[[28,  3],
 [ 5, 24]]
```

---

## Results & Insights

* **Best Model:** Random Forest Classifier
* **Accuracy:** 87%
* **ROC AUC:** 0.92
* **Strengths:**

  * Handles non-linear relationships
  * Robust to outliers
* **Weaknesses:**

  * Less interpretable than logistic regression

---

## Future Work

* Hyperparameter tuning with `GridSearchCV`
* Cross-validation (e.g., `StratifiedKFold`)
* Feature selection using `SelectKBest`
* Balanced class handling with SMOTE
* Deploy as a web app (Streamlit/Flask)
* Integrate SHAP for model interpretability

---

## Usage

1. **Clone the repository:**

   ```bash
   ```

git clone [https://github.com/yourusername/heart-disease-prediction.git](https://github.com/yourusername/heart-disease-prediction.git)
cd heart-disease-prediction

````
2. **Install dependencies:**
```bash
pip install -r requirements.txt
````

3. **Launch the notebook:**

   ```bash
   ```

jupyter notebook

```
4. **Run through each cell in `Heart_Disease_Prediction.ipynb`**

---

## References
- UCI Machine Learning Repository: Heart Disease Dataset
- Scikit-Learn documentation
- Kaggle community notebooks

---

*Created by Prekshith (May 2025)*

```
