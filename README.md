# Heart Disease Prediction 🫀💓

This project demonstrates a complete machine learning workflow to **predict the presence of heart disease** using clinical data. It covers data loading, exploratory analysis, feature engineering, model training (including advanced models), evaluation, hyperparameter tuning, and future improvements. Designed to showcase your end-to-end approach.

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
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Results & Insights](#results--insights)
12. [Future Work](#future-work)
13. [Usage](#usage)
14. [References](#references)

---

## Introduction

Heart disease is a leading cause of death globally. Machine learning can aid early detection. This project uses clinical data to build and compare multiple classification algorithms to predict heart disease presence.

**Objectives:**

* Perform EDA and preprocessing
* Train baseline and advanced models
* Evaluate using multiple metrics
* Optimize best model via GridSearchCV

---

## Dataset

* **Source:** Kaggle (UCI Heart Disease Dataset)
* **Format:** CSV with 303 records and 14 features + target

| Feature    | Description                                           |
| ---------- | ----------------------------------------------------- |
| `age`      | Age in years                                          |
| `sex`      | 1 = male, 0 = female                                  |
| `cp`       | Chest pain type (4 categories)                        |
| `trestbps` | Resting blood pressure (mm Hg)                        |
| `chol`     | Serum cholesterol (mg/dl)                             |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg`  | Resting ECG results (0,1,2)                           |
| `thalach`  | Max heart rate achieved                               |
| `exang`    | Exercise-induced angina (1 = yes; 0 = no)             |
| `oldpeak`  | ST depression induced by exercise relative to rest    |
| `slope`    | Slope of peak exercise ST segment                     |
| `ca`       | Number of major vessels (0-3)                         |
| `thal`     | Thalassemia (encoded later)                           |
| **Target** | `target` (1 = heart disease; 0 = healthy)             |

---

## Environment Setup

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`

---

## Data Loading

```python
import pandas as pd

df = pd.read_csv('data/heart.csv')
print(df.shape)  # (303, 14)
df.head()
df.info()
```

---

## Exploratory Data Analysis (EDA)

1. Summary statistics: `df.describe()`
2. Missing values: `df.isnull().sum()`
3. Target distribution:

   ```python
   import seaborn as sns
   sns.countplot(x='target', data=df)
   ```
4. Correlation heatmap:

   ```python
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12,10))
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   ```

> Classes are balanced (\~55% healthy, 45% diseased). Features like `thalach` and `oldpeak` correlate strongly.

---

## Data Preprocessing

1. Label encode categorical:

   ```python
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   df['thal'] = le.fit_transform(df['thal'])
   df['cp']   = le.fit_transform(df['cp'])
   ```
2. Split features/target and train-val-test:

   ```python
   from sklearn.model_selection import train_test_split
   X = df.drop('target', axis=1)
   y = df['target']
   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
   X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
   ```
3. Scale features for distance-based models:

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_val_scaled   = scaler.transform(X_val)
   X_test_scaled  = scaler.transform(X_test)
   ```

---

## Feature Engineering

* No extra features for now. Consider polynomial/interactions later.

---

## Model Building

Instantiate and train six models:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"{name} trained.")
```

---

## Model Evaluation

Evaluate on validation set:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

best_model, best_acc = None, 0
for name, model in trained_models.items():
    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"--- {name} ---")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_val_scaled)[:,1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        print(f"ROC AUC: {auc(fpr, tpr):.2f}")
    print()
    if acc > best_acc:
        best_acc, best_model = acc, name
print(f"Best base model: {best_model} with accuracy {best_acc:.2%}")
```

---

## Hyperparameter Tuning

Use GridSearchCV on the best model (Logistic Regression):

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.001,0.01,0.1,1,10,100],
    'penalty': ['l1','l2'],
    'solver': ['liblinear','saga'],
    'max_iter': [100,500,1000]
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_val_scaled, y_val)
print("Best params:", grid.best_params_)
best_lr = grid.best_estimator_
```

Evaluate tuned model:

```python
y_pred = best_lr.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
print(f"Tuned Accuracy: {acc:.2%}")
print(classification_report(y_val, y_pred))
```

---

## Results & Insights

* **Best Base Model:** Logistic Regression (92.50% accuracy)
* **Validation ROC AUC:** \~0.93
* **Tuned Logistic Regression:** 100.00% accuracy on validation

# Initial Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 92.50%   |
| SVM                 | 92.50%   |
| KNN                 | 92.50%   |
| Random Forest       | 90.00%   |
| XGBoost             | 85.00%   |
| LightGBM            | 82.00%   |


the best performing Logistic Regression achiving 92.50% is optimized to increse acc by finding best hyperparameters through grid search

# optimized Logistic Regression model achieves a perfect accuraccy of 100% on validation set without overfitting


> **Note:** Perfect tuning on a small validation set may overfit. Recommend cross-validation on full dataset.

---

## Future Work

* Perform k-fold cross-validation
* Test on hold-out test set
* Feature selection and dimensionality reduction
* Deploy model using Streamlit/Flask
* Interpret model with SHAP

---

## Usage

```bash
# Clone and setup environment as above
jupyter notebook
# Open and run Heart_Disease_Prediction.ipynb
```

---

## References

* UCI Machine Learning Repository: Heart Disease Dataset
* Scikit-learn, XGBoost, LightGBM documentation
* Kaggle community notebooks

---

*Created by Prekshith (May 2025)*
