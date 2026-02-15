# 🚢 Titanic Survival Predictor: End-to-End Machine Learning Project

This project predicts the survival probability of Titanic passengers using a **Gradient Boosting Classifier**. It covers the entire machine learning pipeline, from raw data exploration to a deployed web application.

## 🚀 Live Demo
[Insert Streamlit Cloud Link Here]

---

## 📊 Project Workflow

### 1. Exploratory Data Analysis (EDA)
Performed in-depth analysis to understand survival patterns:
* **Gender Bias:** Confirmed the "Women and children first" policy.
* **Social Class:** Passengers in 1st Class had significantly higher survival rates.
* **Fare Distribution:** Identified outliers and skewed data, leading to the need for log transformation.

### 2. Feature Engineering
Transformed raw data into meaningful predictors:
* **Title Extraction:** Extracted titles (Mr, Mrs, Miss, Master, Rare) from passenger names to capture social status.
* **Family Size:** Combined `SibSp` and `Parch` to determine if traveling alone or with family.
* **Log Transformation:** Applied `np.log1p` to the `Fare` column to normalize its distribution.
* **Cabin Indicator:** Created a binary feature indicating whether a passenger had a recorded cabin number.

### 3. Preprocessing
* **Handling Missing Values:** Imputed missing values for `Age`, `Embarked`, and `Fare`.
* **One-Hot Encoding:** Converted categorical variables (`Sex`, `Embarked`, `Title`) into numerical format.
* **Feature Scaling:** Used `StandardScaler` to ensure all numerical features were on a similar scale.

### 4. Model Comparison & Optimization
Evaluated multiple algorithms to find the best performer:
* **Models Tested:** Logistic Regression, Decision Tree, Random Forest, and **Gradient Boosting**.
* **Hyperparameter Tuning:** Used `GridSearchCV` to optimize the Decision Tree and Gradient Boosting parameters.
* **Final Model:** Selected **Gradient Boosting Classifier** for its superior accuracy and robustness.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Environment:** VS Code (Virtual Environment)

---

## 📂 Repository Structure
* `app.py`: The Streamlit web application code.
* `titanic_analysis.ipynb`: The original Jupyter Notebook containing EDA and Model Comparison.
* `generate_model.py`: Script to retrain the final model and export artifacts.
* `best_titanic_model.pkl`: The trained Gradient Boosting model.
* `scaler.pkl`: The fitted StandardScaler object.
* `feature_columns.pkl`: List of features used during training to ensure input consistency.
* `requirements.txt`: List of dependencies for deployment.

---

## 👤 Author
**Niken Larasati** *Data Scientist*
