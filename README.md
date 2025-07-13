
# 🩺 Diabetes Prediction using Machine Learning

This project builds a machine learning pipeline to predict whether a person has diabetes based on synthetic health records. It demonstrates a full ML lifecycle: data generation, preprocessing, exploratory analysis, model training, evaluation, model saving, and real-time prediction.

---

## 📊 Dataset

- **Source:** Synthetic dataset (`diabetes_data.csv`)  
- **Records:** 1,000  
- **Features:**  
  - `Glucose`  
  - `BloodPressure`  
  - `BMI`  
  - `Age`  
  - `Pregnancies`  
- **Target:**  
  - `Has_Diabetes` → `0` (No), `1` (Yes)  
  - ~30% of samples are diabetic

---

## 🧪 Project Workflow

1. **Data Generation**  
   - **Script:** `generate_diabetes_data.py`  
   - Creates realistic health data and saves it to `diabetes_data.csv`

2. **Preprocessing & Scaling**  
   - **Script:** `diabetes_clean_encode.py`  
   - Scales features using `StandardScaler`  
   - Saves to `diabetes_data_cleaned.csv`

3. **Exploratory Data Analysis**  
   - **Script:** `diabetes_eda.py`  
   - Visualizes class distribution, feature correlations, and BMI distribution vs diabetes status

4. **Train-Test Split**  
   - **Script:** `diabetes_split.py`  
   - Performs stratified 80/20 split on scaled data

5. **Model Training & Evaluation**  
   - **Script:** `diabetes_train_evaluate.py`  
   - Trains Logistic Regression and Random Forest  
   - Outputs classification report and confusion matrix

6. **Model Saving & Real-Time Prediction**  
   - **Script:** `diabetes_predict.py`  
   - Saves Random Forest model as `diabetes_rf_model.pkl`  
   - Predicts diabetes status for new patient input

---

## 📈 Model Performance (Sample)

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.87      | 0.76   | 0.81     | 0.89    |
| Random Forest       | 0.91      | 0.85   | 0.88     | 0.94    |

> *Actual results may vary depending on random seed and feature scaling.*

---

## 📁 Folder Structure


---

## ▶️ How to Run

```bash
# 1. Generate synthetic dataset
python generate_diabetes_data.py

# 2. Clean and scale features
python diabetes_clean_encode.py

# 3. Visualize EDA
python diabetes_eda.py

# 4. Train models and evaluate
python diabetes_train_evaluate.py

# 5. Save model and simulate prediction
python diabetes_predict.py
