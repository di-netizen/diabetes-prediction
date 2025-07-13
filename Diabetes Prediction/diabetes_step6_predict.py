import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("diabetes_data_cleaned.csv")
X = df.drop('Has_Diabetes', axis=1)
y = df['Has_Diabetes']

# Train model again (you can skip if you already saved the model)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'diabetes_rf_model.pkl')
print("✅ Model saved as 'diabetes_rf_model.pkl'")

# --------------------------
# Predict new patient
# --------------------------
# Example input (scaled)
new_patient = pd.DataFrame([{
    'Glucose': 1.2,
    'BloodPressure': -0.3,
    'BMI': 0.8,
    'Age': 0.5,
    'Pregnancies': 1.0
}])

# Load the model and predict
loaded_model = joblib.load('diabetes_rf_model.pkl')
prediction = loaded_model.predict(new_patient)[0]

print("🩺 Prediction Result:", "Diabetic" if prediction == 1 else "Non-Diabetic")
