import pandas as pd
import numpy as np

np.random.seed(42)

# Number of records
n = 1000

# Simulate health data
data = {
    'Glucose': np.random.normal(120, 30, n).clip(60, 200),
    'BloodPressure': np.random.normal(70, 10, n).clip(40, 120),
    'BMI': np.random.normal(30, 7, n).clip(15, 60),
    'Age': np.random.randint(21, 80, n),
    'Pregnancies': np.random.randint(0, 15, n),
    'Has_Diabetes': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Save the dataset
df.to_csv("diabetes_data.csv", index=False)
print("✅ Generated 'diabetes_data.csv' with shape:", df.shape)
print(df.head())
