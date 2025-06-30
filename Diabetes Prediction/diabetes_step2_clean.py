import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("diabetes_data.csv")

# Check missing values
print("🔍 Missing values:\n", df.isnull().sum())

# Display summary stats
print("\n📊 Summary:\n", df.describe())

# Scale numeric features
scaler = StandardScaler()
features = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Pregnancies']
df[features] = scaler.fit_transform(df[features])

# Save cleaned data
df.to_csv("diabetes_data_cleaned.csv", index=False)
print("\n✅ Cleaned & scaled data saved as 'diabetes_data_cleaned.csv'")
print(df.head())

