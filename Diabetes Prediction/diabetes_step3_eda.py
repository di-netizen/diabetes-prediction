import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("diabetes_data_cleaned.csv")

# Bar plot: Diabetes count
plt.figure(figsize=(5, 3))
sns.countplot(x='Has_Diabetes', data=df, palette='Set2')
plt.title("Diabetes Distribution")
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Boxplot: BMI vs Diabetes
plt.figure(figsize=(6, 4))
sns.boxplot(x='Has_Diabetes', y='BMI', data=df, palette='Set3')
plt.title("BMI vs Diabetes")
plt.tight_layout()
plt.show()

