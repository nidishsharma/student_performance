import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv('student_data.csv')

print("=== Dataset Info ===")
print(df.head())
print("\nShape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic stats:\n", df.describe())

# ── 2. Drop unused column ──────────────────────────────────
df = df.drop('student_id', axis=1)

# ── 3. Handle missing values (if any) ─────────────────────
df = df.fillna(df.mean(numeric_only=True))

# ── 4. Define features and target ─────────────────────────
X = df.drop('final_marks', axis=1)   # features
y = df['final_marks']                  # target

print("\n=== Features used ===")
print(X.columns.tolist())

# ── 5. Split into train and test sets ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 6. Scale the features ──────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n=== Scaling done ===")
print("Data is ready for model training!")

# ── 7. Save graphs ────────────────────────────────────────
os.makedirs('static', exist_ok=True)

# Graph 1 — Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/heatmap.png')
plt.close()
print("Saved: static/heatmap.png")

# Graph 2 — Study hours vs Final marks
plt.figure(figsize=(7, 5))
plt.scatter(df['study_hours'], df['final_marks'], color='steelblue', alpha=0.7)
plt.xlabel('Study Hours')
plt.ylabel('Final Marks')
plt.title('Study Hours vs Final Marks')
plt.tight_layout()
plt.savefig('static/scatter.png')
plt.close()
print("Saved: static/scatter.png")

# Graph 3 — Distribution of final marks
plt.figure(figsize=(7, 5))
sns.histplot(df['final_marks'], kde=True, color='purple')
plt.xlabel('Final Marks')
plt.title('Distribution of Final Marks')
plt.tight_layout()
plt.savefig('static/distribution.png')
plt.close()
print("Saved: static/distribution.png")

print("\n✓ Step 2 complete — preprocessing done!")