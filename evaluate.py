import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv('student_data.csv')
df = df.drop('student_id', axis=1)
df = df.fillna(df.mean(numeric_only=True))

X = df.drop('final_marks', axis=1)
y = df['final_marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 2. Load saved best model ──────────────────────────────
with open('model.pkl', 'rb') as f:
    best_model = pickle.load(f)

y_pred = best_model.predict(X_test_scaled)

# ── 3. Print evaluation report ───────────────────────────
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print("=" * 45)
print("       MODEL EVALUATION REPORT")
print("=" * 45)
print(f"  R² Score  : {r2:.4f}  (1.0 = perfect)")
print(f"  RMSE      : {rmse:.4f}  (marks error)")
print(f"  MAE       : {mae:.4f}  (avg error)")
print("=" * 45)

# ── 4. Sample predictions table ──────────────────────────
print("\n  Sample Predictions vs Actual:")
print(f"  {'Actual':>8} {'Predicted':>10} {'Difference':>12}")
print("  " + "-" * 33)
for actual, predicted in zip(list(y_test)[:10], y_pred[:10]):
    diff = abs(actual - predicted)
    print(f"  {actual:>8.1f} {predicted:>10.1f} {diff:>12.1f}")

# ── 5. Grading categories ─────────────────────────────────
def get_grade(marks):
    if marks >= 85:   return 'High'
    elif marks >= 55: return 'Medium'
    else:             return 'Low'

y_test_grades = [get_grade(m) for m in y_test]
y_pred_grades = [get_grade(m) for m in y_pred]

correct = sum(a == b for a, b in zip(y_test_grades, y_pred_grades))
grade_accuracy = correct / len(y_test_grades) * 100

print(f"\n  Grade category accuracy: {grade_accuracy:.1f}%")
print("  (High = 85+,  Medium = 55-84,  Low = below 55)")

# ── 6. Cross-validation score ─────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

print(f"\n  Cross-validation R² (5-fold):")
for i, score in enumerate(cv_scores):
    print(f"    Fold {i+1}: {score:.4f}")
print(f"    Mean  : {cv_scores.mean():.4f}")
print("=" * 45)

# ── 7. Save all evaluation graphs ─────────────────────────
os.makedirs('static', exist_ok=True)

# Graph 1 — Actual vs Predicted scatter
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='steelblue', s=80, alpha=0.8, label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect fit')
for i, (actual, predicted) in enumerate(zip(list(y_test)[:5], y_pred[:5])):
    plt.annotate(f'{predicted:.0f}', (actual, predicted),
                 textcoords='offset points', xytext=(6, 4), fontsize=8, color='navy')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted Marks')
plt.legend()
plt.tight_layout()
plt.savefig('static/actual_vs_predicted.png')
plt.close()
print("\n✓ Saved: static/actual_vs_predicted.png")

# Graph 2 — Residuals plot
residuals = np.array(list(y_test)) - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, color='coral', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted Marks')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals Plot')
plt.tight_layout()
plt.savefig('static/residuals.png')
plt.close()
print("✓ Saved: static/residuals.png")

# Graph 3 — Grade distribution pie chart
from collections import Counter
grade_counts = Counter(y_test_grades)

plt.figure(figsize=(6, 6))
colors = ['#55A868', '#4C72B0', '#C44E52']
plt.pie(grade_counts.values(),
        labels=grade_counts.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=140)
plt.title('Grade Distribution (Test Set)')
plt.tight_layout()
plt.savefig('static/grade_distribution.png')
plt.close()
print("✓ Saved: static/grade_distribution.png")

# Graph 4 — Feature importance (always use Random Forest for this)
rf_for_importance = RandomForestRegressor(n_estimators=100, random_state=42)
rf_for_importance.fit(X_train_scaled, y_train)

importances = rf_for_importance.feature_importances_
feat_names  = X.columns.tolist()
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(7, 5))
bars = plt.bar(range(len(feat_names)),
               importances[sorted_idx],
               color=['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974'])
plt.xticks(range(len(feat_names)),
           [feat_names[i] for i in sorted_idx],
           rotation=20)
plt.ylabel('Importance Score')
plt.title('Feature Importance — Which factor matters most?')
for bar, val in zip(bars, importances[sorted_idx]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()
print("✓ Saved: static/feature_importance.png")

# Graph 5 — Cross-validation scores bar
plt.figure(figsize=(7, 4))
fold_labels = [f'Fold {i+1}' for i in range(len(cv_scores))]
plt.bar(fold_labels, cv_scores, color='mediumpurple', alpha=0.85)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--',
            linewidth=1.5, label=f'Mean = {cv_scores.mean():.4f}')
plt.ylabel('R² Score')
plt.title('Cross-Validation Scores (5-Fold)')
plt.ylim(0, 1.1)
plt.legend()
for i, v in enumerate(cv_scores):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('static/cross_validation.png')
plt.close()
print("✓ Saved: static/cross_validation.png")

print("\n✓ Step 4 complete — evaluation done!")