import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── 1. Load and prepare data ──────────────────────────────
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

# ── 2. Define all models ──────────────────────────────────
models = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(random_state=42),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM'               : SVR(kernel='rbf', C=100, gamma=0.1)
}

# ── 3. Train and evaluate each model ─────────────────────
results = {}

print("=" * 55)
print(f"{'Model':<22} {'R² Score':>10} {'RMSE':>10} {'MAE':>10}")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'model': model}
    print(f"{name:<22} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f}")

print("=" * 55)

# ── 4. Pick the best model (highest R² score) ─────────────
best_name = max(results, key=lambda k: results[k]['r2'])
best_model = results[best_name]['model']

print(f"\n✓ Best model: {best_name}")
print(f"  R² Score : {results[best_name]['r2']:.4f}")
print(f"  RMSE     : {results[best_name]['rmse']:.4f}")
print(f"  MAE      : {results[best_name]['mae']:.4f}")

# ── 5. Save best model and scaler ─────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✓ Saved: model.pkl")
print("✓ Saved: scaler.pkl")

# ── 6. Save graphs ────────────────────────────────────────
os.makedirs('static', exist_ok=True)

# Graph 1 — Model comparison bar chart
model_names = list(results.keys())
r2_scores   = [results[m]['r2']   for m in model_names]
rmse_scores = [results[m]['rmse'] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(model_names, r2_scores, color=['#4C72B0','#55A868','#C44E52','#8172B2'])
axes[0].set_title('Model Comparison — R² Score')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1.1)
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

axes[1].bar(model_names, rmse_scores, color=['#4C72B0','#55A868','#C44E52','#8172B2'])
axes[1].set_title('Model Comparison — RMSE')
axes[1].set_ylabel('RMSE (lower is better)')
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)

plt.suptitle('ML Model Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('static/model_comparison.png')
plt.close()
print("✓ Saved: static/model_comparison.png")

# Graph 2 — Actual vs Predicted (best model)
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_best, color='steelblue', alpha=0.8, label='Predictions')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect fit')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title(f'Actual vs Predicted — {best_name}')
plt.legend()
plt.tight_layout()
plt.savefig('static/actual_vs_predicted.png')
plt.close()
print("✓ Saved: static/actual_vs_predicted.png")

# Graph 3 — Feature importance (Random Forest only)
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_names  = X.columns.tolist()

plt.figure(figsize=(7, 5))
sns.barplot(x=importances, y=feat_names, palette='viridis')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()
print("✓ Saved: static/feature_importance.png")

print("\n✓ Step 3 complete — models trained and saved!")