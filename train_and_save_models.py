import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import os

# Load dataset
data = pd.read_csv("employee_salary_data.csv")

# Features & Target
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Encode categorical columns
label_encoders = {}
for col in ["Gender", "Education Level", "Job Title"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Train & Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    results[name] = {"model": model, "R2": r2, "RMSE": rmse}
    
    print(f"{name} -> R²: {r2:.4f}, RMSE: {rmse:.2f}")

# Save models in one pickle
model_data = {
    "models": {name: info["model"] for name, info in results.items()},
    "label_encoders": label_encoders,
    "scaler": scaler,
    "feature_names": X.columns.tolist()
}
joblib.dump(model_data, "all_salary_models.pkl")

print("\n✅ Models saved in all_salary_models.pkl")

# Create comparison graph
r2_scores = [info["R2"] for info in results.values()]
model_names = list(results.keys())

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, r2_scores, color=['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
plt.title("Model Comparison (R² Score)", fontsize=14, fontweight='bold')
plt.ylabel("R² Score")
plt.ylim(0, 1)

# Annotate bars
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
             f"{score:.2f}", ha="center", color="white", fontsize=12, fontweight='bold')

# Save image
os.makedirs("images", exist_ok=True)

plt.savefig("images/model_comparison.png", bbox_inches="tight")

print("✅ Model comparison chart saved as images/model_comparison.png")


# Create RMSE comparison graph
rmse_scores = [info["RMSE"] for info in results.values()]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, rmse_scores, color=['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
plt.title("Model Comparison (RMSE)", fontsize=14, fontweight='bold')
plt.ylabel("RMSE (Lower is Better)")

# Annotate bars
for bar, score in zip(bars, rmse_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - (0.05 * max(rmse_scores)),
             f"{score:.2f}", ha="center", color="white", fontsize=12, fontweight='bold')

# Save image
plt.savefig("images/rmse_comparison.png", bbox_inches="tight")
print("✅ RMSE comparison chart saved as images/rmse_comparison.png")

# Show plots
plt.show()
print("✅ Plots displayed.")    
# End of script