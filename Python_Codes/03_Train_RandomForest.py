"""
=========================================================
03_Train_RandomForest.py

Purpose:
Train a Random Forest model to predict
Radio Refractivity (N)

Training Data : 2010–2024
Testing Data  : 2025

Outputs:
---------
RandomForest_Model.pkl
Model_Performance.csv
Prediction_Results.csv
Feature_Importance.csv
Predicted_vs_Observed.png
Feature_Importance.png

Author:
DRDO Atmospheric EM Propagation Study
=========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("RANDOM FOREST MODEL")
print("=" * 60)

# ----------------------------------------------------------
# Read Dataset
# ----------------------------------------------------------

print("\nReading ML_Features.csv...")

df = pd.read_csv("ML_Features.csv")

print("Dataset Loaded Successfully")
print("Shape :", df.shape)

# ----------------------------------------------------------
# Input Features
# ----------------------------------------------------------

features = [

    "Year",
    "Day_of_Year",
    "Hour",
    "Pressure_hPa",
    "Latitude",
    "Longitude",
    "Month_Sin",
    "Month_Cos",
    "Hour_Sin",
    "Hour_Cos"

]

target = "Radio_Refractivity"

print("\nInput Features")

for feature in features:
    print(" •", feature)

# ----------------------------------------------------------
# Training / Testing Split
# ----------------------------------------------------------

train = df[df["Year"] <= 2024]

test = df[df["Year"] == 2025]

X_train = train[features]

y_train = train[target]

X_test = test[features]

y_test = test[target]

print("\nTraining Samples :", len(train))
print("Testing Samples  :", len(test))

# ----------------------------------------------------------
# Train Random Forest
# ----------------------------------------------------------

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(

    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1

)

model.fit(X_train, y_train)

print("Training Complete")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

print("\nPredicting Refractivity for 2025...")

prediction = model.predict(X_test)

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

mae = mean_absolute_error(y_test, prediction)

rmse = np.sqrt(mean_squared_error(y_test, prediction))

r2 = r2_score(y_test, prediction)

bias = np.mean(prediction - y_test)

print("\nModel Performance")
print("------------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print(f"Bias : {bias:.4f}")

# ----------------------------------------------------------
# Save Metrics
# ----------------------------------------------------------

performance = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2",
        "Bias"
    ],

    "Value": [
        mae,
        rmse,
        r2,
        bias
    ]

})

performance.to_csv(
    "Model_Performance.csv",
    index=False
)

# ----------------------------------------------------------
# Save Predictions
# ----------------------------------------------------------

results = test.copy()

results["Predicted_Refractivity"] = prediction

results.to_csv(
    "Prediction_Results.csv",
    index=False
)

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "Feature_Importance.csv",
    index=False
)

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(
    model,
    "RandomForest_Model.pkl"
)

# ----------------------------------------------------------
# Scatter Plot
# ----------------------------------------------------------

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    prediction,
    s=5
)

minimum = min(y_test.min(), prediction.min())
maximum = max(y_test.max(), prediction.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2
)

plt.xlabel("Observed Refractivity (N)")
plt.ylabel("Predicted Refractivity (N)")
plt.title("Observed vs Predicted Radio Refractivity")

plt.tight_layout()

plt.savefig(
    "Predicted_vs_Observed.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# Feature Importance Plot
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.ylabel("Relative Importance")

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "Feature_Importance.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\nFiles Generated")
print("------------------------------")

print("✓ RandomForest_Model.pkl")
print("✓ Model_Performance.csv")
print("✓ Prediction_Results.csv")
print("✓ Feature_Importance.csv")
print("✓ Predicted_vs_Observed.png")
print("✓ Feature_Importance.png")

print("\nRandom Forest Training Completed Successfully.")
print("=" * 60)