

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("="*60)
print("MODEL VALIDATION")
print("="*60)


print("\nReading Prediction_Results.csv...")

df = pd.read_csv("Prediction_Results.csv")

print("Dataset Loaded")

print(df.shape)

observed = df["Radio_Refractivity"]

predicted = df["Predicted_Refractivity"]

mae = mean_absolute_error(observed,predicted)

rmse = np.sqrt(mean_squared_error(observed,predicted))

r2 = r2_score(observed,predicted)

bias = np.mean(predicted-observed)

print("\nValidation Metrics")

print("------------------------------")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R²   : {r2:.4f}")

print(f"Bias : {bias:.4f}")

metrics = pd.DataFrame({

    "Metric":[
        "MAE",
        "RMSE",
        "R2",
        "Bias"
    ],

    "Value":[
        mae,
        rmse,
        r2,
        bias
    ]

})

metrics.to_csv(

    "Validation_Metrics.csv",

    index=False

)

plt.figure(figsize=(7,7))

plt.scatter(

    observed,

    predicted,

    s=5,

    alpha=0.5

)

minimum=min(observed.min(),predicted.min())

maximum=max(observed.max(),predicted.max())

plt.plot(

    [minimum,maximum],

    [minimum,maximum],

    linewidth=2

)

plt.xlabel("Observed Refractivity (N)")

plt.ylabel("Predicted Refractivity (N)")

plt.title(f"Observed vs Predicted (R² = {r2:.3f})")

plt.tight_layout()

plt.savefig(

    "Observed_vs_Predicted.png",

    dpi=300

)

plt.close()


plt.figure(figsize=(12,5))

plt.plot(

    observed.values[:500],

    label="Observed"

)

plt.plot(

    predicted[:500],

    label="Predicted"

)

plt.xlabel("Sample")

plt.ylabel("Radio Refractivity (N)")

plt.title("Observed vs Predicted Refractivity (First 500 Samples)")

plt.legend()

plt.tight_layout()

plt.savefig(

    "Time_Series_Comparison.png",

    dpi=300

)

plt.close()

errors = predicted-observed

plt.figure(figsize=(8,5))

plt.hist(

    errors,

    bins=40

)

plt.xlabel("Prediction Error (N Units)")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(

    "Prediction_Error_Histogram.png",

    dpi=300

)

plt.close()

plt.figure(figsize=(8,5))

plt.scatter(

    predicted,

    errors,

    s=5,

    alpha=0.5

)

plt.axhline(

    0,

    linewidth=2

)

plt.xlabel("Predicted Refractivity")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(

    "Residual_Plot.png",

    dpi=300

)

plt.close()

print("\nError Statistics")

print("------------------------------")

print("Mean Error      :",round(errors.mean(),3))

print("Std Error       :",round(errors.std(),3))

print("Maximum Error   :",round(errors.max(),3))

print("Minimum Error   :",round(errors.min(),3))

print("\nValidation Complete")

print("------------------------------")

print("Files Generated")

print("✓ Validation_Metrics.csv")

print("✓ Observed_vs_Predicted.png")

print("✓ Time_Series_Comparison.png")

print("✓ Prediction_Error_Histogram.png")

print("✓ Residual_Plot.png")

print("="*60)
