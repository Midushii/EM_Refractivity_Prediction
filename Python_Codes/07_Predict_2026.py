import pandas as pd
import joblib

print("="*60)
print("PREDICTING RADIO REFRACTIVITY FOR 2026")
print("="*60)
print("\nLoading Random Forest Model...")

model = joblib.load("RandomForest_Model.pkl")

print("Model Loaded Successfully")

print("\nReading ML_Features_2026.csv...")

df = pd.read_csv("ML_Features_2026.csv")

print("Dataset Loaded")
print(df.shape)

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

print("\nUsing Features:")

for feature in features:
    print(" •", feature)

print("\nPredicting...")

prediction = model.predict(df[features])

df["Predicted_Refractivity"] = prediction

df.to_csv(

    "Predicted_2026.csv",

    index=False

)

print("\nPrediction Complete")

print(df[[
    "Year",
    "Month",
    "Day",
    "Hour",
    "Pressure_hPa",
    "Radio_Refractivity",
    "Predicted_Refractivity"
]].head())

print("\nTotal Predictions :", len(df))

print("\nAverage Observed N :",
      round(df["Radio_Refractivity"].mean(),2))

print("Average Predicted N :",
      round(df["Predicted_Refractivity"].mean(),2))

print("\nSaved as")

print("Predicted_2026.csv")

print("="*60)
