import xarray as xr
import pandas as pd
import numpy as np

print("="*60)
print("CREATING 2026 MACHINE LEARNING DATASET")
print("="*60)
print("\nReading ERA5_2026.grib ...")

ds = xr.open_dataset(
    "ERA5_2026.grib",
    engine="cfgrib"
)

print("\nDataset Successfully Loaded\n")
print(ds)
temperature = ds["t"]
humidity = ds["q"]
geopotential = ds["z"]

records = []

print("\nCreating Dataset...\n")

for t in range(len(ds.time)):

    current_time = pd.to_datetime(ds.time.values[t])

    year = current_time.year
    month = current_time.month
    day = current_time.day
    hour = current_time.hour

    for p in range(len(ds.isobaricInhPa)):

        pressure = float(ds.isobaricInhPa.values[p])

        for lat in range(len(ds.latitude)):

            latitude = float(ds.latitude.values[lat])

            for lon in range(len(ds.longitude)):

                longitude = float(ds.longitude.values[lon])

                T = float(
                    temperature.values[t, p, lat, lon]
                )

                q = float(
                    humidity.values[t, p, lat, lon]
                )

                Z = float(
                    geopotential.values[t, p, lat, lon]
                )

                # Convert geopotential to height
                height = Z / 9.80665

                # Vapour Pressure (hPa)
                e = (q * pressure) / (0.622 + 0.378*q)

                # Radio Refractivity
                N = (
                    77.6 * pressure / T
                    +
                    3.73e5 * e / (T**2)
                )

                records.append([

                    year,
                    month,
                    day,
                    hour,

                    pressure,

                    latitude,
                    longitude,

                    height,

                    T,
                    q,
                    e,
                    N

                ])


columns = [

    "Year",
    "Month",
    "Day",
    "Hour",

    "Pressure_hPa",

    "Latitude",
    "Longitude",

    "Height_m",

    "Temperature_K",
    "Specific_Humidity",
    "Vapour_Pressure_hPa",

    "Radio_Refractivity"

]

df = pd.DataFrame(
    records,
    columns=columns
)

print(df.head())

print("\nTotal Samples :", len(df))


df.to_csv(
    "ML_Dataset_2026.csv",
    index=False
)

print("\nDataset Saved Successfully")

print("File Created : ML_Dataset_2026.csv")

print("="*60)
