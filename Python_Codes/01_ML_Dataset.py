
import xarray as xr
import pandas as pd
import numpy as np

print("\nReading ERA5 GRIB file...\n")

ds = xr.open_dataset(
    "ERA5.grib",
    engine="cfgrib"
)

print(ds)

temperature = ds["t"]           # Kelvin
humidity = ds["q"]              # kg/kg
geopotential = ds["z"]          # m²/s²

pressure_levels = ds["isobaricInhPa"].values
times = ds["time"].values
latitudes = ds["latitude"].values
longitudes = ds["longitude"].values

rows = []

g = 9.80665

print("\nCreating Machine Learning Dataset...\n")

for t_index, timestamp in enumerate(times):

    year = pd.Timestamp(timestamp).year
    month = pd.Timestamp(timestamp).month
    day = pd.Timestamp(timestamp).day
    hour = pd.Timestamp(timestamp).hour

    for p_index, pressure in enumerate(pressure_levels):

        pressure_hPa = float(pressure)

        for lat_index, lat in enumerate(latitudes):

            for lon_index, lon in enumerate(longitudes):

                T = float(
                    temperature.values[
                        t_index,
                        p_index,
                        lat_index,
                        lon_index
                    ]
                )

                q = float(
                    humidity.values[
                        t_index,
                        p_index,
                        lat_index,
                        lon_index
                    ]
                )

                z = float(
                    geopotential.values[
                        t_index,
                        p_index,
                        lat_index,
                        lon_index
                    ]
                )

              

                height = z / g

                

                e = (q * pressure_hPa) / (0.622 + 0.378 * q)


                N = (
                    (77.6 * pressure_hPa / T)
                    +
                    (3.73e5 * e / (T ** 2))
                )

                rows.append([
                    year,
                    month,
                    day,
                    hour,
                    pressure_hPa,
                    lat,
                    lon,
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

df = pd.DataFrame(rows, columns=columns)


df.to_csv(
    "ML_Dataset.csv",
    index=False
)

print("Machine Learning Dataset Created")


print(df.head())

print("\nTotal Samples :", len(df))

print("\nSaved as : ML_Dataset.csv")
