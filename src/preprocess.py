import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/raw/household_power_consumption.txt")
PROCESSED_PATH = Path("data/processed/cleaned.csv")


def load_data():
    print("Loading raw dataset...")
    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        low_memory=False,
        na_values=["?"]
    )
    return df


def clean_datetime(df):
    print("Parsing datetime...")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime")
    return df


def convert_numeric(df):
    print("Converting numeric columns...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing(df):
    print("Handling missing values...")
    df = df.interpolate(method="time")
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df


def resample_hourly(df):
    print("Resampling to hourly frequency...")
    df_hourly = df.resample("H").mean()
    return df_hourly


def normalize(df):
    print("Normalizing features...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_scaled


def save_processed(df):
    print("Saving processed dataset...")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH)


def main():
    df = load_data()
    df = clean_datetime(df)
    df = convert_numeric(df)
    df = handle_missing(df)
    df = resample_hourly(df)
    df = normalize(df)
    save_processed(df)
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
