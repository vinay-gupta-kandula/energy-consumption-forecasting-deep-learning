import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/cleaned.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def load_data():
    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    return df


def create_calendar_features(df):
    print("Creating calendar features...")
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def create_lag_features(df, target="Global_active_power"):
    print("Creating lag features...")
    df[f"{target}_lag_1"] = df[target].shift(1)
    df[f"{target}_lag_24"] = df[target].shift(24)
    df[f"{target}_lag_168"] = df[target].shift(168)  # weekly
    return df


def create_rolling_features(df, target="Global_active_power"):
    print("Creating rolling statistics...")
    df[f"{target}_roll_mean_24"] = df[target].rolling(window=24).mean()
    df[f"{target}_roll_std_24"] = df[target].rolling(window=24).std()
    return df


def drop_na(df):
    print("Dropping rows with NaN (due to lagging)...")
    df = df.dropna()
    return df


def save_features(df):
    print("Saving feature dataset...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)


def main():
    df = load_data()
    df = create_calendar_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = drop_na(df)
    save_features(df)
    print("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
