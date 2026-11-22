
# src/preprocessing.py

from pathlib import Path

import numpy as np
import pandas as pd
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "household_power_consumption.txt"
)
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "hourly_features.csv"
MEASUREMENT_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TARGET_COLUMN = "Global_active_power"
FEATURE_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]
def load_raw_data(file_path=RAW_DATA_PATH):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print("Loading raw electricity dataset...")

    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["?", ""],
        low_memory=False
    )

    print("Raw dataset shape:", df.shape)

    return df


def clean_data(df):

    df = df.copy()

    print("Cleaning dataset...")
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    df.drop(
        columns=["Date", "Time"],
        inplace=True
    )
    for column in MEASUREMENT_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


        df.dropna(
        subset=["DateTime"],
        inplace=True
    )

    df.sort_values(
        "DateTime",
        inplace=True
    )

    df.drop_duplicates(
        subset=["DateTime"],
        keep="last",
        inplace=True
    )

    df.set_index(
        "DateTime",
        inplace=True
    )

    print("Cleaned dataset shape:", df.shape)

    return df




def resample_hourly(df, minimum_valid_minutes=50):

    df = df.copy()

    print("Resampling minute-level data to hourly data...")

    aggregation = {
        "Global_active_power": "mean",
        "Global_reactive_power": "mean",
        "Voltage": "mean",
        "Global_intensity": "mean",
    }

    for column in [
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]:

        aggregation[column] = (
            lambda values: values.sum(
                min_count=minimum_valid_minutes
            )
        )

    hourly_df = df.resample("h").agg(
        aggregation
    )

    valid_counts = (
        df["Global_active_power"]
        .resample("h")
        .count()
    )

    hourly_df = hourly_df.loc[
        valid_counts >= minimum_valid_minutes
    ]

    print("Hourly dataset shape:", hourly_df.shape)

    return hourly_df

def handle_missing_values(df):

    df = df.copy()

    print("Handling missing values...")

    before = df.isna().sum().sum()


    df[MEASUREMENT_COLUMNS] = (
        df[MEASUREMENT_COLUMNS]
        .ffill(limit=3)
    )

    df.dropna(
        subset=MEASUREMENT_COLUMNS,
        inplace=True
    )

    after = df.isna().sum().sum()

    print("Missing values before:", before)
    print("Missing values after:", after)

    return df

def create_time_features(df):

    df = df.copy()

    print("Creating time features...")

    # Weekend feature.
    df["is_weekend"] = (
        df.index.dayofweek >= 5
    ).astype(int)

    # Hour encoding.
    # Cyclical representation of the hour.
    df["hour_sin"] = np.sin(
        2 * np.pi * df.index.hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df.index.hour / 24
    )

    # Day-of-week encoding.
    df["weekday_sin"] = np.sin(
        2 * np.pi * df.index.dayofweek / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df.index.dayofweek / 7
    )

    return df



def validate_data(df):

    required_columns = FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Processed dataset is empty."
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps found."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not sorted."
        )

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing values remain in model features."
        )

    print("Data validation successful!")


def save_processed_data(df, output_path=OUTPUT_PATH):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index_label="DateTime"
    )

    print(f"Processed data saved to: {output_path}")


def run_preprocessing():

    print("=" * 60)
    print("SMARTGRID AI - PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_raw_data()

    df = clean_data(df)

    df = resample_hourly(df)

    df = handle_missing_values(df)

    df = create_time_features(df)

    validate_data(df)

    save_processed_data(df)

    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFinal dataset shape:", df.shape)

    print("\nFinal features:")
    print(FEATURE_COLUMNS)

    print("\nDataset preview:")
    print(df.head())

    return df

if __name__ == "__main__":

    processed_df = run_preprocessing()
