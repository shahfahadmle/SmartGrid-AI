
# src/dataset.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_PATH = (
    ROOT_DIR / "data" / "processed" / "hourly_features.csv"
)

MODEL_DIR = ROOT_DIR / "models"

SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"

LOOKBACK = 24
BATCH_SIZE = 64

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

TARGET_COLUMN = "Global_active_power"


# =====================================================
# 1. LOAD PROCESSED DATA
# =====================================================

def load_processed_data():

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_PATH}"
        )

    df = pd.read_csv(
        PROCESSED_PATH,
        parse_dates=["DateTime"],
        index_col="DateTime"
    )

    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps found."
        )

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "NaN values found. Clean data before training."
        )

    if not np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Infinite values found in dataset."
        )

    return df


# =====================================================
# 2. CHRONOLOGICAL SPLIT
# =====================================================

def chronological_split(df):

    n = len(df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()

    val_df = df.iloc[train_end:val_end].copy()

    test_df = df.iloc[val_end:].copy()

    print("Dataset split:")

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return train_df, val_df, test_df


# =====================================================
# 3. FIT SCALERS ON TRAINING DATA ONLY
# =====================================================

def scale_data(
    train_df,
    val_df,
    test_df
):

    scaler_X = StandardScaler()

    scaler_y = StandardScaler()

    # Fit only on training data.
    scaler_X.fit(
        train_df[FEATURE_COLUMNS]
    )

    scaler_y.fit(
        train_df[[TARGET_COLUMN]]
    )

    def transform_split(df):

        X = scaler_X.transform(
            df[FEATURE_COLUMNS]
        )

        y = scaler_y.transform(
            df[[TARGET_COLUMN]]
        )

        return X.astype(np.float32), y.astype(np.float32)

    train_X, train_y = transform_split(train_df)

    val_X, val_y = transform_split(val_df)

    test_X, test_y = transform_split(test_df)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler_X,
        SCALER_X_PATH
    )

    joblib.dump(
        scaler_y,
        SCALER_Y_PATH
    )

    return (
        train_X, train_y,
        val_X, val_y,
        test_X, test_y,
        scaler_X, scaler_y
    )


# =====================================================
# 4. SEQUENCE DATASET
# =====================================================

class ElectricityDataset(Dataset):

    def __init__(
        self,
        features,
        targets,
        lookback=24
    ):

        self.features = torch.as_tensor(
            features,
            dtype=torch.float32
        )

        self.targets = torch.as_tensor(
            targets,
            dtype=torch.float32
        )

        self.lookback = lookback

        if len(self.features) != len(self.targets):
            raise ValueError(
                "Feature and target lengths do not match."
            )

        if len(self.features) <= lookback:
            raise ValueError(
                "Not enough observations for the lookback."
            )

    def __len__(self):

        return len(self.features) - self.lookback

    def __getitem__(self, index):

        x = self.features[
            index:index + self.lookback
        ]

        y = self.targets[
            index + self.lookback
        ]

        return x, y


# =====================================================
# 5. CHECK HOURLY CONTINUITY
# =====================================================

def get_continuous_segments(df):

    """
    Find continuous hourly segments.

    A missing timestamp creates a new segment.
    We never create a sequence across that gap.
    """

    time_difference = df.index.to_series().diff()

    new_segment = (
        time_difference != pd.Timedelta(hours=1)
    )

    segment_ids = new_segment.cumsum()

    segments = []

    for _, segment in df.groupby(segment_ids):

        if len(segment) > LOOKBACK:

            segments.append(segment)

    return segments


# =====================================================
# 6. CREATE DATASETS WITH PREVIOUS CONTEXT
# =====================================================

def create_split_dataset(
    context_df,
    target_df,
    scaler_X,
    scaler_y,
    lookback=24
):

    """
    context_df:
        Observations before the target split.

    target_df:
        Observations whose timestamps belong to this split.

    Only target_df observations become prediction targets.
    """

    combined_df = pd.concat(
        [context_df, target_df]
    )

    combined_df = combined_df.sort_index()

    if combined_df.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps in combined split."
        )

    X_all = scaler_X.transform(
        combined_df[FEATURE_COLUMNS]
    ).astype(np.float32)

    y_all = scaler_y.transform(
        combined_df[[TARGET_COLUMN]]
    ).astype(np.float32)

    context_length = len(context_df)

    # Keep only targets belonging to target_df.
    X_targets = X_all[context_length:]
    y_targets = y_all[context_length:]

    # Add previous observations for historical context.
    X_context = X_all[:context_length]
    y_context = y_all[:context_length]

    X_combined = np.concatenate(
        [X_context, X_targets],
        axis=0
    )

    y_combined = np.concatenate(
        [y_context, y_targets],
        axis=0
    )

    # A valid sequence needs 24 preceding observations.
    # We build each segment separately to avoid gaps.
    valid_indices = []

    start_target = context_length

    for segment in get_continuous_segments(combined_df):

        segment_start = combined_df.index.get_loc(
            segment.index[0]
        )

        segment_end = segment_start + len(segment)

        first_target = max(
            start_target,
            segment_start + lookback
        )

        for target_index in range(
            first_target,
            segment_end
        ):

            valid_indices.append(target_index)

    if not valid_indices:
        raise ValueError(
            "No valid sequences found in this split."
        )

    # Custom dataset retaining only valid target positions.
    return IndexedElectricityDataset(
        X_combined,
        y_combined,
        valid_indices,
        lookback
    )


class IndexedElectricityDataset(Dataset):

    def __init__(
        self,
        features,
        targets,
        target_indices,
        lookback
    ):

        self.features = torch.tensor(
            features,
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32
        )

        self.target_indices = target_indices

        self.lookback = lookback

    def __len__(self):

        return len(self.target_indices)

    def __getitem__(self, index):

        target_index = self.target_indices[index]

        x = self.features[
            target_index - self.lookback:target_index
        ]

        y = self.targets[target_index]

        return x, y


# =====================================================
# 7. PREPARE DATA LOADERS
# =====================================================

def prepare_datasets():

    df = load_processed_data()

    train_df, val_df, test_df = chronological_split(df)

    (
        train_X, train_y,
        val_X, val_y,
        test_X, test_y,
        scaler_X, scaler_y
    ) = scale_data(
        train_df,
        val_df,
        test_df
    )

    # Training has no earlier split context.
    train_dataset = create_split_dataset(
        context_df=train_df.iloc[:0],
        target_df=train_df,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        lookback=LOOKBACK
    )

    # Validation receives the previous 24 hours.
    val_context = train_df.tail(LOOKBACK)

    val_dataset = create_split_dataset(
        context_df=val_context,
        target_df=val_df,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        lookback=LOOKBACK
    )

    # Test receives the previous 24 hours.
    test_context = df.iloc[
        max(0, len(train_df) + len(val_df) - LOOKBACK):
        len(train_df) + len(val_df)
    ]

    test_dataset = create_split_dataset(
        context_df=test_context,
        target_df=test_df,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        lookback=LOOKBACK
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    print("\nSequence counts:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler_X,
        scaler_y
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader, _, _ = (
        prepare_datasets()
    )

    x, y = next(iter(train_loader))

    print("\nBatch verification:")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
