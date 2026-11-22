
# src/evaluate.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.dataset import (
    load_processed_data,
    chronological_split,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

from src.model import LSTMModel
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
PLOT_DIR = ROOT_DIR / "reports" / "figures"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"
HISTORY_PATH = MODEL_DIR / "training_history.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
LOOKBACK = 24
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
def load_model():

    import joblib

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Best model checkpoint not found. "
            "Train the model first."
        )

    if not SCALER_X_PATH.exists():
        raise FileNotFoundError(
            "Input scaler not found."
        )

    if not SCALER_Y_PATH.exists():
        raise FileNotFoundError(
            "Target scaler not found."
        )

    print("Loading model checkpoint...")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Read architecture configuration from checkpoint.
    config = checkpoint.get(
        "model_config",
        {
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dim": 64,
            "num_layers": 2,
            "output_dim": 1,
            "dropout": 0.2
        }
    )

    model = LSTMModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_dim=config["output_dim"],
        dropout=config["dropout"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scaler_X = joblib.load(SCALER_X_PATH)

    scaler_y = joblib.load(SCALER_Y_PATH)

    print("Model and scalers loaded successfully!")

    return model, scaler_X, scaler_y


# =====================================================
# 2. GENERATE TEST PREDICTIONS
# =====================================================

def generate_predictions(model, test_loader):

    model.eval()

    predictions = []
    actual_values = []

    print("Generating test predictions...")

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(DEVICE)

            output = model(x)

            predictions.append(
                output.cpu().numpy()
            )

            actual_values.append(
                y.numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0
    ).reshape(-1, 1)

    actual_values = np.concatenate(
        actual_values,
        axis=0
    ).reshape(-1, 1)

    return predictions, actual_values


# =====================================================
# 3. INVERSE TRANSFORM
# =====================================================

def inverse_transform(
    predictions,
    actual_values,
    scaler_y
):

    predictions_original = scaler_y.inverse_transform(
        predictions
    )

    actual_original = scaler_y.inverse_transform(
        actual_values
    )

    return predictions_original, actual_original


# =====================================================
# 4. CALCULATE REGRESSION METRICS
# =====================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual).reshape(-1)

    predicted = np.asarray(predicted).reshape(-1)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# =====================================================
# 5. PERSISTENCE BASELINE
# =====================================================

def generate_persistence_baseline(
    test_loader,
    scaler_X
):

    """
    Persistence forecast:
    Next-hour power = Current-hour power.

    The current-hour power is feature index 0.
    It is scaled using scaler_X, not scaler_y.
    """

    baseline_scaled = []

    for x, y in test_loader:

        # Last observation's active power.
        previous_scaled = x[:, -1, 0].numpy()

        baseline_scaled.append(
            previous_scaled.reshape(-1, 1)
        )

    baseline_scaled = np.concatenate(
        baseline_scaled,
        axis=0
    )

    # Recover original feature units using the
    # first feature's input scaler statistics.
    baseline_original = (
        baseline_scaled * scaler_X.scale_[0]
        + scaler_X.mean_[0]
    )

    return baseline_original.reshape(-1, 1)


# =====================================================
# 6. SAVE PREDICTIONS
# =====================================================

def save_predictions(
    timestamps,
    actual,
    predicted,
    baseline
):

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result = pd.DataFrame({

        "DateTime": timestamps,

        "Actual_Global_active_power": (
            actual.reshape(-1)
        ),

        "Predicted_Global_active_power": (
            predicted.reshape(-1)
        ),

        "Persistence_Baseline": (
            baseline.reshape(-1)
        )

    })

    result.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print(
        f"Predictions saved to: {PREDICTIONS_PATH}"
    )

    return result


# =====================================================
# 7. SAVE METRICS
# =====================================================

def save_metrics(
    lstm_metrics,
    baseline_metrics
):

    metrics = {
        "LSTM": lstm_metrics,
        "Persistence_Baseline": baseline_metrics
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )


# =====================================================
# 8. PLOT ACTUAL VS PREDICTED
# =====================================================

def plot_predictions(result):

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Plot a readable subset instead of millions of points.
    sample = result.head(500)

    plt.figure(figsize=(16, 6))

    plt.plot(
        sample["DateTime"],
        sample["Actual_Global_active_power"],
        label="Actual",
        linewidth=1.5
    )

    plt.plot(
        sample["DateTime"],
        sample["Predicted_Global_active_power"],
        label="LSTM Prediction",
        linewidth=1.5
    )

    plt.plot(
        sample["DateTime"],
        sample["Persistence_Baseline"],
        label="Persistence Baseline",
        linewidth=1,
        alpha=0.7
    )

    plt.title(
        "Actual vs Predicted Electricity Consumption"
    )

    plt.xlabel("DateTime")

    plt.ylabel("Global Active Power (kW)")

    plt.legend()

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "actual_vs_predicted.png",
        dpi=300
    )

    plt.show()

    plt.close()


# =====================================================
# 9. PLOT ERROR DISTRIBUTION
# =====================================================

def plot_error_distribution(result):

    errors = (
        result["Actual_Global_active_power"]
        - result["Predicted_Global_active_power"]
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        errors,
        bins=50
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.title("Prediction Error Distribution")

    plt.xlabel("Actual - Predicted (kW)")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "error_distribution.png",
        dpi=300
    )

    plt.show()

    plt.close()


# =====================================================
# 10. PLOT TRAINING HISTORY
# =====================================================

def plot_training_history():

    if not HISTORY_PATH.exists():

        print("Training history not found.")

        return

    with open(
        HISTORY_PATH,
        "r"
    ) as file:

        history = json.load(file)

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training and Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("MSE Loss (scaled target)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "training_history.png",
        dpi=300
    )

    plt.show()

    plt.close()


# =====================================================
# 11. COMPLETE EVALUATION PIPELINE
# =====================================================

def evaluate_model():

    print("=" * 60)
    print("SMARTGRID AI - MODEL EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)

    # -----------------------------------------------
    # LOAD DATA
    # -----------------------------------------------

    df = load_processed_data()

    # Same chronological split used during training.
    train_df, val_df, test_df = chronological_split(df)

    # -----------------------------------------------
    # LOAD MODEL AND SCALERS
    # -----------------------------------------------

    model, scaler_X, scaler_y = load_model()

    # -----------------------------------------------
    # PREPARE TEST DATA
    # -----------------------------------------------

    from src.dataset import ElectricityDataset

    test_X = scaler_X.transform(
        test_df[FEATURE_COLUMNS]
    )

    test_y = scaler_y.transform(
        test_df[[TARGET_COLUMN]]
    )

    test_dataset = ElectricityDataset(
        test_X,
        test_y,
        lookback=LOOKBACK
    )

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------
    # GENERATE PREDICTIONS
    # -----------------------------------------------

    predictions_scaled, actual_scaled = (
        generate_predictions(
            model,
            test_loader
        )
    )

    # -----------------------------------------------
    # INVERSE TRANSFORM
    # -----------------------------------------------

    predictions, actual = inverse_transform(
        predictions_scaled,
        actual_scaled,
        scaler_y
    )

    # -----------------------------------------------
    # BASELINE
    # -----------------------------------------------

    baseline = generate_persistence_baseline(
        test_loader,
        scaler_X
    )

    # -----------------------------------------------
    # METRICS
    # -----------------------------------------------

    lstm_metrics = calculate_metrics(
        actual,
        predictions
    )

    baseline_metrics = calculate_metrics(
        actual,
        baseline
    )

    # -----------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------

    print("\nLSTM Performance")
    print("-" * 40)

    for name, value in lstm_metrics.items():

        print(f"{name}: {value:.5f}")

    print("\nPersistence Baseline Performance")
    print("-" * 40)

    for name, value in baseline_metrics.items():

        print(f"{name}: {value:.5f}")

    # -----------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------

    # Each dataset window predicts the observation
    # immediately after its 24-hour input window.
    timestamps = test_df.index[LOOKBACK:]

    result = save_predictions(
        timestamps,
        actual,
        predictions,
        baseline
    )

    save_metrics(
        lstm_metrics,
        baseline_metrics
    )

    # -----------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------

    plot_predictions(result)

    plot_error_distribution(result)

    plot_training_history()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    return lstm_metrics, baseline_metrics, result


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    lstm_metrics, baseline_metrics, result = (
        evaluate_model()
    )
