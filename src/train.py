
# src/train.py

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from tqdm.auto import tqdm

from src.dataset import prepare_datasets
from src.model import LSTMModel


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0

PATIENCE = 7
MIN_DELTA = 1e-5

SEED = 42


# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# TRAINING CONFIGURATION
# =====================================================

def get_optimizer(model):

    return Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


def get_loss_function():

    return nn.MSELoss()


# =====================================================
# TRAIN ONE EPOCH
# =====================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for x, y in progress_bar:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Reset old gradients.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass.
        predictions = model(x)

        # Calculate loss.
        loss = loss_fn(predictions, y)

        # Backward pass.
        loss.backward()

        # Prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP
        )

        # Update weights.
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = total_loss / total_samples

    return epoch_loss


# =====================================================
# VALIDATION
# =====================================================

def validate_one_epoch(
    model,
    val_loader,
    loss_fn,
    device
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for x, y in tqdm(
            val_loader,
            desc="Validation",
            leave=False
        ):

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            predictions = model(x)

            loss = loss_fn(predictions, y)

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size

            total_samples += batch_size

    epoch_loss = total_loss / total_samples

    return epoch_loss


# =====================================================
# SAVE CHECKPOINT
# =====================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    path
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_config": {
            "input_dim": 12,
            "hidden_dim": 64,
            "num_layers": 2,
            "output_dim": 1,
            "dropout": 0.2
        }
    }

    torch.save(
        checkpoint,
        path
    )


# =====================================================
# SAVE TRAINING HISTORY
# =====================================================

def save_history(history):

    with open(HISTORY_PATH, "w") as file:

        json.dump(
            history,
            file,
            indent=4
        )


# =====================================================
# EARLY STOPPING
# =====================================================

class EarlyStopping:

    def __init__(
        self,
        patience=PATIENCE,
        min_delta=MIN_DELTA
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0

        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

            return True

        self.counter += 1

        if self.counter >= self.patience:

            self.early_stop = True

        return False


# =====================================================
# COMPLETE TRAINING PIPELINE
# =====================================================

def train_model():

    set_seed()

    print("=" * 60)
    print("SMARTGRID AI - LSTM TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    # -----------------------------------------------
    # LOAD DATA
    # -----------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        scaler_X,
        scaler_y
    ) = prepare_datasets()

    # -----------------------------------------------
    # INITIALIZE MODEL
    # -----------------------------------------------

    model = LSTMModel(
        input_dim=12,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2
    ).to(DEVICE)

    print("\nModel initialized!")

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    # -----------------------------------------------
    # OPTIMIZER AND LOSS
    # -----------------------------------------------

    optimizer = get_optimizer(model)

    loss_fn = get_loss_function()

    # Reduce learning rate when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    early_stopping = EarlyStopping()

    # -----------------------------------------------
    # TRAINING HISTORY
    # -----------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    # -----------------------------------------------
    # EPOCH LOOP
    # -----------------------------------------------

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        )

        # Training.
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            DEVICE
        )

        # Validation.
        val_loss = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            DEVICE
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # Store metrics.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        # Update scheduler.
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"LR:         {current_lr:.8f}")

        # Check whether validation improved.
        improved = early_stopping(val_loss)

        if improved:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_loss=val_loss,
                path=BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

        # Save history after every epoch.
        save_history(history)

        if early_stopping.early_stop:

            print("\nEarly stopping triggered.")

            break

    # -----------------------------------------------
    # TRAINING COMPLETED
    # -----------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("Best validation loss:", early_stopping.best_loss)

    print("Best model:", BEST_MODEL_PATH)

    print("Training history:", HISTORY_PATH)

    return history


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    history = train_model()
