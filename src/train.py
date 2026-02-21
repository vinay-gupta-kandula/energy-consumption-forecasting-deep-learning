import pandas as pd
import numpy as np
import torch
import optuna
import wandb
import logging
import json
import random

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from src.models import LSTMForecaster, get_xgboost_model

DATA_PATH = Path("data/processed/features.csv")
LOG_PATH = Path("logs/training.log")


# ----------------------- REPRODUCIBILITY -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()


# ----------------------- LOGGING -----------------------
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(message)s")


# ----------------------- Load dataset -----------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    target = "Global_active_power"
    X = df.drop(columns=[target]).values
    y = df[target].values
    return X, y


# ----------------------- Sequence builder -----------------------
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


# ----------------------- Optuna objective -----------------------
def objective(trial):

    window = trial.suggest_int("window", 24, 72)
    hidden = trial.suggest_int("hidden", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    wandb.init(project="energy-forecasting", config={
        "window": window,
        "hidden": hidden,
        "lr": lr
    }, reinit=True)

    X, y = load_data()
    Xs, ys = create_sequences(X, y, window)

    split = int(len(Xs) * 0.8)
    X_train, X_val = Xs[:split], Xs[split:]
    y_train, y_val = ys[:split], ys[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=64
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=64
    )

    model = LSTMForecaster(input_size=X.shape[1], hidden_size=hidden, lr=lr)
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)

    trainer.fit(model, train_loader)

    preds = trainer.predict(model, val_loader)
    preds = torch.cat(preds).detach().numpy()

    mae = mean_absolute_error(y_val[:len(preds)], preds)

    wandb.log({"val_mae": mae})
    wandb.finish()

    return mae


# ----------------------- Walk-forward validation -----------------------
def walk_forward_validation():
    logging.info("==== WALK FORWARD VALIDATION STARTED ====")

    X, y = load_data()
    tscv = TimeSeriesSplit(n_splits=3)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):

        logging.info(f"Starting walk-forward fold {fold}")

        # Baseline model
        logging.info("Training baseline model (XGBoost)")
        base = get_xgboost_model()
        base.fit(X[train_idx], y[train_idx])

        preds = base.predict(X[test_idx])
        mae = mean_absolute_error(y[test_idx], preds)

        logging.info(f"Evaluating fold {fold}")
        logging.info(f"Fold {fold} MAE: {mae}")

        logging.info(f"Completed fold {fold}")

    logging.info("==== WALK FORWARD VALIDATION FINISHED ====")


# ----------------------- MAIN -----------------------
def main():

    print("Running Optuna tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    print("Training final model with best parameters...")

    best_params = study.best_params
    window = best_params["window"]
    hidden = best_params["hidden"]
    lr = best_params["lr"]

    X, y = load_data()
    Xs, ys = create_sequences(X, y, window)

    split = int(len(Xs) * 0.8)
    X_train, y_train = Xs[:split], ys[:split]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=64
    )

    model = LSTMForecaster(input_size=X.shape[1], hidden_size=hidden, lr=lr)
    trainer = pl.Trainer(max_epochs=5, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/lstm_model.pth")

    meta = {
        "input_size": X.shape[1],
        "hidden_size": hidden,
        "window_size": window
    }

    with open("results/model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print("Saved model and metadata")

    print("Running walk-forward validation...")
    walk_forward_validation()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
