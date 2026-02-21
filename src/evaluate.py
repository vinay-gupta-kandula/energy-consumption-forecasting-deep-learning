import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models import LSTMForecaster, get_xgboost_model


RESULTS_DIR = "results"
DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "results/lstm_model.pth"
META_PATH = "results/model_meta.json"


# -------------------------
# Sequence builder
# -------------------------
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


def mape(y, pred):
    return float(np.mean(np.abs((y - pred) / (y + 1e-8))) * 100)


# -------------------------
def evaluate():

    print("Running evaluation...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ================= LOAD MODEL METADATA =================
    with open(META_PATH) as f:
        meta = json.load(f)

    window = meta["window_size"]
    hidden = meta["hidden_size"]
    input_size = meta["input_size"]

    print(f"Loaded model config → window={window}, hidden={hidden}, input_size={input_size}")

    # ================= LOAD DATA =================
    df = pd.read_csv(DATA_PATH, index_col=0)

    target_col = "Global_active_power"
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    Xs, ys = create_sequences(X, y, window)

    split = int(len(Xs) * 0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]

    # ================= DEEP LEARNING MODEL =================
    model = LSTMForecaster(input_size=input_size, hidden_size=hidden)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred_dl = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

    dl_mae = mean_absolute_error(y_test, pred_dl)
    dl_rmse = np.sqrt(mean_squared_error(y_test, pred_dl))

    # ================= BASELINE MODEL =================
    base = get_xgboost_model()
    base.fit(X_train.reshape(len(X_train), -1), y_train)
    pred_base = base.predict(X_test.reshape(len(X_test), -1))

    base_mae = mean_absolute_error(y_test, pred_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, pred_base))

    # ================= PREDICTION INTERVALS =================
    residual_std = np.std(y_test - pred_dl)

    lower = pred_dl - 1.96 * residual_std
    upper = pred_dl + 1.96 * residual_std

    # ================= SAVE METRICS =================
    metrics = {
        "deep_learning_model": {
            "mae": float(dl_mae),
            "rmse": float(dl_rmse),
            "mape": mape(y_test, pred_dl),
            "quantile_loss_p50": float(np.mean(np.abs(y_test - pred_dl))),
            "quantile_loss_p95": float(np.mean(np.maximum(0.95*(y_test-pred_dl),0.05*(pred_dl-y_test))))
        },
        "baseline_model": {
            "mae": float(base_mae),
            "rmse": float(base_rmse),
            "mape": mape(y_test, pred_base)
        }
    }

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ================= SAVE FORECAST CSV =================
    timestamps = np.arange(len(y_test))

    df_out = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y_test,
        "prediction": pred_dl,
        "lower_bound": lower,
        "upper_bound": upper
    })

    df_out.to_csv(f"{RESULTS_DIR}/forecasts.csv", index=False)

    # ================= PLOT =================
    plt.figure(figsize=(12,5))
    plt.plot(y_test[:200], label="Actual")
    plt.plot(pred_dl[:200], label="Prediction")
    plt.fill_between(
        np.arange(200),
        lower[:200],
        upper[:200],
        alpha=0.3,
        label="Prediction Interval"
    )
    plt.legend()
    plt.title("Forecast vs Actual with Confidence Interval")
    plt.savefig(f"{RESULTS_DIR}/forecast_visualization.png")
    plt.close()

    print("\nEvaluation finished successfully!")
    print("Files saved inside /results folder")


# -------------------------
if __name__ == "__main__":
    evaluate()
