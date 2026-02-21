import torch
import torch.nn as nn
import pytorch_lightning as pl
from xgboost import XGBRegressor


# -----------------------------
# Deep Learning Model (LSTM)
# -----------------------------
class LSTMForecaster(pl.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

    # -------- TRAINING --------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------- PREDICTION FIX (IMPORTANT) --------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch      # ignore labels during prediction
        y_hat = self(x)
        return y_hat.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -----------------------------
# Baseline Model (XGBoost)
# -----------------------------
def get_xgboost_model():
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    return model
