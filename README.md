# Energy Consumption Forecasting System

**Reproducible Deep Learning + Baseline Time-Series Pipeline**

---

## 1. Project Overview

This project implements a fully containerized end-to-end time-series forecasting system for multi-variate energy consumption data.

The system compares:

* A Deep Learning model (LSTM)
* A Traditional baseline model (XGBoost)

The entire workflow — preprocessing, feature engineering, hyperparameter tuning, walk-forward validation, model training, evaluation, and artifact generation — runs automatically using Docker and Docker Compose.

The objective is not only predictive accuracy, but also production-grade ML system design:

* Deterministic execution
* Proper time-series validation
* Baseline comparison
* Experiment tracking
* Hyperparameter optimization
* Fully reproducible containerized pipeline

---

## 2. Dataset

**Dataset:** Individual Household Electric Power Consumption Dataset

**Target variable:**
`Global_active_power`

The dataset contains:

* Date
* Time
* Voltage
* Global_intensity
* Sub_metering_1
* Sub_metering_2
* Sub_metering_3
* Other electrical measurements

Raw dataset location:

```
data/raw/household_power_consumption.txt
```

---

## 3. Repository Structure

```
energy-forecasting/
│
├── .dockerignore
├── .env
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
├── video.txt
│
├── data/
│   ├── raw/
│   │   └── household_power_consumption.txt
│   │
│   └── processed/
│       ├── cleaned.csv
│       └── features.csv
│
├── logs/
│   └── training.log
│
├── notebooks/
│
├── results/
│   ├── forecasts.csv
│   ├── forecast_visualization.png
│   ├── lstm_model.pth
│   ├── metrics.json
│   └── model_meta.json
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── tests/
│
└── wandb/
```

---

## 4. End-to-End Pipeline

The entire system runs sequentially:

```
Preprocessing → Feature Engineering → Hyperparameter Tuning → Training → Walk-Forward Validation → Evaluation → Artifact Generation
```

Execution is triggered by:

```
docker-compose up --build
```

No manual script execution is required.

---

## 5. System Components

### 5.1 Data Preprocessing (`src/preprocess.py`)

Responsibilities:

* Combine Date and Time columns into datetime index
* Convert numeric columns safely
* Handle missing values using forward/backward fill
* Resample data to hourly frequency
* Normalize features using scaling
* Save cleaned dataset

**Output:**

```
data/processed/cleaned.csv
```

---

### 5.2 Feature Engineering (`src/feature_engineering.py`)

Creates predictive features:

**Calendar Features**

* Hour
* Day of week
* Month

**Lag Features**

* Previous target values

**Rolling Statistics**

* Rolling mean
* Short-term trend windows

**Output:**

```
data/processed/features.csv
```

---

### 5.3 Model Implementation (`src/models.py`)

Two models are implemented:

**Deep Learning Model**

* LSTM (PyTorch)
* Sequential time-series modeling
* Tunable hidden size & learning rate
* Optimized using Optuna

**Baseline Model**

* XGBoost Regressor
* Provides benchmark comparison

---

### 5.4 Hyperparameter Optimization (`src/train.py`)

Uses:

* Optuna for automatic tuning
* PyTorch Lightning for training structure
* Weights & Biases (W&B) for tracking

Optimized parameters:

* Window size
* Hidden layer size
* Learning rate

Saved artifacts:

```
results/lstm_model.pth
results/model_meta.json
```

---

### 5.5 Walk-Forward Validation

Uses `TimeSeriesSplit` instead of random split.

Procedure:

1. Train on earlier time window
2. Validate on next period
3. Slide window forward
4. Repeat across folds

Logs saved:

```
logs/training.log
```

---

### 5.6 Evaluation (`src/evaluate.py`)

Both models evaluated on hold-out dataset.

**Metrics:**

* MAE
* RMSE
* MAPE
* Quantile Loss (p50, p95)

**Artifacts:**

```
results/metrics.json
results/forecasts.csv
results/forecast_visualization.png
```

---

## 6. Forecast Output Format

### `results/metrics.json`

Contains:

```
deep_learning_model:
    mae
    rmse
    mape
    quantile_loss_p50
    quantile_loss_p95

baseline_model:
    mae
    rmse
    mape
```

### `results/forecasts.csv`

Columns:

```
timestamp
actual
prediction
lower_bound
upper_bound
```

### `results/forecast_visualization.png`

Includes:

* Actual values
* LSTM predictions
* Prediction interval band

---

## 7. Docker Execution

### Build & Run Entire Pipeline

```
docker-compose up --build
```

The container automatically:

* Installs dependencies
* Runs preprocessing
* Engineers features
* Tunes hyperparameters
* Trains model
* Validates using walk-forward method
* Evaluates models
* Generates outputs

Successful completion message:

```
PIPELINE FINISHED SUCCESSFULLY
```

---

## 8. Environment Variables

File: `.env.example`

Example:

```
WANDB_API_KEY=your_wandb_api_key_here
WANDB_MODE=offline
```

To run locally:

1. Copy `.env.example` → `.env`
2. Insert W&B key (optional)
3. Offline mode works without credentials

No secrets are committed.

---

## 9. Reproducibility

The project guarantees:

* Identical results across environments
* No dependency conflicts
* No manual configuration
* Fully automated execution

All dependencies defined in:

```
requirements.txt
Dockerfile
docker-compose.yml
```

---

## 10. Engineering Considerations

**Why Walk-Forward Validation?**
Prevents future data leakage and simulates real-world forecasting.

**Why Baseline Comparison?**
Ensures deep learning adds measurable value.

**Why Docker?**
Ensures evaluator runs identical environment.

**Why Optuna?**
Automates hyperparameter search for optimal performance.

---

## 11. How to Verify

1. Clone repository
2. Ensure `.env` exists
3. Run:

```
docker-compose up --build
```

Confirm:

* No runtime errors
* Container exits successfully
* `results/` contains artifacts
* `logs/` contains `training.log`

---

### Reproducibility & Git Tracking Strategy

To ensure the project remains reproducible and lightweight, the repository follows an **artifact-free versioning policy**.

* The folder structure (`data/`, `results/`, `logs/`) is committed to Git so the pipeline has predefined output locations.
* Actual files inside these directories are **ignored using `.gitignore`** because they are automatically generated during execution.
* This guarantees that anyone cloning the repository can recreate the exact outputs by simply running Docker.

Specifically:

* `data/raw/` and `data/processed/` keep only placeholder `.keep` files
  → prevents uploading datasets while preserving directory structure

* `results/` contains generated outputs
  → `metrics.json`, `forecasts.csv`, `forecast_visualization.png` are created during pipeline execution

* `logs/` contains runtime logs
  → `training.log` is produced automatically while training

This approach ensures:

* No large or machine-specific files are stored in Git
* The repository remains clean and lightweight
* The pipeline execution fully demonstrates correctness and reproducibility


---

## 12. Summary

This project demonstrates:

* Time-series feature engineering
* Deep learning sequence modeling
* Baseline benchmarking
* Hyperparameter optimization
* Walk-forward validation
* Probabilistic forecasting
* MLOps integration
* Full Docker containerization

It reflects practical machine learning system design beyond simple model training.


