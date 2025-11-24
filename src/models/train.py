import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow

# -------------------------------
# Paths
# -------------------------------
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CUSTOMERS_CSV = os.path.join(RAW_DATA_DIR, "customers.csv")
METERS_CSV = os.path.join(RAW_DATA_DIR, "meters.csv")
READINGS_CSV = os.path.join(RAW_DATA_DIR, "meter_readings.csv")

print("🔧 [MODULE LOAD] train.py loaded")
print(f"🔧 RAW_DATA_DIR = {RAW_DATA_DIR}")
print(f"🔧 MODEL_DIR = {MODEL_DIR}")

def train_linear_regression(**kwargs):
    """
    Train Linear Regression model to predict 'units' from meter readings.
    """
    print("\n==================== TRAIN LINEAR REGRESSION ====================")

    readings = pd.read_csv(READINGS_CSV)
    print(f"📄 [TRAIN] Loaded meter readings: {readings.shape}")

    # Features & target
    feature_cols = ["voltage", "temperature", "power_factor", "load_kw", "frequency_hz"]
    target_col = "units"

    X = readings[feature_cols]
    y = readings[target_col]

    print(f"🧮 [TRAIN] Feature matrix shape: {X.shape}, target shape: {y.shape}")

    # Fill missing values if any
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🧪 [TRAIN] Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"🧪 [TRAIN] Test shapes: {X_test.shape}, {y_test.shape}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ [TRAIN] Model trained. MSE: {mse:.4f}, R2: {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 [TRAIN] Model saved to: {model_path}")

    # Push metrics to XCom
    kwargs["ti"].xcom_push(key="mse", value=float(mse))
    kwargs["ti"].xcom_push(key="r2", value=float(r2))
    print("==================== END TRAIN LINEAR REGRESSION ====================\n")


def log_model_to_mlflow(**kwargs):
    """
    Logs Linear Regression model and metrics to MLflow under 'Meter_data' experiment.
    """
    from mlflow.tracking import MlflowClient

    print("\n==================== LOG MODEL TO MLFLOW ====================")
    ti = kwargs["ti"]
    mse = ti.xcom_pull(task_ids="train_linear_regression_model", key="mse")
    r2 = ti.xcom_pull(task_ids="train_linear_regression_model", key="r2")

    tracking_uri = "http://mlflow_server:5000"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "Meter_data"
    mlflow.set_experiment(experiment_name)

    model_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = experiment.experiment_id

    run = client.create_run(experiment_id=exp_id)
    run_id = run.info.run_id

    # Log parameters and metrics
    client.log_param(run_id, "model_type", "LinearRegression")
    client.log_metric(run_id, "mse", mse)
    client.log_metric(run_id, "r2", r2)

    # Log artifact (model)
    client.log_artifact(run_id, model_path, artifact_path="model")

    print(f"✅ [LOG] Model and metrics logged to MLflow at: {tracking_uri}")
    print(f"🔗 [LOG] Run URL: {tracking_uri}/#/experiments/{exp_id}/runs/{run_id}")
    print("==================== END LOG MODEL TO MLFLOW ====================\n")
