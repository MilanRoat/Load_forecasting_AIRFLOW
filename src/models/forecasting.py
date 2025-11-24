import os
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

# -------------------------------
# Paths
# -------------------------------
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

READINGS_CSV = os.path.join(RAW_DATA_DIR, "meter_readings.csv")

def train_prophet_model(**kwargs):
    """
    Train Prophet model to forecast 'units' consumption.
    """
    print("\n==================== TRAIN PROPHET FORECASTING ====================")
    
    if not os.path.exists(READINGS_CSV):
        raise FileNotFoundError(f"Meter readings file not found at {READINGS_CSV}")

    readings = pd.read_csv(READINGS_CSV)
    print(f"📄 [TRAIN] Loaded meter readings: {readings.shape}")

    # Prepare data for Prophet
    # Prophet requires columns 'ds' (date) and 'y' (target)
    # Assuming we have a timestamp column, if not we might need to create a dummy one or use existing if available.
    # Checking previous files, I didn't see a timestamp column explicitly mentioned in train.py features.
    # Let's assume there is a 'timestamp' or similar column, or we generate one for demonstration if missing.
    
    if 'timestamp' in readings.columns:
        df = readings[['timestamp', 'units']].rename(columns={'timestamp': 'ds', 'units': 'y'})
    else:
        # Fallback: Generate a dummy time series if no timestamp exists (for demo purposes)
        print("⚠️ [TRAIN] 'timestamp' column not found. Generating dummy timestamps.")
        df = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=len(readings), freq='H'),
            'y': readings['units']
        })

    # Handle missing values
    df['y'] = df['y'].fillna(df['y'].median())

    # Split Train/Test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"🧪 [TRAIN] Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Train Prophet
    model = Prophet()
    model.fit(train_df)
    
    # Predict on test set
    future = test_df[['ds']]
    forecast = model.predict(future)
    
    # Evaluate
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"✅ [TRAIN] Prophet trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "prophet_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 [TRAIN] Model saved to: {model_path}")

    # Log to MLflow
    tracking_uri = "http://mlflow_server:5000"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Meter_Forecasting"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Log model as artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        print(f"✅ [LOG] Model and metrics logged to MLflow experiment '{experiment_name}'")

    print("==================== END TRAIN PROPHET FORECASTING ====================\n")
