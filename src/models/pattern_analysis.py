import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import mlflow
import mlflow.sklearn

# -------------------------------
# Paths
# -------------------------------
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

READINGS_CSV = os.path.join(RAW_DATA_DIR, "meter_readings.csv")

def train_kmeans_clustering(**kwargs):
    """
    Perform K-Means Clustering to identify usage patterns.
    """
    print("\n==================== TRAIN K-MEANS CLUSTERING ====================")
    
    if not os.path.exists(READINGS_CSV):
        raise FileNotFoundError(f"Meter readings file not found at {READINGS_CSV}")

    readings = pd.read_csv(READINGS_CSV)
    print(f"📄 [TRAIN] Loaded meter readings: {readings.shape}")

    # Features for clustering
    feature_cols = ["voltage", "power_factor", "load_kw", "frequency_hz"]
    X = readings[feature_cols]

    # Handle missing values
    X = X.fillna(X.median())

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    k = 3  # Low, Medium, High usage patterns
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    labels = kmeans.labels_
    score = silhouette_score(X_scaled, labels)
    
    print(f"✅ [TRAIN] K-Means trained with K={k}. Silhouette Score: {score:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"💾 [TRAIN] Model saved to: {model_path}")

    # Log to MLflow
    tracking_uri = "http://mlflow_server:5000"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Meter_Patterns"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("k", k)
        mlflow.log_metric("silhouette_score", score)
        
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        print(f"✅ [LOG] Model and metrics logged to MLflow experiment '{experiment_name}'")

    print("==================== END TRAIN K-MEANS CLUSTERING ====================\n")
