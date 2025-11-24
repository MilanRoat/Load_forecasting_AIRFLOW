# src/models/inference.py

import os
import pandas as pd
import joblib
import mlflow
from sklearn.preprocessing import OneHotEncoder

# Paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../mlflow_artifacts/models')

PASSENGERS_CSV = os.path.join(RAW_DATA_DIR, 'passengers.sql.csv')
TICKET_CSV = os.path.join(RAW_DATA_DIR, 'ticket_info.csv')
TARGET_CSV = os.path.join(RAW_DATA_DIR, 'target.csv')

def load_latest_model():
    """
    Loads the latest model from MLflow artifacts folder
    """
    model_path = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    return model

def prepare_features_for_inference():
    """
    Prepares features for inference
    """
    passengers = pd.read_csv(PASSENGERS_CSV)
    ticket_info = pd.read_csv(TICKET_CSV)

    df = passengers.merge(ticket_info, on='PassengerId')

    X = df[['age', 'sex', 'who', 'pclass', 'fare', 'embarked']]

    # Fill missing
    X['age'] = X['age'].fillna(X['age'].median())
    X['fare'] = X['fare'].fillna(X['fare'].median())
    X['embarked'] = X['embarked'].fillna('S')

    # Load encoder
    encoder_path = os.path.join(MODEL_DIR, 'encoder.pkl')
    encoder = joblib.load(encoder_path)
    X_encoded = pd.DataFrame(encoder.transform(X[['sex', 'who', 'embarked']]), columns=encoder.get_feature_names_out(['sex', 'who', 'embarked']))
    X_num = X.drop(columns=['sex', 'who', 'embarked'])
    X_prepared = pd.concat([X_num.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

    print(f"✅ Features prepared for inference. Shape: {X_prepared.shape}")
    return X_prepared

def make_predictions():
    """
    Loads model, prepares features, and makes predictions
    """
    model = load_latest_model()
    X_prepared = prepare_features_for_inference()
    predictions = model.predict(X_prepared)
    print(f"✅ Predictions made. Sample: {predictions[:10]}")

    # Optionally save predictions
    pred_path = os.path.join(RAW_DATA_DIR, 'predictions.csv')
    pd.DataFrame({'PassengerId': pd.read_csv(PASSENGERS_CSV)['PassengerId'], 'predicted_survived': predictions}).to_csv(pred_path, index=False)
    print(f"✅ Predictions saved at {pred_path}")
