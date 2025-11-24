# src/models/train_loan.py

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from mlflow.tracking import MlflowClient

# -------------------------------
# Paths
# -------------------------------

# Data directory (shared with Airflow)
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")

# Artifacts directory INSIDE src/models (and mounted into containers)
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Loan CSVs created by create_datasets.py
LOAN_APPLICANTS_CSV = os.path.join(RAW_DATA_DIR, "loan_applicants.sql.csv")
LOAN_FINANCIALS_CSV = os.path.join(RAW_DATA_DIR, "loan_financials.csv")
LOAN_TARGET_CSV = os.path.join(RAW_DATA_DIR, "loan_target.csv")

print("🔧 [MODULE LOAD] train_loan.py loaded (LOAN VERSION)")
print(f"🔧 [MODULE LOAD] RAW_DATA_DIR          = {RAW_DATA_DIR}")
print(f"🔧 [MODULE LOAD] ARTIFACTS_DIR         = {ARTIFACTS_DIR}")
print(f"🔧 [MODULE LOAD] MODEL_DIR             = {MODEL_DIR}")
print(f"🔧 [MODULE LOAD] LOAN_APPLICANTS_CSV   = {LOAN_APPLICANTS_CSV}")
print(f"🔧 [MODULE LOAD] LOAN_FINANCIALS_CSV   = {LOAN_FINANCIALS_CSV}")
print(f"🔧 [MODULE LOAD] LOAN_TARGET_CSV       = {LOAN_TARGET_CSV}")


def train_loan_model(**kwargs):
    """
    Train Logistic Regression model on Loan Prediction data and push accuracy to XCom.
    """
    print("\n==================== TRAIN LOAN LOGISTIC REGRESSION ====================")
    print(f"📂 [TRAIN] CWD inside task: {os.getcwd()}")
    print(f"📂 [TRAIN] RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"📂 [TRAIN] MODEL_DIR: {MODEL_DIR}")
    print(f"📄 [TRAIN] LOAN_APPLICANTS_CSV: {LOAN_APPLICANTS_CSV} (exists={os.path.exists(LOAN_APPLICANTS_CSV)})")
    print(f"📄 [TRAIN] LOAN_FINANCIALS_CSV: {LOAN_FINANCIALS_CSV} (exists={os.path.exists(LOAN_FINANCIALS_CSV)})")
    print(f"📄 [TRAIN] LOAN_TARGET_CSV: {LOAN_TARGET_CSV} (exists={os.path.exists(LOAN_TARGET_CSV)})")

    # Load data
    applicants = pd.read_csv(LOAN_APPLICANTS_CSV)
    financials = pd.read_csv(LOAN_FINANCIALS_CSV)
    target = pd.read_csv(LOAN_TARGET_CSV)

    print("🧮 [TRAIN] Joining dataframes on Loan_ID...")
    df = (
        applicants.merge(financials, on="Loan_ID")
                  .merge(target, on="Loan_ID")
    )
    print(f"🧮 [TRAIN] Combined DF shape: {df.shape}")

    # Target: Loan_Status (Y/N → 1/0)
    if "Loan_Status" not in df.columns:
        raise ValueError("❌ 'Loan_Status' column missing in merged loan dataset.")

    y = df["Loan_Status"].map({"Y": 1, "N": 0})
    print(f"🧮 [TRAIN] Target distribution:\n{y.value_counts()}")

    # Features: similar style to Titanic example, explicit column list
    feature_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"🧮 [TRAIN] Using feature columns: {feature_cols}")

    X = df[feature_cols]
    print(f"🧮 [TRAIN] Feature matrix shape: {X.shape}, target shape: {y.shape}")

    # Handle missing values (simple strategy)
    if "LoanAmount" in X.columns:
        X["LoanAmount"] = X["LoanAmount"].fillna(X["LoanAmount"].median())
    if "Loan_Amount_Term" in X.columns:
        X["Loan_Amount_Term"] = X["Loan_Amount_Term"].fillna(X["Loan_Amount_Term"].median())
    if "Credit_History" in X.columns:
        X["Credit_History"] = X["Credit_History"].fillna(X["Credit_History"].mode()[0])

    # Categorical and numeric split
    cat_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    cat_features = [c for c in cat_features if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    print(f"🧮 [TRAIN] Categorical features: {cat_features}")
    print(f"🧮 [TRAIN] Numeric features: {num_features}")

    # Encode categorical
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    if cat_features:
        X_cat = pd.DataFrame(
            encoder.fit_transform(X[cat_features]),
            columns=encoder.get_feature_names_out(cat_features),
        )
        X_num = X[num_features].reset_index(drop=True) if num_features else pd.DataFrame()
        X_prepared = pd.concat(
            [X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1
        )
    else:
        X_prepared = X.copy()
        print("⚠️ [TRAIN] No categorical features to encode.")

    print(f"🧮 [TRAIN] Prepared feature matrix shape: {X_prepared.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"🧪 [TRAIN] Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"🧪 [TRAIN] Test shapes: {X_test.shape}, {y_test.shape}")

    # Train model
    print("🤖 [TRAIN] Training LogisticRegression(max_iter=500)...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ [TRAIN] Model trained. Accuracy: {acc}")

    # Save model & encoder to local artifacts dir
    model_path = os.path.join(MODEL_DIR, "loan_logistic_regression_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "loan_encoder.pkl")

    print(f"💾 [TRAIN] Saving model to:   {model_path}")
    print(f"💾 [TRAIN] Saving encoder to: {encoder_path}")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    print(f"📁 [TRAIN] MODEL_DIR listing: {os.listdir(MODEL_DIR)}")

    ti = kwargs["ti"]
    print("📤 [TRAIN] Pushing accuracy/model_path/encoder_path to XCom")
    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="encoder_path", value=encoder_path)

    print("==================== END TRAIN LOAN LOGISTIC REGRESSION ====================\n")


def log_loan_model_to_mlflow(**kwargs):
    """
    Pulls accuracy from XCom and logs model + metrics to MLflow.
    Uses HTTP tracking URI and logs artifacts directly.
    """
    print("\n==================== LOG LOAN MODEL TO MLFLOW ====================")
    print(f"📂 [LOG] CWD inside task: {os.getcwd()}")
    print(f"📂 [LOG] MODEL_DIR: {MODEL_DIR}")

    ti = kwargs["ti"]

    # 👇 MUST match DAG task_id "train_loan_model"
    acc = ti.xcom_pull(task_ids="train_loan_model", key="accuracy")
    model_path = ti.xcom_pull(task_ids="train_loan_model", key="model_path")
    encoder_path = ti.xcom_pull(task_ids="train_loan_model", key="encoder_path")

    print(f"📥 [LOG] Pulled accuracy from XCom: {acc}")
    print(f"📥 [LOG] Pulled model_path from XCom: {model_path}")
    print(f"📥 [LOG] Pulled encoder_path from XCom: {encoder_path}")

    if acc is None:
        raise ValueError("❌ Accuracy value not found in XCom. Did the training task succeed?")

    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "loan_logistic_regression_model.pkl")
    if encoder_path is None:
        encoder_path = os.path.join(MODEL_DIR, "loan_encoder.pkl")

    tracking_uri = "http://mlflow_server:5000"  # same as before
    print(f"🔗 [LOG] Setting MLflow tracking URI to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "loan_logistic_regression"  # 👈 LOAN experiment
    print(f"🧪 [LOG] Setting MLflow experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)

    # Confirm what tracking URI MLflow sees
    effective_uri = mlflow.get_tracking_uri()
    print(f"🔍 [LOG] Effective MLflow tracking URI: {effective_uri}")

    print(f"📄 [LOG] Expecting model at:   {model_path} (exists={os.path.exists(model_path)})")
    print(f"📄 [LOG] Expecting encoder at: {encoder_path} (exists={os.path.exists(encoder_path)})")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"❌ Encoder file not found at {encoder_path}")

    # Use explicit MlflowClient so we know it's bound to HTTP URI
    print("🧑‍💻 [LOG] Creating MlflowClient with tracking URI...")
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("🧪 [LOG] Experiment does not exist yet. Creating...")
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = experiment.experiment_id
        print(f"🧪 [LOG] Found existing experiment id: {exp_id}")

    print("🏃 [LOG] Creating new MLflow run...")
    run = client.create_run(experiment_id=exp_id)
    run_id = run.info.run_id
    print(f"🏃 [LOG] Run created with run_id: {run_id}")

    # Log params and metrics
    print("📊 [LOG] Logging params and metrics...")
    client.log_param(run_id, "model_type", "LogisticRegression")
    client.log_param(run_id, "max_iter", 500)
    client.log_metric(run_id, "accuracy", float(acc))

    # Log artifacts
    print("📦 [LOG] Logging artifacts (model + encoder) to run...")
    client.log_artifact(run_id, model_path, artifact_path="model")
    client.log_artifact(run_id, encoder_path, artifact_path="model")

    print(f"✅ [LOG] Model, encoder, and metrics logged to MLflow at: {tracking_uri}")
    print(f"🔗 [LOG] Run URL: {tracking_uri}/#/experiments/{exp_id}/runs/{run_id}")
    print("==================== END LOG LOAN MODEL TO MLFLOW ====================\n")
