# airflow_dags/training_pipeline_dag.py

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.models.train import train_linear_regression, log_model_to_mlflow

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

print("🚀 [DAG PARSE] Loading meter_training_pipeline_dag...")

with DAG(
    dag_id="meter_training_pipeline_dag",
    description="Training pipeline for Meter Linear Regression model",
    default_args=default_args,
    start_date=datetime(2021, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["meter", "training", "mlflow"],
    max_active_runs=1,
) as dag:

    print("🧱 [DAG PARSE] Defining tasks train_linear_regression_model and log_model_to_mlflow")

    train_model_task = PythonOperator(
        task_id="train_linear_regression_model",
        python_callable=train_linear_regression,
        provide_context=True,
    )

    log_model_task = PythonOperator(
        task_id="log_model_to_mlflow",
        python_callable=log_model_to_mlflow,
        provide_context=True,
    )

    train_model_task >> log_model_task

    print("✅ [DAG PARSE] DAG meter_training_pipeline_dag is fully defined.")
