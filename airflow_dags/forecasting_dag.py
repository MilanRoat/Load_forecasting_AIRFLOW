from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from src.models.forecasting import train_prophet_model

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="forecasting_dag",
    description="Pipeline to train Prophet Forecasting model for Meter Data",
    default_args=default_args,
    start_date=datetime(2025, 11, 18),
    schedule_interval="0 2 * * *",  # Run daily at 2 AM
    catchup=False,
    tags=["meter", "forecasting", "prophet", "mlflow"],
    max_active_runs=1,
) as dag:

    # Wait for data pipeline to finish
    wait_for_data_pipeline = ExternalTaskSensor(
        task_id="wait_for_data_pipeline",
        external_dag_id="data_pipeline_dag",
        external_task_id=None,  # Wait for the entire DAG
        check_existence=True,
        timeout=600,
        allowed_states=["success"],
        mode="reschedule",
    )

    train_forecasting_model = PythonOperator(
        task_id="train_prophet_model",
        python_callable=train_prophet_model,
        provide_context=True,
    )

    wait_for_data_pipeline >> train_forecasting_model
