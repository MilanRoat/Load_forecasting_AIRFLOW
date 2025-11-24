# airflow_dags/data_pipeline_dag.py

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.data.ingestion import (
    check_csv_file_exists,
    load_csv_to_raw_table,
    run_basic_quality_checks_dag,
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="data_pipeline_dag",
    description=(
        "Pipeline to ingest Meter datasets (customers, meters, readings) "
        "into Postgres raw layer and run basic quality checks."
    ),
    default_args=default_args,
    start_date=datetime(2025, 11, 18),
    schedule_interval="*/15 * * * *",
    catchup=False,
    tags=["meter", "ingestion", "postgres"],
    max_active_runs=1,
) as dag:

    # Paths inside the Airflow container
    CUSTOMERS_CSV_PATH = "/opt/airflow/data/raw/customers.csv"
    METERS_CSV_PATH = "/opt/airflow/data/raw/meters.csv"
    READINGS_CSV_PATH = "/opt/airflow/data/raw/meter_readings.csv"

    # --------------------
    # Stage 1: Check CSV Exists
    # --------------------
    check_customers_csv = PythonOperator(
        task_id="check_customers_csv_exists",
        python_callable=check_csv_file_exists,
        op_kwargs={"csv_path": CUSTOMERS_CSV_PATH},
    )

    check_meters_csv = PythonOperator(
        task_id="check_meters_csv_exists",
        python_callable=check_csv_file_exists,
        op_kwargs={"csv_path": METERS_CSV_PATH},
    )

    check_readings_csv = PythonOperator(
        task_id="check_meter_readings_csv_exists",
        python_callable=check_csv_file_exists,
        op_kwargs={"csv_path": READINGS_CSV_PATH},
    )

    # --------------------
    # Stage 2: Load CSV into Postgres
    # --------------------
    load_customers_task = PythonOperator(
        task_id="load_customers_to_postgres",
        python_callable=load_csv_to_raw_table,
        op_kwargs={"csv_path": CUSTOMERS_CSV_PATH, "table_name": "customers_raw"},
    )

    load_meters_task = PythonOperator(
        task_id="load_meters_to_postgres",
        python_callable=load_csv_to_raw_table,
        op_kwargs={"csv_path": METERS_CSV_PATH, "table_name": "meters_raw"},
    )

    load_readings_task = PythonOperator(
        task_id="load_meter_readings_to_postgres",
        python_callable=load_csv_to_raw_table,
        op_kwargs={"csv_path": READINGS_CSV_PATH, "table_name": "meter_readings_raw"},
    )

    # --------------------
    # Stage 3: Run Data Quality Checks
    # --------------------
    quality_check_task = PythonOperator(
        task_id="run_basic_quality_checks",
        python_callable=run_basic_quality_checks_dag,
    )

    # --------------------
    # Task Chaining
    # --------------------
    check_customers_csv >> load_customers_task >> quality_check_task
    check_meters_csv >> load_meters_task >> quality_check_task
    check_readings_csv >> load_readings_task >> quality_check_task
