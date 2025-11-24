# src/data/ingestion.py

import os
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

def get_pg_connection():
    """Return a psycopg2 connection using environment variables."""
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "postgres"),
        port=os.getenv("PG_PORT", "5432"),
        dbname=os.getenv("PG_DB", "airflow"),
        user=os.getenv("PG_USER", "airflow"),
        password=os.getenv("PG_PASSWORD", "airflow"),
    )
    return conn


def check_csv_file_exists(csv_path: str):
    """Check if a CSV exists and can be read."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    pd.read_csv(csv_path, nrows=5)  # Quick read to validate
    logger.info(f"CSV exists and readable: {csv_path}")


def run_basic_quality_checks_dag():
    """
    DAG-friendly wrapper to check meter tables.
    Ensures customers, meters, and readings have rows.
    """
    tables_to_check = ["customers_raw", "meters_raw", "meter_readings_raw"]
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            for table_name in tables_to_check:
                cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                total_rows = cur.fetchone()[0]
                if total_rows == 0:
                    raise ValueError(f"Data quality check failed: {table_name} is empty.")
    finally:
        conn.close()


def load_csv_to_raw_table(csv_path: str, table_name: str):
    """
    Load a CSV into a Postgres raw table.
    All columns are converted to TEXT for simplicity.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty!")

    # Convert numeric columns to Python native types
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else None)

    rows = [tuple(x) for x in df.to_numpy()]
    cols = list(df.columns)

    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join([f"{col} TEXT" for col in cols])}
        );
    """

    insert_query = f"""
        INSERT INTO {table_name} ({', '.join(cols)})
        VALUES %s
    """

    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
                execute_values(cur, insert_query, rows)
        logger.info(f"Ingested {len(rows)} rows into {table_name}.")
    finally:
        conn.close()
