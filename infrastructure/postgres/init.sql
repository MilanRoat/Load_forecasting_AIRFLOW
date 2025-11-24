-- Create the database for MLflow (as you specified)
CREATE DATABASE superset;
CREATE DATABASE mlflow;

-- Connect to the airflow DB to create the project tables
-- (Note: In a real-world setup, a migration tool or Airflow itself would do this)
\c airflow;


-- raw meter data
CREATE TABLE IF NOT EXISTS raw_meter_data (
    id SERIAL PRIMARY KEY,
    reading_date DATE NOT NULL,
    total_units_consumed NUMERIC(10, 2) NOT NULL
);
