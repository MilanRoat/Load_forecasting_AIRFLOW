import os
import pandas as pd

# --- Step 1: Define paths ---
BASE_DIR = os.path.dirname(__file__)

# Raw data directory
RAW_DATA_DIR = os.path.join(BASE_DIR, "../../data/raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Original meter dataset
METER_DATA_FILE = os.path.join(RAW_DATA_DIR, "merged_meter_data.csv")

# Output files
CUSTOMER_FILE = os.path.join(RAW_DATA_DIR, "customers.csv")
METER_FILE = os.path.join(RAW_DATA_DIR, "meters.csv")
READINGS_FILE = os.path.join(RAW_DATA_DIR, "meter_readings.csv")

def main():
    # --- Step 2: Load raw meter dataset ---
    if not os.path.exists(METER_DATA_FILE):
        raise FileNotFoundError(
            f"❌ Could not find {METER_DATA_FILE}. "
            f"Please place your merged_meter_data.csv in {RAW_DATA_DIR}"
        )

    df = pd.read_csv(METER_DATA_FILE)
    print(f"Loaded raw meter dataset with shape: {df.shape}")

    # --- Step 3: Split into separate tables ---

    # 1️⃣ Customers table: unique customer info
    customer_cols = ["customer_id", "name", "mobile_number", "address", "city", "pincode"]
    customers = df[customer_cols].drop_duplicates().reset_index(drop=True)

    # 2️⃣ Meters table: unique meter info
    meter_cols = ["meter_id", "customer_id", "connection_type", "tariff_plan", "connection_date"]
    meters = df[meter_cols].drop_duplicates().reset_index(drop=True)

    # 3️⃣ Meter readings: time-series readings
    reading_cols = [
        "meter_id", "id", "reading_date", "units", "voltage",
        "temperature", "power_factor", "load_kw", "frequency_hz", "phase", "status"
    ]
    readings = df[reading_cols].copy()

    # --- Step 4: Save the separate CSV files ---
    customers.to_csv(CUSTOMER_FILE, index=False)
    meters.to_csv(METER_FILE, index=False)
    readings.to_csv(READINGS_FILE, index=False)

    print("✅ Datasets created:")
    print(f" - Customers -> {CUSTOMER_FILE}")
    print(f" - Meters -> {METER_FILE}")
    print(f" - Meter readings -> {READINGS_FILE}")

if __name__ == "__main__":
    main()
