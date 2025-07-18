#!/usr/bin/env python3
"""
TEMPORARY ONE-USE SCRIPT: Add historical electricity price data to database
This script adds records from July 1, 2023 to June 30, 2024 to fill the gap in historical data.
Should only be run once - includes safety checks to prevent duplicate data.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os
import sys

# Configuration
DB_PATH = r"e:\Python\Huishouden\Elektriciteit\api\db\electricity_prices.db"
CSV_PATH = r"e:\Python\Huishouden\Elektriciteit\BelpexFilter.csv"


def validate_database():
    """Validate database exists and has expected structure."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM electricity_prices")
        current_count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM electricity_prices")
        date_range = cursor.fetchone()

        conn.close()

        print(f"✅ Database found with {current_count} records")
        print(f"✅ Current date range: {date_range[0]} to {date_range[1]}")
        return True

    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False


def validate_csv():
    """Validate CSV file exists and has expected format."""
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found at: {CSV_PATH}")
        return False

    try:
        # Read first few lines to validate format
        df_sample = pd.read_csv(CSV_PATH, sep=';', nrows=5)

        if 'Date' not in df_sample.columns or 'Euro' not in df_sample.columns:
            print(
                f"❌ CSV file missing expected columns. Found: {df_sample.columns.tolist()}")
            return False

        print(f"✅ CSV file found with expected format")
        print(
            f"✅ Sample data: {df_sample.iloc[0]['Date']} -> {df_sample.iloc[0]['Euro']}")
        return True

    except Exception as e:
        print(f"❌ CSV validation failed: {e}")
        return False


def check_historical_gap():
    """Check if there's a gap in historical data that needs filling."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check earliest date in database
    cursor.execute("SELECT MIN(date) FROM electricity_prices")
    earliest_db_date = cursor.fetchone()[0]

    conn.close()

    # Check if CSV has older data
    try:
        df_sample = pd.read_csv(CSV_PATH, sep=';')
        df_sample['timestamp'] = pd.to_datetime(
            df_sample['Date'], format='%d/%m/%Y %H:%M:%S')
        df_sample['date'] = df_sample['timestamp'].dt.strftime('%Y-%m-%d')
        oldest_csv_date = df_sample['date'].min()

        if oldest_csv_date < earliest_db_date:
            print(f"✅ Found historical gap:")
            print(f"   Database starts: {earliest_db_date}")
            print(f"   CSV has data from: {oldest_csv_date}")
            return True
        else:
            print(f"ℹ️ No older historical data available:")
            print(f"   Database starts: {earliest_db_date}")
            print(f"   CSV oldest data: {oldest_csv_date}")
            print("ℹ️ No gap to fill - script will exit")
            return False

    except Exception as e:
        print(f"❌ Error checking CSV data: {e}")
        return False


def parse_and_filter_csv():
    """Parse CSV and filter for historical records only."""
    print("📊 Reading and processing CSV file...")

    # Read the CSV file
    df = pd.read_csv(CSV_PATH, sep=';')

    print(f"✅ Loaded {len(df)} records from CSV")

    # Parse the date and create required columns
    df['timestamp'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['timestamp'].dt.hour

    # Parse Euro column (remove commas and convert to float)
    df['price_eur'] = df['Euro'].str.replace(',', '.').astype(float)
    df['price_raw'] = '€ ' + df['Euro']

    # Get the earliest date currently in the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(date) FROM electricity_prices")
    earliest_db_date = cursor.fetchone()[0]
    conn.close()

    # Filter for data older than what's currently in the database
    historical_df = df[df['date'] < earliest_db_date].copy()

    # Sort by timestamp ascending (oldest first)
    historical_df = historical_df.sort_values('timestamp')

    print(f"✅ Filtered to {len(historical_df)} historical records")

    if len(historical_df) > 0:
        print(
            f"✅ Date range: {historical_df['date'].min()} to {historical_df['date'].max()}")
    else:
        print(
            f"ℹ️ No data found older than current database earliest date: {earliest_db_date}")

    return historical_df


def insert_historical_data(df):
    """Insert historical data into database with safety checks."""
    if len(df) == 0:
        print("ℹ️ No historical data to insert")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check for existing records in the date range (safety check)
        min_date = df['date'].min()
        max_date = df['date'].max()

        cursor.execute(
            "SELECT COUNT(*) FROM electricity_prices WHERE date BETWEEN ? AND ?",
            (min_date, max_date)
        )
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            print(
                f"⚠️ Found {existing_count} existing records in date range {min_date} to {max_date}")
            response = input(
                "Continue with insertion? This might create duplicates (y/N): ")
            if response.lower() != 'y':
                print("❌ Insertion cancelled by user")
                return

        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            records.append((
                row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                row['date'],
                row['hour'],
                row['price_eur'],
                row['price_raw']
            ))

        # Insert records
        print(f"💾 Inserting {len(records)} historical records...")

        cursor.executemany("""
            INSERT OR IGNORE INTO electricity_prices 
            (timestamp, date, hour, price_eur, price_raw)
            VALUES (?, ?, ?, ?, ?)
        """, records)

        rows_inserted = cursor.rowcount
        conn.commit()

        print(f"✅ Successfully inserted {rows_inserted} new records")

        # Verify final count
        cursor.execute("SELECT COUNT(*) FROM electricity_prices")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM electricity_prices")
        date_range = cursor.fetchone()

        print(f"✅ Database now contains {total_count} total records")
        print(f"✅ New date range: {date_range[0]} to {date_range[1]}")

    except Exception as e:
        print(f"❌ Error during insertion: {e}")
        conn.rollback()

    finally:
        conn.close()


def main():
    """Main execution function."""
    print("🔄 TEMPORARY HISTORICAL DATA IMPORT SCRIPT")
    print("=" * 50)

    # Validation steps
    if not validate_database():
        sys.exit(1)

    if not validate_csv():
        sys.exit(1)

    if not check_historical_gap():
        sys.exit(0)

    # Get user confirmation
    print("\n⚠️ This script will add historical electricity price data to your database.")
    print("⚠️ It should only be run once to fill historical gaps.")
    response = input("\n➡️ Do you want to proceed? (y/N): ")

    if response.lower() != 'y':
        print("❌ Operation cancelled by user")
        sys.exit(0)

    # Process and insert data
    try:
        historical_df = parse_and_filter_csv()
        insert_historical_data(historical_df)

        print("\n✅ Historical data import completed successfully!")
        print("ℹ️ This script can now be deleted as it's a one-time use tool.")

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
