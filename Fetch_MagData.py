#!/usr/bin/env python3
"""
Resilient Magnetic Data Fetcher (ID-based chunking for large datasets)
---------------------------------------------------------------------
✅ Fetches data in small ID chunks to avoid server timeouts
✅ Auto-resumes using last fetched ID
✅ Downsamples server-side via MOD(id, N)=0
✅ Efficient CSV streaming per day
✅ Handles millions of rows safely
"""

import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ----------------------------- #
# 🔧 DATABASE CONFIGURATION
# ----------------------------- #
DB_CONFIG = {
    "host": "50.63.129.30",
    "user": "devuser",
    "password": "devuser@221133",
    "database": "dbqnaviitk",
    "port": 3306,
    "connection_timeout": 60,
    "connect_timeout": 60,
    "read_timeout": 120,
    "write_timeout": 120,
    "use_pure": True,
    "consume_results": True
}

# ----------------------------- #
# 📅 DATE RANGE (Edit for desired day)
# ----------------------------- #
TARGET_DATE = datetime(2025, 10, 23)  # only fetch for 23 Oct 2025
NEXT_DATE = TARGET_DATE + timedelta(days=1)

# ----------------------------- #
# ⚙️ PARAMETERS
# ----------------------------- #
CHUNK_SIZE = 5000         # number of rows per chunk
MAX_RETRIES = 999
BASE_BACKOFF = 5
DOWNSAMPLE_MOD = 60       # keep only rows where MOD(id, 60) = 0
OUTPUT_DIR = "Fetched_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, f"MagneticData_{TARGET_DATE.strftime('%Y-%m-%d')}.csv")
RESUME_FILE = os.path.join(OUTPUT_DIR, f"last_id_{TARGET_DATE.strftime('%Y-%m-%d')}.txt")

# ----------------------------- #
# 🧠 QUERY TEMPLATE
# ----------------------------- #
QUERY_TEMPLATE = f"""
SELECT id, sensor_id, timestamp, b_x, b_y, b_z, lat, lon, alt, theta_x, theta_y, theta_z
FROM qnav_magneticdatamodel
WHERE id > %s AND id <= %s
  AND MOD(id, {DOWNSAMPLE_MOD}) = 0
ORDER BY id ASC;
"""

# ----------------------------- #
# 🔁 Function to get min/max ID for the day
# ----------------------------- #
def get_id_range():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MIN(id), MAX(id) FROM qnav_magneticdatamodel "
                "WHERE timestamp BETWEEN %s AND %s;",
                (TARGET_DATE.strftime("%Y-%m-%d 00:00:00"), NEXT_DATE.strftime("%Y-%m-%d 00:00:00"))
            )
            min_id, max_id = cursor.fetchone()
            cursor.close()
            conn.close()
            return min_id or 0, max_id or 0
        except mysql.connector.Error as err:
            retries += 1
            delay = min(BASE_BACKOFF * (2 ** (retries - 1)), 300)
            print(f"   ⚠️ Error getting ID range: {err}, retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"   🚨 Unexpected error: {e}")
            time.sleep(10)

# ----------------------------- #
# 🔁 Fetch chunk by ID
# ----------------------------- #
def fetch_chunk(id_min, id_max, csv_path):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True, buffered=False)
            cursor.execute(QUERY_TEMPLATE, (id_min, id_max))
            rows = []
            count = 0
            for row in cursor:
                rows.append(row)
                count += 1
                if len(rows) >= 2000:
                    df = pd.DataFrame(rows)
                    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
                    rows.clear()
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
            cursor.close()
            conn.close()
            print(f"   ✅ IDs {id_min}-{id_max} | {count} rows")
            return count
        except mysql.connector.Error as err:
            retries += 1
            delay = min(BASE_BACKOFF * (2 ** (retries - 1)), 300)
            print(f"   ⚠️ Lost connection: {err}")
            print(f"   🔄 Retrying after {delay}s (attempt {retries}/{MAX_RETRIES})...")
            time.sleep(delay)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"   🚨 Unexpected error: {e}")
            time.sleep(10)

# ----------------------------- #
# 🧩 MAIN LOOP
# ----------------------------- #
if __name__ == "__main__":
    print(f"\n📅 Fetching magnetic data for {TARGET_DATE.date()}")
    
    # Load last fetched ID if exists (resume feature)
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r") as f:
            last_id = int(f.read().strip())
        print(f"🔁 Resuming from last ID: {last_id}")
    else:
        min_id, max_id = get_id_range()
        last_id = min_id - 1
        print(f"ID range for the day: {min_id}-{max_id}")

    # Determine max_id for the day
    _, max_id = get_id_range()
    
    while last_id < max_id:
        next_id = min(last_id + CHUNK_SIZE, max_id)
        fetch_chunk(last_id, next_id, CSV_FILE)
        last_id = next_id

        # Save last fetched ID to resume file
        with open(RESUME_FILE, "w") as f:
            f.write(str(last_id))

    print(f"\n🎯 Completed fetching for {TARGET_DATE.date()}")
