#!/usr/bin/env python3
"""
Resilient Magnetic Data Fetcher (Optimized for Large Databases)
---------------------------------------------------------------
✅ 15-min query windows (fast and safe)
✅ Streams results without buffering
✅ Downsamples server-side via id%N=0
✅ Auto-resume and retry with exponential backoff
✅ Efficient CSV streaming per day
"""

import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

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
    "use_pure": True,
    "consume_results": True
}

# ----------------------------- #
# 📅 DATE RANGE (EDIT HERE)
# ----------------------------- #
START_DATE = datetime(2025, 9, 26)
END_DATE   = datetime(2025, 9, 27)

# ----------------------------- #
# ⚙️ PARAMETERS
# ----------------------------- #
WINDOW_MINUTES = 15        # fetch data in 15-minute chunks
MAX_RETRIES = 999          # keep retrying until success
BASE_BACKOFF = 5           # seconds
DOWNSAMPLE_MOD = 60        # same as id%60=0 in your example
OUTPUT_DIR = "Fetched_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- #
# 🧠 QUERY TEMPLATE
# ----------------------------- #
QUERY_TEMPLATE = f"""
SELECT id, sensor_id, timestamp, b_x, b_y, b_z, lat, lon, alt, theta_x, theta_y, theta_z
FROM qnav_magneticdatamodel
WHERE timestamp BETWEEN %s AND %s
  AND id %% {DOWNSAMPLE_MOD} = 0
ORDER BY id ASC;
"""

# ----------------------------- #
# 🔁 Function: Fetch Data in Small Time Windows
# ----------------------------- #
def fetch_time_window(date_from, date_to, csv_path):
    """Fetch one small time window and append to CSV."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True, buffered=False)

            cursor.execute(QUERY_TEMPLATE, (
                date_from.strftime("%Y-%m-%d %H:%M:%S"),
                date_to.strftime("%Y-%m-%d %H:%M:%S")
            ))

            rows = []
            count = 0
            for row in cursor:
                rows.append(row)
                count += 1
                if len(rows) >= 5000:
                    df = pd.DataFrame(rows)
                    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
                    rows.clear()

            # flush remaining
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

            cursor.close()
            conn.close()

            print(f"   ✅ {date_from.strftime('%H:%M')}–{date_to.strftime('%H:%M')} | {count} rows")
            return  # success → exit retry loop

        except mysql.connector.Error as err:
            retries += 1
            delay = min(BASE_BACKOFF * (2 ** (retries - 1)), 300)
            print(f"   ⚠️ Lost connection: {err}")
            print(f"   🔄 Retrying after {delay}s (attempt {retries}/{MAX_RETRIES})...")
            time.sleep(delay)
            continue
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"   🚨 Unexpected error: {e}")
            time.sleep(10)
            continue


# ----------------------------- #
# 🧩 MAIN LOOP
# ----------------------------- #
if __name__ == "__main__":
    current_date = START_DATE

    while current_date < END_DATE:
        next_date = current_date + timedelta(days=1)
        csv_path = os.path.join(OUTPUT_DIR, f"MagneticData_{current_date.strftime('%Y-%m-%d')}.csv")

        print(f"\n📅 Fetching data for {current_date.date()}")

        window_start = current_date
        while window_start < next_date:
            window_end = min(window_start + timedelta(minutes=WINDOW_MINUTES), next_date)
            print(f"⏱️  Window: {window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')}")
            fetch_time_window(window_start, window_end, csv_path)
            window_start = window_end

        print(f"✅ Completed {current_date.date()}")
        current_date = next_date

    print("\n🎯 All days fetched successfully.")