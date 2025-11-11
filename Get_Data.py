#!/usr/bin/env python3
"""
Streamlit App for Fetching Magnetic Field Data from Database
------------------------------------------------------------
Allows users to fetch magnetic field data for specific time periods
from the continuous 24/7 database collection.
"""

import streamlit as st
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import time
import io

# Page configuration
st.set_page_config(
    page_title="Magnetic Data Fetcher",
    page_icon="🧲",
    layout="wide"
)

# Database configuration
DB_CONFIG = {
    'host': '50.63.129.30',
    'user': 'devuser',
    'password': 'devuser@221133',
    'database': 'dbqnaviitk',
    'port': 3306,
    'connection_timeout': 60,
    'connect_timeout': 60,
    'read_timeout': 120,
    'write_timeout': 120,
    'use_pure': True,
    'consume_results': True
}

# Table name
TABLE_NAME = "qnav_magneticdatamodel"

# Initialize session state
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None
if 'fetch_status' not in st.session_state:
    st.session_state.fetch_status = None


def test_connection():
    """Test database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True, "Connection successful"
    except mysql.connector.Error as err:
        return False, f"Connection failed: {err}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def get_available_sensors():
    """Get list of available sensors from database"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT sensor_id FROM {TABLE_NAME} ORDER BY sensor_id")
        sensors = [row[0] for row in cursor.fetchall() if row[0] is not None]
        cursor.close()
        conn.close()
        return sensors
    except Exception as e:
        st.error(f"Error fetching sensors: {e}")
        return []


def get_data_range():
    """Get the date range of available data"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE_NAME}")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result and result[0] and result[1]:
            return result[0], result[1]
        return None, None
    except Exception as e:
        st.error(f"Error fetching data range: {e}")
        return None, None


def fetch_data(start_time, end_time, columns=None, sensor_ids=None, downsample=None, 
               chunk_size=10000, order_by="timestamp", order_direction="ASC", 
               limit_rows=None, id_min=None, id_max=None, use_id_range=False):
    """
    Fetch data from database for specified time range
    
    Parameters:
    - start_time: Start timestamp (datetime)
    - end_time: End timestamp (datetime)
    - columns: List of column names to fetch (None = all columns)
    - sensor_ids: List of sensor IDs to filter (None = all sensors)
    - downsample: Downsample factor (None = no downsampling)
    - chunk_size: Number of rows to fetch per chunk
    - order_by: Column to order by ("timestamp", "id", or custom)
    - order_direction: "ASC" or "DESC"
    - limit_rows: Maximum number of rows to fetch (None = no limit)
    - id_min: Minimum ID value (for ID-based filtering)
    - id_max: Maximum ID value (for ID-based filtering)
    - use_id_range: Whether to use ID range instead of timestamp
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True, buffered=False)
        
        # Default columns if not specified
        if columns is None or len(columns) == 0:
            columns = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z", "lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]
        
        # Build SELECT clause
        columns_str = ', '.join(columns)
        
        # Build query
        query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE "
        params = []
        
        # Use ID range or timestamp range
        if use_id_range and id_min is not None and id_max is not None:
            query += "id >= %s AND id <= %s"
            params.extend([id_min, id_max])
        else:
            query += "timestamp >= %s AND timestamp <= %s"
            params.extend([start_time, end_time])
        
        # Add sensor filter if specified
        if sensor_ids and len(sensor_ids) > 0:
            placeholders = ','.join(['%s'] * len(sensor_ids))
            query += f" AND sensor_id IN ({placeholders})"
            params.extend(sensor_ids)
        
        # Add downsampling if specified
        if downsample and downsample > 1:
            query += f" AND MOD(id, {downsample}) = 0"
        
        # Add ID range filter (additional to timestamp)
        if not use_id_range:
            if id_min is not None:
                query += " AND id >= %s"
                params.append(id_min)
            if id_max is not None:
                query += " AND id <= %s"
                params.append(id_max)
        
        # Add ORDER BY
        if order_by:
            query += f" ORDER BY {order_by} {order_direction}"
        else:
            query += " ORDER BY timestamp ASC, id ASC"
        
        # Add LIMIT if specified
        if limit_rows and limit_rows > 0:
            query += f" LIMIT {limit_rows}"
        
        status_text.text("🔄 Executing query...")
        cursor.execute(query, params)
        
        # Fetch data in chunks
        all_rows = []
        total_fetched = 0
        chunk_count = 0
        
        status_text.text("📥 Fetching data...")
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            
            all_rows.extend(rows)
            total_fetched += len(rows)
            chunk_count += 1
            
            # Update progress (approximate)
            progress = min(0.9, chunk_count * 0.1)
            progress_bar.progress(progress)
            status_text.text(f"📥 Fetched {total_fetched:,} rows...")
        
        cursor.close()
        conn.close()
        
        if not all_rows:
            progress_bar.empty()
            status_text.empty()
            return None, "No data found for the specified criteria."
        
        # Convert to DataFrame
        status_text.text("🔄 Converting to DataFrame...")
        df = pd.DataFrame(all_rows)
        
        # Ensure proper data types only for columns that exist
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        numeric_cols = ['b_x', 'b_y', 'b_z', 'lat', 'lon', 'alt', 'theta_x', 'theta_y', 'theta_z']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Successfully fetched {len(df):,} rows")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return df, f"Successfully fetched {len(df):,} rows"
        
    except mysql.connector.Error as err:
        progress_bar.empty()
        status_text.empty()
        return None, f"Database error: {err}"
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        return None, f"Unexpected error: {e}"


def main():
    st.title("🧲 Magnetic Data Fetcher")
    st.markdown("Fetch magnetic field data from the database for specific time periods")
    
    # Sidebar for database info and connection test
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        if st.button("Test Connection", type="primary"):
            success, message = test_connection()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.divider()
        st.header("📊 Database Info")
        
        # Get data range
        with st.spinner("Checking data range..."):
            min_date, max_date = get_data_range()
        
        if min_date and max_date:
            st.info(f"**Available Data Range:**\n\n"
                   f"Start: {min_date}\n\n"
                   f"End: {max_date}\n\n"
                   f"Total Duration: {(max_date - min_date).days} days")
        else:
            st.warning("Could not retrieve data range")
        
        st.divider()
        st.header("ℹ️ About")
        st.markdown("""
        This app allows you to:
        - Fetch data for specific time periods
        - Filter by sensor ID
        - Apply downsampling for large datasets
        - Download data as CSV
        """)
    
    # Main content area
    st.header("📅 Select Time Period")
    
    # Quick preset options
    st.subheader("⚡ Quick Presets")
    preset_cols = st.columns(5)
    
    with preset_cols[0]:
        if st.button("Last Hour", use_container_width=True):
            st.session_state.preset_start = datetime.now() - timedelta(hours=1)
            st.session_state.preset_end = datetime.now()
            st.rerun()
    
    with preset_cols[1]:
        if st.button("Last 24 Hours", use_container_width=True):
            st.session_state.preset_start = datetime.now() - timedelta(days=1)
            st.session_state.preset_end = datetime.now()
            st.rerun()
    
    with preset_cols[2]:
        if st.button("Last Week", use_container_width=True):
            st.session_state.preset_start = datetime.now() - timedelta(days=7)
            st.session_state.preset_end = datetime.now()
            st.rerun()
    
    with preset_cols[3]:
        if st.button("Last Month", use_container_width=True):
            st.session_state.preset_start = datetime.now() - timedelta(days=30)
            st.session_state.preset_end = datetime.now()
            st.rerun()
    
    with preset_cols[4]:
        if st.button("Today", use_container_width=True):
            today = datetime.now().date()
            st.session_state.preset_start = datetime.combine(today, datetime.min.time())
            st.session_state.preset_end = datetime.now()
            st.rerun()
    
    st.divider()
    
    # Time selection method
    time_method = st.radio(
        "Time Selection Method:",
        ["📅 Date & Time Pickers", "⌨️ Direct DateTime Input"],
        horizontal=True
    )
    
    if time_method == "📅 Date & Time Pickers":
        col1, col2 = st.columns(2)
        
        with col1:
            # Use preset if available
            if 'preset_start' in st.session_state:
                default_start = st.session_state.preset_start
                del st.session_state.preset_start
            else:
                default_start = datetime.now() - timedelta(days=1)
            
            start_date = st.date_input(
                "Start Date",
                value=default_start.date(),
                min_value=datetime(2025, 9, 26).date() if min_date is None else min_date.date(),
                max_value=datetime.now().date(),
                key="start_date_picker"
            )
            start_time = st.time_input(
                "Start Time", 
                value=default_start.time(),
                key="start_time_picker"
            )
        
        with col2:
            # Use preset if available
            if 'preset_end' in st.session_state:
                default_end = st.session_state.preset_end
                del st.session_state.preset_end
            else:
                default_end = datetime.now()
            
            end_date = st.date_input(
                "End Date",
                value=default_end.date(),
                min_value=datetime(2025, 9, 26).date() if min_date is None else min_date.date(),
                max_value=datetime.now().date(),
                key="end_date_picker"
            )
            end_time = st.time_input(
                "End Time", 
                value=default_end.time(),
                key="end_time_picker"
            )
        
        # Combine date and time
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
    
    else:  # Direct DateTime Input
        col1, col2 = st.columns(2)
        
        with col1:
            start_str = st.text_input(
                "Start DateTime (YYYY-MM-DD HH:MM:SS)",
                value=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                help="Format: YYYY-MM-DD HH:MM:SS (e.g., 2025-10-15 14:30:00)"
            )
            try:
                start_datetime = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                st.error("Invalid format! Use YYYY-MM-DD HH:MM:SS")
                start_datetime = datetime.now() - timedelta(days=1)
        
        with col2:
            end_str = st.text_input(
                "End DateTime (YYYY-MM-DD HH:MM:SS)",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                help="Format: YYYY-MM-DD HH:MM:SS (e.g., 2025-10-15 18:45:00)"
            )
            try:
                end_datetime = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                st.error("Invalid format! Use YYYY-MM-DD HH:MM:SS")
                end_datetime = datetime.now()
    
    # Validate time range
    if end_datetime <= start_datetime:
        st.error("⚠️ End time must be after start time!")
        st.stop()
    
    if min_date and start_datetime < min_date:
        st.warning(f"⚠️ Start time is before available data range. Earliest data: {min_date}")
    
    if max_date and end_datetime > max_date:
        st.warning(f"⚠️ End time is after available data range. Latest data: {max_date}")
    
    # Calculate duration
    duration = end_datetime - start_datetime
    st.info(f"📊 Selected time range: {duration.days} days, {duration.seconds // 3600} hours, {(duration.seconds % 3600) // 60} minutes")
    
    st.divider()
    
    # Column Selection
    st.header("📋 Column Selection")
    all_columns = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z", "lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]
    
    col_selection_method = st.radio(
        "Column Selection Method:",
        ["Select All Columns", "Select Specific Columns"],
        horizontal=True,
        key="col_method"
    )
    
    if col_selection_method == "Select All Columns":
        selected_columns = all_columns
        st.info("✅ All columns will be fetched")
    else:
        selected_columns = st.multiselect(
            "Select columns to fetch:",
            options=all_columns,
            default=all_columns,
            help="Select which columns you want to include in the fetched data. Fewer columns = faster queries and smaller files."
        )
        if not selected_columns:
            st.warning("⚠️ Please select at least one column!")
            st.stop()
    
    st.divider()
    
    # Filter options
    st.header("🔍 Filter Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sensor selection
        with st.spinner("Loading sensors..."):
            available_sensors = get_available_sensors()
        
        if available_sensors:
            sensor_selection = st.multiselect(
                "Select Sensors (leave empty for all)",
                options=available_sensors,
                help="Select specific sensors or leave empty to fetch data from all sensors"
            )
        else:
            st.warning("Could not load sensor list")
            sensor_selection = []
    
    with col2:
        # Downsampling option
        enable_downsample = st.checkbox("Enable Downsampling", value=False, 
                                        help="Reduce data size by keeping only every Nth row")
        if enable_downsample:
            downsample_factor = st.number_input(
                "Downsample Factor",
                min_value=2,
                max_value=1000,
                value=60,
                help="Keep only rows where MOD(id, factor) = 0. Higher values = more downsampling"
            )
        else:
            downsample_factor = None
    
    st.divider()
    
    # Advanced Query Options - Initialize defaults
    use_id_range = False
    id_min = None
    id_max = None
    order_by = "timestamp"
    order_direction = "ASC"
    limit_rows = None
    chunk_size = 10000
    
    # Advanced Query Options
    with st.expander("⚙️ Advanced Query Options", expanded=False):
        st.markdown("**Configure query parameters for large datasets**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Query method
            query_method = st.radio(
                "Query Method:",
                ["Timestamp-based", "ID-based"],
                help="Timestamp-based: Filter by time range. ID-based: Filter by ID range (faster for large datasets)"
            )
            
            # ID range options
            if query_method == "ID-based":
                use_id_range = True
                id_min = st.number_input(
                    "Minimum ID",
                    min_value=0,
                    value=0,
                    help="Minimum ID value to fetch",
                    key="id_min_input"
                )
                id_max_val = st.number_input(
                    "Maximum ID",
                    min_value=0,
                    value=0,
                    help="Maximum ID value to fetch (0 = no limit)",
                    key="id_max_input"
                )
                id_max = id_max_val if id_max_val > 0 else None
            else:
                use_id_range = False
                # Additional ID filtering for timestamp-based queries
                st.markdown("**Additional ID Filtering (optional)**")
                id_min_val = st.number_input(
                    "Minimum ID (optional)",
                    min_value=0,
                    value=0,
                    help="Additional minimum ID filter (0 = no filter)",
                    key="id_min_optional"
                )
                id_max_val = st.number_input(
                    "Maximum ID (optional)",
                    min_value=0,
                    value=0,
                    help="Additional maximum ID filter (0 = no filter)",
                    key="id_max_optional"
                )
                id_min = id_min_val if id_min_val > 0 else None
                id_max = id_max_val if id_max_val > 0 else None
        
        with col2:
            # Ordering options
            order_by = st.selectbox(
                "Order By:",
                ["timestamp", "id", "sensor_id", "b_x", "b_y", "b_z"],
                help="Column to sort results by",
                key="order_by_select"
            )
            
            order_direction = st.radio(
                "Sort Direction:",
                ["ASC", "DESC"],
                horizontal=True,
                help="Ascending or Descending order",
                key="order_dir_radio"
            )
        
        with col3:
            # Limit and chunk size
            enable_limit = st.checkbox("Limit Rows", value=False,
                                      help="Set maximum number of rows to fetch (useful for testing)",
                                      key="enable_limit_check")
            if enable_limit:
                limit_rows = st.number_input(
                    "Maximum Rows",
                    min_value=1,
                    max_value=10000000,
                    value=10000,
                    help="Maximum number of rows to fetch",
                    key="limit_rows_input"
                )
            else:
                limit_rows = None
            
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Number of rows to fetch per chunk (affects memory usage)",
                key="chunk_size_input"
            )
    
    st.divider()
    
    # Fetch button - Data fetch only happens when this button is clicked
    if st.button("🔍 Fetch Data", type="primary", use_container_width=True):
        with st.spinner("Fetching data from database..."):
            df, message = fetch_data(
                start_datetime,
                end_datetime,
                columns=selected_columns,
                sensor_ids=sensor_selection if sensor_selection else None,
                downsample=downsample_factor,
                chunk_size=chunk_size,
                order_by=order_by,
                order_direction=order_direction,
                limit_rows=limit_rows,
                id_min=id_min,
                id_max=id_max,
                use_id_range=use_id_range
            )
            
            if df is not None:
                st.session_state.fetched_data = df
                st.session_state.fetch_status = message
                st.success(message)
            else:
                st.error(message)
                st.session_state.fetched_data = None
                st.session_state.fetch_status = None
    
    # Display fetched data
    if st.session_state.fetched_data is not None:
        df = st.session_state.fetched_data
        
        st.divider()
        st.header("📊 Fetched Data Preview")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Unique Sensors", df['sensor_id'].nunique())
        with col3:
            if df['timestamp'].notna().any():
                duration_actual = (df['timestamp'].max() - df['timestamp'].min())
                st.metric("Data Duration", f"{duration_actual.days}d {duration_actual.seconds//3600}h")
        with col4:
            if df['timestamp'].notna().any():
                time_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
                st.metric("Time Range", "See below")
        
        # Data preview
        st.subheader("Data Preview (First 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Data statistics
        with st.expander("📈 Data Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Sensor breakdown
        if df['sensor_id'].nunique() > 1:
            with st.expander("🔍 Sensor Breakdown"):
                sensor_counts = df['sensor_id'].value_counts()
                st.bar_chart(sensor_counts)
                st.dataframe(sensor_counts.to_frame("Count"), use_container_width=True)
        
        st.divider()
        
        # Download section
        st.header("💾 Download Data")
        
        # Generate CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Download button
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"magnetic_data_{start_datetime.strftime('%Y%m%d_%H%M%S')}_to_{end_datetime.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Clear data button
        if st.button("🗑️ Clear Fetched Data", use_container_width=True):
            st.session_state.fetched_data = None
            st.session_state.fetch_status = None
            st.rerun()


if __name__ == "__main__":
    main()

