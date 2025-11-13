#!/usr/bin/env python3
"""
Streamlit App for Fetching Magnetic Field Data from Database
------------------------------------------------------------
Allows users to fetch magnetic field data for specific time periods
from the continuous 24/7 database collection.
Optimized for handling very large datasets with efficient chunking and caching.
"""

import streamlit as st
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import time
import io
from mysql.connector import pooling

# Page configuration
st.set_page_config(
    page_title="Magnetic Data Fetcher",
    page_icon="🧲",
    layout="wide"
)

# Database configuration with optimized settings for large datasets
DB_CONFIG = {
    'host': '50.63.129.30',
    'user': 'devuser',
    'password': 'devuser@221133',
    'database': 'dbqnaviitk',
    'port': 3306,
    'connection_timeout': 60,
    'connect_timeout': 60,
    'read_timeout': 600,  # Increased significantly for large datasets
    'write_timeout': 600,  # Increased significantly for large datasets
    'use_pure': True,
    'consume_results': True,
    'buffered': False,  # For large datasets, don't buffer all results
    'autocommit': False,  # Better for large queries
    'sql_mode': 'TRADITIONAL'  # Standard SQL mode
}

# Time window configuration for splitting large queries
TIME_WINDOW_MINUTES = 5  # Split large queries into 5-minute windows (very small for huge datasets)
MAX_RETRIES = 5  # Maximum retry attempts per window
BASE_BACKOFF = 2  # Base delay in seconds for exponential backoff
MAX_BACKOFF = 60  # Maximum delay between retries
WINDOW_FETCH_CHUNK_SIZE = 2000  # Smaller chunks for each window (reduced for huge datasets)
MAX_ROWS_PER_WINDOW = 50000  # Maximum rows to fetch per window to prevent timeouts
ID_CHUNK_SIZE = 100000  # ID range chunk size for ID-based queries (much faster)
USE_ID_BASED_CHUNKING = True  # Use ID-based chunking for large datasets (faster than time-based)

# Connection pool configuration for better performance with large datasets
POOL_CONFIG = {
    'pool_name': 'magnetic_data_pool',
    'pool_size': 3,
    'pool_reset_session': True,
    **DB_CONFIG
}

# Table name
TABLE_NAME = "qnav_magneticdatamodel"

# Initialize session state
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None
if 'fetch_status' not in st.session_state:
    st.session_state.fetch_status = None
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False


@st.cache_resource
def get_connection_pool():
    """Create and cache a connection pool for better performance with large datasets"""
    try:
        pool = pooling.MySQLConnectionPool(**POOL_CONFIG)
        return pool
    except Exception as e:
        st.error(f"Failed to create connection pool: {e}")
        return None


def get_connection():
    """Get a connection from the pool or create a new one with keepalive settings"""
    pool = get_connection_pool()
    if pool:
        try:
            conn = pool.get_connection()
            # Set keepalive to prevent connection timeout using cursor
            try:
                cursor = conn.cursor()
                cursor.execute("SET SESSION wait_timeout=28800")  # 8 hours
                cursor.execute("SET SESSION interactive_timeout=28800")
                cursor.execute("SET SESSION net_read_timeout=600")
                cursor.execute("SET SESSION net_write_timeout=600")
                cursor.close()
            except:
                pass  # Ignore if query fails
            return conn
        except Exception:
            # Fallback to direct connection if pool fails
            conn = mysql.connector.connect(**DB_CONFIG)
            try:
                cursor = conn.cursor()
                cursor.execute("SET SESSION wait_timeout=28800")
                cursor.execute("SET SESSION interactive_timeout=28800")
                cursor.execute("SET SESSION net_read_timeout=600")
                cursor.execute("SET SESSION net_write_timeout=600")
                cursor.close()
            except:
                pass
            return conn
    else:
        conn = mysql.connector.connect(**DB_CONFIG)
        try:
            cursor = conn.cursor()
            cursor.execute("SET SESSION wait_timeout=28800")
            cursor.execute("SET SESSION interactive_timeout=28800")
            cursor.execute("SET SESSION net_read_timeout=600")
            cursor.execute("SET SESSION net_write_timeout=600")
            cursor.close()
        except:
            pass
        return conn


def ping_connection(conn):
    """Ping connection to keep it alive"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        return True
    except:
        return False


def test_connection():
    """Test database connection"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True, "Connection successful"
    except mysql.connector.Error as err:
        return False, f"Connection failed: {err}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_available_sensors():
    """Get list of available sensors from database (cached, optimized for large datasets)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Optimized query: Use GROUP BY instead of DISTINCT (often faster on large tables)
        # Also set a query timeout to prevent hanging
        try:
            cursor.execute("SET SESSION max_execution_time = 30000")  # 30 second timeout
        except:
            pass  # Ignore if not supported
        
        # Try fast query first - GROUP BY is often faster than DISTINCT on indexed columns
        cursor.execute(f"SELECT sensor_id FROM {TABLE_NAME} GROUP BY sensor_id ORDER BY sensor_id")
        sensors = [row[0] for row in cursor.fetchall() if row[0] is not None]
        
        cursor.close()
        conn.close()
        return sensors
    except mysql.connector.Error as e:
        # If GROUP BY fails or times out, try a simpler approach with LIMIT
        try:
            conn = get_connection()
            cursor = conn.cursor()
            # Fallback: Sample-based approach - get sensors from recent data
            cursor.execute(
                f"SELECT DISTINCT sensor_id FROM {TABLE_NAME} "
                f"WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY) "
                f"ORDER BY sensor_id LIMIT 100"
            )
            sensors = [row[0] for row in cursor.fetchall() if row[0] is not None]
            cursor.close()
            conn.close()
            if sensors:
                return sensors
        except:
            pass
        
        # If all else fails, return empty list
        return []
    except Exception as e:
        # Return empty list on any other error (don't show error to avoid blocking UI)
        return []


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_data_range():
    """Get the date range of available data (cached)"""
    try:
        conn = get_connection()
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


def find_id_range_for_time_period(start_time, end_time, sensor_ids=None):
    """
    Find the ID range for a time period - FAST query using indexed columns
    Returns (min_id, max_id) or (None, None) on failure
    This is much faster than timestamp queries for large datasets
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Fast query: find min and max ID for the time range
        query = f"SELECT MIN(id) as min_id, MAX(id) as max_id FROM {TABLE_NAME} WHERE timestamp >= %s AND timestamp <= %s"
        params = [start_time, end_time]
        
        if sensor_ids and len(sensor_ids) > 0:
            placeholders = ','.join(['%s'] * len(sensor_ids))
            query += f" AND sensor_id IN ({placeholders})"
            params.extend(sensor_ids)
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result and result[0] is not None and result[1] is not None:
            return result[0], result[1]
        return None, None
    except Exception as e:
        return None, None


def fetch_id_chunk(id_start, id_end, columns_str, sensor_ids, downsample, 
                   start_time, end_time, order_by, order_direction, status_text, progress_container, 
                   chunk_num, total_chunks):
    """
    Fetch data for an ID chunk - MUCH faster than timestamp queries
    Uses ID index for fast lookup, then filters by timestamp
    Returns list of rows or None on failure
    """
    retries = 0
    conn = None
    cursor = None
    
    while retries < MAX_RETRIES:
        try:
            conn = get_connection()
            if not ping_connection(conn):
                raise mysql.connector.Error("Connection is not alive")
            
            cursor = conn.cursor(dictionary=True, buffered=False)
            
            # Build fast ID-based query with timestamp filter
            # ID filter uses index (fast), timestamp filter is secondary
            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE id >= %s AND id <= %s AND timestamp >= %s AND timestamp <= %s"
            params = [id_start, id_end, start_time, end_time]
            
            if sensor_ids and len(sensor_ids) > 0:
                placeholders = ','.join(['%s'] * len(sensor_ids))
                query += f" AND sensor_id IN ({placeholders})"
                params.extend(sensor_ids)
            
            if downsample and downsample > 1:
                query += f" AND MOD(id, {downsample}) = 0"
            
            # NO ORDER BY for speed - ID is already sequential
            # If order is needed, we can sort in memory after fetching
            
            # Execute query (very fast with ID index)
            cursor.execute(query, params)
            
            # Fetch all rows for this chunk
            chunk_rows = []
            rows_fetched = 0
            
            while True:
                rows = cursor.fetchmany(WINDOW_FETCH_CHUNK_SIZE)
                if not rows:
                    break
                
                chunk_rows.extend(rows)
                rows_fetched += len(rows)
                
                # Ping connection every 10k rows
                if rows_fetched % 10000 == 0:
                    try:
                        if not ping_connection(conn):
                            raise mysql.connector.Error("Connection lost during fetch")
                    except Exception as ping_err:
                        raise mysql.connector.Error(f"Connection ping failed: {ping_err}")
            
            cursor.close()
            conn.close()
            
            return chunk_rows
            
        except mysql.connector.Error as err:
            retries += 1
            error_msg = str(err)
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
            cursor = None
            conn = None
            
            if retries < MAX_RETRIES:
                delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                progress_container.text(
                    f"⚠️ ID Chunk {chunk_num}/{total_chunks} failed: {error_msg[:80]}. "
                    f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                )
                time.sleep(delay)
            else:
                return None
        except Exception as e:
            retries += 1
            error_msg = str(e)
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
            cursor = None
            conn = None
            
            if retries < MAX_RETRIES:
                delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                progress_container.text(
                    f"⚠️ ID Chunk {chunk_num}/{total_chunks} error: {error_msg[:80]}. "
                    f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                )
                time.sleep(delay)
            else:
                return None
    
    return None


def fetch_time_window(window_start, window_end, columns, columns_str, sensor_ids, downsample, 
                     order_by, order_direction, id_min, id_max, use_id_range, status_text, 
                     progress_container, window_num, total_windows):
    """
    Fetch data for a single time window with retry logic.
    Returns list of rows or None on failure.
    """
    retries = 0
    conn = None
    cursor = None
    
    while retries < MAX_RETRIES:
        try:
            # Get fresh connection for each window
            conn = get_connection()
            
            # Validate connection is alive before use
            if not ping_connection(conn):
                raise mysql.connector.Error("Connection is not alive")
            
            cursor = conn.cursor(dictionary=True, buffered=False)
            
            # Build query for this window
            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE "
            params = []
            
            # Use ID range or timestamp range
            if use_id_range and id_min is not None and id_max is not None:
                query += "id >= %s AND id <= %s"
                params.extend([id_min, id_max])
            else:
                query += "timestamp >= %s AND timestamp <= %s"
                params.extend([window_start, window_end])
            
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
            
            # Add LIMIT per window to prevent huge fetches (critical for 50GB+ datasets)
            query += f" LIMIT {MAX_ROWS_PER_WINDOW}"
            
            # Execute query
            cursor.execute(query, params)
            
            # Fetch rows in smaller chunks to prevent timeout during fetch
            window_rows = []
            fetch_chunk_size = WINDOW_FETCH_CHUNK_SIZE  # Use configured chunk size
            rows_fetched = 0
            
            while True:
                rows = cursor.fetchmany(fetch_chunk_size)
                if not rows:
                    break
                
                window_rows.extend(rows)
                rows_fetched += len(rows)
                
                # Ping connection every 5k rows to keep it alive (more frequent for huge datasets)
                if rows_fetched % 5000 == 0:
                    try:
                        if not ping_connection(conn):
                            raise mysql.connector.Error("Connection lost during fetch")
                    except Exception as ping_err:
                        raise mysql.connector.Error(f"Connection ping failed: {ping_err}")
                
                # Safety check: if we hit the limit, warn but continue
                if rows_fetched >= MAX_ROWS_PER_WINDOW:
                    break
            
            cursor.close()
            conn.close()
            
            return window_rows
            
        except mysql.connector.Error as err:
            retries += 1
            error_msg = str(err)
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
            cursor = None
            conn = None
            
            if retries < MAX_RETRIES:
                delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                # Show more detailed error for connection issues
                if "2013" in error_msg or "Lost connection" in error_msg or "timeout" in error_msg.lower():
                    progress_container.text(
                        f"⚠️ Window {window_num}/{total_windows} - Connection timeout. "
                        f"Retrying with fresh connection in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                    )
                else:
                    progress_container.text(
                        f"⚠️ Window {window_num}/{total_windows} failed: {error_msg[:100]}. "
                        f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                    )
                time.sleep(delay)
            else:
                progress_container.text(
                    f"❌ Window {window_num}/{total_windows} failed after {MAX_RETRIES} attempts: {error_msg[:100]}"
                )
                return None
        except Exception as e:
            retries += 1
            error_msg = str(e)
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
            cursor = None
            conn = None
            
            if retries < MAX_RETRIES:
                delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                # Check if it's a connection-related error
                if "2013" in error_msg or "Lost connection" in error_msg or "timeout" in error_msg.lower() or "Connection" in error_msg:
                    progress_container.text(
                        f"⚠️ Window {window_num}/{total_windows} - Connection issue: {error_msg[:80]}. "
                        f"Retrying with fresh connection in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                    )
                else:
                    progress_container.text(
                        f"⚠️ Window {window_num}/{total_windows} error: {error_msg[:100]}. "
                        f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                    )
                time.sleep(delay)
            else:
                progress_container.text(
                    f"❌ Window {window_num}/{total_windows} failed after {MAX_RETRIES} attempts: {error_msg[:100]}"
                )
                return None
    
    return None


def fetch_data(start_time, end_time, columns=None, sensor_ids=None, downsample=None, 
               chunk_size=10000, order_by="timestamp", order_direction="ASC", 
               limit_rows=None, id_min=None, id_max=None, use_id_range=False, 
               time_window_minutes=None):
    """
    Fetch data from database for specified time range
    Optimized for very large datasets using time window splitting and retry logic.
    Prevents connection timeout errors by splitting large queries into smaller windows.
    
    Parameters:
    - start_time: Start timestamp (datetime)
    - end_time: End timestamp (datetime)
    - columns: List of column names to fetch (None = all columns)
    - sensor_ids: List of sensor IDs to filter (None = all sensors)
    - downsample: Downsample factor (None = no downsampling)
    - chunk_size: Number of rows to fetch per chunk (not used in window-based approach)
    - order_by: Column to order by ("timestamp", "id", or custom)
    - order_direction: "ASC" or "DESC"
    - limit_rows: Maximum number of rows to fetch (None = no limit)
    - id_min: Minimum ID value (for ID-based filtering)
    - id_max: Maximum ID value (for ID-based filtering)
    - use_id_range: Whether to use ID range instead of timestamp
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_container = st.empty()
    
    try:
        # Default columns if not specified
        if columns is None or len(columns) == 0:
            columns = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z", "lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]
        
        # Build SELECT clause
        columns_str = ', '.join(columns)
        
        # Use provided time window or default
        if time_window_minutes is None:
            time_window_minutes = TIME_WINDOW_MINUTES
        
        # Calculate time window size based on total duration
        total_duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        # For huge datasets (50GB+), use ID-based chunking (MUCH faster)
        # ID-based queries are 10-100x faster than timestamp queries
        use_id_chunking = False
        id_min_range = None
        id_max_range = None
        
        if USE_ID_BASED_CHUNKING and total_duration > 30 and not use_id_range:
            # Try to find ID range for this time period (fast query)
            status_text.text("🔍 Finding ID range for time period (fast lookup)...")
            progress_container.text("⏳ Querying ID bounds...")
            
            id_min_range, id_max_range = find_id_range_for_time_period(
                start_time, end_time, sensor_ids
            )
            
            if id_min_range is not None and id_max_range is not None:
                total_ids = id_max_range - id_min_range
                use_id_chunking = True
                status_text.text(f"✅ Found ID range: {id_min_range:,} to {id_max_range:,} ({total_ids:,} IDs)")
                progress_container.text(
                    f"🚀 Using ID-based chunking (much faster!). "
                    f"Will fetch in ID chunks of {ID_CHUNK_SIZE:,}"
                )
                time.sleep(1)
        
        # For huge datasets (50GB+), force smaller windows and downsampling
        # If time range > 1 hour and no downsampling, warn user
        if total_duration > 60 and (downsample is None or downsample == 1) and not use_id_chunking:
            status_text.text("⚠️ Large dataset detected - consider enabling downsampling")
            progress_container.text(
                f"💡 Tip: For datasets > 50GB, enable downsampling (e.g., factor 60-100) "
                f"to reduce data size and prevent timeouts"
            )
            time.sleep(2)  # Show warning briefly
        
        # Auto-adjust window size for very large ranges (if not using ID chunking)
        if total_duration > 120 and not use_id_chunking:  # More than 2 hours
            # Use even smaller windows for very large ranges
            if time_window_minutes > 5:
                time_window_minutes = 5
                progress_container.text(
                    f"📊 Large time range detected. Using {time_window_minutes}-minute windows for reliability."
                )
        
        # Use ID-based chunking if available (FASTEST method)
        if use_id_chunking and id_min_range is not None and id_max_range is not None:
            # Calculate ID chunks
            total_ids = id_max_range - id_min_range
            num_chunks = max(1, (total_ids // ID_CHUNK_SIZE) + (1 if total_ids % ID_CHUNK_SIZE > 0 else 0))
            
            status_text.text(f"🔄 Fetching data using ID-based chunks ({num_chunks} chunks)...")
            progress_container.text(f"📊 Processing {num_chunks} ID chunks of ~{ID_CHUNK_SIZE:,} IDs each")
            
            all_rows = []
            total_fetched = 0
            start_time_fetch = time.time()
            failed_chunks = []
            
            # Process each ID chunk
            for chunk_num in range(num_chunks):
                chunk_id_start = id_min_range + (chunk_num * ID_CHUNK_SIZE)
                chunk_id_end = min(id_min_range + ((chunk_num + 1) * ID_CHUNK_SIZE) - 1, id_max_range)
                
                progress_container.text(
                    f"🔄 Processing ID chunk {chunk_num + 1}/{num_chunks}: "
                    f"IDs {chunk_id_start:,} to {chunk_id_end:,}"
                )
                
                chunk_rows = fetch_id_chunk(
                    chunk_id_start, chunk_id_end, columns_str, sensor_ids, downsample,
                    start_time, end_time, order_by, order_direction, status_text, progress_container,
                    chunk_num + 1, num_chunks
                )
                
                if chunk_rows is not None:
                    # Rows are already filtered by timestamp in the query
                    all_rows.extend(chunk_rows)
                    total_fetched += len(chunk_rows)
                    
                    # Update progress
                    progress = (chunk_num + 1) / num_chunks
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time_fetch
                    if elapsed > 0:
                        rate = total_fetched / elapsed
                        remaining_chunks = num_chunks - (chunk_num + 1)
                        est_time = (remaining_chunks * elapsed) / (chunk_num + 1) if chunk_num > 0 else 0
                        progress_container.text(
                            f"✅ Chunk {chunk_num + 1}/{num_chunks} complete ({len(chunk_rows):,} rows) | "
                            f"Total: {total_fetched:,} rows | "
                            f"Rate: {rate:.0f} rows/sec | "
                            f"Est. remaining: {est_time:.0f}s"
                        )
                else:
                    failed_chunks.append((chunk_id_start, chunk_id_end))
                    progress_container.text(
                        f"❌ ID Chunk {chunk_num + 1}/{num_chunks} failed after {MAX_RETRIES} retries"
                    )
            
            if failed_chunks:
                return None, f"Failed to fetch {len(failed_chunks)} out of {num_chunks} ID chunks"
        
        # For ID-based queries or very short time ranges, don't split into windows
        elif use_id_range or total_duration <= time_window_minutes:
            # Use single query approach for small ranges or ID-based queries
            status_text.text("🔄 Executing single query...")
            progress_container.text("⏳ Fetching data...")
            
            conn = get_connection()
            
            # Validate connection is alive before use
            if not ping_connection(conn):
                raise mysql.connector.Error("Connection is not alive")
            
            cursor = conn.cursor(dictionary=True, buffered=False)
            
            # Build query
            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE "
            params = []
            
            if use_id_range and id_min is not None and id_max is not None:
                query += "id >= %s AND id <= %s"
                params.extend([id_min, id_max])
            else:
                query += "timestamp >= %s AND timestamp <= %s"
                params.extend([start_time, end_time])
            
            if sensor_ids and len(sensor_ids) > 0:
                placeholders = ','.join(['%s'] * len(sensor_ids))
                query += f" AND sensor_id IN ({placeholders})"
                params.extend(sensor_ids)
            
            if downsample and downsample > 1:
                query += f" AND MOD(id, {downsample}) = 0"
            
            if not use_id_range:
                if id_min is not None:
                    query += " AND id >= %s"
                    params.append(id_min)
                if id_max is not None:
                    query += " AND id <= %s"
                    params.append(id_max)
            
            if order_by:
                query += f" ORDER BY {order_by} {order_direction}"
            else:
                query += " ORDER BY timestamp ASC, id ASC"
            
            if limit_rows and limit_rows > 0:
                query += f" LIMIT {limit_rows}"
            
            cursor.execute(query, params)
            
            all_rows = []
            total_fetched = 0
            start_time_fetch = time.time()
            
            while True:
                rows = cursor.fetchmany(5000)
                if not rows:
                    break
                all_rows.extend(rows)
                total_fetched += len(rows)
                
                # Ping connection every 10k rows to keep it alive
                if total_fetched % 10000 == 0:
                    try:
                        if not ping_connection(conn):
                            raise mysql.connector.Error("Connection lost during fetch")
                    except Exception as ping_err:
                        raise mysql.connector.Error(f"Connection ping failed: {ping_err}")
                
                elapsed = time.time() - start_time_fetch
                if elapsed > 0:
                    rate = total_fetched / elapsed
                    progress_container.text(f"📊 Fetched: {total_fetched:,} rows | Rate: {rate:.0f} rows/sec")
                    progress_bar.progress(min(0.95, total_fetched / 1000000))  # Approximate progress
            
            cursor.close()
            conn.close()
            
        else:
            # Split into time windows for large time ranges
            window_duration = timedelta(minutes=time_window_minutes)
            windows = []
            current_start = start_time
            
            # Create time windows
            while current_start < end_time:
                current_end = min(current_start + window_duration, end_time)
                windows.append((current_start, current_end))
                current_start = current_end
            
            total_windows = len(windows)
            status_text.text(f"🔄 Splitting query into {total_windows} time windows...")
            progress_container.text(f"📅 Processing {total_windows} windows of {time_window_minutes} minutes each")
            
            all_rows = []
            total_fetched = 0
            start_time_fetch = time.time()
            failed_windows = []
            
            # Process each window
            for window_num, (window_start, window_end) in enumerate(windows, 1):
                progress_container.text(
                    f"🔄 Processing window {window_num}/{total_windows}: "
                    f"{window_start.strftime('%H:%M:%S')} to {window_end.strftime('%H:%M:%S')} "
                    f"(max {MAX_ROWS_PER_WINDOW:,} rows)"
                )
                
                window_rows = fetch_time_window(
                    window_start, window_end, columns, columns_str, sensor_ids, downsample,
                    order_by, order_direction, id_min, id_max, use_id_range,
                    status_text, progress_container, window_num, total_windows
                )
                
                if window_rows is not None:
                    rows_in_window = len(window_rows)
                    all_rows.extend(window_rows)
                    total_fetched += rows_in_window
                    
                    # Warn if window hit row limit (data might be truncated)
                    if rows_in_window >= MAX_ROWS_PER_WINDOW:
                        progress_container.text(
                            f"⚠️ Window {window_num}/{total_windows} hit row limit ({MAX_ROWS_PER_WINDOW:,}). "
                            f"Consider smaller windows or downsampling."
                        )
                        time.sleep(1)  # Brief pause to show warning
                    
                    # Update progress
                    progress = window_num / total_windows
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time_fetch
                    if elapsed > 0:
                        rate = total_fetched / elapsed
                        remaining_windows = total_windows - window_num
                        est_time = (remaining_windows * elapsed) / window_num if window_num > 0 else 0
                        progress_container.text(
                            f"✅ Window {window_num}/{total_windows} complete ({rows_in_window:,} rows) | "
                            f"Total: {total_fetched:,} rows | "
                            f"Rate: {rate:.0f} rows/sec | "
                            f"Est. remaining: {est_time:.0f}s"
                        )
                else:
                    failed_windows.append((window_start, window_end))
                    progress_container.text(
                        f"❌ Window {window_num}/{total_windows} failed after {MAX_RETRIES} retries"
                    )
            
            if failed_windows:
                return None, f"Failed to fetch {len(failed_windows)} out of {total_windows} windows"
        
        # Apply row limit if specified
        if limit_rows and limit_rows > 0 and len(all_rows) > limit_rows:
            all_rows = all_rows[:limit_rows]
            total_fetched = len(all_rows)
        
        if not all_rows:
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            return None, "No data found for the specified criteria."
        
        # Convert to DataFrame
        status_text.text("🔄 Converting to DataFrame...")
        progress_container.text(f"🔄 Processing {len(all_rows):,} rows into DataFrame...")
        
        df = pd.DataFrame(all_rows)
        del all_rows  # Free memory
        
        # Optimize data types
        status_text.text("🔄 Optimizing data types...")
        if 'timestamp' in df.columns:
            # CRITICAL: Convert timestamp to datetime, preserving full datetime info
            # MySQL returns datetime as string, we need to parse it properly
            # Try to preserve the original format from database
            original_timestamps = df['timestamp'].copy()
            
            # Convert to datetime - try multiple methods
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Verify conversion worked - check if we have valid datetimes
            if df['timestamp'].isna().all():
                # Conversion failed - try with original values
                st.warning("⚠️ Timestamp conversion had issues. Trying alternative method...")
                df['timestamp'] = pd.to_datetime(original_timestamps, errors='coerce')
            
            # Ensure we have datetime64[ns] type
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        numeric_cols = ['b_x', 'b_y', 'b_z', 'lat', 'lon', 'alt', 'theta_x', 'theta_y', 'theta_z']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Successfully fetched {len(df):,} rows")
        progress_container.text(f"✅ Complete! Dataset size: {len(df):,} rows × {len(df.columns)} columns")
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        progress_container.empty()
        
        return df, f"Successfully fetched {len(df):,} rows"
        
    except mysql.connector.Error as err:
        progress_bar.empty()
        status_text.empty()
        progress_container.empty()
        return None, f"Database error: {err}"
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        progress_container.empty()
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
        # Load sensors with timeout handling
        sensor_placeholder = st.empty()
        with sensor_placeholder.container():
            with st.spinner("Loading sensors (this may take a moment for large datasets)..."):
                available_sensors = get_available_sensors()
        
        # Clear spinner after loading
        sensor_placeholder.empty()
        
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
    query_method = "Timestamp-based"  # Default query method
    time_window_minutes = TIME_WINDOW_MINUTES  # Default time window size
    
    # Advanced Query Options
    with st.expander("⚙️ Advanced Query Options", expanded=False):
        st.markdown("**Configure query parameters for large datasets**")
        st.info("💡 **Tip:** Large time ranges are automatically split into smaller windows to prevent timeout errors. Adjust window size based on your dataset size.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Query method
            query_method = st.radio(
                "Query Method:",
                ["Timestamp-based", "ID-based"],
                help="Timestamp-based: Filter by time range. ID-based: Filter by ID range (faster for large datasets)",
                key="query_method_radio"
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
        
        # Time window configuration (full width)
        st.divider()
        st.markdown("**⏱️ Time Window Configuration (for large datasets)**")
        col_window1, col_window2 = st.columns(2)
        
        with col_window1:
            time_window_minutes = st.number_input(
                "Time Window Size (minutes)",
                min_value=1,
                max_value=120,
                value=TIME_WINDOW_MINUTES,
                step=1,
                help="Large queries are split into windows of this size. For 50GB+ datasets, use 1-5 minutes. Smaller = safer but slower.",
                key="time_window_input"
            )
            st.caption(f"⚠️ Max {MAX_ROWS_PER_WINDOW:,} rows per window to prevent timeouts")
        
        with col_window2:
            st.markdown("**Window Info:**")
            if duration.days > 0 or duration.seconds > 1800:  # More than 30 minutes
                estimated_windows = max(1, int((duration.total_seconds() / 60) / time_window_minutes))
                st.info(f"Estimated windows: **{estimated_windows}** for this time range")
            else:
                st.info("Time range is small - will use single query")
    
    st.divider()
    
    # Submit button section - Processing only starts when this button is clicked
    st.header("🚀 Submit Query")
    
    # Show query summary before submission
    with st.expander("📋 Query Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Time Range:** {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} to {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Duration:** {duration.days} days, {duration.seconds // 3600} hours")
            st.markdown(f"**Columns:** {len(selected_columns)} selected")
            st.markdown(f"**Sensors:** {len(sensor_selection) if sensor_selection else 'All'}")
        with col2:
            st.markdown(f"**Downsampling:** {'Enabled' if enable_downsample else 'Disabled'}")
            if enable_downsample:
                st.markdown(f"**Downsample Factor:** {downsample_factor}")
            st.markdown(f"**Query Method:** {query_method}")
            st.markdown(f"**Chunk Size:** {chunk_size:,} rows")
            if limit_rows:
                st.markdown(f"**Row Limit:** {limit_rows:,}")
    
    # Warning for large datasets (50GB+ datasets)
    if duration.total_seconds() > 3600:  # More than 1 hour
        if not enable_downsample or downsample_factor < 60:
            st.error("🚨 **CRITICAL: Very Large Dataset Detected**")
            st.warning(
                f"⚠️ **For 50GB+ datasets, you MUST enable downsampling!**\n\n"
                f"**Current settings:**\n"
                f"- Time range: {duration.days} days, {duration.seconds // 3600} hours\n"
                f"- Downsampling: {'Enabled' if enable_downsample else '❌ DISABLED'}\n"
                f"- Downsample factor: {downsample_factor if enable_downsample else 'N/A'}\n\n"
                f"**Recommendations:**\n"
                f"1. ✅ Enable downsampling with factor 60-100 (keeps every 60th-100th row)\n"
                f"2. ✅ Use 1-5 minute time windows\n"
                f"3. ✅ Select only needed columns\n"
                f"4. ✅ Filter by specific sensors if possible\n\n"
                f"**Without downsampling, this query will likely timeout or take hours!**"
            )
        else:
            st.info(f"✅ Downsampling enabled (factor {downsample_factor}). This should help with large datasets.")
    elif duration.days > 7 or (enable_downsample and downsample_factor < 10):
        st.warning("⚠️ **Large Dataset Warning:** This query may take a significant amount of time and memory. Consider using downsampling or limiting the time range for faster results.")
    
    # Submit button - Processing only starts when clicked
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        if st.button("✅ Submit Query & Start Processing", type="primary", use_container_width=True, key="submit_button"):
            st.session_state.processing_started = True
            st.session_state.fetched_data = None
            st.session_state.fetch_status = None
            st.rerun()
    
    # Only process if submit button was clicked
    if st.session_state.processing_started:
        st.info("🔄 **Processing in progress...** Please wait while data is being fetched from the database.")
        
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
                use_id_range=use_id_range,
                time_window_minutes=time_window_minutes
            )
            
            if df is not None:
                st.session_state.fetched_data = df
                st.session_state.fetch_status = message
                st.session_state.processing_started = False  # Reset flag after processing
                st.success(message)
            else:
                st.error(message)
                st.session_state.fetched_data = None
                st.session_state.fetch_status = None
                st.session_state.processing_started = False  # Reset flag on error
    else:
        # Show placeholder message when not processing
        st.info("👆 **Ready to process:** Configure your query parameters above and click 'Submit Query & Start Processing' to begin fetching data.")
    
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
                # Try bar chart, fallback to simple visualization if there's a compatibility issue
                try:
                    st.bar_chart(sensor_counts)
                except (TypeError, AttributeError, Exception) as e:
                    # Handle compatibility issues with Python 3.14 and Altair
                    # Show simple text-based visualization instead
                    st.write("**Sensor Counts:**")
                    max_count = sensor_counts.max()
                    for sensor_id, count in sensor_counts.items():
                        bar_length = int((count / max_count) * 50)  # Scale to 50 chars max
                        bar = "█" * bar_length
                        percentage = (count / len(df)) * 100
                        st.write(f"**Sensor {sensor_id}:** {bar} {count:,} rows ({percentage:.1f}%)")
                st.dataframe(sensor_counts.to_frame("Count"), use_container_width=True)
        
        st.divider()
        
        # Download section
        st.header("💾 Download Data")
        
        # Generate CSV with proper timestamp format matching Data Preview
        # The Data Preview shows timestamps as datetime objects - we need to format them the same way
        csv_buffer = io.StringIO()
        
        # Create a copy for export - don't modify the original df
        df_export = df.copy()
        
        # Format timestamp column to match reference CSV format exactly
        # Reference format: "2025-11-09 08:06:33.120000" (quoted, full date + time + microseconds)
        if 'timestamp' in df_export.columns:
            # CRITICAL: Ensure timestamp is properly converted to datetime
            # Check if it's already datetime
            if not pd.api.types.is_datetime64_any_dtype(df_export['timestamp']):
                # Convert to datetime - try multiple methods
                try:
                    df_export['timestamp'] = pd.to_datetime(df_export['timestamp'], errors='coerce', infer_datetime_format=True)
                except:
                    df_export['timestamp'] = pd.to_datetime(df_export['timestamp'], errors='coerce')
            
            # Verify we have datetime objects
            if not pd.api.types.is_datetime64_any_dtype(df_export['timestamp']):
                # Last attempt - force conversion
                df_export['timestamp'] = pd.to_datetime(df_export['timestamp'], errors='coerce')
            
            # Format each timestamp to exact format: "YYYY-MM-DD HH:MM:SS.microseconds"
            # This matches the reference CSV format exactly: "2025-11-09 08:06:33.120000"
            def format_timestamp_for_csv(ts):
                """Format timestamp to match reference CSV: YYYY-MM-DD HH:MM:SS.microseconds"""
                if pd.isna(ts):
                    return ''
                
                # CRITICAL: Ensure we have a proper datetime object with full date/time
                # If it's a string, parse it first
                if isinstance(ts, str):
                    # If string looks like time only (e.g., "19:47.8"), we have a problem
                    if ':' in ts and len(ts) < 10:  # Likely time-only format
                        # This shouldn't happen - timestamp should have full date
                        # Try to parse anyway
                        try:
                            ts = pd.to_datetime(ts)
                        except:
                            return ts  # Return as-is if can't parse
                    else:
                        try:
                            ts = pd.to_datetime(ts)
                        except:
                            return ts
                
                # Now format as datetime object - MUST have full date and time
                try:
                    # Format: YYYY-MM-DD HH:MM:SS.microseconds (6 digits for microseconds)
                    # This should produce: "2025-11-09 08:06:33.120000"
                    formatted = ts.strftime('%Y-%m-%d %H:%M:%S.%f')
                    
                    # CRITICAL CHECK: Verify format is complete
                    # Should be at least 26 chars: "YYYY-MM-DD HH:MM:SS.microseconds"
                    if len(formatted) < 19:  # Less than "YYYY-MM-DD HH:MM:SS"
                        # Format is incomplete - this is an error
                        # Try to get full datetime info
                        if hasattr(ts, 'year') and hasattr(ts, 'hour'):
                            # We have datetime object - force full format
                            formatted = f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d} {ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}.{ts.microsecond:06d}"
                        else:
                            # Convert to datetime again
                            dt_val = pd.to_datetime(ts)
                            formatted = dt_val.strftime('%Y-%m-%d %H:%M:%S.%f')
                    
                    # Final verification: must start with year (20xx)
                    if not formatted.startswith('20'):
                        # Something is wrong - try one more time
                        dt_val = pd.to_datetime(str(ts))
                        formatted = dt_val.strftime('%Y-%m-%d %H:%M:%S.%f')
                    
                    return formatted
                    
                except (AttributeError, ValueError, TypeError) as e:
                    # If strftime fails, convert to datetime again and retry
                    try:
                        dt_val = pd.to_datetime(ts)
                        formatted = dt_val.strftime('%Y-%m-%d %H:%M:%S.%f')
                        if len(formatted) >= 19 and formatted.startswith('20'):
                            return formatted
                        else:
                            # Still wrong - use pandas string representation
                            return str(dt_val)
                    except:
                        # Last resort
                        return str(ts)
            
            # Apply formatting to each timestamp value
            df_export['timestamp'] = df_export['timestamp'].apply(format_timestamp_for_csv)
            
            # CRITICAL: Verify timestamps are properly formatted
            # Check if any timestamps are missing date (only have time)
            invalid_timestamps = df_export['timestamp'].apply(
                lambda x: x != '' and not pd.isna(x) and not str(x).startswith('20')
            )
            if invalid_timestamps.any():
                # Some timestamps are incorrectly formatted
                bad_samples = df_export.loc[invalid_timestamps, 'timestamp'].head(3).tolist()
                st.error(f"❌ ERROR: Some timestamps are missing date information! Samples: {bad_samples}")
                st.error("This indicates the timestamp data from database may be incomplete.")
        
        # Use pandas to_csv with custom quoting for timestamp
        # This ensures proper CSV formatting while we manually quote the timestamp
        import csv
        
        # Write CSV manually to control timestamp quoting
        columns = df_export.columns.tolist()
        timestamp_col_idx = columns.index('timestamp') if 'timestamp' in columns else None
        
        # Write header
        csv_buffer.write(','.join(columns) + '\n')
        
        # Write data rows
        for _, row in df_export.iterrows():
            row_values = []
            for idx, col in enumerate(columns):
                val = row[col]
                
                if pd.isna(val) or val == '':
                    row_values.append('')
                elif idx == timestamp_col_idx:
                    # Quote timestamp to match reference format: "2025-10-13 00:00:01.011000"
                    # val should already be a formatted string from format_timestamp_for_csv
                    row_values.append(f'"{val}"')
                elif isinstance(val, (int, float)) and pd.notna(val):
                    # Numeric values - write as-is
                    row_values.append(str(val))
                elif isinstance(val, str):
                    # String values - quote if needed
                    if ',' in val or '"' in val or '\n' in val:
                        escaped_val = val.replace('"', '""')
                        row_values.append(f'"{escaped_val}"')
                    else:
                        row_values.append(val)
                else:
                    # Other types - convert to string
                    row_values.append(str(val))
            
            csv_buffer.write(','.join(row_values) + '\n')
        
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
