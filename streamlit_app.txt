# import streamlit as st
# import os
# import pandas as pd
# from sqlalchemy import create_engine, inspect, text
# from sqlalchemy.types import Numeric, Integer, Float, BigInteger

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="TimescaleDB Dashboard",
#     page_icon="🐘",
#     layout="wide"
# )

# st.title("TimescaleDB Analysis Dashboard 📈")
# st.markdown("A tool to run and visualize time-series queries using `time_bucket()`.")

# # --- Database Connection ---
# @st.cache_resource
# def get_db_engine():
#     """
#     Creates and caches a SQLAlchemy engine based on environment variables.
#     """
#     try:
#         user = "tsdbadmin" # os.environ.get("PGUSER")
#         pw = "m6x55zbrm5e93dwv" # os.environ.get("PGPASSWORD")
#         host = "s1l0v9f595.xy2jmmua0e.tsdb.cloud.timescale.com" # os.environ.get("PGHOST")
#         port = 33127 # os.environ.get("PGPORT")
#         db = "tsdb" # os.environ.get("PGDATABASE")

#         if not all([user, pw, host, port, db]):
#             st.error("Missing one or more database environment variables (PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE).")
#             st.stop()

#         connection_string = f"postgresql://{user}:{pw}@{host}:{port}/{db}?sslmode=require"
#         engine = create_engine(connection_string)
        
#         # Test connection
#         with engine.connect() as conn:
#             pass
        
#         return engine
#     except Exception as e:
#         st.error(f"Database connection failed: {e}")
#         st.stop()

# engine = get_db_engine()

# # --- Schema Fetching ---
# @st.cache_data(ttl=600)
# def fetch_schema(_engine):
#     """
#     Fetches schema information, focusing on TimescaleDB hypertables.
#     Caches the result for 10 minutes.
#     """
#     inspector = inspect(_engine)
#     hypertables = []
    
#     # Query TimescaleDB's information view for hypertables
#     try:
#         with _engine.connect() as conn:
#             query = text("SELECT table_name, time_column_name FROM timescaledb_information.hypertables;")
#             result = conn.execute(query)
#             for row in result:
#                 # Use ._asdict() for named-column access
#                 row_dict = row._asdict()
#                 hypertables.append((row_dict['table_name'], row_dict['time_column_name']))
#     except Exception as e:
#         st.sidebar.warning(f"Could not fetch hypertables. (Is TimescaleDB enabled?)\nError: {e}", icon="⚠️")
#         return {}

#     schema_info = {}
#     for table_name, time_col in hypertables:
#         columns = []
#         try:
#             cols = inspector.get_columns(table_name)
#             for col in cols:
#                 columns.append({"name": col['name'], "type": str(col['type'])})
#             schema_info[table_name] = {'time_col': time_col, 'columns': columns}
#         except Exception as e:
#             st.sidebar.error(f"Error fetching columns for {table_name}: {e}")
            
#     return schema_info

# # --- Sidebar ---
# st.sidebar.title("Database Schema")
# st.sidebar.markdown("This panel shows the 'hypertables' (time-series tables) found in your database.")

# with st.sidebar:
#     with st.spinner("Loading schema..."):
#         schema = fetch_schema(engine)

# if not schema:
#     st.sidebar.warning("No hypertables found.")
#     default_query = "SELECT 'No hypertables found' AS status;"
# else:
#     st.sidebar.success(f"Found {len(schema)} hypertable(s).")
    
#     # Generate an intelligent default query
#     first_table = list(schema.keys())[0]
#     time_col = schema[first_table]['time_col']
    
#     # Try to find a numeric column to aggregate
#     value_col = None
#     for col in schema[first_table]['columns']:
#         # Check if column type is numeric
#         try:
#             # Create a temporary instance of the type for checking
#             col_type_instance = col['type']
#             # A bit of a hack: check type name strings
#             type_str = str(col_type_instance).lower()
#             if 'int' in type_str or 'float' in type_str or 'numeric' in type_str or 'double' in type_str:
#                  # Avoid the time column itself
#                 if col['name'] != time_col:
#                     value_col = col['name']
#                     break
#         except Exception:
#             continue # Skip if type instantiation fails
            
#     if not value_col:
#         value_col = "[your_numeric_column]" # Fallback

#     default_query = f"""-- Default query: 1-day buckets for the last 30 days
# SELECT
#     time_bucket('1 day', {time_col}) AS bucket,
#     AVG({value_col}) AS "avg_{value_col}"
# FROM
#     {first_table}
# WHERE
#     {time_col} > now() - INTERVAL '30 days'
# GROUP BY
#     bucket
# ORDER BY
#     bucket DESC;
# """

#     # Display schema in sidebar
#     for table_name, info in schema.items():
#         with st.sidebar.expander(f"🔵 {table_name} (Hypertable)"):
#             st.markdown(f"**Time Column:** `{info['time_col']}`")
#             st.markdown("**All Columns:**")
#             for col in info['columns']:
#                 st.code(f"{col['name']}: {col['type']}", language="sql")


# # --- Main Page: Query Editor ---
# st.header("Query Editor")
# query = st.text_area(
#     "Enter your SQL query. Use `time_bucket()` for time-series analysis.",
#     value=default_query,
#     height=250,
#     key="sql_query"
# )

# if st.button("Run Query", type="primary"):
#     if not query:
#         st.warning("Query is empty.")
#     else:
#         st.subheader("Query Results")
#         try:
#             with st.spinner("Running query..."):
#                 df = pd.read_sql(query, engine)

#             st.dataframe(df, use_container_width=True)

#             # --- Plotting ---
#             st.subheader("Plot")
            
#             # Try to find a time column for plotting
#             plot_col = None
#             if 'bucket' in df.columns:
#                 plot_col = 'bucket'
#             else:
#                 # Find first datetime column
#                 for col_name, dtype in df.dtypes.items():
#                     if pd.api.types.is_datetime64_any_dtype(dtype):
#                         plot_col = col_name
#                         break
            
#             if plot_col:
#                 try:
#                     # Set time column as index for plotting
#                     df_plot = df.set_index(plot_col)
                    
#                     # Identify numeric columns to plot
#                     numeric_cols = df_plot.select_dtypes(include=['number']).columns
#                     if not numeric_cols.empty:
#                         st.line_chart(df_plot[numeric_cols])
#                         st.caption(f"Plotting numeric columns against `{plot_col}`.")
#                     else:
#                         st.warning("No numeric columns found to plot against the time axis.")
#                 except Exception as e:
#                     st.error(f"Failed to create plot: {e}")
#             else:
#                 st.info("No time column (like 'bucket' or another timestamp) found in results to use as X-axis for a plot.")

#         except Exception as e:
#             st.error(f"Query failed to execute:\n\n{e}")






import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.types import Numeric, Integer, Float, BigInteger

# --- Page Configuration ---
st.set_page_config(
    page_title="TimescaleDB Dashboard",
    page_icon="🐘",
    layout="wide"
)

st.title("TimescaleDB Analysis Dashboard 📈")
st.markdown("A tool to run and visualize time-series queries using `time_bucket()`.")

# --- Database Connection ---
@st.cache_resource
def get_db_engine():
    """
    Creates and caches a SQLAlchemy engine based on environment variables.
    """
    try:
        user = "tsdbadmin" # os.environ.get("PGUSER")
        pw = "m6x55zbrm5e93dwv" # os.environ.get("PGPASSWORD")
        host = "s1l0v9f595.xy2jmmua0e.tsdb.cloud.timescale.com" # os.environ.get("PGHOST")
        port = 33127 # os.environ.get("PGPORT")
        db = "tsdb" # os.environ.get("PGDATABASE")

        if not all([user, pw, host, port, db]):
            st.error("Missing one or more database environment variables (PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE).")
            st.stop()

        connection_string = f"postgresql://{user}:{pw}@{host}:{port}/{db}?sslmode=require"
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            pass
        
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

engine = get_db_engine()

# --- Schema Fetching ---
@st.cache_data(ttl=600)
def fetch_schema(_engine):
    """
    Fetches schema information, focusing on TimescaleDB hypertables.
    Caches the result for 10 minutes.
    """
    inspector = inspect(_engine)
    hypertables = []
    
    # Query TimescaleDB's information view for hypertables
    try:
        with _engine.connect() as conn:
            # Join hypertables with dimensions to robustly get the time column
            query = text("""
                SELECT
                    h.hypertable_schema,
                    h.hypertable_name,
                    d.column_name AS time_column_name
                FROM
                    timescaledb_information.hypertables h
                JOIN
                    timescaledb_information.dimensions d
                ON
                    h.hypertable_schema = d.hypertable_schema
                    AND h.hypertable_name = d.hypertable_name
                WHERE
                    d.dimension_type = 'Time'
                ORDER BY
                    h.hypertable_schema,
                    h.hypertable_name;
            """)
            result = conn.execute(query)
            for row in result:
                # Use ._asdict() for named-column access
                row_dict = row._asdict()
                # Store schema, table name, and time column
                hypertables.append((row_dict['hypertable_schema'], row_dict['hypertable_name'], row_dict['time_column_name']))
    except Exception as e:
        st.sidebar.warning(f"Could not fetch hypertables. (Is TimescaleDB enabled?)\nError: {e}", icon="⚠️")
        return {}

    schema_info = {}
    # Unpack schema, table name, and time column
    for table_schema, table_name, time_col in hypertables:
        columns = []
        # Create a qualified name for display and for the default query
        qualified_table_name = f'"{table_schema}"."{table_name}"'
        try:
            # Pass schema to inspector
            cols = inspector.get_columns(table_name, schema=table_schema)
            for col in cols:
                columns.append({"name": col['name'], "type": str(col['type'])})
            # Use the qualified name as the key
            schema_info[qualified_table_name] = {'time_col': time_col, 'columns': columns}
        except Exception as e:
            st.sidebar.error(f"Error fetching columns for {qualified_table_name}: {e}")
            
    return schema_info

# --- Sidebar ---
st.sidebar.title("Database Schema")
st.sidebar.markdown("This panel shows the 'hypertables' (time-series tables) found in your database.")

with st.sidebar:
    with st.spinner("Loading schema..."):
        schema = fetch_schema(engine)

if not schema:
    st.sidebar.warning("No hypertables found.")
    default_query = "SELECT 'No hypertables found' AS status;"
else:
    st.sidebar.success(f"Found {len(schema)} hypertable(s).")
    
    # Generate an intelligent default query
    first_table_qualified = list(schema.keys())[0] # This is now "schema"."table"
    time_col = schema[first_table_qualified]['time_col']
    
    # Try to find a numeric column to aggregate
    value_col = None
    for col in schema[first_table_qualified]['columns']:
        # Check if column type is numeric
        try:
            # Create a temporary instance of the type for checking
            col_type_instance = col['type']
            # A bit of a hack: check type name strings
            type_str = str(col_type_instance).lower()
            if 'int' in type_str or 'float' in type_str or 'numeric' in type_str or 'double' in type_str:
                 # Avoid the time column itself
                if col['name'] != time_col:
                    value_col = col['name']
                    break
        except Exception:
            continue # Skip if type instantiation fails
            
    if not value_col:
        value_col = "[your_numeric_column]" # Fallback
        value_col_quoted = value_col # Don't quote the placeholder
        time_col_quoted = f'"{time_col}"'
        value_col_display = "value" # generic display name
    else:
        value_col_quoted = f'"{value_col}"' # Quote the found column
        time_col_quoted = f'"{time_col}"' # Quote the time column
        value_col_display = value_col # use actual column name

    default_query = f"""-- Default query: 1-day buckets for the last 30 days
SELECT
    time_bucket('1 day', {time_col_quoted}) AS bucket,
    AVG({value_col_quoted}) AS "avg_{value_col_display}"
FROM
    {first_table_qualified}
WHERE
    {time_col_quoted} > now() - INTERVAL '30 days'
GROUP BY
    bucket
ORDER BY
    bucket DESC;
"""

    # Display schema in sidebar
    for table_name_qualified, info in schema.items():
        with st.sidebar.expander(f"🔵 {table_name_qualified} (Hypertable)"):
            st.markdown(f"**Time Column:** `{info['time_col']}`")
            st.markdown("**All Columns:**")
            for col in info['columns']:
                st.code(f"{col['name']}: {col['type']}", language="sql")


# --- Main Page: Query Editor ---
st.header("Query Editor")
query = st.text_area(
    "Enter your SQL query. Use `time_bucket()` for time-series analysis.",
    value=default_query,
    height=250,
    key="sql_query"
)

if st.button("Run Query", type="primary"):
    if not query:
        st.warning("Query is empty.")
    else:
        st.subheader("Query Results")
        try:
            with st.spinner("Running query..."):
                df = pd.read_sql(query, engine)

            st.dataframe(df, use_container_width=True)

            # --- Plotting ---
            st.subheader("Plot")
            
            # Try to find a time column for plotting
            plot_col = None
            if 'bucket' in df.columns:
                plot_col = 'bucket'
            else:
                # Find first datetime column
                for col_name, dtype in df.dtypes.items():
                    if pd.api.types.is_datetime64_any_dtype(dtype):
                        plot_col = col_name
                        break
            
            if plot_col:
                try:
                    # Set time column as index for plotting
                    df_plot = df.set_index(plot_col)
                    
                    # Identify numeric columns to plot
                    numeric_cols = df_plot.select_dtypes(include=['number']).columns
                    if not numeric_cols.empty:
                        st.line_chart(df_plot[numeric_cols])
                        st.caption(f"Plotting numeric columns against `{plot_col}`.")
                    else:
                        st.warning("No numeric columns found to plot against the time axis.")
                except Exception as e:
                    st.error(f"Failed to create plot: {e}")
            else:
                st.info("No time column (like 'bucket' or another timestamp) found in results to use as X-axis for a plot.")

        except Exception as e:
            st.error(f"Query failed to execute:\n\n{e}")


"""
SELECT bucket,
SQRT(avg_b_x*avg_b_x + avg_b_y*avg_b_y + avg_b_z*avg_b_z) as "avg_magnitude",
avg_b_x,
avg_b_y,
avg_b_z
FROM
(SELECT
    time_bucket('5 minute', "timestamp") AS bucket,
    AVG("b_x") AS "avg_b_x",
    AVG("b_y") AS "avg_b_y",
    AVG("b_z") AS "avg_b_z"
FROM
    "public"."sensor_data"
WHERE
    "timestamp" > now() - INTERVAL '30 days'
AND
    sensor_id = 'S20251008_105355_44180345587365_1'
GROUP BY
    bucket
ORDER BY
    bucket DESC);

"""