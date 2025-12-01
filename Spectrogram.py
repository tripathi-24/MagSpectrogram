import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import spectrogram, savgol_filter, butter, filtfilt


# Page configuration
st.set_page_config(
    page_title="Magnetic Field Spectrogram",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Magnetic Field Spectrogram Generator")
st.markdown("Generate spectrograms from magnetic field data")


@st.cache_data
def load_data(file_path_or_buffer):
    """Load CSV data with caching"""
    try:
        df = pd.read_csv(file_path_or_buffer)
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Ensure numeric columns
        for col in ["b_x", "b_y", "b_z"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def compute_resultant(df):
    """Calculate resultant magnetic field magnitude"""
    bx = df["b_x"].astype(float)
    by = df["b_y"].astype(float)
    bz = df["b_z"].astype(float)
    return np.sqrt(bx * bx + by * by + bz * bz)


def estimate_sampling_rate(timestamps):
    """Estimate sampling rate from timestamps"""
    ts = timestamps.dropna().astype("int64").values.astype(float) / 1e9
    if ts.size < 2:
        return 0.0
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        return 0.0
    return 1.0 / median_dt


def compute_spectrogram_simple(
    signal: np.ndarray,
    sampling_hz: float,
    window_seconds: float,
    overlap_ratio: float,
    mode: str = "psd",
):
    """Simple spectrogram computation"""
    from scipy.signal import spectrogram

    if sampling_hz <= 0 or len(signal) < 32:
        return None
    
    # Remove DC component
    signal_processed = signal - np.mean(signal)
    
    # Apply Savitzky-Golay smoothing if possible
    if len(signal_processed) > 21:
        try:
            window_length = min(21, len(signal_processed) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 5:
                signal_processed = savgol_filter(signal_processed, window_length, 3)
        except Exception:
            pass
    
    # Calculate window size
    nperseg = max(8, int(window_seconds * sampling_hz))
    nperseg = min(nperseg, len(signal_processed) // 2)
    noverlap = int(overlap_ratio * nperseg)
    
    try:
        if mode == "psd":
            f, t, sxx = spectrogram(
                signal_processed,
                fs=sampling_hz,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
                mode="psd",
                detrend='linear',
                return_onesided=True,
                window="hann",
            )
        else:
            f, t, sxx = spectrogram(
                signal_processed,
                fs=sampling_hz,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="spectrum",
                mode="magnitude",
                detrend='linear',
                return_onesided=True,
                window="hann",
            )
        
        # Apply log scaling for better visualization
        if mode == "psd":
            sxx = 10 * np.log10(sxx + 1e-15)
        else:
            sxx = np.log10(sxx + 1e-15)
        
        return f, t, sxx
    except Exception as e:
        st.error(f"Spectrogram computation failed: {e}")
        return None


def main():
    # File upload section
    st.header("📁 Data Source")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file with magnetic field data. Required columns: timestamp, b_x, b_y, b_z, sensor_id"
    )
    
    # Default file path
    default_file = "Data_relevant_0900_Onwards.csv"
    
    # Determine which file to use
    if uploaded_file is not None:
        # User uploaded a file
        file_source = uploaded_file.name
        file_to_load = uploaded_file
        
        # Clear cache when new file is uploaded
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
        
        if st.session_state.last_uploaded_file != uploaded_file.name:
            load_data.clear()
            st.session_state.last_uploaded_file = uploaded_file.name
        
        st.info(f"📤 Using uploaded file: **{file_source}**")
    else:
        # Use default file if it exists
        if os.path.exists(default_file):
            file_source = default_file
            file_to_load = default_file
            st.info(f"📂 Using default file: **{file_source}**")
        else:
            st.warning(f"⚠️ No file uploaded and default file '{default_file}' not found. Please upload a CSV file.")
            st.stop()
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(file_to_load)
    
    if df.empty:
        st.error("No data loaded. Please check the CSV file format.")
        st.markdown("""
        **Required columns:**
        - `timestamp`: Date/time information
        - `b_x`, `b_y`, `b_z`: Magnetic field components (nT)
        - `sensor_id`: Sensor identifier
        """)
        return
    
    st.success(f"✅ Loaded {len(df):,} rows from {file_source}")
    
    # Display data range information
    if not df.empty and df["timestamp"].notna().any():
        tmin_all = df["timestamp"].min()
        tmax_all = df["timestamp"].max()
        duration_all = tmax_all - tmin_all
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Data Range:**\n{tmin_all.strftime('%Y-%m-%d %H:%M:%S')} to {tmax_all.strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.info(f"**Duration:**\n{duration_all}")
        with col3:
            valid_timestamps = df["timestamp"].notna().sum()
            st.info(f"**Valid Timestamps:**\n{valid_timestamps:,} / {len(df):,} rows")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Sensor selection
    sensors = sorted(df["sensor_id"].dropna().unique().tolist()) if not df.empty else []
    if not sensors:
        st.error("No sensors found in data")
        return
    
    selected_sensor = st.sidebar.selectbox(
        "Select Sensor",
        options=sensors,
        index=0
    )
    
    # Show sensor-specific time range
    sensor_df = df[df["sensor_id"] == selected_sensor].copy()
    if not sensor_df.empty and sensor_df["timestamp"].notna().any():
        tmin_sensor = sensor_df["timestamp"].min()
        tmax_sensor = sensor_df["timestamp"].max()
        duration_sensor = tmax_sensor - tmin_sensor
        st.sidebar.caption(f"**Sensor {selected_sensor} range:**\n{tmin_sensor.strftime('%H:%M:%S')} to {tmax_sensor.strftime('%H:%M:%S')}\n({duration_sensor})")
    
    # Field selection
    field_options = ["b_x", "b_y", "b_z", "resultant"]
    selected_field = st.sidebar.selectbox(
        "Select Field",
        options=field_options,
        index=3  # Default to resultant
    )
    
    # Time range selection - use full dataset range, not sensor-specific
    if not df.empty and df["timestamp"].notna().any():
        tmin = df["timestamp"].min()
        tmax = df["timestamp"].max()
        
        # Use sensor-specific range as default if available
        if not sensor_df.empty and sensor_df["timestamp"].notna().any():
            tmin_sensor = sensor_df["timestamp"].min()
            tmax_sensor = sensor_df["timestamp"].max()
            default_min = tmin_sensor.to_pydatetime()
            default_max = tmax_sensor.to_pydatetime()
        else:
            default_min = tmin.to_pydatetime()
            default_max = tmax.to_pydatetime()
        
        time_range = st.sidebar.slider(
            "Time Range",
            min_value=tmin.to_pydatetime(),
            max_value=tmax.to_pydatetime(),
            value=(default_min, default_max),
            step=pd.Timedelta(seconds=1),
            format="YYYY-MM-DD HH:mm:ss",
        )
        st.sidebar.caption(f"**Full dataset range:** {tmin.strftime('%H:%M:%S')} to {tmax.strftime('%H:%M:%S')}")
    else:
        st.error("No valid timestamps found")
        return
    
    # Spectrogram parameters
    st.sidebar.subheader("📊 Spectrogram Parameters")
    window_seconds = st.sidebar.number_input(
        "Window Size (seconds)",
        min_value=0.1,
        max_value=60.0,
        value=1.0,
        step=0.1,
        help="Size of each FFT window"
    )
    overlap_ratio = st.sidebar.slider(
        "Overlap Ratio",
        min_value=0.0,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Overlap between windows (0 = no overlap, 0.95 = 95% overlap)"
    )
    spec_mode = st.sidebar.selectbox(
        "Mode",
        options=["psd", "magnitude"],
        index=0,
        help="PSD = Power Spectral Density, Magnitude = Magnitude Spectrum"
    )
    
    # Display options
    st.sidebar.subheader("🎨 Display Options")
    colormap = st.sidebar.selectbox(
        "Colormap",
        options=["Viridis", "Plasma", "Inferno", "Turbo", "Cividis", "Jet"],
        index=0
    )
    log_freq = st.sidebar.checkbox(
        "Log Frequency Axis",
        value=True,
        help="Use logarithmic scale for frequency axis"
    )
    
    # Time alignment option
    st.sidebar.markdown("---")
    time_alignment_mode = st.sidebar.radio(
        "Time Alignment Mode",
        options=["Window Center (Standard)", "Window Start (Better Event Alignment)"],
        index=0,
        help="Window Center: Shows when window is centered (standard). Window Start: Shows when window begins (better for event timing)."
    )
    use_window_start = (time_alignment_mode == "Window Start (Better Event Alignment)")
    
    # Filter data
    filtered_df = df[df["sensor_id"] == selected_sensor].copy()
    t0, t1 = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
    filtered_df = filtered_df[
        (filtered_df["timestamp"] >= t0) & (filtered_df["timestamp"] <= t1)
    ].sort_values("timestamp")
    
    if filtered_df.empty:
        st.warning(f"No data for sensor {selected_sensor} in the selected time range ({t0.strftime('%H:%M:%S')} to {t1.strftime('%H:%M:%S')})")
        # Show what data is actually available for this sensor
        if not sensor_df.empty and sensor_df["timestamp"].notna().any():
            sensor_tmin = sensor_df["timestamp"].min()
            sensor_tmax = sensor_df["timestamp"].max()
            st.info(f"**Available data for {selected_sensor}:** {sensor_tmin.strftime('%H:%M:%S')} to {sensor_tmax.strftime('%H:%M:%S')}")
        return
    
    # Show filtered data info
    filtered_tmin = filtered_df["timestamp"].min()
    filtered_tmax = filtered_df["timestamp"].max()
    st.caption(f"📊 Filtered data: {len(filtered_df):,} rows from {filtered_tmin.strftime('%H:%M:%S')} to {filtered_tmax.strftime('%H:%M:%S')}")
    
    # Get signal values
    if selected_field == "resultant":
        signal_values = compute_resultant(filtered_df).values
    else:
        signal_values = pd.to_numeric(filtered_df[selected_field], errors="coerce").dropna().values
    
    if len(signal_values) < 10:
        st.warning(f"Insufficient data points ({len(signal_values)}) for spectrogram")
        return
    
    # Estimate sampling rate
    fs = estimate_sampling_rate(filtered_df["timestamp"])
    if fs <= 0:
        st.error("Cannot estimate sampling rate")
        return
    
    # Frequency range selection (after we have fs estimate)
    st.sidebar.subheader("🔬 Frequency Range Selection")
    
    # Calculate Nyquist and min frequency estimates
    nyquist_estimate = fs / 2.0
    min_freq_estimate = 1.0 / window_seconds if window_seconds > 0 else 0.001
    
    # Frequency range presets
    st.sidebar.markdown("**Quick Presets:**")
    preset_cols = st.sidebar.columns(3)
    with preset_cols[0]:
        if st.button("Full Range", use_container_width=True):
            st.session_state.min_freq = 0.0
            st.session_state.max_freq = 0.0  # 0 means auto (Nyquist)
            st.session_state.use_custom_range = False
            st.rerun()
    with preset_cols[1]:
        if st.button("Ultra Low", use_container_width=True, help="0 to 1 μHz"):
            st.session_state.min_freq = 0.0
            st.session_state.max_freq = 1e-6
            st.session_state.use_custom_range = True
            st.rerun()
    with preset_cols[2]:
        if st.button("Very Low", use_container_width=True, help="0 to 1 mHz"):
            st.session_state.min_freq = 0.0
            st.session_state.max_freq = 0.001
            st.session_state.use_custom_range = True
            st.rerun()
    
    preset_cols2 = st.sidebar.columns(2)
    with preset_cols2[0]:
        if st.button("Low Freq", use_container_width=True, help="0 to 0.1 Hz"):
            st.session_state.min_freq = 0.0
            st.session_state.max_freq = 0.1
            st.session_state.use_custom_range = True
            st.rerun()
    with preset_cols2[1]:
        if st.button("Reset", use_container_width=True):
            st.session_state.min_freq = 0.0
            st.session_state.max_freq = 0.0
            st.session_state.use_custom_range = False
            st.rerun()
    
    # Custom frequency range inputs
    use_custom_range = st.sidebar.checkbox(
        "Use Custom Frequency Range",
        value=st.session_state.get('use_custom_range', False),
        help="Enable to set custom min/max frequency limits"
    )
    
    if use_custom_range:
        min_freq = st.sidebar.number_input(
            "Min Frequency (Hz)",
            min_value=0.0,
            max_value=nyquist_estimate,
            value=st.session_state.get('min_freq', 0.0),
            step=0.0001,
            format="%.6f",
            help=f"Minimum frequency to display (0 to {nyquist_estimate:.6f} Hz)"
        )
        max_freq = st.sidebar.number_input(
            "Max Frequency (Hz)",
            min_value=0.0,
            max_value=nyquist_estimate * 1.1,  # Allow slightly above Nyquist for display
            value=st.session_state.get('max_freq', nyquist_estimate if nyquist_estimate > 0 else 1.0),
            step=0.0001,
            format="%.6f",
            help=f"Maximum frequency to display (0 = auto/Nyquist, max: {nyquist_estimate:.6f} Hz)"
        )
        
        # Validate range
        if max_freq > 0 and min_freq >= max_freq:
            st.sidebar.error("Min frequency must be less than max frequency")
            max_freq = min_freq + 0.0001
        
        st.session_state.min_freq = min_freq
        st.session_state.max_freq = max_freq
    else:
        min_freq = 0.0
        max_freq = 0.0  # 0 means use full range
        st.session_state.min_freq = 0.0
        st.session_state.max_freq = 0.0
    
    st.sidebar.caption(f"**Theoretical Limits:**\nMin: {min_freq_estimate:.6f} Hz | Max (Nyquist): {nyquist_estimate:.6f} Hz")
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", f"{len(signal_values):,}")
    with col2:
        st.metric("Sampling Rate", f"{fs:.2f} Hz")
    with col3:
        signal_mean = np.mean(signal_values)
        st.metric("Mean Value", f"{signal_mean:.2f} nT")
    with col4:
        signal_std = np.std(signal_values)
        st.metric("Std Dev", f"{signal_std:.2f} nT")
    
    # Time series plot
    st.subheader("📈 Time Series")
    time_series_df = pd.DataFrame({
        "timestamp": filtered_df["timestamp"].values[:len(signal_values)],
        selected_field: signal_values
    })
    
    import plotly.express as px
    fig_ts = px.line(
        time_series_df,
        x="timestamp",
        y=selected_field,
        title=f"Time Series: {selected_field} (nT)",
        labels={"timestamp": "Time", selected_field: f"{selected_field} (nT)"}
    )
    fig_ts.update_layout(hovermode="x unified")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Compute spectrogram
    st.subheader("🎵 Spectrogram")
    
    with st.spinner("Computing spectrogram..."):
        result = compute_spectrogram_simple(
            signal_values,
            fs,
            window_seconds,
            overlap_ratio,
            mode=spec_mode
        )
    
    if result is None:
        st.error("Failed to compute spectrogram")
        return
    
    f, t, sxx = result
    
    # Apply frequency range filtering if custom range is set
    nyquist_freq = fs / 2.0
    freq_mask = np.ones_like(f, dtype=bool)
    
    if use_custom_range:
        # Apply min frequency filter
        if min_freq > 0:
            freq_mask &= f >= min_freq
            if not np.any(freq_mask):
                st.warning(f"No frequencies >= {min_freq:.6f} Hz. Available range: {f[0]:.6f} to {f[-1]:.6f} Hz")
                min_freq = f[0]  # Reset to minimum available
        
        # Apply max frequency filter
        if max_freq > 0:
            # Use Nyquist if max_freq exceeds it
            effective_max = min(max_freq, nyquist_freq)
            freq_mask &= f <= effective_max
            if not np.any(freq_mask):
                st.warning(f"No frequencies <= {max_freq:.6f} Hz. Available range: {f[0]:.6f} to {f[-1]:.6f} Hz")
                max_freq = f[-1]  # Reset to maximum available
        else:
            # max_freq = 0 means use Nyquist
            effective_max = nyquist_freq
            freq_mask &= f <= effective_max
        
        # Filter frequency arrays
        if np.any(freq_mask):
            f_filtered = f[freq_mask]
            sxx_filtered = sxx[freq_mask, :]
            f_display = f_filtered
            sxx_display = sxx_filtered
            freq_range_info = f" (Filtered: {f_filtered[0]:.6f} - {f_filtered[-1]:.6f} Hz)"
        else:
            st.error("No frequencies in selected range!")
            f_display = f
            sxx_display = sxx
            freq_range_info = ""
    else:
        # Use full range
        f_display = f
        sxx_display = sxx
        freq_range_info = ""
    
    # Convert time bins to actual timestamps
    # IMPORTANT: scipy.signal.spectrogram returns 't' as the CENTER time of each window, not the start
    start_time = filtered_df["timestamp"].min()
    
    # Calculate window parameters for time alignment explanation
    nperseg = max(8, int(window_seconds * fs))
    nperseg = min(nperseg, len(signal_values) // 2)
    noverlap = int(overlap_ratio * nperseg)
    window_advance = nperseg - noverlap  # Samples between window starts
    window_advance_seconds = window_advance / fs  # Time between window starts
    
    # The 't' array from spectrogram represents window CENTER times
    # Window start = center - window_seconds/2
    # Window end = center + window_seconds/2
    window_center_times = [start_time + pd.Timedelta(seconds=float(t_bin)) for t_bin in t]
    window_start_times = [t_center - pd.Timedelta(seconds=window_seconds/2) for t_center in window_center_times]
    window_end_times = [t_center + pd.Timedelta(seconds=window_seconds/2) for t_center in window_center_times]
    
    # Choose which time to display based on user selection
    if use_window_start:
        # Use window START times for better event alignment
        actual_times = window_start_times
        time_label = "Window Start Times"
        alignment_note = "Using window START times - events appear closer to their actual occurrence time"
    else:
        # Use window CENTER times (standard spectrogram behavior)
        actual_times = window_center_times
        time_label = "Window Center Times"
        alignment_note = "Using window CENTER times (standard spectrogram behavior)"
    
    # Create spectrogram plot
    heatmap = go.Heatmap(
        x=actual_times,
        y=f_display,
        z=sxx_display,
        colorscale=colormap.lower(),
        colorbar=dict(
            title="Power (dB)" if spec_mode == "psd" else "Magnitude (log)"
        ),
        zsmooth="best",
    )
    
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=f"Spectrogram: {selected_field} - {selected_sensor}{freq_range_info}",
        xaxis_title=f"Time ({time_label})",
        yaxis_title="Frequency (Hz)",
        yaxis_type='log' if (log_freq and len(f_display) > 0 and np.min(f_display) > 0) else 'linear',
        height=600,
    )
    
    # Show alignment note
    if use_window_start:
        st.info(f"✅ **{alignment_note}** - Events should appear closer to their actual occurrence time in the time series plot.")
    else:
        st.caption(f"ℹ️ {alignment_note} - Switch to 'Window Start' mode for better event alignment.")
    
    # Set y-axis range if custom range is used
    if use_custom_range and len(f_display) > 0:
        fig.update_yaxes(range=[f_display[0], f_display[-1]])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time alignment explanation
    with st.expander("⏰ Understanding Time Alignment in Spectrograms", expanded=False):
        st.markdown(f"""
        ### **Why Anomalies Appear at Different Times:**
        
        **Time Series Plot:**
        - Shows **exact timestamps** of each data point
        - Anomaly at **11:00-11:15** = exact time when the event occurred
        
        **Spectrogram Plot:**
        - Shows **window center times** (not window start/end)
        - Each time point represents the **center** of a time window
        - Window size: **{window_seconds:.1f} seconds**
        - Window overlap: **{overlap_ratio*100:.1f}%** ({noverlap} samples)
        - Time between windows: **{window_advance_seconds:.2f} seconds**
        
        ### **How Windows Work:**
        
        **Current Mode:** {time_label}
        
        If an anomaly appears in the spectrogram:
        - **Window center** time: {window_center_times[0].strftime('%H:%M:%S') if len(window_center_times) > 0 else 'N/A'}
        - **Window start** time: {window_start_times[0].strftime('%H:%M:%S') if len(window_start_times) > 0 else 'N/A'}
        - **Window end** time: {window_end_times[0].strftime('%H:%M:%S') if len(window_end_times) > 0 else 'N/A'}
        
        **With Window Start Mode:**
        - Events appear at the **start** of the window that captures them
        - Better alignment with time series plot
        - Event at 11:00 will appear closer to 11:00 (or slightly before)
        
        **With Window Center Mode (Standard):**
        - Events appear at the **center** of the window
        - Standard spectrogram behavior
        - Event at 11:00 might appear at 10:40 if window is large
        
        **Why the Shift?**
        1. **Window Size Effect**: With a {window_seconds:.1f}s window, an event at 11:00 might be captured in a window that **starts earlier** (e.g., 10:50) but has its **center** at 10:55 or 10:40
        2. **Overlap Effect**: With {overlap_ratio*100:.0f}% overlap, windows overlap significantly, so an event can appear in multiple windows
        3. **Frequency Analysis Delay**: The spectrogram analyzes frequency content over a window, so the time shown is when the **analysis window** is centered, not when the event starts
        
        ### **Example:**
        - **Event occurs**: 11:00:00 (visible in time series)
        - **Window size**: {window_seconds:.1f} seconds
        - **Window that captures it**:
          - Starts: 10:59:30 (if window_seconds = 1s, or earlier if larger)
          - Center: 11:00:00 (or 10:40 if window is very large)
          - Ends: 11:00:30
        - **Spectrogram shows**: Window center time (11:00:00 or 10:40 depending on window size)
        
        ### **To Match Times Better:**
        1. **Use smaller windows** ({window_seconds/2:.1f}s or less) for better time resolution
        2. **Check window start/end times** - the event might be in a window that starts before the center time
        3. **Remember**: Spectrogram time = window **center**, not event **start**
        
        **Current Window Info:**
        - Window size: {window_seconds:.1f} seconds
        - Window advance: {window_advance_seconds:.2f} seconds
        - First window center: {actual_times[0].strftime('%H:%M:%S') if len(actual_times) > 0 else 'N/A'}
        - First window range: {window_start_times[0].strftime('%H:%M:%S') if len(window_start_times) > 0 else 'N/A'} to {window_end_times[0].strftime('%H:%M:%S') if len(window_end_times) > 0 else 'N/A'}
        """)
    
    # Spectrogram statistics
    st.subheader("📊 Spectrogram Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        if use_custom_range and len(f_display) > 0:
            st.metric("Frequency Range (Displayed)", f"{f_display[0]:.6f} - {f_display[-1]:.6f} Hz",
                     help="Filtered frequency range based on your selection")
        else:
            st.metric("Frequency Range", f"{f[0]:.6f} - {f[-1]:.6f} Hz",
                     help="Full frequency range available")
    with col2:
        st.metric("Frequency Bins", len(f_display) if use_custom_range else len(f),
                 help="Number of frequency bins in displayed range")
    with col3:
        st.metric("Time Bins", len(t))
    
    # Show full vs filtered range comparison
    if use_custom_range and len(f_display) > 0:
        st.info(f"📊 **Range Filter Applied:** Showing {len(f_display)}/{len(f)} frequency bins "
               f"({f_display[0]:.6f} - {f_display[-1]:.6f} Hz) out of full range "
               f"({f[0]:.6f} - {f[-1]:.6f} Hz)")
    
    # Frequency limits and harmonics information
    nyquist_freq = fs / 2.0
    min_resolvable_freq = 1.0 / window_seconds  # Minimum frequency that can be resolved
    frequency_resolution = fs / (2 * len(f))  # Frequency bin width
    
    st.subheader("🔬 Frequency Range Limits & Harmonics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nyquist Frequency", f"{nyquist_freq:.6f} Hz", 
                 help="Maximum frequency that can be detected (fs/2). This is the absolute upper limit.")
    with col2:
        st.metric("Min Resolvable", f"{min_resolvable_freq:.6f} Hz",
                 help=f"Minimum frequency that can be resolved with {window_seconds}s window")
    with col3:
        st.metric("Frequency Resolution", f"{frequency_resolution:.6f} Hz",
                 help="Smallest frequency difference that can be distinguished")
    with col4:
        st.metric("Sampling Rate", f"{fs:.4f} Hz",
                 help="Data sampling frequency")
    
    # Harmonics explanation
    with st.expander("📚 Understanding Frequency Limits & Harmonics", expanded=True):
        st.markdown("""
        ### **Frequency Range Limits:**
        
        **Upper Limit (Y-axis Top):**
        - **Nyquist Frequency = fs/2** = Maximum frequency that can be detected
        - This is a **hard limit** - frequencies above this cannot be detected (aliasing occurs)
        - For your data: **{:.6f} Hz** is the absolute maximum
        
        **Lower Limit (Y-axis Bottom):**
        - **Minimum resolvable frequency ≈ 1/window_size**
        - With {:.1f}s window: minimum ≈ **{:.6f} Hz**
        - Longer windows = better low-frequency resolution
        - Shorter windows = better time resolution
        
        **Frequency Resolution:**
        - **Bin width = {:.6f} Hz** (smallest distinguishable frequency difference)
        - Determined by: window size and sampling rate
        - More bins = finer frequency resolution
        
        ### **Harmonics in Spectrograms:**
        
        **What are Harmonics?**
        - Harmonics are integer multiples of a fundamental frequency
        - If fundamental = f₀, harmonics appear at: 2f₀, 3f₀, 4f₀, etc.
        - They appear as **parallel horizontal bands** in the spectrogram
        
        **How Far Can Harmonics Go?**
        - **Upward (Higher Frequencies):**
          - Limited by **Nyquist frequency (fs/2)**
          - Example: If fs = 10 Hz, Nyquist = 5 Hz
          - You can see harmonics up to 5 Hz maximum
          - Higher harmonics may be aliased (folded back) if they exceed Nyquist
        
        - **Downward (Lower Frequencies):**
          - Limited by **1/window_size**
          - Example: With 1s window, minimum ≈ 1 Hz
          - With 10s window, minimum ≈ 0.1 Hz
          - With 100s window, minimum ≈ 0.01 Hz
        
        **Practical Limits for Your Data:**
        - **Sampling Rate:** {:.4f} Hz
        - **Maximum Detectable Frequency:** {:.6f} Hz (Nyquist)
        - **Minimum Resolvable Frequency:** {:.6f} Hz (with current window)
        - **Harmonic Range:** Can detect harmonics from {:.6f} Hz to {:.6f} Hz
        
        **Example:**
        - If fundamental frequency = 0.1 Hz
        - Harmonics: 0.2 Hz, 0.3 Hz, 0.4 Hz, 0.5 Hz, ...
        - All harmonics must be < {:.6f} Hz (Nyquist limit)
        - With {:.1f}s window, you can resolve down to {:.6f} Hz
        
        ### **Tips for Detecting Harmonics:**
        1. **Use longer windows** (10-60s) to see lower frequency harmonics
        2. **Check for parallel horizontal bands** at integer multiples
        3. **Look for patterns** that repeat at regular frequency intervals
        4. **Adjust window size** to balance frequency vs. time resolution
        """.format(
            nyquist_freq,           # 0: line 593
            window_seconds,         # 1: line 597
            min_resolvable_freq,    # 2: line 597
            frequency_resolution,   # 3: line 602
            fs,                     # 4: line 627 (sampling rate)
            nyquist_freq,           # 5: line 628
            min_resolvable_freq,    # 6: line 629
            min_resolvable_freq,    # 7: line 630 (min)
            nyquist_freq,           # 8: line 630 (max)
            nyquist_freq,           # 9: line 635
            window_seconds,         # 10: line 636
            min_resolvable_freq     # 11: line 636
        ))
    
    # Frequency info
    st.caption(f"📊 **Current Settings:** Nyquist = {nyquist_freq:.6f} Hz | Window = {window_seconds}s | Overlap = {overlap_ratio*100:.1f}% | Resolution = {frequency_resolution:.6f} Hz/bin")


if __name__ == "__main__":
    main()