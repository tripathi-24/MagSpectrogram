import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data(show_spinner=False)
def read_single_csv(file_path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(file_path_or_buffer)
    # Normalize expected columns
    expected = [
        "id",
        "b_x",
        "b_y", 
        "b_z",
        "timestamp",
        "lat",
        "lon",
        "alt",
        "theta_x",
        "theta_y",
        "theta_z",
        "sensor_id",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Ensure numeric where applicable
    for c in ["b_x", "b_y", "b_z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def read_multiple_csvs(files: List) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(read_single_csv(f))
        except Exception as e:
            st.warning(f"Failed to read {getattr(f, 'name', str(f))}: {e}")
    if not frames:
        return pd.DataFrame(columns=[
            "id", "b_x", "b_y", "b_z", "timestamp", "lat", "lon", "alt",
            "theta_x", "theta_y", "theta_z", "sensor_id"
        ])
    df = pd.concat(frames, ignore_index=True)
    # Sort by time for better UX
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df


def compute_resultant(df: pd.DataFrame) -> pd.Series:
    """Calculate resultant magnetic field magnitude in nT (nanotesla)"""
    bx = df["b_x"].astype(float)
    by = df["b_y"].astype(float)
    bz = df["b_z"].astype(float)
    return np.sqrt(bx * bx + by * by + bz * bz)


def layout_sidebar(df: pd.DataFrame) -> Tuple[str, str, Tuple[pd.Timestamp, pd.Timestamp], float, float]:
    st.sidebar.header("Controls")
    # Sensor selection
    sensor_options = sorted(df["sensor_id"].dropna().unique().tolist()) if not df.empty else []
    selected_sensor = st.sidebar.selectbox("Sensor ID", options=["All"] + sensor_options, index=0)

    # Axis or resultant
    series_options = ["b_x", "b_y", "b_z", "resultant"]
    selected_series = st.sidebar.selectbox("Series", options=series_options, index=0)

    # Time window
    if not df.empty and df["timestamp"].notna().any():
        tmin = df["timestamp"].min()
        tmax = df["timestamp"].max()
    else:
        tmin = pd.Timestamp.now() - pd.Timedelta(hours=1)
        tmax = pd.Timestamp.now()
    time_range = st.sidebar.slider(
        "Time range",
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
        value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
        step=pd.Timedelta(seconds=1),
        format="YYYY-MM-DD HH:mm:ss",
    )

    # Spectrogram settings
    bin_width_seconds = st.sidebar.number_input(
        "Spectrogram window (s)", min_value=0.05, max_value=12000.0, value=1.0, step=0.05
    )
    overlap_ratio = st.sidebar.slider("Overlap", min_value=0.0, max_value=0.99, value=0.5, step=0.05)

    return selected_sensor, selected_series, (pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])), bin_width_seconds, overlap_ratio


def filter_df(df: pd.DataFrame, sensor: str, time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df
    if sensor != "All":
        filtered = filtered[filtered["sensor_id"] == sensor]
    t0, t1 = time_range
    if "timestamp" in filtered.columns and pd.notna(t0) and pd.notna(t1):
        mask = (filtered["timestamp"] >= t0) & (filtered["timestamp"] <= t1)
        filtered = filtered.loc[mask]
    return filtered


def plot_time_series(df: pd.DataFrame, series: str):
    import plotly.express as px

    if df.empty:
        st.info("No data to plot.")
        return
    plot_df = df.copy()
    if series == "resultant":
        plot_df["resultant"] = compute_resultant(plot_df)
    y_col = series
    fig = px.line(
        plot_df,
        x="timestamp",
        y=y_col,
        color="sensor_id",
        title=f"Time Series: {y_col} (nT)",
        labels={"timestamp": "Time", y_col: f"{y_col} (nT)", "sensor_id": "Sensor"},
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def estimate_sampling_rate(timestamps: pd.Series) -> float:
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


def compute_spectrogram(
    signal: np.ndarray,
    sampling_hz: float,
    window_seconds: float,
    overlap_ratio: float,
    mode: str = "psd",
    window: str = "hann",
):
    """
    Ultra-enhanced spectrogram computation for extremely weak magnetic field data
    Implements aggressive signal processing to reveal patterns in flat spectrograms
    """
    from scipy.signal import spectrogram, savgol_filter

    if sampling_hz <= 0:
        return None
    
    if len(signal) < 32:
        return None
    
    # Ultra-aggressive preprocessing for extremely weak signals
    signal_processed = signal.copy()
    
    # 1. Remove DC component
    signal_processed = signal_processed - np.mean(signal_processed)
    
    # 2. Apply Savitzky-Golay smoothing to reduce noise while preserving features
    if len(signal_processed) > 21:
        try:
            window_length = min(21, len(signal_processed) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 5:
                signal_processed = savgol_filter(signal_processed, window_length, 3)
        except Exception:
            pass
    
    # 3. Apply high-pass filter to remove very low frequency drift
    if len(signal_processed) > 100:
        try:
            from scipy.signal import butter, filtfilt
            nyquist = sampling_hz / 2
            # More aggressive high-pass filtering
            cutoff = max(0.0001, 0.0001 * nyquist)  # Even lower cutoff
            if cutoff < nyquist * 0.99:
                b, a = butter(6, cutoff, btype='high', fs=sampling_hz)
                signal_processed = filtfilt(b, a, signal_processed)
        except Exception:
            pass
    
    # 4. Ultra-aggressive signal amplification
    signal_std = np.std(signal_processed)
    if signal_std > 0:
        # Much more aggressive amplification for extremely weak signals
        amplification_factor = min(1000, max(10, 0.01 / signal_std))  # 10-1000x amplification
        # Check for potential overflow before multiplication
        if np.max(np.abs(signal_processed)) * amplification_factor < 1e10:
            signal_processed = signal_processed * amplification_factor
        else:
            # Use safer amplification to avoid overflow
            signal_processed = signal_processed * min(amplification_factor, 100)
    
    # 5. Apply additional enhancement techniques
    # Remove linear and quadratic trends
    from scipy.signal import detrend
    signal_processed = detrend(signal_processed, type='linear')
    
    # Apply median filtering to remove outliers
    if len(signal_processed) > 5:
        from scipy.signal import medfilt
        signal_processed = medfilt(signal_processed, kernel_size=min(5, len(signal_processed)))
    
    # Adaptive window sizing optimized for weak signals
    signal_duration = len(signal_processed) / sampling_hz
    
    # Use much longer windows for extremely weak signals
    if signal_duration > 3600:  # > 1 hour
        optimal_window = min(600, signal_duration / 10)  # Up to 10 minutes
    elif signal_duration > 360:  # > 6 minutes
        optimal_window = min(120, signal_duration / 5)   # Up to 2 minutes
    else:
        optimal_window = min(30, signal_duration / 3)     # Up to 30 seconds
    
    # Use the larger of user-specified or optimal window
    window_seconds = max(window_seconds, optimal_window)
    
    nperseg = max(128, int(window_seconds * sampling_hz))  # Minimum 128 samples
    nperseg = min(nperseg, len(signal_processed) // 2)
    noverlap = int(overlap_ratio * nperseg)
    
    try:
        # Use multiple methods and combine results
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
                window=window,
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
                window=window,
            )
        
        # Ultra-aggressive post-processing for flat spectrograms
        if sxx.size > 0:
            # 1. Remove noise floor more aggressively
            noise_floor = np.percentile(sxx, 1)  # Bottom 1% instead of 5%
            sxx = np.maximum(sxx, noise_floor)
            
            # 2. Apply contrast stretching
            sxx_min, sxx_max = np.percentile(sxx, [2, 98])  # Use 2nd and 98th percentiles
            if sxx_max > sxx_min:
                sxx = (sxx - sxx_min) / (sxx_max - sxx_min)
                sxx = np.clip(sxx, 0, 1)
            
            # 3. Apply gamma correction for better contrast
            gamma = 0.5  # Gamma < 1 brightens low values
            sxx = np.power(sxx, gamma)
            
            # 4. Apply log scaling with offset to avoid log(0)
            if mode == "psd":
                sxx = 10 * np.log10(sxx + 1e-15)  # Smaller offset
            else:
                sxx = np.log10(sxx + 1e-15)
            
            # 5. Apply additional contrast enhancement
            sxx_mean = np.mean(sxx)
            sxx_std = np.std(sxx)
            if sxx_std > 0:
                # Z-score normalization for better contrast
                sxx = (sxx - sxx_mean) / sxx_std
                # Clip extreme values
                sxx = np.clip(sxx, -3, 3)
        
        return f, t, sxx
    except Exception as e:
        print(f"Ultra-enhanced spectrogram computation failed: {e}")
        return None


def detect_anomalies_in_spectrogram(sxx: np.ndarray, f: np.ndarray, t: np.ndarray, threshold: float = 2.0):
    """
    Detect anomalies in spectrogram using statistical methods
    Returns coordinates of anomalous frequency-time points
    """
    if sxx.size == 0:
        return [], []
    
    # Calculate z-score for each frequency bin across time
    sxx_mean = np.mean(sxx, axis=1, keepdims=True)
    sxx_std = np.std(sxx, axis=1, keepdims=True) + 1e-12
    z_scores = (sxx - sxx_mean) / sxx_std
    
    # Find anomalies (z-score > threshold)
    anomaly_mask = np.abs(z_scores) > threshold
    
    # Get coordinates of anomalies
    anomaly_freqs = []
    anomaly_times = []
    for i in range(sxx.shape[0]):
        for j in range(sxx.shape[1]):
            if anomaly_mask[i, j]:
                anomaly_freqs.append(f[i])
                anomaly_times.append(t[j])
    
    return anomaly_freqs, anomaly_times


def compute_correlation_analysis(df: pd.DataFrame, sensor1: str, sensor2: str, field1: str, field2: str) -> dict:
    """
    Compute correlation analysis between two sensors for a specific field
    Returns correlation statistics and processed data
    """
    # Filter data for both sensors
    sensor1_data = df[df["sensor_id"] == sensor1].sort_values("timestamp")
    sensor2_data = df[df["sensor_id"] == sensor2].sort_values("timestamp")
    
    if sensor1_data.empty or sensor2_data.empty:
        return {"error": "One or both sensors have no data"}
    
    # Get field values per sensor
    if field1 == "resultant":
        sensor1_values = compute_resultant(sensor1_data).values
    else:
        sensor1_values = pd.to_numeric(sensor1_data[field1], errors="coerce").dropna().values

    if field2 == "resultant":
        sensor2_values = compute_resultant(sensor2_data).values
    else:
        sensor2_values = pd.to_numeric(sensor2_data[field2], errors="coerce").dropna().values
    
    if len(sensor1_values) == 0 or len(sensor2_values) == 0:
        return {"error": f"No valid data for one or both sensors (fields: {field1}, {field2})"}
    
    # Align data by timestamp for proper correlation
    # Create time-aligned series and handle duplicate timestamps
    sensor1_series = pd.Series(sensor1_values, index=sensor1_data["timestamp"])
    sensor2_series = pd.Series(sensor2_values, index=sensor2_data["timestamp"])
    
    # Handle duplicate timestamps by taking the mean
    if not sensor1_series.index.is_unique:
        sensor1_series = sensor1_series.groupby(level=0).mean()
    if not sensor2_series.index.is_unique:
        sensor2_series = sensor2_series.groupby(level=0).mean()
    
    # Find common time range
    common_start = max(sensor1_series.index.min(), sensor2_series.index.min())
    common_end = min(sensor1_series.index.max(), sensor2_series.index.max())
    
    if common_start >= common_end:
        return {"error": "No overlapping time range between sensors"}
    
    # Resample to common time grid for alignment
    time_step = pd.Timedelta(seconds=1)  # 1 second resolution
    time_index = pd.date_range(start=common_start, end=common_end, freq=time_step)
    
    sensor1_aligned = sensor1_series.reindex(time_index).interpolate(method="time").dropna()
    sensor2_aligned = sensor2_series.reindex(time_index).interpolate(method="time").dropna()
    
    # Find common indices where both sensors have data
    common_idx = sensor1_aligned.index.intersection(sensor2_aligned.index)
    
    if len(common_idx) < 10:
        return {"error": "Insufficient overlapping data points for correlation analysis"}
    
    # Get aligned values
    x_values = sensor1_aligned.loc[common_idx].values
    y_values = sensor2_aligned.loc[common_idx].values
    
    # Compute correlations
    try:
        pearson_corr, pearson_p = pearsonr(x_values, y_values)
        spearman_corr, spearman_p = spearmanr(x_values, y_values)
    except Exception as e:
        return {"error": f"Correlation computation failed: {str(e)}"}
    
    # Compute additional statistics
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    x_std = np.std(x_values)
    y_std = np.std(y_values)
    
    # Compute R-squared
    r_squared = pearson_corr ** 2
    
    return {
        "sensor1": sensor1,
        "sensor2": sensor2,
        "field1": field1,
        "field2": field2,
        "n_points": len(common_idx),
        "time_range": (common_start, common_end),
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "r_squared": r_squared,
        "sensor1_mean": x_mean,
        "sensor2_mean": y_mean,
        "sensor1_std": x_std,
        "sensor2_std": y_std,
        "x_values": x_values,
        "y_values": y_values,
        "timestamps": common_idx
    }


def plot_correlation_analysis(corr_data: dict):
    """
    Create comprehensive correlation analysis plots with multiple visualization methods
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if "error" in corr_data:
        st.error(f"Correlation Analysis Error: {corr_data['error']}")
        return
    
    st.subheader(
        f"🔗 Correlation Analysis: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]"
    )
    
    # Display correlation statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pearson Correlation", f"{corr_data['pearson_correlation']:.4f}")
    with col2:
        st.metric("Spearman Correlation", f"{corr_data['spearman_correlation']:.4f}")
    with col3:
        st.metric("R²", f"{corr_data['r_squared']:.4f}")
    with col4:
        st.metric("Data Points", f"{corr_data['n_points']}")
    
    # P-values
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pearson P-value", f"{corr_data['pearson_p_value']:.2e}")
    with col2:
        st.metric("Spearman P-value", f"{corr_data['spearman_p_value']:.2e}")
    
    # Interpretation
    pearson_strength = "strong" if abs(corr_data['pearson_correlation']) > 0.7 else "moderate" if abs(corr_data['pearson_correlation']) > 0.3 else "weak"
    correlation_direction = "positive" if corr_data['pearson_correlation'] > 0 else "negative"
    
    st.info(f"📊 **Interpretation**: {pearson_strength.capitalize()} {correlation_direction} linear correlation (r = {corr_data['pearson_correlation']:.3f})")
    
    # Visualization method selection
    st.subheader("📊 Visualization Methods")
    viz_method = st.selectbox(
        "Choose visualization method:",
        [
            "Comprehensive Dashboard", 
            "Simple X-Y Scatter", 
            "Time Series Overlay", 
            "Density Heatmap", 
            "3D Scatter Plot",
            "Seaborn Scatterplot",
            "Seaborn Heatmap",
            "Seaborn Correlogram",
            "Seaborn Bubble Plot",
            "Seaborn Connected Scatter"
        ],
        key="corr_viz_method"
    )
    
    x_vals = corr_data['x_values']
    y_vals = corr_data['y_values']
    timestamps = corr_data['timestamps']
    
    if viz_method == "Comprehensive Dashboard":
        # Create comprehensive subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Scatter Plot: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]",
                "Time Series Comparison",
                "Residuals Plot",
                "Correlation Matrix"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Scatter plot with regression line
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers',
                name='Data Points',
                marker=dict(color='blue', opacity=0.6, size=6),
                hovertemplate=f"{corr_data['sensor1']}: %{{x:.2f}}<br>{corr_data['sensor2']}: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add regression line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = p(x_line)
        
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                name=f'Regression Line (r={corr_data["pearson_correlation"]:.3f})',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # 2. Time series comparison
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=x_vals,
                mode='lines',
                name=corr_data['sensor1'],
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=y_vals,
                mode='lines',
                name=corr_data['sensor2'],
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # 3. Residuals plot
        y_pred = p(x_vals)
        residuals = y_vals - y_pred
        
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', opacity=0.6, size=5)
            ),
            row=2, col=1
        )
        
        # Add zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Correlation matrix heatmap
        corr_matrix = np.array([[1.0, corr_data['pearson_correlation']],
                               [corr_data['pearson_correlation'], 1.0]])
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=[corr_data['sensor1'], corr_data['sensor2']],
                y=[corr_data['sensor1'], corr_data['sensor2']],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=True
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Comprehensive Correlation Analysis: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text=f"{corr_data['sensor1']} ({corr_data['field1']})", row=1, col=1)
        fig.update_yaxes(title_text=f"{corr_data['sensor2']} ({corr_data['field2']})", row=1, col=1)
        
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text=f"{corr_data['field1']} / {corr_data['field2']} (nT)", row=1, col=2)
        
        fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Simple X-Y Scatter":
        # Simple scatter plot with enhanced features
        fig = go.Figure()
        
        # Add scatter points with color gradient based on time
        colors = np.arange(len(x_vals))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(
                color=colors,
                colorscale='Viridis',
                size=8,
                opacity=0.7,
                colorbar=dict(title="Time Index")
            ),
            name='Data Points',
            hovertemplate=f"{corr_data['sensor1']}: %{{x:.2f}}<br>{corr_data['sensor2']}: %{{y:.2f}}<br>Index: %{{marker.color}}<extra></extra>"
        ))
        
        # Add regression line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = p(x_line)
        
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            name=f'Regression Line (r={corr_data["pearson_correlation"]:.3f})',
            line=dict(color='red', width=3)
        ))
        
        # Add confidence interval
        if len(x_vals) > 10:
            # Calculate confidence interval
            from scipy import stats
            n = len(x_vals)
            x_mean = np.mean(x_vals)
            t_val = stats.t.ppf(0.975, n-2)  # 95% confidence
            s_err = np.sqrt(np.sum((y_vals - p(x_vals))**2) / (n-2))
            conf_int = t_val * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_vals - x_mean)**2))
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line + conf_int,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line - conf_int,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"Scatter Plot: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]",
            xaxis_title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]",
            yaxis_title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Time Series Overlay":
        # Time series comparison with dual y-axis
        fig = go.Figure()
        
        # Add first sensor
        fig.add_trace(go.Scatter(
            x=timestamps, y=x_vals,
            mode='lines',
            name=corr_data['sensor1'],
            line=dict(color='blue', width=2),
            yaxis='y'
        ))
        
        # Add second sensor on secondary y-axis
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_vals,
            mode='lines',
            name=corr_data['sensor2'],
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        # Add correlation info as text
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=f"Correlation: r = {corr_data['pearson_correlation']:.3f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"Time Series Comparison: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]",
            xaxis_title="Time",
            yaxis=dict(title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]", side='left'),
            yaxis2=dict(title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]", side='right', overlaying='y'),
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Density Heatmap":
        # 2D density plot
        fig = go.Figure()
        
        # Create 2D histogram
        fig.add_trace(go.Histogram2d(
            x=x_vals, y=y_vals,
            colorscale='Blues',
            nbinsx=30,
            nbinsy=30,
            showscale=True,
            colorbar=dict(title="Density")
        ))
        
        # Add regression line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = p(x_line)
        
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            name=f'Regression Line (r={corr_data["pearson_correlation"]:.3f})',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title=f"Density Heatmap: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]",
            xaxis_title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]",
            yaxis_title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "3D Scatter Plot":
        # 3D scatter plot with time as z-axis
        fig = go.Figure()
        
        # Convert timestamps to numeric for 3D plotting
        if isinstance(timestamps, pd.DatetimeIndex):
            time_numeric = (timestamps - timestamps.min()).total_seconds().values
        else:
            time_numeric = (timestamps - timestamps.min()).dt.total_seconds().values
        
        # Ensure numpy arrays
        x_vals_array = np.array(x_vals)
        y_vals_array = np.array(y_vals)
        time_numeric_array = np.array(time_numeric)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals_array, 
            y=y_vals_array, 
            z=time_numeric_array,
            mode='markers',
            marker=dict(
                size=4,
                color=time_numeric_array,
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Time (seconds)")
            ),
            name='Data Points',
            hovertemplate=f"{corr_data['sensor1']}: %{{x:.2f}}<br>{corr_data['sensor2']}: %{{y:.2f}}<br>Time: %{{z:.1f}}s<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"3D Scatter: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}] over Time",
            scene=dict(
                xaxis_title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]",
                yaxis_title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]",
                zaxis_title="Time (seconds)"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Seaborn Scatterplot":
        # Seaborn scatterplot with regression line
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create DataFrame for seaborn
        plot_df = pd.DataFrame({
            corr_data['sensor1']: x_vals,
            corr_data['sensor2']: y_vals
        })
        
        # Scatter plot with regression line
        sns.scatterplot(
            data=plot_df,
            x=corr_data['sensor1'],
            y=corr_data['sensor2'],
            alpha=0.6,
            s=50,
            ax=ax
        )
        
        # Add regression line
        sns.regplot(
            data=plot_df,
            x=corr_data['sensor1'],
            y=corr_data['sensor2'],
            scatter=False,
            color='red',
            line_kws={'linewidth': 2},
            ax=ax
        )
        
        ax.set_xlabel(f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]")
        ax.set_ylabel(f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]")
        ax.set_title(f"Scatter Plot: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]\n(r = {corr_data['pearson_correlation']:.3f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_method == "Seaborn Heatmap":
        # Create correlation matrix heatmap
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create correlation matrix
        corr_matrix = np.array([
            [1.0, corr_data['pearson_correlation']],
            [corr_data['pearson_correlation'], 1.0]
        ])
        
        # Create labels
        labels = [corr_data['sensor1'], corr_data['sensor2']]
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f"Correlation Heatmap: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_method == "Seaborn Correlogram":
        # Create correlogram (pair plot style) with multiple correlation metrics
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Create DataFrame with all relevant data
        plot_df = pd.DataFrame({
            f"{corr_data['sensor1']} ({corr_data['field1']})": x_vals,
            f"{corr_data['sensor2']} ({corr_data['field2']})": y_vals
        })
        
        # 1. Scatter plot with regression
        sns.scatterplot(
            data=plot_df,
            x=plot_df.columns[0],
            y=plot_df.columns[1],
            alpha=0.6,
            ax=axes[0, 0]
        )
        sns.regplot(
            data=plot_df,
            x=plot_df.columns[0],
            y=plot_df.columns[1],
            scatter=False,
            color='red',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title(f"Scatter Plot\n(r = {corr_data['pearson_correlation']:.3f})")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of sensor 1
        sns.histplot(
            data=plot_df,
            x=plot_df.columns[0],
            kde=True,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title(f"Distribution: {corr_data['sensor1']}")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of sensor 2
        sns.histplot(
            data=plot_df,
            x=plot_df.columns[1],
            kde=True,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title(f"Distribution: {corr_data['sensor2']}")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Correlation metrics text
        axes[1, 1].axis('off')
        stats_text = f"""
        Correlation Statistics
        
        Pearson Correlation: {corr_data['pearson_correlation']:.4f}
        P-value: {corr_data['pearson_p_value']:.2e}
        
        Spearman Correlation: {corr_data['spearman_correlation']:.4f}
        P-value: {corr_data['spearman_p_value']:.2e}
        
        R²: {corr_data['r_squared']:.4f}
        
        Data Points: {corr_data['n_points']}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f"Correlogram: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]", 
                     fontsize=14, y=0.995)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_method == "Seaborn Bubble Plot":
        # Bubble plot with time progression
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert timestamps to numeric for time progression
        if isinstance(timestamps, pd.DatetimeIndex):
            time_numeric = (timestamps - timestamps.min()).total_seconds().values
        else:
            time_numeric = (timestamps - timestamps.min()).dt.total_seconds().values
        
        # Create DataFrame with time information
        sensor1_col = f"{corr_data['sensor1']}_x"
        sensor2_col = f"{corr_data['sensor2']}_y"
        plot_df = pd.DataFrame({
            sensor1_col: np.array(x_vals),
            sensor2_col: np.array(y_vals),
            'time': time_numeric
        })
        
        # Normalize time for bubble size
        time_min = plot_df['time'].min()
        time_max = plot_df['time'].max()
        bubble_sizes = (plot_df['time'] - time_min) / (time_max - time_min + 1e-10)
        bubble_sizes = 50 + bubble_sizes * 200  # Scale between 50 and 250
        
        # Create scatter plot with varying bubble sizes
        scatter = ax.scatter(
            plot_df[sensor1_col],
            plot_df[sensor2_col],
            s=bubble_sizes,
            c=plot_df['time'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Add regression line
        x_vals_array = np.array(x_vals)
        y_vals_array = np.array(y_vals)
        z = np.polyfit(x_vals_array, y_vals_array, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals_array.min(), x_vals_array.max(), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression (r={corr_data["pearson_correlation"]:.3f})')
        
        ax.set_xlabel(f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]")
        ax.set_ylabel(f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]")
        ax.set_title(f"Bubble Plot: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]\n(Bubble size and color represent time progression)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (seconds from start)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_method == "Seaborn Connected Scatter":
        # Connected scatter plot showing time progression
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert timestamps to numeric for time progression
        if isinstance(timestamps, pd.DatetimeIndex):
            time_numeric = (timestamps - timestamps.min()).total_seconds().values
        else:
            time_numeric = (timestamps - timestamps.min()).dt.total_seconds().values
        
        # Create DataFrame with proper column names
        sensor1_col = f"{corr_data['sensor1']}_x"
        sensor2_col = f"{corr_data['sensor2']}_y"
        plot_df = pd.DataFrame({
            sensor1_col: np.array(x_vals),
            sensor2_col: np.array(y_vals),
            'time': timestamps,
            'time_numeric': time_numeric
        })
        
        # Sort by time
        plot_df = plot_df.sort_values('time')
        
        # Plot connected scatter
        ax.plot(
            plot_df[sensor1_col],
            plot_df[sensor2_col],
            'o-',
            alpha=0.6,
            markersize=4,
            linewidth=1,
            color='blue',
            label='Time progression'
        )
        
        # Color code by time (optional gradient)
        scatter = ax.scatter(
            plot_df[sensor1_col],
            plot_df[sensor2_col],
            c=plot_df['time_numeric'],
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
            zorder=5
        )
        
        # Add regression line
        x_vals_array = np.array(x_vals)
        y_vals_array = np.array(y_vals)
        z = np.polyfit(x_vals_array, y_vals_array, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals_array.min(), x_vals_array.max(), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Regression (r={corr_data["pearson_correlation"]:.3f})')
        
        ax.set_xlabel(f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]")
        ax.set_ylabel(f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]")
        ax.set_title(f"Connected Scatter: {corr_data['sensor1']} [{corr_data['field1']}] vs {corr_data['sensor2']} [{corr_data['field2']}]\n(Connected points show temporal sequence)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (seconds from start)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Additional statistics
    with st.expander("📈 Detailed Statistics"):
        st.write("**Data Summary:**")
        st.write(f"- Time range: {corr_data['time_range'][0]} to {corr_data['time_range'][1]}")
        st.write(f"- Number of data points: {corr_data['n_points']}")
        st.write(f"- {corr_data['sensor1']} mean: {corr_data['sensor1_mean']:.2f} nT, std: {corr_data['sensor1_std']:.2f} nT")
        st.write(f"- {corr_data['sensor2']} mean: {corr_data['sensor2_mean']:.2f} nT, std: {corr_data['sensor2_std']:.2f} nT")
        
        st.write("**Correlation Interpretation:**")
        if abs(corr_data['pearson_correlation']) > 0.8:
            st.write("🔴 **Very Strong Correlation**: Sensors show very similar behavior")
        elif abs(corr_data['pearson_correlation']) > 0.6:
            st.write("🟠 **Strong Correlation**: Sensors show strong relationship")
        elif abs(corr_data['pearson_correlation']) > 0.4:
            st.write("🟡 **Moderate Correlation**: Sensors show moderate relationship")
        elif abs(corr_data['pearson_correlation']) > 0.2:
            st.write("🟢 **Weak Correlation**: Sensors show weak relationship")
        else:
            st.write("⚪ **Very Weak/No Correlation**: Sensors appear independent")
        
        if corr_data['pearson_p_value'] < 0.001:
            st.write("✅ **Highly Significant** (p < 0.001)")
        elif corr_data['pearson_p_value'] < 0.01:
            st.write("✅ **Very Significant** (p < 0.01)")
        elif corr_data['pearson_p_value'] < 0.05:
            st.write("✅ **Significant** (p < 0.05)")
        else:
            st.write("❌ **Not Significant** (p ≥ 0.05)")


def correlation_study_interface(df: pd.DataFrame):
    """
    Create the correlation study interface
    """
    st.subheader("🔗 Correlation Study")
    st.markdown("Analyze the relationship between different sensors and magnetic field components.")
    
    if df.empty:
        st.warning("No data available for correlation analysis.")
        return
    
    # Get available sensors
    available_sensors = sorted(df["sensor_id"].dropna().unique().tolist())
    
    if len(available_sensors) < 2:
        st.warning("At least 2 sensors are required for correlation analysis.")
        return
    
    # Sensor selection
    col1, col2 = st.columns(2)
    with col1:
        sensor1 = st.selectbox(
            "Select First Sensor",
            options=available_sensors,
            key="corr_sensor1"
        )
    with col2:
        # Filter out sensor1 from options
        sensor2_options = [s for s in available_sensors if s != sensor1]
        sensor2 = st.selectbox(
            "Select Second Sensor",
            options=sensor2_options,
            key="corr_sensor2"
        )
    
    # Field selection (per sensor)
    field_options = ["b_x", "b_y", "b_z", "resultant"]
    field_labels = {
        "b_x": "Bx (X-component)",
        "b_y": "By (Y-component)", 
        "b_z": "Bz (Z-component)",
        "resultant": "Resultant Magnitude"
    }

    st.markdown("**Field Selection for Each Sensor**")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        selected_field1 = st.selectbox(
            f"Field for {sensor1}",
            options=field_options,
            format_func=lambda x: field_labels[x],
            key="corr_field_sensor1"
        )
    with fcol2:
        selected_field2 = st.selectbox(
            f"Field for {sensor2}",
            options=field_options,
            format_func=lambda x: field_labels[x],
            key="corr_field_sensor2"
        )
    
    # Time range selection for correlation
    if not df.empty and df["timestamp"].notna().any():
        tmin = df["timestamp"].min()
        tmax = df["timestamp"].max()
    else:
        tmin = pd.Timestamp.now() - pd.Timedelta(hours=1)
        tmax = pd.Timestamp.now()
    
    st.markdown("**Time Range for Correlation Analysis**")
    time_range_corr = st.slider(
        "Select time range",
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
        value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
        step=pd.Timedelta(seconds=1),
        format="YYYY-MM-DD HH:mm:ss",
        key="corr_time_range"
    )
    
    # Filter data for correlation analysis
    df_filtered = df.copy()
    t0, t1 = pd.Timestamp(time_range_corr[0]), pd.Timestamp(time_range_corr[1])
    if "timestamp" in df_filtered.columns and pd.notna(t0) and pd.notna(t1):
        mask = (df_filtered["timestamp"] >= t0) & (df_filtered["timestamp"] <= t1)
        df_filtered = df_filtered.loc[mask]
    
    # Create a key to identify the current analysis parameters
    analysis_key = f"{sensor1}_{sensor2}_{selected_field1}_{selected_field2}_{t0}_{t1}"
    
    # Check if parameters have changed - if so, clear stored correlation data
    if 'corr_analysis_key' not in st.session_state or st.session_state.corr_analysis_key != analysis_key:
        if 'corr_analysis_key' in st.session_state:
            # Parameters changed, clear old data
            if 'corr_data' in st.session_state:
                del st.session_state.corr_data
    
    # Run correlation analysis
    if st.button("🔍 Run Correlation Analysis", type="primary"):
        with st.spinner("Computing correlation analysis..."):
            corr_data = compute_correlation_analysis(df_filtered, sensor1, sensor2, selected_field1, selected_field2)
            # Store correlation data in session state
            st.session_state.corr_data = corr_data
            st.session_state.corr_analysis_key = analysis_key
    
    # Display stored correlation data if available (allows dropdown to work)
    if 'corr_data' in st.session_state and st.session_state.corr_analysis_key == analysis_key:
        plot_correlation_analysis(st.session_state.corr_data)
    
    # Quick correlation matrix for all sensors
    if st.checkbox("Show Correlation Matrix for All Sensors", value=False):
        st.subheader("📊 Correlation Matrix - All Sensors")
        
        # Create correlation matrix for all available fields
        correlation_data = {}
        
        for field in field_options:
            field_correlations = {}
            
            for i, sensor1 in enumerate(available_sensors):
                for j, sensor2 in enumerate(available_sensors):
                    if i < j:  # Only compute upper triangle
                        corr_result = compute_correlation_analysis(df_filtered, sensor1, sensor2, field, field)
                        if "error" not in corr_result:
                            field_correlations[f"{sensor1} vs {sensor2}"] = corr_result['pearson_correlation']
            
            if field_correlations:
                correlation_data[field_labels[field]] = field_correlations
        
        if correlation_data:
            # Display correlation summary
            for field_name, correlations in correlation_data.items():
                st.write(f"**{field_name}:**")
                for pair, corr in correlations.items():
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                    direction = "positive" if corr > 0 else "negative"
                    st.write(f"- {pair}: {corr:.3f} ({strength} {direction})")
                st.write("")
            
            # Add visual correlation matrix
            st.subheader("📈 Visual Correlation Matrix")
            viz_type = st.selectbox(
                "Choose visualization type:",
                ["Heatmap Matrix", "Network Graph", "Bar Chart Comparison"],
                key="matrix_viz_type"
            )
            
            if viz_type == "Heatmap Matrix":
                # Create heatmap for each field
                for field_name, correlations in correlation_data.items():
                    if correlations:
                        # Create matrix data
                        sensor_pairs = list(correlations.keys())
                        corr_values = list(correlations.values())
                        
                        # Create a simple matrix visualization
                        fig = go.Figure(data=go.Heatmap(
                            z=[corr_values],
                            x=sensor_pairs,
                            y=[field_name],
                            colorscale='RdBu',
                            zmid=0,
                            text=[f"{v:.3f}" for v in corr_values],
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            showscale=True
                        ))
                        
                        fig.update_layout(
                            title=f"Correlation Heatmap: {field_name}",
                            height=200,
                            xaxis_title="Sensor Pairs",
                            yaxis_title="Field"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Network Graph":
                # Create network-style visualization
                
                for field_name, correlations in correlation_data.items():
                    if correlations:
                        # Extract sensor names and create nodes
                        sensor_names = set()
                        for pair in correlations.keys():
                            sensor1, sensor2 = pair.split(" vs ")
                            sensor_names.add(sensor1)
                            sensor_names.add(sensor2)
                        
                        sensor_list = list(sensor_names)
                        n_sensors = len(sensor_list)
                        
                        # Create positions for nodes
                        angles = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
                        x_pos = np.cos(angles)
                        y_pos = np.sin(angles)
                        
                        # Create node positions
                        node_x = x_pos.tolist()
                        node_y = y_pos.tolist()
                        
                        # Create edges
                        edge_x = []
                        edge_y = []
                        edge_info = []
                        
                        for pair, corr in correlations.items():
                            sensor1, sensor2 = pair.split(" vs ")
                            idx1 = sensor_list.index(sensor1)
                            idx2 = sensor_list.index(sensor2)
                            
                            edge_x.extend([x_pos[idx1], x_pos[idx2], None])
                            edge_y.extend([y_pos[idx1], y_pos[idx2], None])
                            edge_info.append(corr)
                        
                        # Create the network plot
                        fig = go.Figure()
                        
                        # Add edges
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            mode='lines',
                            line=dict(width=2, color='lightgray'),
                            hoverinfo='none',
                            showlegend=False
                        ))
                        
                        # Add nodes
                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            marker=dict(size=20, color='lightblue'),
                            text=sensor_list,
                            textposition="middle center",
                            showlegend=False,
                            hovertemplate="%{text}<extra></extra>"
                        ))
                        
                        # Add correlation strength as edge colors
                        for i, (pair, corr) in enumerate(correlations.items()):
                            sensor1, sensor2 = pair.split(" vs ")
                            idx1 = sensor_list.index(sensor1)
                            idx2 = sensor_list.index(sensor2)
                            
                            # Color based on correlation strength
                            color = 'red' if abs(corr) > 0.7 else 'orange' if abs(corr) > 0.3 else 'gray'
                            width = 5 if abs(corr) > 0.7 else 3 if abs(corr) > 0.3 else 1
                            
                            fig.add_trace(go.Scatter(
                                x=[x_pos[idx1], x_pos[idx2]], 
                                y=[y_pos[idx1], y_pos[idx2]],
                                mode='lines',
                                line=dict(width=width, color=color),
                                showlegend=False,
                                hovertemplate=f"{pair}<br>Correlation: {corr:.3f}<extra></extra>"
                            ))
                        
                        fig.update_layout(
                            title=f"Network Graph: {field_name}",
                            showlegend=False,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Bar Chart Comparison":
                # Create bar chart comparison
                for field_name, correlations in correlation_data.items():
                    if correlations:
                        pairs = list(correlations.keys())
                        corr_values = list(correlations.values())
                        
                        # Color bars based on correlation strength
                        colors = ['red' if abs(v) > 0.7 else 'orange' if abs(v) > 0.3 else 'lightblue' for v in corr_values]
                        
                        fig = go.Figure(data=go.Bar(
                            x=pairs,
                            y=corr_values,
                            marker_color=colors,
                            text=[f"{v:.3f}" for v in corr_values],
                            textposition='auto',
                        ))
                        
                        fig.update_layout(
                            title=f"Correlation Comparison: {field_name}",
                            xaxis_title="Sensor Pairs",
                            yaxis_title="Correlation Coefficient",
                            height=400,
                            xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to compute correlations for the selected time range.")


def plot_spectrogram(df: pd.DataFrame, series: str, window_seconds: float, overlap_ratio: float):
    import plotly.graph_objects as go

    if df.empty:
        st.info("No data for spectrogram.")
        return
    work = df.copy()
    if series == "resultant":
        work["resultant"] = compute_resultant(work)
    y_col = series
    
    # Enhanced signal quality diagnostics
    st.subheader("🔍 Enhanced Signal Quality Analysis")
    for sensor_id in work["sensor_id"].dropna().unique():
        sensor_data = work[work["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
            
        if series == "resultant":
            signal_values = compute_resultant(sensor_data).values
        else:
            signal_values = pd.to_numeric(sensor_data[series], errors="coerce").dropna().values
        
        if len(signal_values) < 10:
            st.warning(f"Sensor {sensor_id}: Insufficient data points ({len(signal_values)})")
            continue
            
        signal_mean = np.mean(signal_values)
        signal_std = np.std(signal_values)
        signal_range = np.max(signal_values) - np.min(signal_values)
        
        # Enhanced quality checks
        if signal_std < 1e-6:
            st.error(f"⚠️ Sensor {sensor_id}: Signal appears constant (std={signal_std:.2e})")
            st.info("💡 Try: Different time range, enable signal enhancement, or check data quality")
            continue
        elif signal_std < signal_mean * 0.01:
            st.warning(f"⚠️ Sensor {sensor_id}: Very low signal variation (std={signal_std:.2f}, mean={signal_mean:.2f})")
            st.info("💡 Try: Enable signal enhancement or check for constant signals")
        
        # Estimate sampling rate
        fs = estimate_sampling_rate(sensor_data['timestamp'])
        if fs <= 0:
            st.warning(f"Cannot estimate sampling rate for sensor {sensor_id}")
            continue
        elif fs < 0.1:
            st.warning(f"⚠️ Sensor {sensor_id}: Very low sampling rate ({fs:.3f} Hz)")
        elif fs > 1000:
            st.info(f"ℹ️ Sensor {sensor_id}: High sampling rate ({fs:.1f} Hz) - consider downsampling")
        
        # Signal-to-noise ratio estimation
        signal_power = np.mean(signal_values**2)
        noise_estimate = np.var(np.diff(signal_values))
        snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-12))
        
        if snr_db < 10:
            st.warning(f"⚠️ Sensor {sensor_id}: Low SNR ({snr_db:.1f} dB)")
        else:
            st.success(f"✅ Sensor {sensor_id}: Good SNR ({snr_db:.1f} dB)")
        
        # Display quality metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Sensor {sensor_id} - Mean", f"{signal_mean:.2f} nT")
        with col2:
            st.metric(f"Sensor {sensor_id} - Std", f"{signal_std:.2f} nT")
        with col3:
            st.metric(f"Sensor {sensor_id} - Range", f"{signal_range:.2f} nT")
        with col4:
            st.metric(f"Sensor {sensor_id} - Sampling Rate", f"{fs:.2f} Hz")
    
    # Enhanced spectrogram controls
    with st.sidebar.expander("🔧 Enhanced Spectrogram Settings", expanded=True):
        st.markdown("**📊 Frequency Range Selection**")
        # Quick presets for magnetic data
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Ultra Low (0-1 μHz)"):
                st.session_state.min_freq = 0.0
                st.session_state.max_freq = 1e-6
                st.rerun()
        with col2:
            if st.button("Very Low (0-1 mHz)"):
                st.session_state.min_freq = 0.0
                st.session_state.max_freq = 0.001
                st.rerun()
        with col3:
            if st.button("Low (0-0.1 Hz)"):
                st.session_state.min_freq = 0.0
                st.session_state.max_freq = 0.1
                st.rerun()
        
        # Frequency range
        min_freq = st.number_input("Min frequency (Hz)", min_value=0.0, value=st.session_state.get('min_freq', 0.0), step=0.001)
        max_freq = st.number_input("Max frequency (Hz, 0 = auto)", min_value=0.0, value=st.session_state.get('max_freq', 0.0), step=0.001)
        
        st.markdown("**🔧 Ultra-Aggressive Signal Enhancement**")
        enable_enhancement = st.checkbox("Enable ultra-aggressive enhancement", value=True, help="For extremely weak signals like your flat spectrogram")
        enhancement_factor = st.slider("Enhancement factor", min_value=1.0, max_value=500.0, value=50.0, step=5.0, help="Use smaller values first to avoid saturation")
        
        st.markdown("**🎯 Flat Spectrogram Solutions**")
        enable_contrast_stretch = st.checkbox("Enable contrast stretching", value=True, help="Essential for flat spectrograms")
        enable_gamma_correction = st.checkbox("Enable gamma correction", value=False, help="Brightens low values")
        gamma_value = st.slider("Gamma value", min_value=0.1, max_value=2.0, value=0.8, step=0.1, help="Lower values brighten weak signals more")
        enable_zscore_norm = st.checkbox("Enable z-score normalization (global)", value=False, help="Global z-score can wash out later segments; keep off for stability")

        st.markdown("**🧮 Normalization & Scaling**")
        per_freq_norm = st.checkbox("Normalize per-frequency (row-wise)", value=True, help="Stabilizes colors across time; avoids early windows dominating")
        stable_scaling = st.checkbox("Stable percentile scaling (middle time only)", value=True, help="Compute stretch percentiles on middle 80% time bins to avoid edge transients")
        trim_edge_bins = st.slider("Trim edge time bins (columns)", min_value=0, max_value=5, value=1, step=1, help="Ignore this many columns at each edge when scaling")
        
        st.markdown("**🎛️ Spectrogram Parameters**")
        spec_mode = st.selectbox("Mode", options=["psd", "magnitude"], index=1, help="Magnitude mode often better for weak signals")
        window_fn = st.selectbox("Window function", options=["hann", "hamming", "blackman", "bartlett"], index=0)
        
        st.markdown("**🔍 Anomaly Detection**")
        enable_anomaly_detection = st.checkbox("Enable anomaly detection", value=True)
        anomaly_threshold = st.slider("Anomaly threshold (z-score)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        
        st.markdown("**📈 Display Options**")
        colormap = st.selectbox("Colormap", options=["Viridis", "Plasma", "Inferno", "Turbo", "Cividis"], index=0)
        log_freq_axis = st.checkbox("Log frequency axis", value=True, help="Recommended for magnetic field analysis")
        show_anomalies = st.checkbox("Highlight anomalies", value=True)

    # Process each sensor
    sensors = work["sensor_id"].dropna().unique().tolist()
    for sid in sensors:
        sub = work[work["sensor_id"] == sid].sort_values("timestamp")
        
        fs = estimate_sampling_rate(sub["timestamp"])
        if fs <= 0:
            st.warning(f"Cannot estimate sampling rate for sensor {sid}; skipping spectrogram.")
            continue
        
        # Resample to uniform grid
        series_values = pd.to_numeric(sub[y_col], errors="coerce")
        ts_indexed = pd.Series(series_values.values, index=sub["timestamp"]).dropna()
        if not ts_indexed.index.is_unique:
            ts_indexed = ts_indexed.groupby(level=0).mean()
        if ts_indexed.empty:
            st.warning(f"No valid samples for sensor {sid} spectrogram.")
            continue
        
        # Build uniform time grid
        start_time = ts_indexed.index.min()
        end_time = ts_indexed.index.max()
        grid_step = pd.Timedelta(seconds=max(1.0 / fs, 1e-6))
        uniform_index = pd.date_range(start=start_time, end=end_time, freq=grid_step)
        uniform_series = ts_indexed.reindex(uniform_index).interpolate(method="time").bfill().ffill()
        values = uniform_series.values
        
        if values.size < 32:  # Increased minimum
            st.warning(f"Not enough samples for sensor {sid} spectrogram.")
            continue
        
        # Enhanced preprocessing
        # Remove DC component
        values = values - np.mean(values)
        
        # Apply ultra-aggressive enhancement if enabled
        if enable_enhancement:
            # Safer enhancement: standardize first, then scale with strict caps
            values = values.astype(np.float64)
            signal_std = float(np.std(values))
            if np.isfinite(signal_std) and signal_std > 0:
                values = values / signal_std
                safe_factor = min(enhancement_factor, 100.0)
                values = np.clip(values * safe_factor, -1e6, 1e6)
                st.caption(f"🔧 Safe enhancement applied (std-normalized, ×{safe_factor:.1f})")
        
        # Apply additional preprocessing for flat spectrograms
        if enable_contrast_stretch:
            # Apply contrast stretching to the time series
            values_min, values_max = np.percentile(values, [1, 99])
            if values_max > values_min:
                values = (values - values_min) / (values_max - values_min)
                values = values * 2 - 1  # Center around 0
                st.caption("🔧 Contrast stretching applied to time series")
        
        # Apply Savitzky-Golay smoothing for better signal quality
        if len(values) > 21:
            try:
                from scipy.signal import savgol_filter
                window_length = min(21, len(values) // 10)
                if window_length % 2 == 0:
                    window_length += 1
                if window_length >= 5:
                    values = savgol_filter(values, window_length, 3)
                    st.caption("🔧 Savitzky-Golay smoothing applied")
            except Exception:
                pass
        
        # Compute enhanced spectrogram
        result = compute_spectrogram(
            values,
            fs,
            window_seconds,
            overlap_ratio,
            mode=spec_mode,
            window=window_fn,
        )
        if result is None:
            st.warning(f"Failed to compute spectrogram for sensor {sid}")
            continue
        f, t, sxx = result
        
        # Convert time bins to actual timestamps
        actual_times = [start_time + pd.Timedelta(seconds=float(t_bin)) for t_bin in t]
        
        # Apply frequency band mask
        freq_mask = np.ones_like(f, dtype=bool)
        if min_freq > 0:
            freq_mask &= f >= min_freq
        if max_freq > 0:
            freq_mask &= f <= max_freq
        elif max_freq == 0:
            # Auto limit to reasonable range for magnetic data
            if fs < 0.1:
                default_max = min(0.01, fs / 2.0)
            elif fs < 1.0:
                default_max = min(0.1, fs / 2.0)
            else:
                default_max = min(0.5, fs / 2.0)
            freq_mask &= f <= default_max
        
        if np.any(freq_mask):
            f_filtered = f[freq_mask]
            sxx_filtered = sxx[freq_mask, :]
            st.caption(f"📊 Frequency range: {f_filtered[0]:.6f} to {f_filtered[-1]:.6f} Hz ({len(f_filtered)} bins)")
            f = f_filtered
            sxx = sxx_filtered
        else:
            st.warning(f"No frequencies in range! Available: {f[0]:.6f} to {f[-1]:.6f} Hz")
            continue

        # Detect anomalies if enabled
        anomaly_freqs, anomaly_times = [], []
        if enable_anomaly_detection:
            anomaly_freqs, anomaly_times = detect_anomalies_in_spectrogram(sxx, f, t, anomaly_threshold)
            if anomaly_freqs:
                st.success(f"🔍 Found {len(anomaly_freqs)} anomalies in sensor {sid}")

        # Apply additional post-processing for flat spectrograms
        sxx_processed = sxx.copy()
        # Guard numerics
        sxx_processed = np.where(np.isfinite(sxx_processed), sxx_processed, 0.0)
        
        # Optional per-frequency normalization (row-wise)
        if per_freq_norm and sxx_processed.size > 0:
            row_mean = np.mean(sxx_processed, axis=1, keepdims=True)
            row_std = np.std(sxx_processed, axis=1, keepdims=True) + 1e-12
            sxx_processed = (sxx_processed - row_mean) / row_std
            sxx_processed = np.clip(sxx_processed, -3, 3)
            st.caption("🔧 Per-frequency normalization applied")

        # Stable percentile contrast stretching
        if enable_contrast_stretch and sxx_processed.size > 0:
            stretch_source = sxx_processed
            if stable_scaling and stretch_source.shape[1] > (2 * trim_edge_bins):
                stretch_source = stretch_source[:, trim_edge_bins:-trim_edge_bins]
            sxx_min, sxx_max = np.percentile(stretch_source, [2, 98])
            if np.isfinite(sxx_max) and np.isfinite(sxx_min) and sxx_max > sxx_min:
                sxx_processed = (sxx_processed - sxx_min) / (sxx_max - sxx_min)
                sxx_processed = np.clip(sxx_processed, 0, 1)
                st.caption("🔧 Stable contrast stretching applied")
        
        if enable_gamma_correction:
            # Apply gamma correction
            sxx_processed = np.power(sxx_processed, gamma_value)
            st.caption(f"🔧 Gamma correction applied (γ={gamma_value:.1f})")
        
        if enable_zscore_norm:
            # Apply z-score normalization
            sxx_mean = np.mean(sxx_processed)
            sxx_std = np.std(sxx_processed)
            if sxx_std > 0:
                sxx_processed = (sxx_processed - sxx_mean) / sxx_std
                sxx_processed = np.clip(sxx_processed, -3, 3)
                st.caption("🔧 Z-score normalization applied")
        
        # Create enhanced heatmap with better color mapping
        heatmap = go.Heatmap(
            x=actual_times,
            y=f,
            z=sxx_processed,
            colorscale=colormap,
            colorbar=dict(title="Enhanced Power" if spec_mode == "psd" else "Enhanced Magnitude"),
            zsmooth="best",
            zmin=np.percentile(sxx_processed, 5) if sxx_processed.size > 0 else None,
            zmax=np.percentile(sxx_processed, 95) if sxx_processed.size > 0 else None,
        )
        fig = go.Figure(data=heatmap)
        
        # Add anomaly markers if enabled and anomalies found
        if show_anomalies and anomaly_freqs:
            anomaly_x = [start_time + pd.Timedelta(seconds=float(t_anom)) for t_anom in anomaly_times]
            fig.add_trace(go.Scattergl(
                x=anomaly_x,
                y=anomaly_freqs,
                mode="markers",
                marker=dict(color="red", size=6, opacity=0.8, symbol="x"),
                name=f"Anomalies (z>{anomaly_threshold:.1f})",
                hovertemplate="Time: %{x}<br>Freq: %{y:.6f} Hz<extra>Anomaly</extra>"
            ))
        
        fig.update_layout(
            title=f"Enhanced Spectrogram ({y_col}) - {sid}",
            xaxis_title="Time",
            yaxis_title="Frequency (Hz)",
            yaxis_type='log' if (log_freq_axis and len(f) > 0 and np.min(f) > 0) else 'linear',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show enhanced spectrogram statistics
        sxx_range = np.max(sxx_processed) - np.min(sxx_processed)
        sxx_mean = np.mean(sxx_processed)
        sxx_std = np.std(sxx_processed)
        
        st.caption(f"📊 Enhanced spectrogram stats: mean={sxx_mean:.3f}, std={sxx_std:.3f}, range={sxx_range:.3f}")
        
        # Enhanced troubleshooting for flat spectrograms
        if sxx_range < 1e-6:
            st.error(f"🚨 Sensor {sid}: Spectrogram still appears flat (range={sxx_range:.2e})")
            st.markdown("""
            **🔧 Ultra-Aggressive Solutions for Flat Spectrograms:**
            
            1. **Increase enhancement factor to 500-1000x**
            2. **Enable all processing options** (contrast stretch, gamma correction, z-score norm)
            3. **Try different frequency ranges** (use presets)
            4. **Check if signal has any variation** in time series plot
            5. **Try different sensors** if available
            6. **Use longer time windows** (1-10 minutes)
            """)
        elif sxx_range < 0.1:
            st.warning(f"⚠️ Sensor {sid}: Low spectrogram variation (range={sxx_range:.3f})")
            st.info("💡 Try: Higher enhancement factor (200-500x), enable all processing options")
        else:
            st.success(f"✅ Sensor {sid}: Spectrogram shows good variation (range={sxx_range:.3f})")
        
        # Show processing summary
        processing_applied = []
        if enable_enhancement:
            processing_applied.append(f"Enhancement ({enhancement_factor:.0f}x)")
        if enable_contrast_stretch:
            processing_applied.append("Contrast stretch")
        if enable_gamma_correction:
            processing_applied.append(f"Gamma (γ={gamma_value:.1f})")
        if enable_zscore_norm:
            processing_applied.append("Z-score norm")
        
        if processing_applied:
            st.caption(f"🔧 Processing applied: {', '.join(processing_applied)}")


def plot_3d_vector_field(df: pd.DataFrame):
    """Plot 3D vector field visualization of magnetic field"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    
    # Sample data for performance (if too many points)
    max_points = 1000
    if len(df) > max_points:
        df_sample = df.sample(n=max_points).sort_values("timestamp")
        st.caption(f"📊 Showing {max_points} sampled points from {len(df)} total points for performance")
    else:
        df_sample = df.sort_values("timestamp")
    
    fig = go.Figure()
    
    # Plot vectors for each sensor
    for sensor_id in df_sample["sensor_id"].dropna().unique():
        sensor_data = df_sample[df_sample["sensor_id"] == sensor_id]
        
        # Create quiver plot (3D arrows)
        x = sensor_data["b_x"].values
        y = sensor_data["b_y"].values
        z = sensor_data["b_z"].values
        
        # Color by magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=magnitude,
                colorscale='Viridis',
                colorbar=dict(title="Magnitude (nT)"),
                opacity=0.7
            ),
            name=f"Sensor {sensor_id}",
            customdata=magnitude,
            hovertemplate="Bx: %{x:.2f} nT<br>By: %{y:.2f} nT<br>Bz: %{z:.2f} nT<br>Magnitude: %{customdata:.2f} nT<extra></extra>"
        ))
    
    fig.update_layout(
        title="3D Magnetic Field Vector Space",
        scene=dict(
            xaxis_title="Bx (nT)",
            yaxis_title="By (nT)",
            zaxis_title="Bz (nT)",
            aspectmode='cube'
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_distribution_histograms(df: pd.DataFrame):
    """Plot distribution histograms for b_x, b_y, b_z"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Bx Distribution", "By Distribution", "Bz Distribution"),
        horizontal_spacing=0.1
    )
    
    components = ["b_x", "b_y", "b_z"]
    colors = ["blue", "green", "red"]
    
    for idx, (comp, color) in enumerate(zip(components, colors), 1):
        values = pd.to_numeric(df[comp], errors="coerce").dropna()
        
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=50,
                name=comp,
                marker_color=color,
                opacity=0.7,
                showlegend=True
            ),
            row=1, col=idx
        )
        
        # Add statistics
        mean_val = np.mean(values)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Mean: {mean_val:.2f}",
            row=1, col=idx
        )
    
    fig.update_xaxes(title_text="Magnetic Field Value (nT)", row=1, col=1)
    fig.update_xaxes(title_text="Magnetic Field Value (nT)", row=1, col=2)
    fig.update_xaxes(title_text="Magnetic Field Value (nT)", row=1, col=3)
    fig.update_yaxes(title_text="Count (Number of Occurrences)", row=1, col=1)
    fig.update_yaxes(title_text="Count (Number of Occurrences)", row=1, col=2)
    fig.update_yaxes(title_text="Count (Number of Occurrences)", row=1, col=3)
    
    fig.update_layout(
        title="Magnetic Field Component Distributions",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    for idx, (comp, col) in enumerate(zip(components, [col1, col2, col3])):
        values = pd.to_numeric(df[comp], errors="coerce").dropna()
        with col:
            st.metric(f"{comp} Mean", f"{np.mean(values):.2f} nT")
            st.metric(f"{comp} Std", f"{np.std(values):.2f} nT")
            st.metric(f"{comp} Range", f"{np.min(values):.2f} to {np.max(values):.2f} nT")


def plot_box_plots(df: pd.DataFrame):
    """Plot box plots comparing distributions across sensors"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    sensors = df["sensor_id"].dropna().unique()
    if len(sensors) == 0:
        st.warning("No sensor data available.")
        return
    
    components = ["b_x", "b_y", "b_z"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Bx Comparison", "By Comparison", "Bz Comparison"),
        horizontal_spacing=0.1
    )
    
    for idx, comp in enumerate(components, 1):
        data_for_box = []
        labels_for_box = []
        
        for sensor in sensors:
            sensor_data = df[df["sensor_id"] == sensor]
            values = pd.to_numeric(sensor_data[comp], errors="coerce").dropna()
            if len(values) > 0:
                data_for_box.append(values.values)
                labels_for_box.append(sensor)
        
        if data_for_box:
            # Prepare data for box plot
            if len(data_for_box) == 1:
                # Single sensor
                box_y = data_for_box[0]
                box_x = [labels_for_box[0]] * len(data_for_box[0])
            else:
                # Multiple sensors
                box_y = np.concatenate(data_for_box)
                box_x = [label for label, data in zip(labels_for_box, data_for_box) for _ in range(len(data))]
            
            fig.add_trace(
                go.Box(
                    y=box_y,
                    x=box_x,
                    name=comp,
                    boxpoints='outliers',
                    showlegend=False
                ),
                row=1, col=idx
            )
    
    fig.update_yaxes(title_text="Value (nT)", row=1, col=1)
    fig.update_layout(
        title="Magnetic Field Component Distributions by Sensor",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_statistical_dashboard(df: pd.DataFrame):
    """Create a comprehensive statistical dashboard"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    st.subheader("📊 Statistical Summary Dashboard")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", f"{len(df):,}")
    with col2:
        st.metric("Unique Sensors", f"{df['sensor_id'].nunique()}")
    with col3:
        if not df.empty and df["timestamp"].notna().any():
            duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
            st.metric("Duration", f"{duration:.2f} hours")
    with col4:
        if not df.empty and df["timestamp"].notna().any():
            fs = estimate_sampling_rate(df["timestamp"])
            st.metric("Avg Sampling Rate", f"{fs:.3f} Hz")
    
    # Component statistics
    st.subheader("📈 Component Statistics")
    components = ["b_x", "b_y", "b_z"]
    stats_data = []
    
    for comp in components:
        values = pd.to_numeric(df[comp], errors="coerce").dropna()
        if len(values) > 0:
            stats_data.append({
                "Component": comp,
                "Mean (nT)": np.mean(values),
                "Std (nT)": np.std(values),
                "Min (nT)": np.min(values),
                "Max (nT)": np.max(values),
                "Range (nT)": np.max(values) - np.min(values),
                "Median (nT)": np.median(values)
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    # Resultant statistics
    st.subheader("🧲 Resultant Magnetic Field Statistics")
    resultant = compute_resultant(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Resultant", f"{np.mean(resultant):.2f} nT")
    with col2:
        st.metric("Std Resultant", f"{np.std(resultant):.2f} nT")
    with col3:
        st.metric("Min Resultant", f"{np.min(resultant):.2f} nT")
    with col4:
        st.metric("Max Resultant", f"{np.max(resultant):.2f} nT")
    
    # Sensor-specific statistics
    if df["sensor_id"].nunique() > 1:
        st.subheader("🔍 Per-Sensor Statistics")
        sensors = df["sensor_id"].dropna().unique()
        for sensor in sensors:
            sensor_data = df[df["sensor_id"] == sensor]
            with st.expander(f"Sensor: {sensor}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Data Points:** {len(sensor_data):,}")
                with col2:
                    if sensor_data["timestamp"].notna().any():
                        duration = (sensor_data["timestamp"].max() - sensor_data["timestamp"].min()).total_seconds() / 3600
                        st.write(f"**Duration:** {duration:.2f} hours")
                with col3:
                    resultant_sensor = compute_resultant(sensor_data)
                    st.write(f"**Mean Resultant:** {np.mean(resultant_sensor):.2f} nT")


def plot_frequency_domain_analysis(df: pd.DataFrame, series: str):
    """Plot frequency domain analysis (FFT and PSD)"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import signal
    
    sensors = df["sensor_id"].dropna().unique()
    
    for sensor_id in sensors:
        sensor_data = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
        
        if series == "resultant":
            signal_values = compute_resultant(sensor_data).values
        else:
            signal_values = pd.to_numeric(sensor_data[series], errors="coerce").dropna().values
        
        if len(signal_values) < 32:
            st.warning(f"Insufficient data for frequency analysis: {sensor_id}")
            continue
        
        # Estimate sampling rate
        fs = estimate_sampling_rate(sensor_data['timestamp'])
        if fs <= 0:
            st.warning(f"Cannot estimate sampling rate for {sensor_id}")
            continue
        
        # Remove DC component
        signal_values = signal_values - np.mean(signal_values)
        
        # Compute FFT
        n = len(signal_values)
        fft_vals = np.fft.fft(signal_values)
        fft_freq = np.fft.fftfreq(n, 1/fs)
        
        # Only positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_freq_positive = fft_freq[positive_freq_mask]
        fft_magnitude = np.abs(fft_vals[positive_freq_mask])
        fft_power = fft_magnitude ** 2
        
        # Compute PSD using Welch's method
        nperseg = min(256, len(signal_values) // 4)
        if nperseg < 32:
            nperseg = 32
        
        try:
            psd_freq, psd = signal.welch(signal_values, fs=fs, nperseg=nperseg, scaling='density')
        except Exception:
            psd_freq = fft_freq_positive
            psd = fft_power / fs
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"FFT Magnitude Spectrum - {sensor_id}",
                f"Power Spectral Density (PSD) - {sensor_id}"
            ),
            vertical_spacing=0.15
        )
        
        # FFT plot
        fig.add_trace(
            go.Scatter(
                x=fft_freq_positive,
                y=fft_magnitude,
                mode='lines',
                name='FFT Magnitude',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # PSD plot
        fig.add_trace(
            go.Scatter(
                x=psd_freq,
                y=10 * np.log10(psd + 1e-15),  # Convert to dB
                mode='lines',
                name='PSD (dB)',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1, type='log')
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1, type='log')
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_yaxes(title_text="PSD (dB)", row=2, col=1)
        
        fig.update_layout(
            title=f"Frequency Domain Analysis: {series} - {sensor_id}",
            height=700,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


def plot_multi_component_comparison(df: pd.DataFrame):
    """Plot all three components (b_x, b_y, b_z) side by side"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    sensors = df["sensor_id"].dropna().unique()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Bx Component", "By Component", "Bz Component"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    components = ["b_x", "b_y", "b_z"]
    colors = ["blue", "green", "red"]
    
    for sensor_id in sensors:
        sensor_data = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
        
        for idx, (comp, color) in enumerate(zip(components, colors), 1):
            values = pd.to_numeric(sensor_data[comp], errors="coerce")
            
            fig.add_trace(
                go.Scatter(
                    x=sensor_data["timestamp"],
                    y=values,
                    mode='lines',
                    name=f"{comp} - {sensor_id}",
                    line=dict(color=color, width=1.5),
                    showlegend=(idx == 1)  # Only show legend for first component
                ),
                row=idx, col=1
            )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Bx (nT)", row=1, col=1)
    fig.update_yaxes(title_text="By (nT)", row=2, col=1)
    fig.update_yaxes(title_text="Bz (nT)", row=3, col=1)
    
    fig.update_layout(
        title="Multi-Component Magnetic Field Comparison",
        height=900,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_component_heatmap(df: pd.DataFrame):
    """Plot heatmap showing how components vary over time"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    
    sensors = df["sensor_id"].dropna().unique()
    
    for sensor_id in sensors:
        sensor_data = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
        
        # Resample to uniform time grid for better heatmap
        sensor_data = sensor_data.copy()
        sensor_data["timestamp"] = pd.to_datetime(sensor_data["timestamp"])
        
        # Create time bins (e.g., 1 minute bins)
        time_range = (sensor_data["timestamp"].max() - sensor_data["timestamp"].min()).total_seconds()
        n_bins = min(100, max(20, int(time_range / 60)))  # ~1 minute bins, max 100
        
        sensor_data["time_bin"] = pd.cut(
            (sensor_data["timestamp"] - sensor_data["timestamp"].min()).dt.total_seconds(),
            bins=n_bins,
            labels=False
        )
        
        # Aggregate by time bin
        components = ["b_x", "b_y", "b_z"]
        heatmap_data = []
        time_labels = []
        
        for bin_idx in range(n_bins):
            bin_data = sensor_data[sensor_data["time_bin"] == bin_idx]
            if not bin_data.empty:
                time_labels.append(bin_data["timestamp"].iloc[0])
                heatmap_data.append([
                    pd.to_numeric(bin_data["b_x"], errors="coerce").mean(),
                    pd.to_numeric(bin_data["b_y"], errors="coerce").mean(),
                    pd.to_numeric(bin_data["b_z"], errors="coerce").mean()
                ])
        
        if not heatmap_data:
            continue
        
        heatmap_array = np.array(heatmap_data).T  # Shape: (3, n_bins)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_array,
            x=time_labels,
            y=components,
            colorscale='RdBu',
            colorbar=dict(title="Magnetic Field (nT)"),
            text=[[f"{val:.1f}" for val in row] for row in heatmap_array],
            texttemplate="%{text}",
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title=f"Component Heatmap Over Time - {sensor_id}",
            xaxis_title="Time",
            yaxis_title="Component",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def plot_polar_direction(df: pd.DataFrame):
    """Plot polar plot showing magnetic field direction and magnitude"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    import plotly.graph_objects as go
    
    # Sample for performance
    max_points = 500
    if len(df) > max_points:
        df_sample = df.sample(n=max_points)
        st.caption(f"📊 Showing {max_points} sampled points from {len(df)} total points")
    else:
        df_sample = df
    
    # Calculate direction angles (azimuth and elevation)
    b_x = pd.to_numeric(df_sample["b_x"], errors="coerce").values
    b_y = pd.to_numeric(df_sample["b_y"], errors="coerce").values
    b_z = pd.to_numeric(df_sample["b_z"], errors="coerce").values
    
    # Azimuth angle (in XY plane)
    azimuth = np.arctan2(b_y, b_x) * 180 / np.pi  # Convert to degrees
    
    # Magnitude
    magnitude = np.sqrt(b_x**2 + b_y**2 + b_z**2)
    
    fig = go.Figure()
    
    # Color by sensor
    sensors = df_sample["sensor_id"].dropna().unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, sensor_id in enumerate(sensors):
        sensor_mask = df_sample["sensor_id"] == sensor_id
        if not sensor_mask.any():
            continue
        
        fig.add_trace(go.Scatterpolar(
            r=magnitude[sensor_mask],
            theta=azimuth[sensor_mask],
            mode='markers',
            name=f"Sensor {sensor_id}",
            marker=dict(
                size=6,
                color=colors[idx % len(colors)],
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            hovertemplate="Magnitude: %{r:.2f} nT<br>Azimuth: %{theta:.1f}°<extra></extra>"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                title="Magnitude (nT)",
                range=[0, np.max(magnitude) * 1.1]
            ),
            angularaxis=dict(
                thetaunit="degrees",
                rotation=90,
                direction="counterclockwise"
            )
        ),
        title="Magnetic Field Direction (Polar Plot)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="MagSpectrogram - Simplified", layout="wide")
    st.title("Magnetic Sensor Data Visualizer (Simplified)")

    st.markdown(
        "Upload CSV files with magnetic field data. The resultant magnetic field is calculated as √(b_x² + b_y² + b_z²) and displayed in **nT (nanotesla)** units."
    )

    # File upload
    with st.expander("Import Data", expanded=True):
        uploaded = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        dataset_dir = os.path.join(os.getcwd(), "Dataset")
        quick_load = st.checkbox("Load all CSVs from ./Dataset", value=False)
        files_from_dir = []
        if quick_load and os.path.isdir(dataset_dir):
            for name in os.listdir(dataset_dir):
                if name.lower().endswith(".csv"):
                    files_from_dir.append(os.path.join(dataset_dir, name))

    df = read_multiple_csvs((uploaded or []) + files_from_dir)
    st.caption(f"Loaded {len(df)} rows from {len((uploaded or [])) + len(files_from_dir)} files")

    if df.empty:
        st.stop()

    # Main navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "🔗 Correlation Study", "ℹ️ About"])
    
    with tab1:
        selected_sensor, selected_series, time_range, bin_width_seconds, overlap_ratio = layout_sidebar(df)
        view_df = filter_df(df, selected_sensor, time_range)
        
        st.caption(f"🔍 After filtering: {len(view_df)} rows (from {len(df)} total)")

        # Visualization mode selector
        st.subheader("📊 Visualization Options")
        viz_mode = st.selectbox(
            "Choose visualization type:",
            [
                "Time Series & Spectrogram (Classic)",
                "3D Vector Field",
                "Distribution Histograms",
                "Box Plots (Multi-Sensor Comparison)",
                "Statistical Dashboard",
                "Frequency Domain Analysis (FFT/PSD)",
                "Multi-Component Comparison",
                "Component Heatmap",
                "Polar Direction Plot"
            ],
            key="viz_mode_selector"
        )

        if viz_mode == "Time Series & Spectrogram (Classic)":
            # Time series plot
            st.subheader("Time Series")
            plot_time_series(view_df, selected_series)

            # Spectrogram
            st.subheader("Spectrogram")
            plot_spectrogram(view_df, selected_series, bin_width_seconds, overlap_ratio)
        
        elif viz_mode == "3D Vector Field":
            st.subheader("🧲 3D Magnetic Field Vector Space")
            st.markdown("Visualize magnetic field vectors in 3D space. Each point represents a measurement with color indicating magnitude.")
            plot_3d_vector_field(view_df)
        
        elif viz_mode == "Distribution Histograms":
            st.subheader("📊 Component Distribution Histograms")
            st.markdown("View the distribution of magnetic field components (Bx, By, Bz) to understand data characteristics.")
            plot_distribution_histograms(view_df)
        
        elif viz_mode == "Box Plots (Multi-Sensor Comparison)":
            st.subheader("📦 Box Plots - Sensor Comparison")
            st.markdown("Compare the distribution of magnetic field components across different sensors.")
            plot_box_plots(view_df)
        
        elif viz_mode == "Statistical Dashboard":
            st.subheader("📈 Statistical Summary Dashboard")
            st.markdown("Comprehensive statistical overview of your magnetic field data.")
            plot_statistical_dashboard(view_df)
        
        elif viz_mode == "Frequency Domain Analysis (FFT/PSD)":
            st.subheader("🔊 Frequency Domain Analysis")
            st.markdown("Analyze the frequency content of your magnetic field data using FFT and Power Spectral Density (PSD).")
            plot_frequency_domain_analysis(view_df, selected_series)
        
        elif viz_mode == "Multi-Component Comparison":
            st.subheader("📉 Multi-Component Time Series Comparison")
            st.markdown("View all three magnetic field components (Bx, By, Bz) simultaneously for easy comparison.")
            plot_multi_component_comparison(view_df)
        
        elif viz_mode == "Component Heatmap":
            st.subheader("🔥 Component Heatmap Over Time")
            st.markdown("Heatmap visualization showing how each component varies over time. Color intensity represents field strength.")
            plot_component_heatmap(view_df)
        
        elif viz_mode == "Polar Direction Plot":
            st.subheader("🧭 Polar Direction Plot")
            st.markdown("Polar plot showing magnetic field direction (azimuth) and magnitude. Useful for understanding field orientation patterns.")
            plot_polar_direction(view_df)
    
    with tab2:
        correlation_study_interface(df)
    
    with tab3:
        st.subheader("ℹ️ About This Enhanced Application")
        
        st.markdown("""
        **Units:**
        - Magnetic field components (b_x, b_y, b_z): **nT (nanotesla)**
        - Resultant magnetic field: **nT (nanotesla)** - calculated as √(b_x² + b_y² + b_z²)
        
        **Enhanced Features:**
        - **Robust DFT-based spectrogram** optimized for magnetic field data
        - **Automatic signal enhancement** for weak magnetic variations
        - **Anomaly detection** using statistical z-score analysis
        - **Adaptive window sizing** based on signal characteristics
        - **Advanced preprocessing** with DC removal and high-pass filtering
        - **Quick frequency presets** for magnetic field analysis (μHz, mHz, Hz)
        - **🔗 NEW: Correlation Study** - Analyze relationships between sensors and field components
        
        **Correlation Study Features:**
        - **Sensor Comparison**: Compare any two sensors from your dataset
        - **Field Selection**: Analyze Bx, By, Bz, or resultant magnetic field
        - **Statistical Analysis**: Pearson and Spearman correlation coefficients
        - **Comprehensive Visualizations**: Scatter plots, time series, residuals, correlation matrix
        - **Significance Testing**: P-values and statistical interpretation
        - **Time Range Selection**: Focus on specific time periods for analysis
        
        **Solving Flat Spectrogram Issues:**
        
        **Common Causes:**
        1. **Constant or near-constant signals** - Check signal variation
        2. **DC offset masking variations** - Enable DC removal
        3. **Insufficient signal amplification** - Use enhancement factor
        4. **Wrong frequency range** - Try magnetic field presets
        5. **Poor window sizing** - Use adaptive window sizing
        
        **Enhanced Spectrogram Parameters:**
        - **Adaptive window sizing**: Automatically optimized for magnetic data
        - **Signal enhancement**: Amplifies weak magnetic variations (1-100x)
        - **DC removal**: Eliminates constant magnetic field bias
        - **High-pass filtering**: Removes very low frequency drift
        - **Anomaly detection**: Identifies unusual frequency-time patterns
        - **Frequency presets**: Quick setup for magnetic field analysis
        
        **Magnetic Field Frequency Ranges:**
        - **Ultra-low (μHz)**: Geomagnetic diurnal variations, solar wind effects
        - **Very low (mHz)**: Magnetic storms, geomagnetic pulsations  
        - **Low (Hz)**: Magnetic anomalies, sensor drift
        
        **Anomaly Detection:**
        - Uses z-score analysis to identify unusual patterns
        - Highlights anomalies as red X markers on spectrogram
        - Adjustable threshold (1.0-5.0 z-score)
        - Helps identify magnetic field disturbances and anomalies
        
        **Flat Spectrogram Solutions:**
        
        **If your spectrogram appears completely flat (uniform color):**
        
        1. **Ultra-Aggressive Enhancement**: Set enhancement factor to 500-1000x
        2. **Enable All Processing**: Check all enhancement options
        3. **Use Magnetic Presets**: Try "Very Low (0-1 mHz)" or "Low (0-0.1 Hz)"
        4. **Longer Windows**: Use 2-10 minute windows for better resolution
        5. **Check Time Series**: Ensure the time series shows some variation
        6. **Try Different Sensors**: Some sensors may have better signal quality
        
        **Processing Pipeline for Flat Spectrograms:**
        - **Signal Enhancement**: 100-1000x amplification
        - **Contrast Stretching**: Expands dynamic range
        - **Gamma Correction**: Brightens weak signals (γ=0.5)
        - **Z-score Normalization**: Statistical contrast enhancement
        - **Savitzky-Golay Smoothing**: Reduces noise while preserving features
        - **Adaptive Windowing**: Longer windows for weak signals
        
        **Correlation Study Usage:**
        
        1. **Navigate to Correlation Study tab**
        2. **Select two sensors** from the dropdown menus
        3. **Choose field** (Bx, By, Bz, or resultant)
        4. **Set time range** for analysis
        5. **Click "Run Correlation Analysis"** to generate comprehensive results
        
        **Correlation Interpretation:**
        - **|r| > 0.8**: Very strong correlation
        - **|r| > 0.6**: Strong correlation  
        - **|r| > 0.4**: Moderate correlation
        - **|r| > 0.2**: Weak correlation
        - **|r| ≤ 0.2**: Very weak/no correlation
        
        **Statistical Significance:**
        - **p < 0.001**: Highly significant
        - **p < 0.01**: Very significant
        - **p < 0.05**: Significant
        - **p ≥ 0.05**: Not significant
        """)


if __name__ == "__main__":
    main()