# 📊 MagSpectrogram v2 - Comprehensive Technical Documentation

## 🎯 **Application Overview**

The **MagSpectrogram v2** is an advanced web application designed for analyzing magnetic field data using sophisticated signal processing techniques. It transforms raw magnetic sensor data into powerful visualizations that reveal hidden patterns, anomalies, and frequency characteristics in magnetic field measurements. The application now includes a comprehensive **Correlation Study** feature for analyzing relationships between sensors and field components.

---

## 🔬 **Core Scientific Concepts**

### **1. Magnetic Field Analysis**
**What it is**: The application analyzes three-dimensional magnetic field data from sensors that measure the Earth's magnetic field strength in three directions.

**Why it matters**: Magnetic field variations can indicate:
- **Geomagnetic activity** (solar wind effects, magnetic storms)
- **Geological anomalies** (mineral deposits, underground structures)
- **Environmental changes** (pollution, industrial activity)
- **Sensor drift** or calibration issues

**Example**: A magnetometer measuring 50,000 nT in the X direction, 20,000 nT in the Y direction, and 40,000 nT in the Z direction has a resultant field strength of √(50,000² + 20,000² + 40,000²) = 67,082 nT.

### **2. Spectrogram Analysis**
**What it is**: A spectrogram is a 2D visualization showing how the frequency content of a signal changes over time. Think of it as a "musical score" for magnetic field data.

**How it works**:
1. **Time Windows**: The signal is divided into overlapping time segments
2. **Frequency Analysis**: Each window is analyzed to find its frequency components
3. **Visualization**: Results are displayed as a heatmap where:
   - **X-axis** = Time
   - **Y-axis** = Frequency
   - **Color** = Signal strength at that frequency and time

**Example**: A spectrogram might show that at 10:00 AM, there was a strong signal at 0.001 Hz (very low frequency), indicating a slow magnetic field variation.

### **3. Signal Processing Pipeline**
The application uses a sophisticated multi-step process to enhance weak magnetic signals:

#### **Step 1: DC Component Removal**
**What it is**: Removes the constant offset (average value) from the signal.
**Why it's needed**: Magnetic sensors often have a constant bias that masks small variations.
**Example**: If a sensor reads 50,000 nT ± 10 nT, removing the DC component (50,000 nT) reveals the ±10 nT variations.

#### **Step 2: Savitzky-Golay Smoothing**
**What it is**: A smart smoothing technique that reduces noise while preserving important signal features.
**Why it's better than simple averaging**: It maintains sharp changes while removing random noise.
**Example**: A noisy signal with random spikes gets smoothed to show the underlying trend.

#### **Step 3: High-Pass Filtering**
**What it is**: Removes very slow changes (drift) that can hide interesting patterns.
**Why it's needed**: Magnetic field data often has slow drift that masks faster variations.
**Example**: Removes changes slower than 0.0001 Hz to focus on more interesting frequencies.

#### **Step 4: Signal Amplification**
**What it is**: Amplifies weak signals by 10x to 1000x to make them visible.
**Why it's needed**: Magnetic field variations can be extremely small (nanotesla level).
**Example**: A 1 nT variation gets amplified to 100 nT for better visualization.

#### **Step 5: Contrast Enhancement**
**What it is**: Stretches the signal range to use the full display range.
**Why it's needed**: Weak signals might only use 1% of the available range.
**Example**: A signal ranging from 0.1 to 0.2 gets stretched to 0.0 to 1.0.

---

## 🛠️ **Technical Implementation Details**

### **1. Data Structure and Validation**

#### **CSV File Format**
The application expects CSV files with specific columns:
```csv
id,b_x,b_y,b_z,timestamp,lat,lon,alt,theta_x,theta_y,theta_z,sensor_id
1,50000,20000,40000,2024-01-01 10:00:00,40.7128,-74.0060,100,0,0,0,sensor_1
```

**Column Descriptions**:
- **id**: Unique measurement identifier
- **b_x, b_y, b_z**: Magnetic field components in nanotesla (nT)
- **timestamp**: When the measurement was taken
- **lat, lon, alt**: GPS coordinates and altitude
- **theta_x, theta_y, theta_z**: Sensor orientation angles
- **sensor_id**: Which sensor took the measurement

#### **Data Validation Process**
```python
# Check for required columns
expected_columns = ["id", "b_x", "b_y", "b_z", "timestamp", "lat", "lon", "alt", 
                   "theta_x", "theta_y", "theta_z", "sensor_id"]
missing_columns = [col for col in expected_columns if col not in df.columns]

# Convert data types
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["b_x"] = pd.to_numeric(df["b_x"], errors="coerce")
```

### **2. Resultant Magnetic Field Calculation**

The total magnetic field strength is calculated using the 3D Pythagorean theorem:

```python
def compute_resultant(df):
    bx = df["b_x"].astype(float)  # X component (north-south)
    by = df["b_y"].astype(float)  # Y component (east-west)
    bz = df["b_z"].astype(float)  # Z component (up-down)
    
    # 3D Pythagorean theorem
    resultant = np.sqrt(bx * bx + by * by + bz * bz)
    return resultant
```

**Example**: If b_x = 50,000 nT, b_y = 20,000 nT, b_z = 40,000 nT:
Resultant = √(50,000² + 20,000² + 40,000²) = 67,082 nT

### **3. Sampling Rate Estimation**

The application automatically calculates how often the sensor was taking measurements:

```python
def estimate_sampling_rate(timestamps):
    # Convert timestamps to seconds
    ts = timestamps.dropna().astype("int64").values.astype(float) / 1e9
    
    # Calculate time differences between measurements
    dt = np.diff(ts)
    dt = dt[dt > 0]  # Remove invalid differences
    
    # Use median to avoid outliers
    median_dt = float(np.median(dt))
    
    # Sampling rate = 1 / time_between_samples
    return 1.0 / median_dt
```

**Example**: If measurements are taken every 0.1 seconds, the sampling rate is 10 Hz.

### **4. Spectrogram Computation**

The core spectrogram calculation uses advanced signal processing:

```python
def compute_spectrogram(signal, sampling_hz, window_seconds, overlap_ratio):
    # Preprocessing steps
    signal_processed = signal.copy()
    
    # 1. Remove DC component
    signal_processed = signal_processed - np.mean(signal_processed)
    
    # 2. Apply smoothing
    signal_processed = savgol_filter(signal_processed, window_length, 3)
    
    # 3. Apply high-pass filter
    cutoff = max(0.0001, 0.0001 * nyquist)
    b, a = butter(6, cutoff, btype='high', fs=sampling_hz)
    signal_processed = filtfilt(b, a, signal_processed)
    
    # 4. Apply amplification
    amplification_factor = min(1000, max(10, 0.01 / signal_std))
    signal_processed = signal_processed * amplification_factor
    
    # 5. Calculate spectrogram
    f, t, sxx = spectrogram(
        signal_processed,
        fs=sampling_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd"
    )
    
    return f, t, sxx
```

### 4.1 Stabilized Scaling and Enhancement (30 Oct 2025)

- Safer enhancement: Standardize the time series first, then apply a capped scale factor and clipping to avoid numeric overflow and saturation.
- Per-frequency normalization (row-wise): Optional post-processing that normalizes each frequency band across time, preventing early bright windows from compressing later time to the low end of the colormap.
- Stable percentile scaling: Percentiles for contrast stretching are computed on the middle portion of the spectrogram (optionally trimming a few edge columns) so edge transients do not dominate the scaling.
- Defaults updated: Gamma and global z-score are OFF by default.

Why initial “blue/yellow then purple” happens:
- A brief high-power patch early can set the global scale (after noise-floor clamp, percentile stretch, gamma, log, and z-score), making subsequent low-power regions appear uniformly dark/purple.
- When the time range slider changes, adaptive windowing and re-scaling can reintroduce this effect unless stabilized normalization/scaling is enabled.

Frequency (y-axis) limits:
- Upper limit is Nyquist (fs/2) from the estimated sampling rate.
- Practical minimum resolvable frequency ≈ 1 / window_seconds; longer windows resolve lower μHz–mHz bands.

### **5. Frequency Range Selection**

The application provides intelligent frequency range selection:

#### **Preset Options**:
- **Micro Hz (μHz)**: 0 to 1 μHz (ultra-low frequencies)
- **Milli Hz (mHz)**: 0 to 1 mHz (very low frequencies)
- **Hertz (Hz)**: 0 to 1 Hz (low frequencies)
- **Full Data Range**: Shows all available frequencies

#### **Dynamic Frequency Limits**:
```python
# Calculate Nyquist frequency (maximum possible frequency)
nyquist_freq = sampling_rate / 2.0

# Smart frequency range based on sampling rate
if sampling_rate < 0.01:  # Very low sampling rate
    default_max = min(0.001, nyquist_freq)
elif sampling_rate < 0.1:  # Low sampling rate
    default_max = min(0.01, nyquist_freq)
elif sampling_rate < 1.0:  # Medium-low sampling rate
    default_max = min(0.1, nyquist_freq)
# ... and so on
```

### **6. Anomaly Detection System**

The application uses statistical analysis to identify unusual patterns:

#### **Z-Score Analysis**:
```python
def detect_anomalies_in_spectrogram(sxx, f, t, threshold=2.0):
    # Calculate mean and standard deviation for each frequency
    sxx_mean = np.mean(sxx, axis=1, keepdims=True)
    sxx_std = np.std(sxx, axis=1, keepdims=True) + 1e-12
    
    # Calculate z-scores
    z_scores = (sxx - sxx_mean) / sxx_std
    
    # Find points above threshold
    anomaly_mask = np.abs(z_scores) > threshold
    
    return anomaly_freqs, anomaly_times
```

**How it works**:
1. **Calculate normal behavior**: For each frequency, find the average and variation
2. **Calculate z-scores**: How many standard deviations each point is from normal
3. **Identify anomalies**: Points with z-score > 2.0 are considered unusual
4. **Visualize**: Anomalies are marked with red X markers

**Example**: If a frequency normally has values around 0.5 ± 0.1, a value of 0.8 would have a z-score of 3.0, indicating an anomaly.

---

## 🎨 **User Interface Components**

### **1. File Upload System**
- **Multiple CSV support**: Upload several files at once
- **Dataset folder**: Quick load from local Dataset directory
- **Data validation**: Automatic checking for required columns
- **Error handling**: Graceful handling of corrupted files

### **2. Control Panel (Sidebar)**
- **Sensor selection**: Choose specific sensor or analyze all
- **Series selection**: Pick magnetic field component (X, Y, Z, or resultant)
- **Time range**: Select specific time period to analyze
- **Spectrogram parameters**: Window size, overlap ratio

### **3. Frequency Range Controls**
- **Quick presets**: One-click selection for common frequency ranges
- **Manual controls**: Fine-tune frequency range with number inputs
- **Range slider**: Visual adjustment of frequency range
- **Smart suggestions**: Automatic recommendations based on data

### **4. Signal Enhancement Controls**
- **Ultra-aggressive enhancement**: 1x to 1000x signal amplification
- **Contrast stretching**: Expands dynamic range for better visibility
- **Gamma correction**: Brightens weak signals (γ=0.1 to 2.0)
- **Z-score normalization**: Statistical contrast enhancement

### **5. Visualization Options**
- **Colormap selection**: Viridis, Plasma, Inferno, Turbo, Cividis
- **Log/linear frequency axis**: Choose appropriate scale
- **Anomaly highlighting**: Show/hide anomaly markers
- **Interactive plots**: Zoom, pan, hover for detailed inspection

---

## 🔗 **Correlation Study Feature**

### **1. Overview**
The Correlation Study feature enables comprehensive analysis of relationships between different magnetic sensors and field components. This powerful tool helps researchers understand sensor behavior, validate data quality, and identify patterns in multi-sensor datasets.

### **2. Statistical Analysis Methods**

#### **Pearson Correlation Coefficient**
```python
def compute_pearson_correlation(x_values, y_values):
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(x_values, y_values)
    return correlation, p_value
```

**What it measures**: Linear relationship strength between two variables
**Range**: -1 (perfect negative) to +1 (perfect positive)
**Interpretation**: 
- |r| > 0.8: Very strong correlation
- |r| > 0.6: Strong correlation
- |r| > 0.4: Moderate correlation
- |r| > 0.2: Weak correlation
- |r| ≤ 0.2: Very weak/no correlation

#### **Spearman Rank Correlation**
```python
def compute_spearman_correlation(x_values, y_values):
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(x_values, y_values)
    return correlation, p_value
```

**What it measures**: Monotonic relationship strength (any consistent trend)
**Advantage**: Not affected by outliers or non-linear relationships
**Use case**: When data doesn't follow linear patterns

### **3. Data Alignment and Preprocessing**

#### **Time Synchronization**
```python
def align_sensor_data(sensor1_data, sensor2_data):
    # Find common time range
    common_start = max(sensor1_data.index.min(), sensor2_data.index.min())
    common_end = min(sensor1_data.index.max(), sensor2_data.index.max())
    
    # Create uniform time grid
    time_step = pd.Timedelta(seconds=1)
    time_index = pd.date_range(start=common_start, end=common_end, freq=time_step)
    
    # Interpolate to common grid
    sensor1_aligned = sensor1_data.reindex(time_index).interpolate(method="time")
    sensor2_aligned = sensor2_data.reindex(time_index).interpolate(method="time")
    
    return sensor1_aligned, sensor2_aligned
```

**Purpose**: Ensures both sensors are compared at the same time points
**Method**: Linear interpolation between measurement points
**Resolution**: 1-second intervals for accurate correlation

#### **Data Quality Validation**
```python
def validate_correlation_data(x_values, y_values):
    # Check minimum data points
    if len(x_values) < 10:
        return False, "Insufficient data points"
    
    # Check for valid numeric data
    if np.any(np.isnan(x_values)) or np.any(np.isnan(y_values)):
        return False, "Invalid data (NaN values)"
    
    # Check for constant signals
    if np.std(x_values) < 1e-6 or np.std(y_values) < 1e-6:
        return False, "Constant signal detected"
    
    return True, "Data validation passed"
```

### **4. Comprehensive Visualizations**

#### **Scatter Plot with Regression Line**
```python
def create_scatter_plot(x_values, y_values, correlation):
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values,
        mode='markers',
        name='Data Points',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Add regression line
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    y_line = p(x_line)
    
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        name=f'Regression Line (r={correlation:.3f})',
        line=dict(color='red', width=2)
    ))
```

#### **Time Series Comparison**
- Overlays both sensors' data over time
- Different colors for each sensor
- Helps identify temporal synchronization
- Shows data quality and gaps

#### **Residuals Analysis**
```python
def create_residuals_plot(x_values, y_values):
    # Calculate regression line
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    y_pred = p(x_values)
    
    # Calculate residuals
    residuals = y_values - y_pred
    
    # Create residuals plot
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='green', opacity=0.6)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
```

**Purpose**: Shows how well the linear model fits the data
**Interpretation**: Points close to zero line indicate good linear fit

#### **Correlation Matrix Heatmap**
```python
def create_correlation_matrix(correlation_value):
    corr_matrix = np.array([[1.0, correlation_value],
                           [correlation_value, 1.0]])
    
    fig.add_trace(go.Heatmap(
        z=corr_matrix,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 3),
        texttemplate="%{text}",
        showscale=True
    ))
```

### **5. Statistical Significance Testing**

#### **P-Value Interpretation**
```python
def interpret_p_value(p_value):
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant (p < 0.01)"
    elif p_value < 0.05:
        return "Significant (p < 0.05)"
    else:
        return "Not significant (p ≥ 0.05)"
```

**Purpose**: Determines if correlation is statistically meaningful
**Threshold**: p < 0.05 is generally considered significant
**Caution**: Large sample sizes can make weak correlations significant

### **6. Multi-Sensor Correlation Matrix**

#### **All-Pairs Analysis**
```python
def compute_all_correlations(df, field_options, available_sensors):
    correlation_data = {}
    
    for field in field_options:
        field_correlations = {}
        
        for i, sensor1 in enumerate(available_sensors):
            for j, sensor2 in enumerate(available_sensors):
                if i < j:  # Only compute upper triangle
                    corr_result = compute_correlation_analysis(df, sensor1, sensor2, field)
                    if "error" not in corr_result:
                        field_correlations[f"{sensor1} vs {sensor2}"] = corr_result['pearson_correlation']
        
        if field_correlations:
            correlation_data[field] = field_correlations
    
    return correlation_data
```

**Purpose**: Shows correlations between all sensor pairs
**Efficiency**: Only computes upper triangle (symmetric matrix)
**Output**: Organized by field type for easy comparison

### **7. Practical Applications**

#### **Sensor Calibration Validation**
- **High correlation** (r > 0.8): Sensors are well-calibrated and measuring similar conditions
- **Low correlation** (r < 0.3): Possible calibration issues or different measurement conditions
- **Negative correlation**: Sensors may be measuring opposite effects

#### **Environmental Analysis**
- **Spatial correlation**: How magnetic field varies across sensor locations
- **Temporal correlation**: How sensors respond to time-varying conditions
- **Field component analysis**: Which magnetic field directions are most correlated

#### **Quality Control**
- **Data consistency**: Identify sensors with unusual behavior
- **Sensor drift**: Detect changes in sensor performance over time
- **Environmental effects**: Understand how external factors affect measurements

### **8. Advanced Features**

#### **Time Range Selection**
- Focus analysis on specific time periods
- Compare different environmental conditions
- Analyze before/after events
- Seasonal pattern analysis

#### **Field Component Analysis**
- **Bx (X-component)**: North-south magnetic field
- **By (Y-component)**: East-west magnetic field
- **Bz (Z-component)**: Up-down magnetic field
- **Resultant**: Total magnetic field strength

#### **Data Quality Metrics**
- Number of overlapping data points
- Time range coverage
- Signal-to-noise ratios
- Data completeness statistics

---

## 🔍 **Advanced Features**

### **1. Signal Quality Analysis**

The application automatically assesses data quality:

#### **Signal-to-Noise Ratio (SNR)**:
```python
signal_power = np.mean(signal_values**2)
noise_estimate = np.var(np.diff(signal_values))
snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-12))
```

**SNR Interpretation**:
- **> 20 dB**: Excellent signal quality
- **10-20 dB**: Good signal quality
- **< 10 dB**: Poor signal quality, enhancement needed

#### **Signal Variation Assessment**:
```python
signal_std = np.std(signal_values)
signal_mean = np.mean(signal_values)

if signal_std < 1e-6:
    st.error("Signal appears constant")
elif signal_std < signal_mean * 0.01:
    st.warning("Very low signal variation")
```

### **2. Adaptive Window Sizing**

The application automatically optimizes window size based on signal characteristics:

```python
signal_duration = len(signal_processed) / sampling_hz

if signal_duration > 3600:  # More than 1 hour
    optimal_window = min(600, signal_duration / 10)  # Up to 10 minutes
elif signal_duration > 360:  # More than 6 minutes
    optimal_window = min(120, signal_duration / 5)   # Up to 2 minutes
else:
    optimal_window = min(30, signal_duration / 3)     # Up to 30 seconds
```

### **3. Post-Processing Enhancement**

Multiple enhancement techniques are applied to improve visualization:

#### **Contrast Stretching**:
```python
sxx_min, sxx_max = np.percentile(sxx_processed, [2, 98])
if sxx_max > sxx_min:
    sxx_processed = (sxx_processed - sxx_min) / (sxx_max - sxx_min)
    sxx_processed = np.clip(sxx_processed, 0, 1)
```

#### **Gamma Correction**:
```python
gamma = 0.5  # Gamma < 1 brightens low values
sxx_processed = np.power(sxx_processed, gamma)
```

#### **Z-Score Normalization**:
```python
sxx_mean = np.mean(sxx_processed)
sxx_std = np.std(sxx_processed)
if sxx_std > 0:
    sxx_processed = (sxx_processed - sxx_mean) / sxx_std
    sxx_processed = np.clip(sxx_processed, -3, 3)
```

### **4. Troubleshooting System**

The application provides intelligent troubleshooting for common issues:

#### **Flat Spectrogram Detection**:
```python
sxx_range = np.max(sxx_processed) - np.min(sxx_processed)

if sxx_range < 1e-6:
    st.error("Spectrogram still appears flat")
    st.markdown("""
    **Ultra-Aggressive Solutions:**
    1. Increase enhancement factor to 500-1000x
    2. Enable all processing options
    3. Try different frequency ranges
    4. Use longer time windows
    5. Check signal variation in time series
    """)
```

#### **Signal Quality Warnings**:
- **Constant signals**: Alert when signal has no variation
- **Low sampling rate**: Warn when data is undersampled
- **Poor SNR**: Suggest enhancement when signal quality is low
- **Missing data**: Handle gaps in time series gracefully

---

## 📊 **Magnetic Field Frequency Ranges**

### **1. Ultra-Low Frequencies (μHz)**
- **Range**: 0.000001 to 0.000001 Hz (1 μHz)
- **Applications**: 
  - Geomagnetic diurnal variations
  - Solar wind effects
  - Long-term magnetic field changes
- **Example**: Daily variations in Earth's magnetic field

### **2. Very Low Frequencies (mHz)**
- **Range**: 0.000001 to 0.001 Hz (1 mHz)
- **Applications**:
  - Magnetic storms
  - Geomagnetic pulsations
  - Ionospheric disturbances
- **Example**: Magnetic field variations during solar storms

### **3. Low Frequencies (Hz)**
- **Range**: 0.001 to 1 Hz
- **Applications**:
  - Magnetic anomalies
  - Sensor drift
  - Industrial interference
- **Example**: Magnetic field variations near power lines

### **4. Full Data Range**
- **Range**: 0 to Nyquist frequency (sampling_rate / 2)
- **Applications**:
  - Complete frequency analysis
  - Research and development
  - Comprehensive signal characterization
- **Example**: Full spectrum analysis of magnetic field data

---

## 🚀 **Performance Optimizations**

### **1. Caching System**
```python
@st.cache_data(show_spinner=False)
def read_single_csv(file_path_or_buffer):
    # Cached function to avoid re-reading files
    return pd.read_csv(file_path_or_buffer)
```

### **2. Efficient Data Processing**
- **Vectorized operations**: Use NumPy for fast mathematical operations
- **Memory management**: Process data in chunks for large datasets
- **Lazy loading**: Only load data when needed

### **3. Interactive Visualizations**
- **Plotly integration**: Fast, interactive plots
- **Responsive design**: Adapts to different screen sizes
- **Real-time updates**: Immediate response to user changes

---

## 🔧 **Troubleshooting Guide**

### **Common Issues and Solutions**

#### **1. Flat Spectrogram (Uniform Color)**
**Symptoms**: Spectrogram appears as a single color with no variation
**Causes**:
- Constant or near-constant magnetic field
- Insufficient signal amplification
- Wrong frequency range selection
- Poor data quality

**Solutions**:
1. **Ultra-Aggressive Enhancement**: Set enhancement factor to 500-1000x
2. **Enable All Processing**: Check all enhancement options
3. **Use Magnetic Presets**: Try "Very Low (0-1 mHz)" or "Low (0-0.1 Hz)"
4. **Longer Windows**: Use 2-10 minute windows for better resolution
5. **Check Time Series**: Ensure the time series shows some variation

#### **2. No Data Loaded**
**Symptoms**: "No data for spectrogram" message
**Causes**:
- Missing required columns in CSV
- Invalid data format
- Empty files

**Solutions**:
1. **Check CSV Format**: Ensure all required columns are present
2. **Validate Data**: Check for missing or invalid values
3. **File Size**: Ensure files are not empty

#### **3. Poor Signal Quality**
**Symptoms**: Low SNR warnings, noisy spectrograms
**Causes**:
- High noise levels
- Poor sensor calibration
- Environmental interference

**Solutions**:
1. **Signal Enhancement**: Increase enhancement factor
2. **Filtering**: Enable high-pass filtering
3. **Smoothing**: Apply Savitzky-Golay smoothing
4. **Time Range**: Select cleaner time periods

#### **4. Frequency Range Issues**
**Symptoms**: "No frequencies in range" warnings
**Causes**:
- Frequency range too narrow
- Sampling rate too low
- Wrong frequency units

**Solutions**:
1. **Use Presets**: Try different frequency range presets
2. **Manual Adjustment**: Manually set frequency range
3. **Check Sampling Rate**: Ensure adequate sampling rate
4. **Full Range**: Use "Full Data Range" option

---

## 📈 **Best Practices**

### **1. Data Preparation**
- **Clean Data**: Remove outliers and fill missing values
- **Consistent Format**: Ensure all CSV files have the same structure
- **Time Synchronization**: Align timestamps across multiple sensors
- **Quality Control**: Check for sensor drift and calibration

### **2. Analysis Workflow**
1. **Start with Time Series**: Always examine the raw data first
2. **Check Signal Quality**: Review SNR and variation metrics
3. **Select Appropriate Frequency Range**: Use presets for common applications
4. **Enable Enhancement**: Use signal enhancement for weak signals
5. **Detect Anomalies**: Enable anomaly detection for pattern recognition
6. **Iterate**: Adjust parameters based on results

### **3. Parameter Selection**
- **Window Size**: Longer windows for better frequency resolution
- **Overlap Ratio**: 50% overlap for smooth results
- **Enhancement Factor**: Start with 100x, increase for weak signals
- **Frequency Range**: Use presets for magnetic field analysis
- **Processing Options**: Enable all for maximum enhancement

### **4. Interpretation Guidelines**
- **Color Intensity**: Brighter colors indicate stronger signals
- **Frequency Patterns**: Look for consistent frequency components
- **Time Patterns**: Identify periodic or transient events
- **Anomalies**: Red X markers indicate unusual events
- **Trends**: Long-term changes in frequency content

---

## 🎯 **Application Architecture**

### **1. Main Components**
- **Data Loading**: CSV file processing and validation
- **Signal Processing**: Advanced preprocessing pipeline
- **Spectrogram Computation**: FFT-based frequency analysis
- **Visualization**: Interactive plots and charts
- **User Interface**: Streamlit-based web application

### **2. Key Functions**
- **`read_single_csv()`**: Load and validate individual CSV files
- **`read_multiple_csvs()`**: Combine multiple files into dataset
- **`compute_resultant()`**: Calculate total magnetic field strength
- **`estimate_sampling_rate()`**: Calculate measurement frequency
- **`compute_spectrogram()`**: Generate frequency-time analysis
- **`detect_anomalies_in_spectrogram()`**: Find unusual patterns
- **`plot_spectrogram()`**: Create interactive visualizations
- **`compute_correlation_analysis()`**: Calculate correlations between sensors
- **`plot_correlation_analysis()`**: Create comprehensive correlation visualizations
- **`correlation_study_interface()`**: Main UI for correlation analysis

### **3. Data Flow**
1. **Input**: CSV files with magnetic field data
2. **Validation**: Check data format and quality
3. **Preprocessing**: Clean and enhance signals
4. **Analysis**: Compute spectrograms and detect anomalies
5. **Correlation Analysis**: Compare sensors and field components (NEW)
6. **Visualization**: Display interactive plots and correlation results
7. **Output**: Enhanced spectrograms with anomaly markers and correlation insights

---

## 🔬 **Scientific Applications**

### **1. Geomagnetic Research**
- **Solar Wind Effects**: Study how solar wind affects Earth's magnetic field
- **Magnetic Storms**: Analyze geomagnetic disturbances
- **Diurnal Variations**: Examine daily magnetic field changes

### **2. Geological Exploration**
- **Mineral Deposits**: Detect magnetic anomalies from ore bodies
- **Underground Structures**: Identify buried metallic objects
- **Geological Mapping**: Map magnetic field variations

### **3. Environmental Monitoring**
- **Pollution Detection**: Monitor magnetic field changes from industrial activity
- **Climate Studies**: Analyze long-term magnetic field trends
- **Space Weather**: Study space weather effects on magnetic field

### **4. Sensor Calibration**
- **Drift Detection**: Identify sensor calibration drift
- **Quality Control**: Assess sensor performance
- **Calibration Validation**: Verify sensor accuracy
- **Correlation Analysis**: Compare sensor behavior and identify calibration issues

---

## 🎓 **Educational Value**

### **1. Signal Processing Concepts**
- **Fourier Analysis**: Understanding frequency domain analysis
- **Filtering**: High-pass, low-pass, and band-pass filters
- **Smoothing**: Noise reduction techniques
- **Enhancement**: Signal amplification and contrast improvement

### **2. Data Analysis Skills**
- **Statistical Analysis**: Mean, standard deviation, z-scores
- **Pattern Recognition**: Identifying trends and anomalies
- **Data Visualization**: Creating effective plots and charts
- **Quality Assessment**: Evaluating data quality and reliability
- **Correlation Analysis**: Understanding relationships between variables
- **Statistical Significance**: Interpreting p-values and confidence levels

### **3. Scientific Method**
- **Hypothesis Testing**: Formulating and testing hypotheses
- **Data Interpretation**: Drawing conclusions from analysis
- **Error Analysis**: Understanding measurement uncertainties
- **Reproducibility**: Ensuring consistent results

---

## 🚀 **Future Enhancements**

### **1. Advanced Features**
- **Machine Learning**: Automated pattern recognition
- **Real-time Processing**: Live data analysis
- **Multi-sensor Fusion**: Combining data from multiple sensors
- **Advanced Filtering**: More sophisticated signal processing

### **2. User Experience**
- **Mobile Support**: Responsive design for mobile devices
- **Export Options**: Save results in various formats
- **Batch Processing**: Analyze multiple datasets automatically
- **Custom Algorithms**: User-defined processing pipelines

### **3. Scientific Capabilities**
- **3D Visualization**: Three-dimensional spectrogram displays
- **Spectral Analysis**: Advanced frequency domain analysis
- **Time-Frequency Analysis**: Wavelet transforms and other methods
- **Statistical Modeling**: Advanced statistical analysis tools

---

## 📚 **Conclusion**

The **MagSpectrogram v2** application represents a sophisticated tool for magnetic field data analysis, combining advanced signal processing techniques with an intuitive user interface. It enables researchers, students, and professionals to:

- **Analyze complex magnetic field data** with advanced signal processing
- **Detect anomalies and patterns** using statistical methods
- **Visualize frequency content** through interactive spectrograms
- **Enhance weak signals** with ultra-aggressive processing
- **Understand data quality** through comprehensive diagnostics
- **Study sensor relationships** through comprehensive correlation analysis
- **Validate sensor performance** using statistical correlation methods

The application serves as both a practical analysis tool and an educational platform for learning about signal processing, data analysis, and magnetic field phenomena. Its comprehensive feature set and user-friendly interface make it accessible to users with varying levels of technical expertise while providing the depth and sophistication needed for advanced research applications.

---

*This documentation provides a comprehensive understanding of the MagSpectrogram v2 application, its scientific foundations, technical implementation, and practical applications. For additional support or questions, please refer to the application's built-in help system or consult the source code for detailed implementation information.*
