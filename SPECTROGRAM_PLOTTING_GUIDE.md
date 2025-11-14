# 📊 Spectrogram Plotting Guide: Step-by-Step Explanation

## Overview

This guide explains how `app_12_Nov.py` (and `spectrogram_app.py`) transforms magnetic field data from a CSV file into a spectrogram visualization and aligns it with the actual timeline. The implementation uses a simplified, working approach that produces non-flat spectrograms.

---

## 🔄 **Complete Process Flow**

```
CSV File → Load Data → Filter & Process → Compute Spectrogram → Align Timeline → Visualize
```

---

## 📋 **Step-by-Step Process**

### **STEP 1: Load CSV Data** (Lines 20-34, 169-181)

**What happens:**
1. CSV file is read using `pd.read_csv()`
2. Timestamp column is parsed to `datetime` format
3. Magnetic field columns (`b_x`, `b_y`, `b_z`) are converted to numeric

**Code:**
```python
df = pd.read_csv(file_path_or_buffer)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
for col in ["b_x", "b_y", "b_z"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

**Result:** DataFrame with properly formatted timestamps and numeric magnetic field values

---

### **STEP 2: Filter Data by Sensor and Time Range** (Lines 297-316)

**What happens:**
1. Filter data for selected sensor: `df[df["sensor_id"] == selected_sensor]`
2. Filter by time range: `(timestamp >= t0) & (timestamp <= t1)`
3. Sort by timestamp to ensure chronological order

**Code:**
```python
filtered_df = df[df["sensor_id"] == selected_sensor].copy()
t0, t1 = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
filtered_df = filtered_df[
    (filtered_df["timestamp"] >= t0) & (filtered_df["timestamp"] <= t1)
].sort_values("timestamp")
```

**Result:** Filtered DataFrame containing only the selected sensor's data in the specified time range

---

### **STEP 3: Extract Signal Values** (Lines 318-326)

**What happens:**
1. If "resultant" is selected: Calculate `√(b_x² + b_y² + b_z²)`
2. Otherwise: Extract the selected field (`b_x`, `b_y`, or `b_z`)
3. Convert to numpy array for signal processing

**Code:**
```python
if selected_field == "resultant":
    signal_values = compute_resultant(filtered_df).values
else:
    signal_values = pd.to_numeric(filtered_df[selected_field], errors="coerce").dropna().values
```

**Result:** 1D numpy array of signal values (e.g., `[45870.19, 45871.23, 45869.45, ...]`)

---

### **STEP 4: Estimate Sampling Rate** (Lines 45-57, 328-332)

**What happens:**
1. Convert timestamps to seconds (nanoseconds → seconds)
2. Calculate time differences between consecutive samples
3. Use median time difference to estimate sampling rate
4. Sampling rate = 1 / median_time_difference

**Code:**
```python
ts = timestamps.dropna().astype("int64").values.astype(float) / 1e9
dt = np.diff(ts)
dt = dt[dt > 0]  # Remove invalid differences
median_dt = float(np.median(dt))
fs = 1.0 / median_dt  # Sampling rate in Hz
```

**Example:**
- If timestamps are: `[10:00:00, 10:00:02, 10:00:04, ...]`
- Time differences: `[2s, 2s, 2s, ...]`
- Median: `2 seconds`
- Sampling rate: `1/2 = 0.5 Hz`

**Result:** Sampling rate `fs` in Hz (e.g., `0.5 Hz`)

---

### **STEP 5: Preprocess Signal** (app_12_Nov.py Lines 121-181)

**What happens:**
1. **Remove DC component**: Subtract mean to center signal around zero
   ```python
   signal_processed = signal - np.mean(signal)
   ```
   This removes the constant offset, revealing variations in the signal.

2. **Apply smoothing**: Savitzky-Golay filter to reduce noise while preserving features
   ```python
   if len(signal_processed) > 21:
       window_length = min(21, len(signal_processed) // 10)
       if window_length % 2 == 0:
           window_length += 1  # Must be odd
       if window_length >= 5:
           signal_processed = savgol_filter(signal_processed, window_length, 3)
   ```
   This smooths the signal to reduce high-frequency noise without losing important features.

**Key Point:** The preprocessing is minimal and focused - only DC removal and optional smoothing. No aggressive enhancement that could flatten the spectrogram.

**Result:** Preprocessed signal ready for frequency analysis

---

### **STEP 6: Calculate Window Parameters** (app_12_Nov.py Lines 141-144)

**What happens:**
1. Calculate window size in samples: `nperseg = window_seconds × fs`
2. Ensure minimum window size: `nperseg = max(8, nperseg)` (at least 8 samples)
3. Ensure maximum window size: `nperseg = min(nperseg, len(signal) // 2)` (not more than half the signal)
4. Calculate overlap in samples: `noverlap = overlap_ratio × nperseg`

**Code:**
```python
nperseg = max(8, int(window_seconds * sampling_hz))
nperseg = min(nperseg, len(signal_processed) // 2)
noverlap = int(overlap_ratio * nperseg)
```

**Example:**
- Window size: `1.0 seconds`
- Sampling rate: `0.5 Hz`
- `nperseg = max(8, int(1.0 × 0.5)) = max(8, 0) = 8` (minimum enforced)
- Overlap ratio: `0.5` (50%)
- `noverlap = int(0.5 × 8) = 4 samples`

**Result:** Window parameters (`nperseg`, `noverlap`) for FFT analysis

---

### **STEP 7: Compute Spectrogram** (app_12_Nov.py Lines 146-178)

**What happens:**
1. Call `scipy.signal.spectrogram()` with preprocessed signal
2. Apply log scaling to the output for better visualization

**Code:**
```python
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
    sxx = 10 * np.log10(sxx + 1e-15)  # Convert to dB
else:
    sxx = np.log10(sxx + 1e-15)  # Log magnitude
```

**What `scipy.signal.spectrogram()` does internally:**
1. **Divide signal into overlapping windows**
   - Window 1: samples [0 to nperseg-1]
   - Window 2: samples [nperseg-noverlap to 2×nperseg-noverlap-1]
   - Window 3: samples [2×(nperseg-noverlap) to 3×nperseg-2×noverlap-1]
   - ... and so on

2. **Apply window function** (Hann window) to each segment to reduce spectral leakage

3. **Detrend** each window (remove linear trend) to eliminate DC and low-frequency drift

4. **Compute FFT** for each window to get frequency content

5. **Return:**
   - `f`: Array of frequencies (Hz) - **Y-axis values**
   - `t`: Array of time points (seconds) - **Window CENTER times** (relative to signal start)
   - `sxx`: 2D array of power/magnitude - **Z-axis (color) values**

**Important:** The output `sxx` is then converted to logarithmic scale (dB for PSD, log for magnitude) to improve visualization of weak signals.

**Result:**
- `f`: `[0.0, 0.05, 0.1, 0.15, ...]` Hz (frequency bins)
- `t`: `[0.0, 2.0, 4.0, 6.0, ...]` seconds (window center times, relative)
- `sxx`: 2D array `[frequencies × time_windows]` (log-scaled power/magnitude values)

---

### **STEP 8: Align with Actual Timeline** (app_12_Nov.py Lines 647-649)

**⚠️ CRITICAL: This is the key step for timeline alignment!**

**What happens:**

1. **Get the first timestamp** from filtered data:
   ```python
   start_time = sub["timestamp"].min()
   ```
   Example: `start_time = "2025-11-09 04:41:29"`

2. **Convert relative times to absolute timestamps:**
   ```python
   window_center_times = [start_time + pd.Timedelta(seconds=float(t_bin)) for t_bin in t]
   ```
   
   **Important:** The `t` array from `spectrogram()` contains **window CENTER times** in seconds relative to the start of the signal, NOT absolute timestamps!

   **Example:**
   - `t = [0.0, 2.0, 4.0, 6.0, ...]` (seconds from signal start)
   - `start_time = "2025-11-09 04:41:29"`
   - `window_center_times = ["2025-11-09 04:41:29", "2025-11-09 04:41:31", "2025-11-09 04:41:33", ...]`

3. **Understanding window boundaries** (for reference):
   ```python
   # Window start = center - window_seconds/2
   # Window end = center + window_seconds/2
   ```
   
   This shows the actual time range each window covers:
   - Window center: `04:41:29`
   - Window start: `04:41:28.5` (if window_seconds = 1s)
   - Window end: `04:41:29.5`

**Result:** Array of actual timestamps (`window_center_times`) aligned with the original CSV data timeline

---

### **STEP 9: Apply Frequency Filtering (Optional)** (app_12_Nov.py Lines 619-645)

**What happens:**
1. If custom frequency range is specified, filter the frequency and power arrays
2. Otherwise, use full frequency range up to Nyquist frequency

**Code:**
```python
nyquist_freq = fs / 2.0
freq_mask = np.ones_like(f, dtype=bool)

if use_custom_range:
    if min_freq > 0:
        freq_mask &= f >= min_freq
    if max_freq > 0:
        effective_max = min(max_freq, nyquist_freq)
        freq_mask &= f <= effective_max
    else:
        freq_mask &= f <= nyquist_freq
    
    if np.any(freq_mask):
        f_display = f[freq_mask]
        sxx_display = sxx[freq_mask, :]
    else:
        f_display = f
        sxx_display = sxx
else:
    f_display = f
    sxx_display = sxx
```

**Result:** Filtered frequency and power arrays ready for visualization

---

### **STEP 10: Create Visualization** (app_12_Nov.py Lines 675-725)

**What happens:**
1. Create Plotly heatmap with:
   - **X-axis**: `window_center_times` (aligned timestamps)
   - **Y-axis**: `f_display` (frequencies, optionally filtered)
   - **Z-axis (color)**: `sxx_display` (log-scaled power/magnitude values)

**Code:**
```python
heatmap = go.Heatmap(
    x=window_center_times,  # Actual timestamps from CSV (window centers)
    y=f_display,            # Frequencies (Hz)
    z=sxx_display,          # Log-scaled power values (dB or log magnitude)
    colorscale=colormap.lower(),
    colorbar=dict(
        title="Power (dB)" if spec_mode == "psd" else "Magnitude (log)"
    ),
    zsmooth="best",
)

fig = go.Figure(data=heatmap)
fig.update_layout(
    title=f"Spectrogram: {y_col} - {sid}",
    xaxis_title="Time (Window Center)",
    yaxis_title="Frequency (Hz)",
    yaxis_type='log' if log_freq else 'linear',
    height=600,
)
```

**Key Points:**
- **No additional normalization**: The `sxx_display` is used directly from the spectrogram computation (after log scaling)
- **Direct visualization**: This prevents flattening that can occur with excessive post-processing
- **Timeline alignment**: X-axis uses actual timestamps, so events align with time series plots

**Result:** Interactive spectrogram plot aligned with actual timeline, showing frequency content over time

---

## 🎯 **Key Concepts for Timeline Alignment**

### **1. Window Center Times**

**Critical Understanding:**
- `scipy.signal.spectrogram()` returns `t` as **window CENTER times**, not window start times
- Each time point represents when the **analysis window is centered**, not when it starts

**Example:**
- If window size = 1 second
- Window that analyzes data from `10:00:00` to `10:00:01`:
  - **Window start**: `10:00:00`
  - **Window center**: `10:00:00.5` ← This is what `t` contains
  - **Window end**: `10:00:01`

### **2. Time Alignment Formula**

```python
actual_timestamp = first_timestamp_in_data + window_center_time
```

**Step-by-step:**
1. Find first timestamp: `start_time = filtered_df["timestamp"].min()`
2. `t` contains relative times: `[0.0, 2.0, 4.0, ...]` seconds
3. Add relative time to start time:
   ```python
   actual_times[0] = start_time + 0.0 seconds = start_time
   actual_times[1] = start_time + 2.0 seconds
   actual_times[2] = start_time + 4.0 seconds
   ```

### **3. Why Times Might Not Match Exactly**

**Scenario:** Anomaly appears at `11:00` in time series but `10:40` in spectrogram

**Explanation:**
- **Time Series**: Shows exact timestamp when event occurred
- **Spectrogram**: Shows window **center** time
- If window size is large (e.g., 20 minutes):
  - Window that captures 11:00 event might:
    - Start at: `10:20:00`
    - Center at: `10:40:00` ← Spectrogram shows this
    - End at: `11:00:00` ← Event occurs here

**Solution:** Use smaller windows for better time resolution

---

## 📊 **Visual Example**

### **Data Flow:**

```
CSV File:
timestamp,              b_x,     b_y,     b_z
2025-11-09 04:41:29,   45870,   10838,   32901
2025-11-09 04:41:31,   45871,   10839,   32902
2025-11-09 04:41:33,   45872,   10840,   32903
...

↓ Filter & Extract

Signal Array:
[45870.19, 45871.23, 45872.15, ...]

↓ Compute Spectrogram

Spectrogram Output:
f = [0.0, 0.05, 0.1, 0.15, ...] Hz
t = [0.0, 2.0, 4.0, 6.0, ...] seconds (relative)
sxx = [[power values for each frequency at each time]]

↓ Align Timeline

Actual Times:
[2025-11-09 04:41:29, 2025-11-09 04:41:31, 2025-11-09 04:41:33, ...]

↓ Visualize

Spectrogram Plot:
X-axis: Actual timestamps (aligned with CSV)
Y-axis: Frequencies (Hz)
Color: Power (dB)
```

---

## 🔍 **Detailed Timeline Alignment Process**

### **Step-by-Step Alignment:**

1. **Original Data Timestamps:**
   ```
   Row 0: 2025-11-09 04:41:29
   Row 1: 2025-11-09 04:41:31
   Row 2: 2025-11-09 04:41:33
   ...
   ```

2. **Signal Array Index:**
   ```
   Index 0 → Row 0 data
   Index 1 → Row 1 data
   Index 2 → Row 2 data
   ...
   ```

3. **Spectrogram Windows:**
   ```
   Window 0: Analyzes indices [0 to nperseg-1]
   Window 1: Analyzes indices [nperseg-noverlap to 2×nperseg-noverlap-1]
   ...
   ```

4. **Window Center Times (from scipy):**
   ```
   t[0] = 0.0 seconds (center of window 0)
   t[1] = 2.0 seconds (center of window 1)
   t[2] = 4.0 seconds (center of window 2)
   ...
   ```

5. **Convert to Absolute Times:**
   ```
   start_time = 2025-11-09 04:41:29
   actual_times[0] = 2025-11-09 04:41:29 + 0.0s = 2025-11-09 04:41:29
   actual_times[1] = 2025-11-09 04:41:29 + 2.0s = 2025-11-09 04:41:31
   actual_times[2] = 2025-11-09 04:41:29 + 4.0s = 2025-11-09 04:41:33
   ...
   ```

6. **Plot with Aligned Times:**
   - X-axis uses `actual_times` (aligned with CSV timestamps)
   - Each point on spectrogram corresponds to actual time in your data

---

## ⚠️ **Important Notes**

### **1. Window Center vs. Window Start**

- **Spectrogram shows**: Window **center** times
- **Window actually covers**: `[center - window_size/2, center + window_size/2]`
- **To find window start**: `window_start = center - window_seconds/2`

### **2. Time Resolution**

- **Small windows** (0.1-1s): Better time resolution, events appear closer to actual time
- **Large windows** (10-60s): Better frequency resolution, but events may appear shifted

### **3. Overlap Effect**

- With overlap, same event appears in multiple windows
- Each window shows the event at its **center time**
- This can make events appear to "spread" across time

---

## 🛠️ **Code Reference**

### **Key Functions:**

1. **`load_data()`** (Lines 20-34): Loads and parses CSV
2. **`estimate_sampling_rate()`** (Lines 45-57): Calculates fs from timestamps
3. **`compute_spectrogram_simple()`** (Lines 60-127): Computes spectrogram
4. **Timeline alignment** (Lines 511-527): Converts relative times to absolute timestamps

### **Critical Alignment Code:**

```python
# Line 513: Get first timestamp
start_time = filtered_df["timestamp"].min()

# Line 525: Convert relative times to absolute timestamps
actual_times = [start_time + pd.Timedelta(seconds=float(t_bin)) for t_bin in t]

# Line 531: Use aligned times in plot
heatmap = go.Heatmap(x=actual_times, y=f_display, z=sxx_display, ...)
```

---

## 📝 **Summary - Complete Process**

1. **Load CSV** → Parse timestamps and numeric values
2. **Filter data** → Select sensor and time range
3. **Extract signal** → Get 1D array of values directly from dataframe
4. **Estimate sampling rate** → Calculate fs from timestamp differences
5. **Preprocess** → Remove DC component, apply Savitzky-Golay smoothing (minimal processing)
6. **Calculate window parameters** → Determine nperseg and noverlap based on window size and overlap ratio
7. **Compute spectrogram** → Call `scipy.signal.spectrogram()` to get frequencies, times (relative), and power
8. **Apply log scaling** → Convert power to dB (PSD) or log magnitude for better visualization
9. **Filter frequencies (optional)** → Apply min/max frequency limits if specified
10. **Align timeline** → Convert relative times to absolute timestamps using first timestamp
11. **Visualize** → Plot with aligned timestamps on X-axis, using sxx directly (no extra normalization)

**The key to alignment:** Add the first timestamp from your filtered data to each relative time from the spectrogram output!

**The key to non-flat spectrograms:** Use minimal preprocessing (DC removal + smoothing), apply log scaling, and use the spectrogram output directly without excessive normalization or post-processing.

---

*This guide explains the complete process from CSV to aligned spectrogram visualization.*

