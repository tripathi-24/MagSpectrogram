# 📊 Spectrogram Implementation Guide - app_12_Nov.py

## Overview

This guide provides a detailed, step-by-step explanation of how the spectrogram is computed and plotted in `app_12_Nov.py`. The implementation uses a simplified, working approach that produces non-flat spectrograms by avoiding excessive post-processing.

---

## 🔄 Complete Process Flow

```
CSV Data → Filter by Sensor/Time → Extract Signal → Preprocess → Compute Spectrogram → 
Apply Log Scaling → Filter Frequencies → Align Timeline → Visualize
```

---

## 📋 Detailed Step-by-Step Process

### **STEP 1: Data Loading and Filtering**

**Location:** `plot_spectrogram()` function (Lines 564-589)

**What happens:**
1. Filter dataframe by sensor ID
2. Extract signal values directly from the dataframe (no resampling)
3. Calculate resultant if needed: `√(b_x² + b_y² + b_z²)`

**Code:**
```python
# Filter by sensor
sub = work[work["sensor_id"] == sid].sort_values("timestamp")

# Get signal values directly (like spectrogram_app.py)
if series == "resultant":
    signal_values = compute_resultant(sub).values
else:
    signal_values = pd.to_numeric(sub[series], errors="coerce").dropna().values
```

**Key Point:** Signal values are extracted **directly** from the dataframe without complex resampling. This preserves the original data characteristics.

**Result:** 1D numpy array of signal values: `[45870.19, 45871.23, 45869.45, ...]`

---

### **STEP 2: Estimate Sampling Rate**

**Location:** `estimate_sampling_rate()` function (Lines 45-57)

**What happens:**
1. Convert timestamps to seconds (nanoseconds → seconds)
2. Calculate time differences between consecutive samples
3. Use median time difference to estimate sampling rate

**Code:**
```python
ts = timestamps.dropna().astype("int64").values.astype(float) / 1e9
dt = np.diff(ts)
dt = dt[dt > 0]  # Remove invalid differences
median_dt = float(np.median(dt))
fs = 1.0 / median_dt  # Sampling rate in Hz
```

**Example:**
- Timestamps: `[10:00:00, 10:00:02, 10:00:04, ...]`
- Time differences: `[2s, 2s, 2s, ...]`
- Median: `2 seconds`
- Sampling rate: `1/2 = 0.5 Hz`

**Result:** Sampling rate `fs` in Hz

---

### **STEP 3: Signal Preprocessing**

**Location:** `compute_spectrogram()` function (Lines 127-139)

**What happens:**
1. **Remove DC component**: Subtract mean to center signal around zero
   ```python
   signal_processed = signal - np.mean(signal)
   ```
   This removes the constant offset (e.g., 50,000 nT baseline), revealing variations.

2. **Apply Savitzky-Golay smoothing**: Reduce noise while preserving features
   ```python
   if len(signal_processed) > 21:
       window_length = min(21, len(signal_processed) // 10)
       if window_length % 2 == 0:
           window_length += 1  # Must be odd
       if window_length >= 5:
           signal_processed = savgol_filter(signal_processed, window_length, 3)
   ```
   This smooths the signal to reduce high-frequency noise without losing important features.

**Key Point:** Preprocessing is **minimal** - only DC removal and optional smoothing. No aggressive enhancement that could distort the signal.

**Result:** Preprocessed signal ready for frequency analysis

---

### **STEP 4: Calculate Window Parameters**

**Location:** `compute_spectrogram()` function (Lines 141-144)

**What happens:**
1. Calculate window size in samples: `nperseg = window_seconds × fs`
2. Enforce minimum: `nperseg = max(8, nperseg)` (at least 8 samples for valid FFT)
3. Enforce maximum: `nperseg = min(nperseg, len(signal) // 2)` (not more than half the signal)
4. Calculate overlap: `noverlap = overlap_ratio × nperseg`

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

### **STEP 5: Compute Spectrogram**

**Location:** `compute_spectrogram()` function (Lines 146-178)

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

**Log Scaling:**
- For PSD mode: `sxx = 10 * np.log10(sxx + 1e-15)` converts to decibels (dB)
- For magnitude mode: `sxx = np.log10(sxx + 1e-15)` converts to log magnitude
- The `1e-15` offset prevents log(0) errors

**Result:**
- `f`: `[0.0, 0.05, 0.1, 0.15, ...]` Hz (frequency bins)
- `t`: `[0.0, 2.0, 4.0, 6.0, ...]` seconds (window center times, relative)
- `sxx`: 2D array `[frequencies × time_windows]` (log-scaled power/magnitude values)

---

### **STEP 6: Apply Frequency Filtering (Optional)**

**Location:** `plot_spectrogram()` function (Lines 619-645)

**What happens:**
1. If custom frequency range is specified, filter the frequency and power arrays
2. Otherwise, use full frequency range up to Nyquist frequency

**Code:**
```python
nyquist_freq = fs / 2.0
freq_mask = np.ones_like(f, dtype=bool)
use_custom_range = (min_freq > 0) or (max_freq > 0)

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

**Result:** Filtered frequency and power arrays (`f_display`, `sxx_display`) ready for visualization

---

### **STEP 7: Align with Actual Timeline**

**Location:** `plot_spectrogram()` function (Lines 647-649)

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

**Result:** Array of actual timestamps (`window_center_times`) aligned with the original CSV data timeline

---

### **STEP 8: Create Visualization**

**Location:** `plot_spectrogram()` function (Lines 675-725)

**What happens:**
1. Create Plotly heatmap with aligned timestamps, frequencies, and power values
2. Use `sxx_display` directly without additional normalization

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

## 🎯 Key Implementation Details

### **Why This Approach Works (Non-Flat Spectrograms)**

1. **Minimal Preprocessing**: Only DC removal and optional smoothing
   - No aggressive signal enhancement
   - No excessive filtering
   - Preserves original signal characteristics

2. **Direct Use of Spectrogram Output**: 
   - Log scaling is applied (dB or log magnitude)
   - No additional normalization or contrast stretching
   - No per-frequency normalization that could flatten features

3. **Simple Window Calculation**:
   - Uses standard scipy parameters
   - No adaptive window sizing that could distort results

4. **Direct Signal Extraction**:
   - No complex resampling or interpolation
   - Uses signal values directly from dataframe

### **Comparison with Previous Approach**

**Previous (Flat Spectrograms):**
- Complex resampling to uniform grid
- Multiple normalization steps (per-frequency, contrast stretching, gamma correction)
- Aggressive signal enhancement
- Z-score normalization that could wash out features

**Current (Working Spectrograms):**
- Direct signal extraction
- Minimal preprocessing (DC removal + smoothing)
- Log scaling only
- Direct visualization without extra normalization

---

## 📊 Visual Example

### **Data Flow:**

```
CSV File:
timestamp,              b_x,     b_y,     b_z,     sensor_id
2025-11-09 04:41:29,   45870,   10838,   32901,   sensor1
2025-11-09 04:41:31,   45871,   10839,   32902,   sensor1
2025-11-09 04:41:33,   45872,   10840,   32903,   sensor1
...

↓ Filter & Extract (Step 1)

Signal Array:
[45870.19, 45871.23, 45872.15, ...]

↓ Preprocess (Step 3)

Preprocessed Signal:
[0.0, 1.04, 1.96, ...]  (DC removed, smoothed)

↓ Compute Spectrogram (Step 5)

Spectrogram Output:
f = [0.0, 0.05, 0.1, 0.15, ...] Hz
t = [0.0, 2.0, 4.0, 6.0, ...] seconds (relative)
sxx = [[log-scaled power values for each frequency at each time]]

↓ Align Timeline (Step 7)

Actual Times:
[2025-11-09 04:41:29, 2025-11-09 04:41:31, 2025-11-09 04:41:33, ...]

↓ Visualize (Step 8)

Spectrogram Plot:
X-axis: Actual timestamps (aligned with CSV)
Y-axis: Frequencies (Hz)
Color: Log-scaled Power (dB) or Magnitude (log)
```

---

## ⚠️ Important Notes

### **1. Window Center Times**

- `scipy.signal.spectrogram()` returns `t` as **window CENTER times**, not window start times
- Each time point represents when the **analysis window is centered**, not when it starts
- Window start = `center - window_seconds/2`
- Window end = `center + window_seconds/2`

### **2. Why No Extra Normalization?**

- Log scaling (dB or log magnitude) already provides good dynamic range
- Additional normalization can flatten the spectrogram by compressing the dynamic range
- The direct approach preserves the natural frequency content

### **3. Signal Extraction**

- Signal values are extracted **directly** from the dataframe
- No resampling to uniform grid (which can introduce artifacts)
- Preserves original sampling characteristics

---

## 🛠️ Code Reference

### **Key Functions:**

1. **`compute_spectrogram()`** (Lines 121-181): 
   - Preprocesses signal (DC removal, smoothing)
   - Calculates window parameters
   - Calls `scipy.signal.spectrogram()`
   - Applies log scaling

2. **`plot_spectrogram()`** (Lines 564-728):
   - Filters data by sensor
   - Extracts signal values
   - Estimates sampling rate
   - Applies frequency filtering
   - Aligns timeline
   - Creates visualization

3. **`estimate_sampling_rate()`** (Lines 45-57):
   - Calculates sampling rate from timestamp differences

### **Critical Code Sections:**

```python
# Signal extraction (direct, no resampling)
signal_values = compute_resultant(sub).values  # or sub[series].values

# Preprocessing (minimal)
signal_processed = signal - np.mean(signal)
signal_processed = savgol_filter(signal_processed, window_length, 3)

# Spectrogram computation
f, t, sxx = spectrogram(signal_processed, fs=fs, nperseg=nperseg, ...)

# Log scaling
sxx = 10 * np.log10(sxx + 1e-15)  # PSD mode

# Timeline alignment
start_time = sub["timestamp"].min()
window_center_times = [start_time + pd.Timedelta(seconds=float(t_bin)) for t_bin in t]

# Visualization (direct, no extra normalization)
heatmap = go.Heatmap(x=window_center_times, y=f_display, z=sxx_display, ...)
```

---

## 📝 Summary

The spectrogram implementation in `app_12_Nov.py` follows these principles:

1. ✅ **Direct signal extraction** - No complex resampling
2. ✅ **Minimal preprocessing** - Only DC removal and smoothing
3. ✅ **Standard spectrogram computation** - Using scipy.signal.spectrogram
4. ✅ **Log scaling** - For better visualization of weak signals
5. ✅ **Direct visualization** - No excessive normalization
6. ✅ **Timeline alignment** - Convert relative times to absolute timestamps

**Result:** Non-flat spectrograms that accurately represent the frequency content of your magnetic field data!

---

*This guide explains the complete implementation from data loading to visualization in `app_12_Nov.py`.*

