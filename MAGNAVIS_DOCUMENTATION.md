# MagNav Anomaly Detection System - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Voting and Fusion Mechanisms](#voting-and-fusion-mechanisms)
9. [Visualization and Export](#visualization-and-export)
10. [Configuration and Parameters](#configuration-and-parameters)
11. [Usage Guide](#usage-guide)
12. [Technical Details](#technical-details)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## Overview

**MagNav Anomaly Detection System** (`magnavis_anomaly_maVo.py`) is a comprehensive Streamlit-based application designed for detecting anomalies in magnetic field data collected from multiple sensors and observatories. The system employs Long Short-Term Memory (LSTM) autoencoders to learn normal patterns in magnetic field measurements and identify deviations that may indicate anomalies.

### Key Features

- **Multi-Sensor Support**: Handles dynamic numbers of sensors and observatories
- **Per-Channel Training**: Individual LSTM-AE models for each sensor-component combination
- **Robust Data Handling**: Safe conversion from long-format to wide-format with duplicate removal
- **Flexible Voting System**: Supports both absolute and fraction-based majority voting
- **Global Majority Voting**: Combines results from all channels across all observatories using majority voting
- **Full-Length Anomaly Timeline**: Expands window-level detections to complete time series
- **Interactive Visualization**: Plotly-based charts with anomaly highlighting
- **Model Persistence**: Save and load trained models, scalers, and thresholds
- **Session State Caching**: Efficient result caching to avoid recomputation

### Use Cases

- Magnetic navigation system validation
- Geomagnetic anomaly detection
- Sensor fault detection
- Data quality assessment
- Research and analysis of magnetic field variations

---

## System Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                        │
│  (File Upload, Parameter Configuration, Visualization)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                       │
│  (Long→Wide Conversion, Index Alignment, Gap Filling)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Training Layer                            │
│  (LSTM-AE Training, Scaler Fitting, Threshold Computation)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Layer                            │
│  (Model Matching, Window Creation, Prediction, Expansion)    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Voting & Fusion Layer                       │
│  (Per-Observatory Voting, Global Majority Voting)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                Visualization & Export Layer                   │
│  (Plotly Charts, CSV Export, Summary Statistics)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Processing Utilities

#### `ensure_datetime(df, ts_col='timestamp')`
Converts timestamp columns to pandas datetime format, ensuring consistent time handling throughout the pipeline.

**Parameters:**
- `df`: Input DataFrame
- `ts_col`: Name of timestamp column (default: 'timestamp')

**Returns:** DataFrame with converted timestamp column

#### `safe_build_global_index(df)`
Creates a unified, sorted, unique timestamp index from all sensors in the dataset. This global index serves as the reference timeline for all subsequent operations.

**Key Features:**
- Removes duplicate timestamps
- Handles timezone-aware and timezone-naive timestamps
- Ensures uniqueness even with precision issues

**Returns:** DatetimeIndex with unique, sorted timestamps

#### `build_wide_df_from_df(df)`
The core data transformation function that converts long-format data to wide-format.

**Input Format (Long):**
```
timestamp          sensor_id    b_x    b_y    b_z
2025-10-08 10:00  SENSOR1      100.5  200.3  300.1
2025-10-08 10:01  SENSOR1      101.2  201.0  301.5
2025-10-08 10:00  SENSOR2      150.2  250.1  350.3
```

**Output Format (Wide):**
```
timestamp          SENSOR1__b_x  SENSOR1__b_y  SENSOR1__b_z  SENSOR2__b_x  ...
2025-10-08 10:00  100.5         200.3         300.1         150.2         ...
2025-10-08 10:01  101.2         201.0         301.5         150.2         ...
```

**Processing Steps:**
1. Extract unique timestamps to create global index
2. For each sensor:
   - Remove duplicate timestamps (keep first)
   - Extract numeric components (b_x, b_y, b_z)
   - Reindex to global timeline (creates NaNs for missing timestamps)
   - Rename columns with sensor prefix (e.g., `SENSOR1__b_x`)
3. Concatenate all sensor data horizontally
4. Forward-fill then backward-fill to handle small gaps

**Handles:**
- Duplicate timestamps within sensors
- Missing timestamps (gaps in data)
- Multiple sensors with different sampling rates
- Timezone inconsistencies

### 2. Window Creation

#### `create_windows(series, window_size, step)`
Creates sliding windows from a time series for LSTM processing.

**Parameters:**
- `series`: 1D numpy array of time series values
- `window_size`: Number of samples per window
- `step`: Step size between windows (overlap = window_size - step)

**Returns:**
- `X`: Array of shape (n_windows, window_size)
- `idxs`: List of (start_idx, end_idx) tuples for each window

**Example:**
```python
series = [1, 2, 3, 4, 5, 6, 7, 8]
window_size = 3, step = 2
# Returns:
# X = [[1,2,3], [3,4,5], [5,6,7]]
# idxs = [(0,2), (2,4), (4,6)]
```

### 3. Model Architecture

#### `build_lstm_autoencoder(window_size, latent_dim=32)`
Constructs a symmetric LSTM autoencoder for time series reconstruction.

**Architecture:**
```
Input: (batch, window_size, 1)
    │
    ▼
LSTM(64, return_sequences=True)  ──┐
    │                              │ Encoder
    ▼                              │
LSTM(32, return_sequences=False) ─┘
    │
    ▼
RepeatVector(window_size)  ───────┐
    │                              │ Decoder
    ▼                              │
LSTM(32, return_sequences=True) ──┤
    │                              │
    ▼                              │
LSTM(64, return_sequences=True) ──┤
    │                              │
    ▼                              │
TimeDistributed(Dense(1)) ─────────┘
    │
    ▼
Output: (batch, window_size, 1)
```

**Design Rationale:**
- **Encoder**: Compresses input to latent representation (32 dimensions)
- **Decoder**: Reconstructs from latent space back to original dimensions
- **Symmetric Structure**: Encoder and decoder mirror each other for balanced learning
- **TimeDistributed Dense**: Applies dense layer to each time step independently

**Hyperparameters:**
- `window_size`: Input sequence length (configurable, default: 64)
- `latent_dim`: Bottleneck dimension (configurable, default: 32)
- Activation: `tanh` for LSTM layers
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

### 4. Threshold Computation

#### `compute_threshold(window_errors, k=6.0)`
Computes anomaly threshold using Median Absolute Deviation (MAD) method.

**Formula:**
```
threshold = median(window_errors) + k × MAD(window_errors)
```

**Why MAD?**
- Robust to outliers (unlike standard deviation)
- Works well with non-normal distributions
- Commonly used in anomaly detection

**Parameter `k`:**
- Controls sensitivity (higher k = fewer false positives, more false negatives)
- Default: 6.0 (approximately 3 standard deviations in normal distribution)
- Tune based on labeled data if available

### 5. Window Expansion

#### `expand_window_flags_to_full(length, window_idxs, window_flags)`
Expands window-level anomaly flags to full-length time series.

**Problem:** LSTM-AE processes data in windows, producing one flag per window. We need flags for every timestamp.

**Solution:** Mark all timestamps within an anomalous window as anomalous.

**Example:**
```
length = 10
window_idxs = [(0,2), (3,5), (6,8)]
window_flags = [False, True, False]

Result: [False, False, False, True, True, True, False, False, False, False]
         └─window1─┘  └──window2──┘  └─window3─┘
```

### 6. Anomaly Region Detection

#### `find_anomaly_regions(flag_bool)`
Identifies continuous regions of anomalies for visualization.

**Returns:** List of (start_idx, end_idx) tuples representing continuous anomaly regions.

**Example:**
```
flags = [False, False, True, True, True, False, True, True, False]
regions = [(2, 4), (6, 7)]
```

---

## Data Processing Pipeline

### Input Requirements

**Training CSV Format:**
- Required columns: `timestamp`, `sensor_id`, `b_x`, `b_y`, `b_z`
- Optional: Additional metadata columns (ignored)
- Timestamp format: Any pandas-readable datetime format

**Detection CSV Format:**
- Same structure as training CSV
- Can include different sensors/observatories than training data
- System will attempt to match models to available sensors

### Processing Flow

1. **Load CSV**: Read file (uploaded or from fixed path)
2. **Validate**: Check for required columns
3. **Convert Timestamps**: Ensure datetime format
4. **Build Wide Format**: Transform to sensor-component columns
5. **Handle Gaps**: Forward-fill and backward-fill small gaps
6. **Validate Output**: Ensure no duplicate indices

### Error Handling

- Missing columns: Error message with guidance
- No numeric channels: Error message
- Insufficient data: Warning, skip channel
- Duplicate timestamps: Automatically handled (keep first, group by mean)

---

## Model Architecture

### LSTM Autoencoder Design

The LSTM autoencoder is specifically designed for time series anomaly detection:

**Encoder Path:**
- Reduces temporal sequence to fixed-size latent vector
- Captures temporal dependencies and patterns
- Learns compressed representation of normal behavior

**Decoder Path:**
- Reconstructs sequence from latent vector
- Should accurately reconstruct normal patterns
- Will struggle with anomalous patterns (high reconstruction error)

**Training Objective:**
- Minimize reconstruction error on normal data
- Anomalies will have high reconstruction error during inference

### Why LSTM?

- **Temporal Dependencies**: Magnetic field data has temporal correlations
- **Variable Length**: Can handle sequences of different lengths (via windowing)
- **Memory**: LSTM cells maintain memory of past patterns
- **Proven**: Widely used in time series anomaly detection

### Alternative Architectures Considered

- **1D CNN**: Faster but less effective for long-term dependencies
- **Transformer**: More complex, requires more data
- **Simple Dense AE**: Cannot capture temporal patterns

---

## Training Pipeline

### Overview

The training process creates one LSTM-AE model for each sensor-component combination (channel).

### Step-by-Step Process

1. **Load Training Data**
   - Read CSV file
   - Convert to wide format
   - Display preview

2. **For Each Channel:**
   ```
   a. Extract time series (drop NaNs)
   b. Check sufficient length (>= window_size)
   c. Fit StandardScaler on training data
   d. Normalize time series
   e. Create windows (window_size, step_size)
   f. Check sufficient windows (>= 10)
   g. Build LSTM-AE model
   h. Train model (epochs, batch_size, validation_split=0.05)
   i. Compute reconstruction errors on training windows
   j. Compute threshold using MAD method
   k. Store: model, scaler, threshold
   ```

3. **Save Models** (if auto_save enabled)
   - Models: `model_{safe_channel_name}.h5`
   - Scalers: `scalers.pkl`
   - Thresholds: `thresholds.json`
   - Metadata: `metadata.json` (window_size, step_size, latent_dim)

### Training Parameters

- **window_size**: Length of input sequences (default: 64)
- **step_size**: Stride between windows (default: window_size, no overlap)
- **batch_size**: Training batch size (default: 64)
- **epochs**: Number of training epochs (default: 10)
- **latent_dim**: Bottleneck dimension (default: 32)
- **resid_k**: MAD multiplier for threshold (default: 6.0)

### Validation

- 5% of training data used for validation
- No early stopping (can be added)
- Training progress not displayed (verbose=0)

### Model Storage

**Directory Structure:**
```
saved_models/
├── model_SENSOR1__b_x.h5
├── model_SENSOR1__b_y.h5
├── model_SENSOR1__b_z.h5
├── model_SENSOR2__b_x.h5
├── ...
├── scalers.pkl
├── thresholds.json
└── metadata.json
```

**File Formats:**
- Models: Keras H5 format (TensorFlow SavedModel compatible)
- Scalers: Pickle format
- Thresholds: JSON format (human-readable)
- Metadata: JSON format

---

## Inference Pipeline

### Overview

The inference pipeline applies trained models to detection data, identifies anomalies, and combines results through voting mechanisms.

### Step-by-Step Process

1. **Load Detection Data**
   - Read CSV file
   - Convert to wide format
   - Display preview

2. **Load Models**
   - Load from model directory
   - Extract metadata (window_size, step_size)
   - Fallback to sidebar parameters if metadata missing

3. **Model-to-Column Matching**
   
   The system uses a sophisticated matching algorithm to pair trained models with detection columns:
   
   **Matching Strategy (in order of priority):**
   
   a. **Exact Match**: Model key exactly matches column name
      ```
      Model: "SENSOR1__b_x" → Column: "SENSOR1__b_x" ✓
      ```
   
   b. **Safe Name Match**: Handle special characters
      ```
      Model: "SENSOR/1__b_x" (saved as "SENSOR_1__b_x")
      → Column: "SENSOR_1__b_x" ✓
      ```
   
   c. **Heuristic Match by Sensor Key and Axis**:
      - Extract sensor identifier (ignoring timestamp prefix)
      - Extract axis (b_x, b_y, b_z)
      - Match by sensor number and axis
      ```
      Model: "S20251008_105355_44180345587365_1__b_x"
      → Sensor key: "_1", Axis: "b_x"
      → Column: "S20251107_154804_OBS1_1__b_x"
      → Sensor key: "OBS1_1" (contains "_1"), Axis: "b_x" ✓
      ```
   
   d. **Axis-Only Fallback**: Match by axis only (use first available sensor)
      ```
      Model: "SENSOR1__b_x" → Column: "SENSOR2__b_x" (if SENSOR1 not available)
      ```

4. **Sensor Key Extraction**
   
   The `extract_sensor_key()` function identifies meaningful sensor identifiers:
   
   - **OBS Pattern**: `OBS1_1`, `OBS2_3` → Returns `("OBS1_1", 1)`
   - **Numeric Suffix**: `S2025..._1` → Returns `("_1", 1)`
   - **Fallback**: Returns full sensor ID
   
   This allows matching sensors across different timestamps or observatories.

5. **Per-Channel Inference**
   
   For each matched model-column pair:
   ```
   a. Get model and determine its window_size
   b. Extract detection time series
   c. Check sufficient length (>= window_size)
   d. Get or create scaler (prefer saved, else fit on detection data)
   e. Normalize time series
   f. Create windows (model-specific window_size and step_size)
   g. Predict with LSTM-AE
   h. Compute window-level RMSE
   i. Get threshold (from saved or compute new)
   j. Flag windows as anomalous (RMSE > threshold)
   k. Expand window flags to full timeline
   l. Store channel flags and errors
   ```

6. **Observatory Detection**
   
   Automatically identifies observatories from column names:
   - Searches for `OBS1`, `OBS2`, etc. patterns
   - Groups channels by observatory
   - Fallback: `OBS_ALL` if no pattern found

7. **Per-Observatory Voting**
   
   For each observatory:
   - Collect flags from all channels in that observatory
   - Apply majority voting (absolute or fraction-based)
   - Store observatory-level flags and vote counts

8. **Global Majority Voting (Final Decision)**
   
   - Collect flags from ALL channels (across all observatories)
   - Apply majority voting across all channels
   - This is the final anomaly flag
   - **Note**: Per-observatory voting is computed for reference/export but the final decision uses global voting across all channels

### Window Size Handling

The system intelligently handles different window sizes:

- **From Metadata**: Prefers window_size from saved metadata
- **From Model**: Extracts from model input_shape if metadata missing
- **From Sidebar**: Fallback to user-specified value
- **Per-Model**: Each model can have different window_size (adjusted automatically)

### Step Size Adjustment

If model window_size differs from global:
```
adjusted_step_size = global_step_size × (model_window_size / global_window_size)
```

This maintains proportional overlap between windows.

---

## Voting and Fusion Mechanisms

**Overview**: The system uses **majority voting** (not AND fusion) to combine results from multiple channels. The final anomaly decision is made through **global majority voting** across all channels from all observatories. Per-observatory voting is computed for analysis purposes but does not determine the final flag.

### Voting Modes

#### 1. Absolute Voting
Requires a fixed number of channels to vote for anomaly.

**Example:**
- 10 channels total
- `min_channels_required = 5`
- Anomaly if ≥ 5 channels flag it

**Use Case:** When you know the minimum number of sensors that should agree.

#### 2. Fraction-Based Voting
Requires a fraction of channels to vote for anomaly.

**Example:**
- 10 channels total
- `vote_fraction = 0.56`
- Required: ⌈10 × 0.56⌉ = 6 channels
- Anomaly if ≥ 6 channels flag it

**Use Case:** When sensor count varies, want proportional agreement.

### Voting Hierarchy

```
Channel-Level Flags (per sensor-component)
    │
    ├─→ Per-Observatory Voting
    │   (Majority vote within OBS1, OBS2, etc.)
    │
    └─→ Global Voting
        (Majority vote across ALL channels)
        │
        └─→ Final Anomaly Flag
```

**Important Note:** The final anomaly flag uses **global majority voting** across ALL channels from all observatories. While per-observatory voting is computed (and available in exported results), it is **not used for the final decision**. The final flag requires a majority of all channels (not per-observatory) to agree on an anomaly.

### Vote Counting

The system tracks:
- **Channel flags**: Boolean array per channel
- **Vote counts**: Integer array showing how many channels voted for each timestamp
- **Observatory flags**: Boolean array per observatory
- **Observatory votes**: Integer array per observatory

### Example Voting Scenario

```
Timestamps:     T1    T2    T3    T4    T5
Channel 1:      ✓     ✗     ✓     ✗     ✗
Channel 2:      ✗     ✗     ✓     ✓     ✗
Channel 3:      ✓     ✓     ✗     ✓     ✗
Channel 4:      ✗     ✗     ✓     ✗     ✗
Channel 5:      ✓     ✗     ✓     ✓     ✗

Votes:           3     1     4     3     0

With min_channels=3:
Final Flag:      ✓     ✗     ✓     ✓     ✗
```

---

## Visualization and Export

### Visualization Components

#### 1. B Components Plot
For each sensor, displays:
- **b_x**: Blue line
- **b_y**: Green line
- **b_z**: Orange line
- **Anomaly Regions**: Red shaded vertical rectangles
- **Anomaly Points**: Red 'x' markers

**Features:**
- Interactive Plotly charts (zoom, pan, hover)
- Grouped by observatory
- Full-width display
- Legend with horizontal layout

#### 2. Summary Statistics
- Total flagged timestamps
- Percentage of data flagged
- Number of channels processed
- Number of observatories detected

### Export Functionality

**CSV Export Includes:**
- All original detection data columns
- Per-channel flags: `{channel_name}_flag` (0/1)
- Per-observatory votes: `{obs_name}_votes` (integer) - computed for reference but not used for final decision
- Final flag: `final_cross_obs_flag` (0/1) - determined by **global majority voting** across all channels

**File Format:**
- CSV with comma separator
- Includes index (timestamp)
- All numeric flags as integers

### Session State Caching

Results are cached in Streamlit session state to:
- Avoid recomputation when parameters change
- Enable faster re-rendering
- Maintain state across widget interactions

**Cache Keys:**
- `inference_results`: Full inference results dictionary
- `inference_wide_detect`: Wide-format detection DataFrame

**Cache Invalidation:**
- When detection file changes (shape mismatch)
- When "Run inference" button is clicked again

---

## Configuration and Parameters

### File Configuration

#### Fixed File Paths Mode
- **Training CSV path**: Path to training data file
- **Detection CSV path**: Path to detection data file
- **Model directory**: Directory for saving/loading models (default: `saved_models`)

#### File Upload Mode
- Upload files directly through Streamlit interface
- Files are read into memory

### Model Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `window_size` | 64 | 8-2048 | Length of input sequences |
| `step_size` | window_size | 1-window_size | Stride between windows |
| `batch_size` | 64 | 8-1024 | Training batch size |
| `epochs` | 10 | 1-200 | Number of training epochs |
| `latent_dim` | 32 | 8-256 | LSTM bottleneck dimension |
| `resid_k` | 6.0 | 1.0-20.0 | MAD multiplier for threshold |

### Voting Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_absolute_voting` | True | Use absolute vs fraction-based voting |
| `min_channels_required` | 5 | Required channels for absolute voting |
| `vote_fraction` | 0.56 | Required fraction for fraction-based voting |

### Auto-Save

- **Auto-save trained models**: Automatically save models after training
- Saves to specified model directory
- Includes models, scalers, thresholds, and metadata

---

## Usage Guide

### Basic Workflow

1. **Prepare Data**
   - Ensure CSV files have required columns: `timestamp`, `sensor_id`, `b_x`, `b_y`, `b_z`
   - Timestamps should be in pandas-readable format

2. **Configure Parameters**
   - Set window size (should match expected anomaly duration)
   - Set step size (smaller = more overlap, smoother results)
   - Adjust voting parameters based on number of sensors

3. **Train Models** (Optional if models already exist)
   - Upload or specify training CSV path
   - Click "Start training per-channel models"
   - Wait for training to complete
   - Models are auto-saved if enabled

4. **Run Inference**
   - Upload or specify detection CSV path
   - Click "Run inference on detection file"
   - Review matched models count
   - Wait for inference to complete

5. **Review Results**
   - Examine B component plots
   - Check summary statistics
   - Zoom into specific time regions
   - Download results CSV

### Advanced Usage

#### Training on Subset of Sensors
- Filter training CSV to include only desired sensors
- Train models
- Use same models for inference on different sensors (will use axis-only matching)

#### Different Window Sizes
- Train models with different window sizes
- System automatically handles per-model window sizes during inference
- Adjust step sizes proportionally

#### Threshold Tuning
- Train models
- Manually adjust `resid_k` parameter
- Re-run inference to see effect
- For fine-tuning, modify thresholds in `thresholds.json` directly

#### Multi-Observatory Analysis
- Include multiple observatories in detection CSV
- System automatically detects and groups by observatory
- Review per-observatory votes in exported CSV

### Command Line Alternative

While this is a Streamlit app, you can run it with:

```bash
streamlit run magnavis_anomaly_maVo.py
```

Or with custom port:

```bash
streamlit run magnavis_anomaly_maVo.py --server.port 8502
```

---

## Technical Details

### Data Structures

#### Wide DataFrame Structure
```
Index: DatetimeIndex (unique, sorted)
Columns: 
  - {sensor_id}__b_x
  - {sensor_id}__b_y
  - {sensor_id}__b_z
  - ... (for each sensor)
Values: Float (magnetic field in nT, typically)
```

#### Channel Flags Dictionary
```python
{
  'SENSOR1__b_x': np.array([False, True, False, ...]),  # length = n_points
  'SENSOR1__b_y': np.array([False, False, True, ...]),
  ...
}
```

#### Inference Results Dictionary
```python
{
  'channel_flags': {channel: bool_array},
  'channel_errors': {channel: float_array},
  'obs_flags': {obs: bool_array},
  'obs_votes': {obs: int_array},
  'final_flag': bool_array,
  'n_points': int,
  'det_index': DatetimeIndex,
  'matched_pairs': [(model_key, detection_column), ...],
  'obs_prefixes': ['OBS1', 'OBS2', ...]
}
```

### Performance Considerations

#### Memory Usage
- Wide DataFrames: Can be large with many sensors and long time series
- Model Storage: Each model ~100KB-1MB depending on window_size
- Session State: Caches full results (can be memory-intensive)

#### Computational Complexity
- Training: O(n_channels × n_samples × epochs × window_size)
- Inference: O(n_channels × n_windows × window_size)
- Voting: O(n_channels × n_timestamps)

#### Optimization Tips
- Use non-overlapping windows (step_size = window_size) for faster inference
- Reduce window_size for faster training (if anomaly duration allows)
- Use smaller batch_size if memory constrained
- Clear session state if memory issues occur

### Dependencies

**Required Packages:**
- `streamlit`: Web interface
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: StandardScaler
- `scipy`: Statistical functions (MAD)
- `tensorflow`: LSTM models
- `plotly`: Interactive visualization

**Version Compatibility:**
- Python 3.7+
- TensorFlow 2.x
- Streamlit 1.0+

### Error Handling

The system includes comprehensive error handling:

- **File Not Found**: Warning message, graceful degradation
- **Missing Columns**: Error message with guidance
- **Insufficient Data**: Warning, skip channel
- **Model Loading Failure**: Warning, skip model
- **Mismatched Window Sizes**: Automatic adjustment with info message
- **No Models Available**: Warning, prevent inference

---

## Best Practices

### Data Preparation

1. **Clean Data Beforehand**
   - Remove obvious outliers
   - Handle missing values appropriately
   - Ensure consistent timestamp formats

2. **Sufficient Training Data**
   - At least 10× window_size samples per channel
   - More data = better model performance
   - Include diverse normal conditions

3. **Consistent Sensor Naming**
   - Use consistent sensor_id format
   - Include observatory prefix (OBS1, OBS2) if applicable
   - Avoid special characters in sensor names

### Model Training

1. **Window Size Selection**
   - Should match expected anomaly duration
   - Too small: May miss long anomalies
   - Too large: May smooth out short anomalies
   - Start with 64, adjust based on results

2. **Step Size Selection**
   - `step_size = window_size`: No overlap, faster
   - `step_size < window_size`: Overlap, smoother results
   - Overlap of 50% (step_size = window_size/2) is common

3. **Epochs**
   - Start with 10-20 epochs
   - Monitor validation loss (if added)
   - Avoid overfitting (model too specific to training data)

4. **Threshold Tuning**
   - Start with default k=6.0
   - Increase k for fewer false positives
   - Decrease k for fewer false negatives
   - Use labeled validation data if available

### Inference

1. **Model Matching**
   - Ensure sensor naming is consistent between training and detection
   - Review matched pairs count after inference
   - Manually verify critical sensor matches

2. **Voting Parameters**
   - For N sensors, typically require N/2 to N*0.6 agreement
   - Adjust based on sensor reliability
   - Use absolute voting if sensor count is fixed
   - Use fraction-based if sensor count varies

3. **Result Validation**
   - Review plots for obvious false positives/negatives
   - Check summary statistics (flag percentage)
   - Compare with known anomalies if available

### System Maintenance

1. **Model Versioning**
   - Use different model directories for different experiments
   - Include metadata in directory name (e.g., `models_v1_window64`)
   - Document parameter settings used for each model set

2. **Regular Retraining**
   - Retrain when sensor characteristics change
   - Retrain when new normal patterns emerge
   - Keep training data representative of current conditions

3. **Performance Monitoring**
   - Track inference time
   - Monitor memory usage
   - Log matched model counts

---

## Troubleshooting

### Common Issues

#### 1. "No models available"
**Cause:** No models in model directory or not trained yet.

**Solution:**
- Train models first, or
- Place saved models in model directory, or
- Check model directory path is correct

#### 2. "Not enough samples"
**Cause:** Time series shorter than window_size.

**Solution:**
- Reduce window_size, or
- Use longer time series, or
- Skip that channel

#### 3. "Models not matching detection columns"
**Cause:** Sensor naming mismatch between training and detection.

**Solution:**
- Check sensor_id format in both CSVs
- Review matching algorithm output
- Use consistent naming conventions
- Check for timestamp prefixes affecting matching

#### 4. "High false positive rate"
**Cause:** Threshold too low or insufficient training data.

**Solution:**
- Increase `resid_k` parameter
- Add more diverse training data
- Review and adjust thresholds manually

#### 5. "High false negative rate"
**Cause:** Threshold too high or anomalies similar to normal data.

**Solution:**
- Decrease `resid_k` parameter
- Increase model capacity (latent_dim)
- Review training data for anomaly contamination

#### 6. "Memory errors"
**Cause:** Large datasets or too many models.

**Solution:**
- Reduce batch_size
- Process data in chunks
- Clear session state
- Use smaller window_size

#### 7. "Inference results not updating"
**Cause:** Session state caching.

**Solution:**
- Click "Run inference" again
- Clear browser cache
- Restart Streamlit app

### Debugging Tips

1. **Enable Verbose Output**
   - Modify training to set `verbose=1` in model.fit()
   - Add print statements in inference loop

2. **Check Intermediate Results**
   - Print matched_pairs to verify matching
   - Print channel_flags to verify detection
   - Print vote counts to verify voting

3. **Validate Data**
   - Check wide DataFrame shape and columns
   - Verify no duplicate indices
   - Check for NaN values

4. **Model Inspection**
   - Load model and check input_shape
   - Verify scaler statistics
   - Check threshold values

---

## Future Enhancements

### Potential Improvements

1. **Early Stopping**: Add callback to prevent overfitting
2. **Learning Rate Scheduling**: Adaptive learning rate
3. **Attention Mechanisms**: For better temporal modeling
4. **Ensemble Methods**: Combine multiple models
5. **Online Learning**: Update models with new data
6. **Real-time Processing**: Stream data instead of batch
7. **Anomaly Scoring**: Continuous scores instead of binary flags
8. **Multi-scale Detection**: Detect anomalies at different time scales
9. **Transfer Learning**: Pre-trained models for new sensors
10. **Interactive Threshold Tuning**: Slider to adjust thresholds in real-time

### Code Improvements

1. **Type Hints**: Add type annotations throughout
2. **Unit Tests**: Comprehensive test suite
3. **Logging**: Structured logging instead of print statements
4. **Configuration File**: YAML/JSON config instead of sidebar
5. **Modularization**: Split into separate modules
6. **Documentation**: Inline docstrings for all functions
7. **Error Messages**: More descriptive error messages
8. **Progress Bars**: Better progress indication for long operations

---

## Conclusion

The MagNav Anomaly Detection System is a comprehensive solution for detecting anomalies in multi-sensor magnetic field data. Its modular architecture, robust data handling, and flexible voting mechanisms make it suitable for various use cases in magnetic navigation and geophysical research.

The system's strength lies in its ability to:
- Handle dynamic sensor configurations
- Automatically match models to detection data
- Combine results from multiple sensors and observatories using **global majority voting**
- Provide intuitive visualization and export capabilities

With proper data preparation, parameter tuning, and validation, the system can provide reliable anomaly detection for magnetic field monitoring applications.

---

## Appendix

### A. Glossary

- **Channel**: A sensor-component combination (e.g., SENSOR1__b_x)
- **Observatory**: A group of sensors (e.g., OBS1, OBS2)
- **Window**: A contiguous sequence of samples used for LSTM processing
- **MAD**: Median Absolute Deviation, a robust measure of variability
- **RMSE**: Root Mean Squared Error, reconstruction error metric
- **Wide Format**: DataFrame with sensors as columns
- **Long Format**: DataFrame with sensors as rows
- **Global Majority Voting**: The final anomaly decision mechanism that requires a majority of all channels (across all observatories) to agree on an anomaly. This is different from per-observatory voting, which is computed for reference but not used for the final decision.

### B. References

- LSTM Autoencoders for Anomaly Detection: [Various papers on time series anomaly detection]
- MAD Threshold: Robust Statistics literature
- Streamlit Documentation: https://docs.streamlit.io
- TensorFlow/Keras Documentation: https://www.tensorflow.org

### C. Contact and Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Script Version**: magnavis_anomaly_maVo.py (897 lines)

