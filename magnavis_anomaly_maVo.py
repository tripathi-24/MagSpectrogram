# magnavis_streamlit_app_final.py
"""
MagNav — Final Streamlit app (dynamic sensors & observatories)

OVERVIEW:
This application detects anomalies in magnetic field data using LSTM autoencoders.
The system works in two phases:
1. TRAINING: Learns normal patterns from training data (optional if models already exist)
2. INFERENCE: Detects anomalies in new data by reconstructing it and measuring errors

KEY CONCEPT: LSTM Autoencoder RECONSTRUCTS input data (not predicts future values).
High reconstruction error = Anomaly detected!

Features:
 - Safe wide-format builder (duplicates removed, common timeline)
 - Per-channel LSTM-AE training with scalers & thresholds
 - Inference with full-length anomaly timeline (window expansion)
 - Majority voting per observatory (strict majority threshold)
 - Global majority voting across all channels (final decision)
 - Intuitive visualization using Plotly (shaded anomaly regions + residuals)
 - Save/load models, scalers, thresholds, metadata
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, io, pickle, json, re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import median_abs_deviation
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="MagNav — Final")

# -------------------- Utilities --------------------

def ensure_datetime(df, ts_col='timestamp'):
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])
    return df

def safe_build_global_index(df):
    """Return sorted unique timestamps from df as DatetimeIndex."""
    idx = df['timestamp'].drop_duplicates().sort_values()
    # If index has timezone-naive vs tz-aware issues, normalize to naive
    try:
        idx = pd.DatetimeIndex(idx)
    except Exception:
        idx = pd.to_datetime(idx).tz_localize(None)
    # Ensure uniqueness (in case drop_duplicates didn't work due to precision issues)
    idx = idx.drop_duplicates()
    return idx

def build_wide_df_from_df(df):
    """
    Build safe wide-format DataFrame from long-format data with columns:
    ['timestamp','sensor_id','b_x','b_y','b_z', ...]
    
    CONVERTS FROM LONG TO WIDE FORMAT:
    Long format: Each row is one sensor at one timestamp
    Wide format: Each column is one sensor-component, rows are timestamps
    
    WHY WIDE FORMAT?
    - Each column (e.g., SENSOR1__b_x) becomes a separate time series
    - One LSTM model is trained per column
    - Makes it easy to process each sensor-component independently
    
    Ensures:
     - Unique global timestamp index (all sensors aligned to same timeline)
     - Duplicate timestamps within sensor are removed (keep first)
     - All sensors reindexed to the global index (NaNs for missing), then forward/backfilled
    """
    df = df.copy()
    df = ensure_datetime(df, 'timestamp')
    df = df.sort_values('timestamp')
    global_index = safe_build_global_index(df)

    parts = []
    sensors = sorted(df['sensor_id'].unique())
    for sid in sensors:
        g = df[df['sensor_id'] == sid].copy()
        # Drop duplicates on timestamp for the sensor (keep first)
        g = g.drop_duplicates(subset='timestamp', keep='first').sort_values('timestamp')
        # Keep numeric components only
        numeric_cols = [c for c in ['b_x','b_y','b_z'] if c in g.columns]
        if not numeric_cols:
            continue
        g = g.set_index('timestamp')[numeric_cols]
        # Handle any remaining duplicate timestamps by grouping and taking mean
        if not g.index.is_unique:
            g = g.groupby(g.index).mean()
        # Reindex to global index (align)
        g = g.reindex(global_index)
        # Ensure the reindexed DataFrame has a unique index (group by index if needed)
        if not g.index.is_unique:
            g = g.groupby(g.index).mean()
        # Rename columns to include sensor id
        g = g.rename(columns={c: f"{sid}__{c}" for c in g.columns})
        parts.append(g)

    if not parts:
        return None

    # Ensure all parts have unique indices before concatenation
    for i, part in enumerate(parts):
        if not part.index.is_unique:
            parts[i] = part.groupby(part.index).mean()
    
    wide = pd.concat(parts, axis=1)
    # Ensure the final DataFrame has a unique index
    if not wide.index.is_unique:
        wide = wide.groupby(wide.index).mean()
    # Fill small gaps from forward then backward (so models have contiguous values)
    wide = wide.ffill().bfill()
    wide.index.name = 'timestamp'
    return wide

def create_windows(series, window_size, step):
    """
    Create sliding windows from a time series for LSTM processing.
    
    WHY WINDOWS? LSTM needs sequences (not single values). The model was trained
    on windows of consecutive samples, so we must process data the same way.
    
    Example:
        series = [1, 2, 3, 4, 5, 6, 7, 8]
        window_size = 3, step = 2
        Returns:
        X = [[1,2,3], [3,4,5], [5,6,7]]  ← Overlapping windows
        idxs = [(0,2), (2,4), (4,6)]     ← Original indices for each window
    """
    X = []
    idxs = []
    n = len(series)
    for start in range(0, n - window_size + 1, step):
        X.append(series[start:start+window_size])
        idxs.append((start, start+window_size-1))
    return np.array(X), idxs

def build_lstm_autoencoder(window_size, latent_dim=32):
    """
    Build an LSTM Autoencoder for time series anomaly detection.
    
    ARCHITECTURE EXPLANATION:
    This is an AUTOENCODER (not a predictor). It learns to RECONSTRUCT input data.
    
    Encoder (Compression):
    - Input: [window_size, 1] - Your time series window
    - LSTM(64) → LSTM(32) → Latent vector (32 dimensions)
    - Compresses input to a compact representation
    
    Decoder (Reconstruction):
    - Latent vector → RepeatVector → LSTM(32) → LSTM(64) → Output
    - Reconstructs the original window from the compressed representation
    
    HOW IT DETECTS ANOMALIES:
    - During training: Learns to reconstruct NORMAL patterns well
    - During inference: 
      * Normal data → Reconstructs well → Low error → No anomaly
      * Anomalous data → Struggles to reconstruct → High error → Anomaly!
    
    The model does NOT predict future values - it reconstructs the input you give it.
    """
    inp = layers.Input((window_size, 1))
    # Encoder: Compress input to latent representation
    x = layers.LSTM(64, activation='tanh', return_sequences=True)(inp)
    x = layers.LSTM(32, activation='tanh', return_sequences=False)(x)  # Latent vector
    # Decoder: Reconstruct from latent representation
    x = layers.RepeatVector(window_size)(x)  # Repeat latent vector for each time step
    x = layers.LSTM(32, activation='tanh', return_sequences=True)(x)
    x = layers.LSTM(64, activation='tanh', return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(1))(x)  # Output: reconstructed window
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error: measures reconstruction quality
    return model

def compute_threshold(window_errors, k=6.0):
    """
    Compute anomaly threshold using Median Absolute Deviation (MAD) method.
    
    WHAT IS THIS?
    During training, we compute reconstruction errors for all normal data windows.
    This function finds the "maximum normal error" - anything above this is anomalous.
    
    FORMULA: threshold = median(error) + k × MAD(error)
    
    WHY MAD?
    - More robust to outliers than standard deviation
    - Works well with non-normal distributions
    - Commonly used in anomaly detection
    
    PARAMETER k:
    - Controls sensitivity (higher k = fewer false positives, more false negatives)
    - Default: 6.0 (approximately 3 standard deviations in normal distribution)
    - Tune based on your data if you have labeled anomalies
    """
    med = np.median(window_errors)
    mad = median_abs_deviation(window_errors)
    return float(med + k * mad)

def expand_window_flags_to_full(length, window_idxs, window_flags):
    """
    Expand window-level anomaly flags to full-length time series.
    
    PROBLEM: LSTM processes data in windows, producing one flag per window.
    But we need flags for EVERY timestamp in the original data.
    
    SOLUTION: Mark all timestamps within an anomalous window as anomalous.
    
    Example:
        length = 10
        window_idxs = [(0,2), (3,5), (6,8)]
        window_flags = [False, True, False]  ← Window 2 is anomalous
        
        Result: [False, False, False, True, True, True, False, False, False, False]
                 └─window1─┘  └──window2──┘  └─window3─┘
                 (normal)      (anomalous)    (normal)
    """
    flags = np.zeros(length, dtype=bool)
    for (start, end), f in zip(window_idxs, window_flags):
        if f:  # If this window is flagged as anomalous
            s = max(0, start)
            e = min(length-1, end)
            flags[s:e+1] = True  # Mark all timestamps in this window as anomalous
    return flags

def find_anomaly_regions(flag_bool):
    """Find continuous anomaly regions from boolean flag array. Returns list of (start_idx, end_idx) tuples."""
    regions = []
    in_reg = False
    start_idx = None
    for i, v in enumerate(flag_bool):
        if v and not in_reg:
            in_reg = True
            start_idx = i
        elif (not v) and in_reg:
            in_reg = False
            regions.append((start_idx, i-1))
    if in_reg:
        regions.append((start_idx, len(flag_bool)-1))
    return regions

def majority_vote(channel_flags_dict, vote_fraction=None, min_channels=None):
    """
    Majority voting function to combine results from multiple channels.
    
    HOW IT WORKS:
    - Each channel votes (True/False) for each timestamp
    - Count votes: How many channels say "anomaly" for each timestamp?
    - If votes >= required threshold → Final flag = Anomaly
    
    VOTING MODES:
    1. Absolute voting: Requires fixed number of channels (e.g., 5 out of 10)
    2. Fraction-based: Requires fraction of channels (e.g., 0.56 = 6 out of 10)
    
    Args:
        channel_flags_dict: dict of channel -> boolean array (True = anomaly, False = normal)
        vote_fraction: fraction of channels required (if None, uses min_channels)
        min_channels: absolute number of channels required (if None, uses vote_fraction)
    Returns:
        (voted_bool, votes) tuple
        - voted_bool: Final anomaly flags (True/False for each timestamp)
        - votes: Vote counts (how many channels voted for anomaly at each timestamp)
    """
    keys = sorted(channel_flags_dict.keys())
    if len(keys) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
    stacked = np.vstack([channel_flags_dict[k].astype(int) for k in keys])
    votes = stacked.sum(axis=0)
    n_channels = stacked.shape[0]
    
    if min_channels is not None:
        # Absolute voting: require at least min_channels
        required = min_channels
    else:
        # Fraction-based voting
        required = int(np.ceil(n_channels * vote_fraction))
    
    return votes >= required, votes

def safe_save_models(models_store, scalers_store, thresholds_store, meta, model_dir):
    """
    Save trained models, scalers, thresholds, and metadata to disk.
    
    WHAT IS SAVED:
    - Models: LSTM autoencoder models (one per channel) - .h5 format
    - Scalers: Normalization parameters (CRITICAL for inference!) - .pkl format
    - Thresholds: Anomaly detection thresholds (from training data) - .json format
    - Metadata: Window size, step size, latent dim - .json format
    
    WHY SAVE SCALERS?
    During inference, you MUST use the same scaler that was used during training.
    Otherwise, the model sees data in a different scale and detection breaks!
    
    Directory structure:
    saved_models/
    ├── model_SENSOR1__b_x.h5
    ├── model_SENSOR1__b_y.h5
    ├── scalers.pkl
    ├── thresholds.json
    └── metadata.json
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # Save models (one file per channel)
    for ch, model in models_store.items():
        safe = ch.replace('/', '_').replace('\\', '_')  # Make filename-safe
        path = os.path.join(model_dir, f"model_{safe}.h5")
        model.save(path)
    # Save scalers & thresholds & meta (shared across all channels)
    with open(os.path.join(model_dir, "scalers.pkl"), 'wb') as f:
        pickle.dump(scalers_store, f)
    with open(os.path.join(model_dir, "thresholds.json"), 'w') as f:
        json.dump(thresholds_store, f)
    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f)

def safe_load_models(model_dir):
    """
    Load pre-trained models, scalers, thresholds, and metadata from disk.
    
    WHEN IS THIS CALLED?
    - At app startup: Loads existing models from "saved_models" folder
    - Allows you to skip training if models already exist
    
    WHAT IS LOADED:
    - Models: Pre-trained LSTM autoencoders (one per channel)
    - Scalers: Normalization parameters (must match training!)
    - Thresholds: Anomaly detection thresholds (from training)
    - Metadata: Window size, step size, etc. (for proper inference)
    
    RETURNS:
    - models_store: Dictionary of {channel_name: model}
    - scalers_store: Dictionary of {channel_name: scaler}
    - thresholds_store: Dictionary of {channel_name: threshold}
    - meta: Dictionary with window_size, step_size, latent_dim
    """
    models_store = {}
    scalers_store = {}
    thresholds_store = {}
    meta = {}
    if not os.path.exists(model_dir):
        return models_store, scalers_store, thresholds_store, meta

    # Load scalers, thresholds, meta if present
    sc_path = os.path.join(model_dir, "scalers.pkl")
    thr_path = os.path.join(model_dir, "thresholds.json")
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(sc_path):
        with open(sc_path, 'rb') as f:
            scalers_store = pickle.load(f)
    if os.path.exists(thr_path):
        with open(thr_path, 'r') as f:
            thresholds_store = json.load(f)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    # Load models (all .h5 files starting with "model_")
    for fn in os.listdir(model_dir):
        if fn.startswith("model_") and fn.endswith(".h5"):
            safe_name = fn[len("model_"):-3]  # Remove "model_" prefix and ".h5" suffix
            p = os.path.join(model_dir, fn)
            try:
                m = tf.keras.models.load_model(p, compile=False)
                m.compile(optimizer='adam', loss='mse')  # Recompile for inference
                models_store[safe_name] = m
            except Exception:
                try:
                    m = tf.keras.models.load_model(p)  # Try loading with default compile
                    models_store[safe_name] = m
                except Exception:
                    continue  # Skip if model can't be loaded
    return models_store, scalers_store, thresholds_store, meta

# -------------------- Streamlit UI --------------------

# Initialize session state for inference results
if 'inference_results' not in st.session_state:
    st.session_state['inference_results'] = None
if 'inference_wide_detect' not in st.session_state:
    st.session_state['inference_wide_detect'] = None

st.title("MagNav — Final Anomaly Detector (dynamic sensors/obs)")

st.markdown("""
**How to use**
1. Upload TRAIN CSV (optional) — used to train per-channel models (if you already have models, you can skip training).
2. Upload DETECTION CSV — the app will run inference (can include multiple observatories).
3. Use the controls to set window size, step, thresholds, and vote fraction.
4. Visualize and download results.
""")

# Sidebar controls
st.sidebar.header("Files")
use_fixed_files = st.sidebar.checkbox("Use fixed file paths", value=True, help="If checked, uses fixed file paths instead of upload")
if use_fixed_files:
    train_file_path = st.sidebar.text_input("Training CSV path", value="Data_AllSensors_10Oct_to_15Oct1030h_Downsample_60.csv")
    detect_file_path = st.sidebar.text_input("Detection CSV path", value="Data_relevant_0900_Onwards.csv")
    train_file = train_file_path if os.path.exists(train_file_path) else None
    detect_file = detect_file_path if os.path.exists(detect_file_path) else None
    if train_file is None:
        st.sidebar.warning(f"Training file not found: {train_file_path}")
    if detect_file is None:
        st.sidebar.warning(f"Detection file not found: {detect_file_path}")
else:
    train_file = st.sidebar.file_uploader("Upload TRAIN CSV (for training)", type=['csv'], key='train')
    detect_file = st.sidebar.file_uploader("Upload DETECTION CSV (for inference)", type=['csv'], key='detect')

st.sidebar.markdown("---")
st.sidebar.header("Model params")
window_size = st.sidebar.number_input("Window size (samples)", min_value=8, max_value=2048, value=64, step=8)
step_size = st.sidebar.number_input("Step size (samples)", min_value=1, max_value=window_size, value=window_size, step=1)
batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=1024, value=64, step=8)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=200, value=10, step=1)
latent_dim = st.sidebar.number_input("LSTM latent dim", min_value=8, max_value=256, value=32, step=8)
resid_k = st.sidebar.number_input("MAD multiplier k", min_value=1.0, max_value=20.0, value=6.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Voting & models")
use_absolute_voting = st.sidebar.checkbox("Use absolute voting (require N channels)", value=True, help="If checked, requires at least N channels to vote. If unchecked, uses fraction-based voting.")
if use_absolute_voting:
    min_channels_required = st.sidebar.number_input("Minimum channels required for anomaly", min_value=1, max_value=20, value=5, step=1, help="Anomaly is flagged if at least this many channels vote for it")
    vote_fraction = None
else:
    vote_fraction = st.sidebar.slider("Majority vote fraction (Option C)", min_value=0.1, max_value=1.0, value=0.56)
    min_channels_required = None
model_dir = st.sidebar.text_input("Model directory", value="saved_models")
auto_save = st.sidebar.checkbox("Auto-save trained models", value=True)

# Load pre-trained models if they exist in the model directory
# This allows you to skip training and go straight to inference
models_store, scalers_store, thresholds_store, meta_loaded = safe_load_models(model_dir)
if len(models_store) > 0:
    st.sidebar.success(f"Loaded {len(models_store)} models from {model_dir}")

# -------------------- Load CSVs --------------------
"""
DATA LOADING:
=============

CSV FORMAT REQUIREMENTS:
- Required columns: timestamp, sensor_id, b_x, b_y, b_z
- Long format: Each row = one sensor at one timestamp
- Multiple sensors can be in the same CSV

TRAINING CSV (optional):
- Used to train LSTM models
- Should contain NORMAL data only (no anomalies)
- Models learn: "What does normal magnetic field data look like?"

DETECTION CSV (required for inference):
- Your new data to check for anomalies
- Can include different sensors than training (system will try to match)
- Can include multiple observatories
"""

train_df = None
if train_file is not None:
    try:
        if isinstance(train_file, str):  # File path
            train_df = pd.read_csv(train_file)
        else:  # File object (uploaded)
            train_df = pd.read_csv(train_file)
        train_df = ensure_datetime(train_df, 'timestamp')
        st.write("Training CSV loaded — shape:", train_df.shape)
        if 'sensor_id' not in train_df.columns:
            st.error("TRAIN CSV must contain a 'sensor_id' column.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading training CSV: {e}")
        train_df = None

detect_df = None
if detect_file is not None:
    try:
        if isinstance(detect_file, str):  # File path
            detect_df = pd.read_csv(detect_file)
        else:  # File object (uploaded)
            detect_df = pd.read_csv(detect_file)
        detect_df = ensure_datetime(detect_df, 'timestamp')
        st.write("Detection CSV loaded — shape:", detect_df.shape)
        if 'sensor_id' not in detect_df.columns:
            st.error("DETECTION CSV must contain a 'sensor_id' column.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading detection CSV: {e}")
        detect_df = None

# -------------------- Build wide dataframes --------------------
"""
DATA FORMAT CONVERSION:
=======================

Converts from LONG format to WIDE format:

LONG FORMAT (input):
timestamp          sensor_id    b_x    b_y    b_z
2025-10-08 10:00  SENSOR1      100.5  200.3  300.1
2025-10-08 10:01  SENSOR1      101.2  201.0  301.5
2025-10-08 10:00  SENSOR2      150.2  250.1  350.3

WIDE FORMAT (output):
timestamp          SENSOR1__b_x  SENSOR1__b_y  SENSOR1__b_z  SENSOR2__b_x  ...
2025-10-08 10:00  100.5         200.3         300.1         150.2         ...
2025-10-08 10:01  101.2         201.0         301.5         150.2         ...

WHY WIDE FORMAT?
- Each column becomes a separate time series
- One LSTM model per column (e.g., one model for SENSOR1__b_x)
- Makes it easy to process each sensor-component independently
"""

wide_train = None
if train_df is not None:
    wide_train = build_wide_df_from_df(train_df)
    if wide_train is None:
        st.error("TRAIN CSV: no numeric channels found (b_x/b_y/b_z).")
        st.stop()
    st.subheader("Training data (wide) preview")
    st.write("Columns (train):", wide_train.columns.tolist()[:20])
    st.dataframe(wide_train.head(60))

wide_detect = None
if detect_df is not None:
    wide_detect = build_wide_df_from_df(detect_df)
    if wide_detect is None:
        st.error("DETECT CSV: no numeric channels found (b_x/b_y/b_z).")
        st.stop()
    st.subheader("Detection data (wide) preview")
    st.write("Columns (detect):", wide_detect.columns.tolist()[:30])
    st.dataframe(wide_detect.head(60))

# -------------------- TRAINING --------------------
"""
TRAINING WORKFLOW EXPLANATION:
==============================

Training is OPTIONAL if you already have pre-trained models in "saved_models" folder.

WHAT HAPPENS DURING TRAINING:
1. For each channel (sensor-component combination, e.g., SENSOR1__b_x):
   a. Extract time series from training CSV
   b. Fit StandardScaler (normalizes data to mean=0, std=1)
   c. Create windows (split into sequences)
   d. Build LSTM autoencoder
   e. Train model to RECONSTRUCT normal data
   f. Compute reconstruction errors on training data
   g. Compute threshold (maximum "normal" error)
   h. Save: model, scaler, threshold

KEY INSIGHT: Model learns to reconstruct NORMAL patterns only.
During inference, it will struggle to reconstruct anomalies → High error → Anomaly detected!

The scaler is CRITICAL - it must be saved and reused during inference!
"""

st.markdown("---")
st.header("Training per-channel LSTM-AE (optional)")

train_btn = st.button("Start training per-channel models")

if train_btn:
    if wide_train is None:
        st.error("Please upload TRAIN CSV to train models.")
    else:
        channels = wide_train.columns.tolist()
        st.info(f"Training {len(channels)} channels.")
        new_models = {}
        new_scalers = {}
        new_thresholds = {}
        pb = st.progress(0)
        for i, ch in enumerate(channels):
            st.write(f"[{i+1}/{len(channels)}] {ch}")
            # Extract time series for this channel (e.g., SENSOR1__b_x values)
            series = wide_train[ch].dropna().values.reshape(-1,1)
            if len(series) < window_size:
                st.warning(f"Not enough samples for {ch}. Need >= {window_size}. Skipping.")
                continue
            
            # Fit scaler on training data - this will be saved and reused during inference
            # CRITICAL: Same scaler must be used for inference to maintain consistent normalization
            scaler = StandardScaler()
            scaler.fit(series)
            s_series = scaler.transform(series).flatten()
            
            # Create windows for training - LSTM needs sequences
            X_train, _ = create_windows(s_series, window_size, step_size)
            if len(X_train) < 10:
                st.warning(f"Not enough windows for {ch} after windowing. Skipping.")
                continue
            X_train = X_train.reshape((-1, window_size, 1))
            
            # Build LSTM autoencoder - this will learn to RECONSTRUCT normal patterns
            model = build_lstm_autoencoder(window_size, latent_dim=int(latent_dim))
            
            # Train the model to reconstruct normal data
            # Input = X_train, Target = X_train (same data!)
            # The model learns: "Given this input, I should output the same thing"
            with st.spinner(f"Training {ch} ..."):
                model.fit(X_train, X_train, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.05, verbose=0)
            
            # Compute reconstruction errors on training data
            # This tells us: "What's the normal reconstruction error?"
            preds = model.predict(X_train)
            window_rmse = np.sqrt(np.mean((X_train.reshape(X_train.shape[0], -1) - preds.reshape(preds.shape[0], -1))**2, axis=1))
            
            # Compute threshold: maximum "normal" reconstruction error
            # During inference, errors above this threshold indicate anomalies
            thr = compute_threshold(window_rmse, k=float(resid_k))
            new_models[ch] = model
            new_scalers[ch] = scaler
            new_thresholds[ch] = thr
            pb.progress((i+1)/len(channels))
        # Merge newly trained models into the stores (combines with any pre-loaded models)
        models_store.update(new_models)
        scalers_store.update(new_scalers)
        thresholds_store.update(new_thresholds)
        
        # Save models to disk if auto-save is enabled
        # This allows you to use these models later without retraining
        if auto_save and len(new_models) > 0:
            meta = {"window_size": int(window_size), "step_size": int(step_size), "latent_dim": int(latent_dim)}
            try:
                safe_save_models(models_store, scalers_store, thresholds_store, meta, model_dir)
                st.success(f"Saved {len(new_models)} models to {model_dir}")
            except Exception as e:
                st.error(f"Error saving models: {e}")
        st.success("Training complete.")

# -------------------- INFERENCE --------------------
"""
INFERENCE WORKFLOW EXPLANATION:
================================

When you load a new CSV file for inference, here's what happens:

1. LOAD PRE-TRAINED MODELS
   - Loads LSTM models, scalers, and thresholds from "saved_models" folder
   - These models learned normal patterns from training data

2. MATCH MODELS TO YOUR DATA
   - Tries to match pre-trained models to columns in your new CSV
   - Uses smart matching (handles different sensor ID formats)

3. FOR EACH MATCHED CHANNEL:
   a. Extract your new data (e.g., b_x values from SENSOR1)
   b. Normalize using SAME scaler from training (critical!)
   c. Create windows (split into sequences of 64 samples)
   d. LSTM RECONSTRUCTS each window (key step!)
   e. Calculate reconstruction error (RMSE between original and reconstructed)
   f. Compare error with threshold → Flag as anomaly if error > threshold
   g. Expand window flags to full timeline

4. COMBINE ALL CHANNELS
   - Global majority voting across all channels
   - Final anomaly flag requires majority of channels to agree

KEY INSIGHT: The LSTM RECONSTRUCTS (recreates) your input data, not predicts future values.
High reconstruction error = Model couldn't recreate it = Anomaly detected!
"""

st.markdown("---")
st.header("Inference & Anomaly Detection")

if wide_detect is None:
    st.info("Upload a detection CSV to run inference (or place models in the model directory and upload detection file).")
    # Clear inference results if no detection file
    st.session_state['inference_results'] = None
    st.session_state['inference_wide_detect'] = None
else:
    # Check if we have cached results for this detection file
    use_cached = (st.session_state['inference_results'] is not None and 
                  st.session_state['inference_wide_detect'] is not None and
                  st.session_state['inference_wide_detect'].shape == wide_detect.shape)
    
    run_inf = st.button("Run inference on detection file")
    if run_inf:
        if len(models_store) == 0:
            st.warning("No models available. Train models or put saved models in the model directory.")
        else:
            st.info("Running inference...")
            
            # ============================================================
            # STEP 1: Determine window_size and step_size for inference
            # ============================================================
            # These must match the models' training parameters for correct inference
            # Priority: metadata > model input shape > sidebar value
            inference_window_size = meta_loaded.get('window_size', None)
            inference_step_size = meta_loaded.get('step_size', None)
            
            if inference_window_size is None and len(models_store) > 0:
                # Try to get window_size from first model's input shape
                first_model = list(models_store.values())[0]
                if hasattr(first_model, 'input_shape') and first_model.input_shape:
                    # input_shape is (None, window_size, 1) or (batch, window_size, 1)
                    if len(first_model.input_shape) >= 2:
                        inference_window_size = int(first_model.input_shape[1])
            
            if inference_window_size is None:
                inference_window_size = window_size  # Fallback to sidebar value
                st.warning(f"⚠️ Could not determine model's window_size. Using sidebar value: {inference_window_size}")
            else:
                st.info(f"📐 Using window_size={inference_window_size} from loaded model metadata")
            
            if inference_step_size is None:
                inference_step_size = step_size  # Fallback to sidebar value
                if inference_window_size != window_size:
                    # If we're using a different window_size, adjust step_size proportionally
                    inference_step_size = int(inference_step_size * (inference_window_size / window_size))
                    st.info(f"📐 Adjusted step_size to {inference_step_size} to match window_size")

            det_index = wide_detect.index
            n_points = len(det_index)

            # ============================================================
            # STEP 2: Match pre-trained models to your detection columns
            # ============================================================
            # The system tries to match models (trained on old data) to your new data columns
            # Uses smart matching to handle different sensor ID formats
            det_cols = list(wide_detect.columns)  # Columns in your new CSV
            model_keys = list(models_store.keys())  # Pre-trained model names

            matched_pairs = []
            used_det = set()

            # exact match
            for mk in model_keys:
                if mk in det_cols:
                    matched_pairs.append((mk, mk))
                    used_det.add(mk)

            # safe match (safe mk -> det col)
            for mk in model_keys:
                if any(mk == p[0] for p in matched_pairs): continue
                safe_mk = mk.replace('/', '_').replace('\\', '_')
                for dc in det_cols:
                    if dc == safe_mk and dc not in used_det:
                        matched_pairs.append((mk, dc))
                        used_det.add(dc)
                        break

            # Extract meaningful sensor identifier (observatory + sensor number) ignoring timestamp prefix
            def extract_sensor_key(sensor_id):
                """
                Extract the meaningful part of sensor ID for matching.
                Examples:
                - 'S20251008_105355_44180345587365_1' -> ('_1', 1)  # (key, sensor_number)
                - 'S20251107_154804_OBS1_1' -> ('OBS1_1', 1)
                - 'S20251008_105355_44180345587365_2' -> ('_2', 2)
                Returns tuple: (sensor_key, sensor_number)
                """
                # Try to find OBS pattern first (most reliable)
                obs_match = re.search(r'(OBS\d+)_(\d+)', sensor_id.upper())
                if obs_match:
                    return (obs_match.group(1) + '_' + obs_match.group(2), int(obs_match.group(2)))
                
                # Try to find trailing number pattern (like _1, _2, _3)
                num_match = re.search(r'_(\d+)$', sensor_id)
                if num_match:
                    sensor_num = int(num_match.group(1))
                    return ('_' + num_match.group(1), sensor_num)
                
                # Fallback: return last part after last underscore
                parts = sensor_id.split('_')
                if len(parts) > 1:
                    try:
                        sensor_num = int(parts[-1])
                        return ('_' + parts[-1], sensor_num)
                    except ValueError:
                        pass
                
                return (sensor_id, None)
            
            # Helper to check if two sensor keys match (handles both OBS1_1 and _1 patterns)
            def sensor_keys_match(key1, key2, num1, num2):
                """Check if two sensor keys match, considering both OBS pattern and numeric suffix"""
                # If both have sensor numbers, match by number
                if num1 is not None and num2 is not None:
                    if num1 == num2:
                        return True
                
                # Direct key match
                if key1 == key2:
                    return True
                
                # Check if one contains the other (e.g., "OBS1_1" contains "_1")
                if key1 in key2 or key2 in key1:
                    return True
                
                return False
            
            # heuristic match by axis & sensor key (observatory + sensor number)
            def extract_sensor_axis(name):
                if '__' in name:
                    sensor, axis = name.split('__', 1)
                    return sensor, axis
                # otherwise try suffix
                for ax in ['b_x','b_y','b_z']:
                    if name.endswith(ax):
                        sensor = name[:-len(ax)-2]
                        return sensor, ax
                return None, None

            # Match by axis and sensor key (ignoring timestamp prefix)
            for mk in model_keys:
                if any(mk == p[0] for p in matched_pairs): continue
                sensor, axis = extract_sensor_axis(mk)
                if sensor is None or axis is None: continue
                
                # Extract sensor key (meaningful part for matching)
                mk_sensor_key, mk_sensor_num = extract_sensor_key(sensor)
                
                # try find detection column with same axis and matching sensor key
                for dc in det_cols:
                    if dc in used_det: continue
                    if '__' not in dc: continue
                    dsensor, daxis = dc.split('__', 1)
                    if daxis != axis: continue
                    
                    # Extract sensor key from detection column
                    dc_sensor_key, dc_sensor_num = extract_sensor_key(dsensor)
                    
                    # Match if sensor keys match (ignoring timestamp prefix)
                    if sensor_keys_match(mk_sensor_key, dc_sensor_key, mk_sensor_num, dc_sensor_num):
                        matched_pairs.append((mk, dc))
                        used_det.add(dc)
                        break

            # axis-only fallback (match by axis only, use first available sensor)
            for mk in model_keys:
                if any(mk == p[0] for p in matched_pairs): continue
                _, axis = extract_sensor_axis(mk)
                if axis is None: continue
                for dc in det_cols:
                    if dc in used_det: continue
                    if '__' in dc and dc.split('__',1)[1] == axis:
                        matched_pairs.append((mk, dc))
                        used_det.add(dc)
                        break

            st.write(f"Matched {len(matched_pairs)}/{len(model_keys)} models to detection columns.")

            # ============================================================
            # STEP 3: Run inference for each matched model-column pair
            # ============================================================
            # This is the core inference loop - processes each channel independently
            channel_flags = {}  # Will store anomaly flags for each channel
            channel_errors = {}  # Will store reconstruction errors for each channel
            progress_bar = st.progress(0)
            total_pairs = len(matched_pairs)
            
            for idx, (mk, dc) in enumerate(matched_pairs):
                # mk = model key (e.g., "SENSOR1__b_x")
                # dc = detection column (e.g., "SENSOR1__b_x")
                progress_bar.progress((idx + 1) / total_pairs)
                
                # Get the pre-trained model for this channel
                model = models_store.get(mk, None)
                if model is None:
                    # try safe name
                    model = models_store.get(mk.replace('/', '_').replace('\\', '_'), None)
                if model is None:
                    st.warning(f"Model {mk} not found in memory. Skipping.")
                    channel_flags[mk] = np.zeros(n_points, dtype=bool)
                    channel_errors[mk] = np.zeros(n_points, dtype=float)
                    continue
                
                # Determine window_size for THIS specific model
                model_window_size = inference_window_size  # Start with global value
                if hasattr(model, 'input_shape') and model.input_shape:
                    # input_shape is (None, window_size, 1) or (batch, window_size, 1)
                    if len(model.input_shape) >= 2:
                        model_window_size = int(model.input_shape[1])
                        if model_window_size != inference_window_size:
                            st.info(f"Model {mk} uses window_size={model_window_size} (different from global {inference_window_size})")
                
                # Determine step_size for this model
                model_step_size = inference_step_size
                if model_window_size != inference_window_size:
                    # Adjust step_size proportionally
                    model_step_size = int(inference_step_size * (model_window_size / inference_window_size))
                    if model_step_size < 1:
                        model_step_size = 1
                
                # ============================================================
                # STEP 3a: Extract your new data from the detection CSV
                # ============================================================
                # This is YOUR actual data that you want to check for anomalies
                series = wide_detect[dc].dropna().values.reshape(-1,1)
                if len(series) < model_window_size:
                    st.warning(f"Not enough samples for {dc}. Need >= {model_window_size}. Skipping and marking as no anomaly.")
                    channel_flags[mk] = np.zeros(n_points, dtype=bool)
                    channel_errors[mk] = np.zeros(n_points, dtype=float)
                    continue

                # ============================================================
                # STEP 3b: Normalize using the SAME scaler from training
                # ============================================================
                # CRITICAL: Must use the same normalization as training!
                # The model learned patterns in normalized space (mean=0, std=1)
                # Your new data must be in the same scale, otherwise detection breaks
                scaler = None
                if mk in scalers_store:
                    scaler = scalers_store[mk]  # Use saved scaler from training (preferred)
                elif dc in scalers_store:
                    scaler = scalers_store[dc]  # Try detection column name
                else:
                    # Fallback: create new scaler (not optimal - should use training scaler)
                    scaler = StandardScaler()
                    scaler.fit(series)
                    st.info(f"Fitted ad-hoc scaler for {mk} using detection series (not optimal).")

                # Normalize your new data to match training scale
                s_series = scaler.transform(series).flatten()
                
                # ============================================================
                # STEP 3c: Create windows from your time series
                # ============================================================
                # LSTM needs sequences (windows), not individual values
                # Split your data into windows of consecutive samples
                X_test, idxs = create_windows(s_series, model_window_size, model_step_size)
                if len(X_test) == 0:
                    st.warning(f"No windows created for {dc}. Skipping.")
                    channel_flags[mk] = np.zeros(n_points, dtype=bool)
                    channel_errors[mk] = np.zeros(n_points, dtype=float)
                    continue
                # Reshape for LSTM: (n_windows, window_size, 1)
                X_test = X_test.reshape((-1, model_window_size, 1))
                
                # ============================================================
                # STEP 3d: LSTM RECONSTRUCTION (THE KEY STEP!)
                # ============================================================
                # The model tries to RECONSTRUCT each window of your new data
                # It does NOT predict future values - it recreates the input you gave it
                #
                # What happens inside:
                # 1. Encoder: Compresses your window to latent representation
                # 2. Decoder: Reconstructs the window from latent representation
                # 3. Output: Reconstructed window (should match input if data is normal)
                #
                # Example:
                #   Input:  [100.5, 101.2, 150.0, 101.5]  ← Your new data
                #   Output: [100.3, 101.1, 101.0, 101.4]  ← Model's reconstruction attempt
                #   The spike (150.0) can't be reconstructed → High error → Anomaly!
                preds = model.predict(X_test)
                
                # ============================================================
                # STEP 3e: Calculate reconstruction error (RMSE)
                # ============================================================
                # Compare original windows with reconstructed windows
                # High error = Model couldn't reconstruct well = Anomaly!
                # Low error = Model reconstructed well = Normal data
                window_rmse = np.sqrt(np.mean((X_test.reshape(X_test.shape[0], -1) - preds.reshape(preds.shape[0], -1))**2, axis=1))
                
                # ============================================================
                # STEP 3f: Get threshold and flag anomalies
                # ============================================================
                # Threshold defines "maximum normal reconstruction error"
                # Computed during training from normal data statistics
                thr = thresholds_store.get(mk, None)
                if thr is None:
                    # If no saved threshold, compute new one (not ideal - should use training threshold)
                    thr = compute_threshold(window_rmse, k=float(resid_k))
                    thresholds_store[mk] = thr

                # Flag windows as anomalous if reconstruction error exceeds threshold
                wflags = window_rmse > thr  # Boolean array: True = anomaly, False = normal
                
                # ============================================================
                # STEP 3g: Expand window-level flags to full timeline
                # ============================================================
                # We have one flag per window, but need flags for every timestamp
                # Solution: Mark all timestamps in anomalous windows as anomalous
                full_flags = expand_window_flags_to_full(n_points, idxs, wflags)
                # Store reconstruction errors for each timestamp (for visualization/analysis)
                errors_full = np.zeros(n_points, dtype=float)
                for (start, end), we in zip(idxs, window_rmse):
                    s = max(0, start); e = min(n_points-1, end)
                    errors_full[s:e+1] = np.maximum(errors_full[s:e+1], we)  # Use maximum error in window
                
                # Store results for this channel
                channel_flags[mk] = full_flags  # Anomaly flags (True/False for each timestamp)
                channel_errors[mk] = errors_full  # Reconstruction errors (for analysis)

            # If some models not matched, add zero arrays to channel_flags for consistent downstream operations
            for mk in models_store.keys():
                if mk not in channel_flags:
                    channel_flags[mk] = np.zeros(n_points, dtype=bool)
                    channel_errors[mk] = np.zeros(n_points, dtype=float)

            # ============================================================
            # STEP 4: Detect observatories and group channels
            # ============================================================
            # Automatically identifies observatories (OBS1, OBS2, etc.) from column names
            # Groups channels by observatory for per-observatory voting (reference only)
            def detect_obs_prefixes(cols):
                """
                Automatically detect observatory prefixes from column names.
                Looks for patterns like "OBS1", "OBS2" in sensor IDs.
                """
                obs = set()
                for c in cols:
                    if '__' not in c: continue
                    sid = c.split('__')[0]
                    match = re.search(r'(OBS\d+)', sid.upper())
                    if match:
                        obs.add(match.group(1))
                if not obs:
                    # fallback: try any token that starts with 'OBS'
                    for c in cols:
                        if '__' not in c: continue
                        sid = c.split('__')[0]
                        toks = sid.split('_')
                        for t in toks:
                            if t.upper().startswith('OBS'):
                                obs.add(t.upper())
                if not obs:
                    obs.add('OBS_ALL')  # Default if no observatory pattern found
                return sorted(list(obs))

            obs_prefixes = detect_obs_prefixes(list(wide_detect.columns))

            # Map channels to observatory by looking at sensor key (meaningful part) in channel name
            # This groups channels by observatory for per-observatory voting (computed but not used for final decision)
            obs_channels_map = {obs: [] for obs in obs_prefixes}
            for ch in channel_flags.keys():
                # channel keys are mk (like 'S2025...__b_x' or safe name); extract sensor id
                if '__' not in ch:
                    # fallback -> first obs
                    obs_channels_map[obs_prefixes[0]].append(ch)
                    continue
                    
                sensor_id = ch.split('__')[0]
                # Extract meaningful sensor key (ignoring timestamp prefix)
                sensor_key, sensor_num = extract_sensor_key(sensor_id)
                
                assigned = False
                # Try to match by observatory in sensor key
                for obs in obs_prefixes:
                    # Check if observatory is in the sensor key (e.g., OBS1 in "OBS1_1")
                    if obs in sensor_key.upper():
                        obs_channels_map[obs].append(ch)
                        assigned = True
                        break
                
                if not assigned:
                    # Try to find matching detection column and use its observatory
                    for dc in det_cols:
                        if '__' not in dc: continue
                        dc_sensor_id = dc.split('__')[0]
                        dc_sensor_key, dc_sensor_num = extract_sensor_key(dc_sensor_id)
                        
                        # If sensor keys match, use the detection column's observatory
                        if sensor_keys_match(sensor_key, dc_sensor_key, sensor_num, dc_sensor_num):
                            for obs in obs_prefixes:
                                if obs in dc_sensor_key.upper():
                                    obs_channels_map[obs].append(ch)
                                    assigned = True
                                    break
                        if assigned: break
                
                if not assigned:
                    # fallback -> first obs
                    obs_channels_map[obs_prefixes[0]].append(ch)

            # ============================================================
            # STEP 4: Combine results from all channels using voting
            # ============================================================
            
            # STEP 4a: Per-observatory voting (computed for reference/export)
            # Note: This is computed but NOT used for final decision
            # It's available in exported CSV for analysis purposes
            obs_flags = {}
            obs_votes = {}
            for obs, chs in obs_channels_map.items():
                if len(chs) == 0:
                    obs_flags[obs] = np.zeros(n_points, dtype=bool)
                    obs_votes[obs] = np.zeros(n_points, dtype=int)
                    continue
                # Collect flags from all channels in this observatory
                subset = {ch: channel_flags[ch] for ch in chs}
                if use_absolute_voting:
                    voted_bool, votes = majority_vote(subset, min_channels=min_channels_required)
                else:
                    voted_bool, votes = majority_vote(subset, vote_fraction=float(vote_fraction))
                obs_flags[obs] = voted_bool
                obs_votes[obs] = votes

            # ============================================================
            # STEP 4b: Global majority voting (FINAL DECISION)
            # ============================================================
            # This is what determines the final anomaly flag
            # Requires majority of ALL channels (across all observatories) to agree
            # Example: If 5+ out of 10 channels flag a timestamp → Final flag = Anomaly
            if use_absolute_voting:
                global_voted_bool, global_votes = majority_vote(channel_flags, min_channels=min_channels_required)
            else:
                # Use fraction-based if absolute voting is disabled
                global_voted_bool, global_votes = majority_vote(channel_flags, vote_fraction=float(vote_fraction))
            
            # Final anomaly flag: True = anomaly detected, False = normal
            final_flag = global_voted_bool
            st.success(f"Inference completed. {len(channel_flags)} channels processed, {int(np.sum(final_flag))} timestamps flagged as anomalies.")
            
            # Store results in session state
            st.session_state['inference_results'] = {
                'channel_flags': channel_flags,
                'channel_errors': channel_errors,
                'obs_flags': obs_flags,
                'obs_votes': obs_votes,
                'final_flag': final_flag,
                'n_points': n_points,
                'det_index': det_index,
                'matched_pairs': matched_pairs,
                'obs_prefixes': obs_prefixes
            }
            st.session_state['inference_wide_detect'] = wide_detect.copy()
    
    # Display results if available (either from just-run inference or cached)
    if st.session_state['inference_results'] is not None and wide_detect is not None:
        results = st.session_state['inference_results']
        channel_flags = results['channel_flags']
        channel_errors = results['channel_errors']
        obs_flags = results['obs_flags']
        obs_votes = results['obs_votes']
        final_flag = results['final_flag']
        n_points = results['n_points']
        det_index = results['det_index']
        
        # Use cached wide_detect or current one
        display_wide_detect = st.session_state.get('inference_wide_detect', wide_detect)
        obs_prefixes = results.get('obs_prefixes', [])

        # ============================================================
        # STEP 6: Visualization - Display Results
        # ============================================================
        # Creates interactive plots showing:
        # - Original magnetic field data (b_x, b_y, b_z)
        # - Anomaly regions highlighted in red
        # - Anomaly points marked with red 'x' markers
        
        st.subheader("B Components for All Sensors with Anomalies")
        
        ts = display_wide_detect.index  # Timestamps for x-axis
        
        # Extract sensor information from detection columns
        sensor_components = {}  # {sensor_id: {'b_x': col_name, 'b_y': col_name, 'b_z': col_name}}
        for col in display_wide_detect.columns:
            if '__' in col:
                sensor_id, component = col.split('__', 1)
                if component in ['b_x', 'b_y', 'b_z']:
                    if sensor_id not in sensor_components:
                        sensor_components[sensor_id] = {}
                    sensor_components[sensor_id][component] = col
        
        # Group sensors by observatory for better organization
        sensors_by_obs = {}
        for sensor_id in sensor_components:
            # Extract observatory from sensor_id
            obs = None
            for obs_prefix in obs_prefixes:
                if obs_prefix in sensor_id.upper():
                    obs = obs_prefix
                    break
            if obs is None:
                obs = 'UNKNOWN'
            if obs not in sensors_by_obs:
                sensors_by_obs[obs] = []
            sensors_by_obs[obs].append(sensor_id)
        
        # Plot each sensor's B components
        for obs in sorted(sensors_by_obs.keys()):
            for sensor_id in sorted(sensors_by_obs[obs]):
                if sensor_id not in sensor_components:
                    continue
                comps = sensor_components[sensor_id]
                if len(comps) < 3:  # Need all three components
                    continue
                
                fig_sensor = go.Figure()
                
                # Plot b_x, b_y, b_z
                colors = {'b_x': 'blue', 'b_y': 'green', 'b_z': 'orange'}
                for comp in ['b_x', 'b_y', 'b_z']:
                    if comp in comps:
                        col_name = comps[comp]
                        fig_sensor.add_trace(go.Scatter(
                            x=ts, 
                            y=display_wide_detect[col_name].values, 
                            name=f"{comp}", 
                            mode='lines', 
                            line=dict(color=colors[comp], width=1.5)
                        ))
                
                # Add shaded regions for anomalies (red)
                flag_bool = final_flag.astype(bool)
                regions = find_anomaly_regions(flag_bool)
                for (s, e) in regions:
                    fig_sensor.add_vrect(x0=ts[s], x1=ts[e], fillcolor="red", opacity=0.4, line_width=0, layer="below")
                
                # Mark anomaly points with red markers
                anomaly_indices = np.where(flag_bool)[0]
                if len(anomaly_indices) > 0:
                    # Add markers for each component at anomaly points
                    for comp in ['b_x', 'b_y', 'b_z']:
                        if comp in comps:
                            col_name = comps[comp]
                            anomaly_values = display_wide_detect[col_name].iloc[anomaly_indices].values
                            fig_sensor.add_trace(go.Scatter(
                                x=ts[anomaly_indices], 
                                y=anomaly_values,
                                mode='markers',
                                marker=dict(color='red', size=6, symbol='x'),
                                name=f'{comp} anomalies',
                                showlegend=(comp == 'b_x')  # Only show legend once
                            ))
                
                fig_sensor.update_layout(
                    title=f"B Components for {sensor_id} ({obs})",
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Magnetic Field (nT)",
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                st.plotly_chart(fig_sensor, use_container_width=True)

        # ============================================================
        # STEP 5: Summary & Export Results
        # ============================================================
        st.subheader("Summary & Export")
        st.write(f"**Final flagged timestamps:** {int(np.sum(final_flag))} / {n_points} ({100.0 * np.sum(final_flag) / n_points:.2f}%)")

        # Prepare export DataFrame with all results
        out_df = display_wide_detect.copy()
        # Add per-channel flags (0 = normal, 1 = anomaly)
        for mk, farr in channel_flags.items():
            arr = np.array(farr, dtype=int)
            if len(arr) < n_points:
                arr = np.concatenate([arr, np.zeros(n_points - len(arr), dtype=int)])
            out_df[mk + "_flag"] = arr
        # Add per-observatory vote counts (for reference - not used for final decision)
        for obs, votes in obs_votes.items():
            arr = np.array(votes, dtype=int)
            if len(arr) < n_points:
                arr = np.concatenate([arr, np.zeros(n_points - len(arr), dtype=int)])
            out_df[obs + "_votes"] = arr
        # Add final anomaly flag (determined by global majority voting)
        out_df['final_cross_obs_flag'] = final_flag.astype(int)

        csvbuf = io.StringIO()
        out_df.to_csv(csvbuf)
        st.download_button("Download full detection results CSV", data=csvbuf.getvalue().encode(), file_name="magnavis_detection_results_full.csv")
        st.success("Inference complete. You can zoom into plots to inspect anomalies.")

st.markdown("---")
st.markdown("""
**Notes & recommendations**
- If your TRAIN CSV contained only OBS1 sensors, then inference on OBS2 will be unreliable unless you train OBS2 models too. The app will run but uses per-channel scalers/models where available.
- Persist scalers per-channel during training (we save them in `scalers.pkl`) for consistent normalization during inference.
- Use overlapping windows (step < window_size) for smoother anomaly timelines (but more compute).
- Tune `resid_k` (MAD multiplier) per-channel if you have labeled data.
""")