# How LSTM Autoencoder Works During Inference in magnavis_anomaly_maVo.py

## Overview

Yes, the LSTM autoencoder **does reconstruct a new time series** during inference, but this reconstructed series is used to **detect anomalies**, not to replace your original data. The anomaly detection works by comparing the **reconstruction error** between the original input and the model's reconstruction.

## How It Works: Step-by-Step

### 1. **Training Phase** (Learning Normal Patterns)

During training, the LSTM autoencoder learns to:
- **Encode**: Compress normal time series patterns into a latent representation
- **Decode**: Reconstruct the original time series from the latent representation
- **Goal**: Minimize reconstruction error for **normal data**

The model learns: *"What does normal magnetic field data look like?"*

### 2. **Inference Phase** (Detecting Anomalies in New Data)

When you load a new CSV file for inference, here's what happens:

#### Step 1: Load and Preprocess New Data
```python
# Line 610: Extract time series from detection CSV
series = wide_detect[dc].dropna().values.reshape(-1,1)

# Line 628: Normalize using the SAME scaler from training
s_series = scaler.transform(series).flatten()
```

**Key Point**: The new data is normalized using the **same scaler** that was fitted during training. This ensures the data is in the same scale as what the model learned.

#### Step 2: Create Windows
```python
# Line 629: Split time series into windows
X_test, idxs = create_windows(s_series, model_window_size, model_step_size)
# X_test shape: (n_windows, window_size)
```

The new time series is split into overlapping or non-overlapping windows (e.g., 64 samples per window).

#### Step 3: **LSTM Reconstruction** (This is the key step!)
```python
# Line 637: Model predicts/reconstructs each window
preds = model.predict(X_test)
```

**What happens here:**
- Input: `X_test` - Your new data windows (shape: `[n_windows, window_size, 1]`)
- Output: `preds` - **Reconstructed time series** (shape: `[n_windows, window_size, 1]`)

The LSTM autoencoder:
1. Takes your new data window
2. Encodes it to latent space (compresses to 32 dimensions)
3. **Decodes it back** to reconstruct the original window
4. Returns the reconstructed window

**Important**: The model tries to reconstruct your new data based on what it learned from **normal training data**.

#### Step 4: Calculate Reconstruction Error
```python
# Line 638: Calculate RMSE between original and reconstructed
window_rmse = np.sqrt(np.mean(
    (X_test.reshape(X_test.shape[0], -1) - 
     preds.reshape(preds.shape[0], -1))**2, 
    axis=1
))
```

**This is the core of anomaly detection:**
- **Low RMSE**: The model successfully reconstructed the window → Data looks "normal" → **No anomaly**
- **High RMSE**: The model struggled to reconstruct the window → Data doesn't match normal patterns → **Anomaly detected!**

#### Step 5: Compare Against Threshold
```python
# Line 639-642: Get threshold (from training or compute new)
thr = thresholds_store.get(mk, None)

# Line 644: Flag windows as anomalous if error > threshold
wflags = window_rmse > thr
```

The reconstruction error is compared against a threshold computed during training (using MAD method).

#### Step 6: Expand to Full Timeline
```python
# Line 645: Expand window-level flags to full time series
full_flags = expand_window_flags_to_full(n_points, idxs, wflags)
```

Window-level anomaly flags are expanded to mark every timestamp in anomalous windows.

## Visual Example

### Normal Data (Low Reconstruction Error)
```
Original:     [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.0]
                ↓ (LSTM encodes & decodes)
Reconstructed: [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.0]
                ↓ (Calculate error)
RMSE: 0.05  →  ✅ Normal (error < threshold)
```

### Anomalous Data (High Reconstruction Error)
```
Original:     [1.0, 1.1, 5.0, 1.2, 1.1, 1.0, 1.1, 1.0]  ← Sudden spike!
                ↓ (LSTM encodes & decodes)
Reconstructed: [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.0]  ← Model can't reconstruct spike
                ↓ (Calculate error)
RMSE: 1.25  →  ❌ Anomaly! (error > threshold)
```

## Key Concepts

### 1. **Reconstruction, Not Prediction**
The LSTM autoencoder is **not predicting future values**. It's trying to:
- Reconstruct the **same input** it received
- If it can't reconstruct well, the input is anomalous

### 2. **Unsupervised Learning**
- The model learns from **normal data only** during training
- During inference, it flags anything that doesn't match normal patterns
- No labeled anomalies needed!

### 3. **Why Reconstruction Error Works**
- **Normal patterns**: Model learned these during training → Low reconstruction error
- **Anomalous patterns**: Model never saw these → High reconstruction error
- **Threshold**: Separates normal from anomalous based on training data statistics

## Code Flow Diagram

```
New CSV Data
    ↓
Extract Time Series (per channel)
    ↓
Normalize (using training scaler)
    ↓
Create Windows (64 samples each)
    ↓
┌─────────────────────────────────┐
│  LSTM Autoencoder Inference     │
│                                  │
│  Input Window → Encode → Decode │
│                                  │
│  Original: [1.0, 1.1, 1.0, ...] │
│  Reconstructed: [1.0, 1.1, ...] │
└─────────────────────────────────┘
    ↓
Calculate RMSE (reconstruction error)
    ↓
Compare with Threshold
    ↓
Flag as Anomaly if RMSE > Threshold
    ↓
Expand to Full Timeline
    ↓
Majority Voting (across channels)
    ↓
Final Anomaly Flags
```

## Important Details from the Code

### 1. **Same Scaler from Training**
```python
# Line 619-626: Uses saved scaler from training
if mk in scalers_store:
    scaler = scalers_store[mk]  # ← Uses training scaler!
```

**Why this matters**: The model expects data in the same scale as training. Using a different scaler would break the anomaly detection.

### 2. **Window-Based Processing**
```python
# Line 629: Creates windows
X_test, idxs = create_windows(s_series, model_window_size, model_step_size)
```

**Why windows**: LSTM needs sequences. The model processes data in windows (e.g., 64 consecutive samples), not individual points.

### 3. **Reconstruction Error Calculation**
```python
# Line 638: RMSE between original and reconstructed
window_rmse = np.sqrt(np.mean(
    (X_test.reshape(...) - preds.reshape(...))**2, 
    axis=1
))
```

**What this measures**: How well the model reconstructed each window. Higher error = more anomalous.

### 4. **Threshold from Training**
```python
# Line 639: Get threshold computed during training
thr = thresholds_store.get(mk, None)
```

**Why this matters**: The threshold is based on reconstruction errors from **normal training data**. It defines what "normal" reconstruction error looks like.



**Q: Does LSTM construct a new time series during inference?**

**A: Yes, but it's a RECONSTRUCTION, not a new independent time series.**

- The LSTM **reconstructs** your input data based on learned normal patterns
- The reconstruction is compared to the original to detect anomalies
- **High reconstruction error = Anomaly**
- **Low reconstruction error = Normal**

**Q: How does it infer anomalies in newly loaded data?**

**A: Through reconstruction error analysis:**

1. **Load new CSV** → Extract time series
2. **Normalize** → Use training scaler
3. **Create windows** → Split into sequences
4. **LSTM reconstructs** → Model tries to recreate each window
5. **Calculate error** → RMSE between original and reconstructed
6. **Compare threshold** → Error > threshold = Anomaly
7. **Expand flags** → Mark all timestamps in anomalous windows

## Example Scenario

### Training Data (Normal)
```
Time:  [0, 1, 2, 3, 4, 5, 6, 7]
Value: [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.0]
```
Model learns: "Normal magnetic field varies smoothly around 1.0-1.2"

### Inference Data (Contains Anomaly)
```
Time:  [0, 1, 2, 3, 4, 5, 6, 7]
Value: [1.0, 1.1, 5.0, 1.2, 1.1, 1.0, 1.1, 1.0]
                    ↑
              Anomaly at t=2
```

### What Happens:
1. **Window [0-3]**: `[1.0, 1.1, 5.0, 1.2]`
   - Model reconstructs: `[1.0, 1.1, 1.0, 1.2]` (can't reconstruct the spike)
   - RMSE: **1.25** (high!)
   - Threshold: 0.3
   - **Flagged as ANOMALY** ✅

2. **Window [1-4]**: `[1.1, 5.0, 1.2, 1.1]`
   - Model reconstructs: `[1.1, 1.0, 1.2, 1.1]`
   - RMSE: **1.0** (high!)
   - **Flagged as ANOMALY** ✅

3. **Window [4-7]**: `[1.1, 1.0, 1.1, 1.0]`
   - Model reconstructs: `[1.1, 1.0, 1.1, 1.0]` (matches normal pattern)
   - RMSE: **0.05** (low)
   - **Normal** ✅

## Summary

1. **Yes, LSTM reconstructs time series** during inference
2. **Reconstruction is compared to original** to detect anomalies
3. **High reconstruction error** = Anomaly (model can't reconstruct it)
4. **Low reconstruction error** = Normal (model successfully reconstructs it)
5. **Threshold** separates normal from anomalous based on training data
6. **Window-based processing** allows detection of local anomalies
7. **Majority voting** combines results from multiple channels/sensors

The key insight: **The model doesn't predict the future; it tries to reconstruct the present. If it fails, that's an anomaly!**

