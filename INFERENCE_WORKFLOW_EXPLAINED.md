# How Inference Works with Pre-Trained Models - Detailed Explanation

## Quick Answer to Your Main Question

**The LSTM RECONSTRUCTS the newly loaded data, it does NOT predict future values.**

Think of it like this:
- **Reconstruction**: "Here's your data back, I tried to recreate it based on what I learned"
- **Prediction**: "Here's what will happen next based on past patterns"

The LSTM autoencoder does **reconstruction** - it takes your new data, tries to recreate it, and if it fails (high error), that's an anomaly!

---

## Complete Inference Workflow: Step-by-Step

### Phase 1: Loading and Preparation

#### Step 1: You Load a New CSV File
```
You click "Upload DETECTION CSV" or provide file path
→ File contains: timestamp, sensor_id, b_x, b_y, b_z columns
→ This is your NEW data that you want to check for anomalies
```

**Example of your new CSV:**
```
timestamp          sensor_id    b_x    b_y    b_z
2025-11-15 10:00  SENSOR1      100.5  200.3  300.1
2025-11-15 10:01  SENSOR1      101.2  201.0  301.5
2025-11-15 10:02  SENSOR1      150.0  250.0  350.0  ← Possible anomaly?
2025-11-15 10:03  SENSOR1      101.5  201.2  301.8
```

#### Step 2: System Loads Pre-Trained Models
```
System looks in "saved_models" folder
→ Finds: model_SENSOR1__b_x.h5, model_SENSOR1__b_y.h5, etc.
→ Loads: Models, Scalers, Thresholds, Metadata
→ These models were trained on NORMAL data (your training CSV)
```

**What's in saved_models folder:**
```
saved_models/
├── model_SENSOR1__b_x.h5      ← Pre-trained LSTM model
├── model_SENSOR1__b_y.h5      ← Pre-trained LSTM model
├── model_SENSOR1__b_z.h5      ← Pre-trained LSTM model
├── scalers.pkl                ← Normalization parameters from training
├── thresholds.json            ← Anomaly thresholds from training
└── metadata.json              ← Window size, step size, etc.
```

#### Step 3: Convert Your CSV to Wide Format
```
Your CSV (long format):
timestamp          sensor_id    b_x    b_y    b_z
2025-11-15 10:00  SENSOR1      100.5  200.3  300.1

Becomes (wide format):
timestamp          SENSOR1__b_x  SENSOR1__b_y  SENSOR1__b_z
2025-11-15 10:00  100.5         200.3         300.1
```

**Why?** Each sensor-component combination (like `SENSOR1__b_x`) becomes a separate column. This matches how models were trained (one model per column).

---

### Phase 2: Model Matching

#### Step 4: Match Models to Your Data Columns
```
System tries to match:
- Pre-trained model: "SENSOR1__b_x" 
- Your data column: "SENSOR1__b_x"
→ Match found! ✅

If names don't match exactly:
- System tries smart matching (ignores timestamp prefixes)
- Example: Model "S20251008_105355_SENSOR1__b_x" 
  matches Column "S20251115_100000_SENSOR1__b_x"
  (both have SENSOR1 and b_x)
```

**What happens:**
- System finds which of your columns have matching pre-trained models
- For each match, it will run inference
- If no match found, that column is skipped (no anomaly detection for it)

---

### Phase 3: Per-Channel Inference (The Core Process)

For each matched model-column pair (e.g., `SENSOR1__b_x`), here's what happens:

#### Step 5: Extract Your New Data
```python
# Line 610: Get your new data
series = [100.5, 101.2, 150.0, 101.5, 102.0, ...]  # Your b_x values
```

**This is YOUR actual data** - the magnetic field measurements you just loaded.

#### Step 6: Normalize Using Training Scaler
```python
# Lines 619-626: Use the SAME scaler from training
scaler = scalers_store['SENSOR1__b_x']  # Loaded from saved_models/scalers.pkl
normalized_series = scaler.transform(series)
```

**Why this matters:**
- The model was trained on normalized data (mean=0, std=1)
- Your new data must be normalized the SAME way
- Uses the scaler that was saved during training
- This ensures your data is in the same "scale" as training data

**Example:**
```
Your raw data:     [100.5, 101.2, 150.0, 101.5]
After normalization: [-0.1, 0.0, 5.2, 0.1]  ← Now in same scale as training
```

#### Step 7: Create Windows
```python
# Line 629: Split your data into windows
# Window size = 64 (from training)
# Step size = 64 (no overlap, or smaller for overlap)

Your data: [100.5, 101.2, 150.0, 101.5, 102.0, ...] (1000 values)
↓
Windows:
  Window 1: [100.5, 101.2, 150.0, ..., 102.3]  (64 values)
  Window 2: [102.3, 102.5, 101.8, ..., 103.1]  (64 values)
  Window 3: [103.1, 103.0, 102.9, ..., 101.5]  (64 values)
  ...
```

**Why windows?**
- LSTM needs sequences (not single values)
- Model was trained on windows of 64 samples
- Each window is processed independently

#### Step 8: **LSTM RECONSTRUCTION** (This is the key step!)

```python
# Line 637: Model tries to RECONSTRUCT your windows
preds = model.predict(X_test)
```

**What happens inside the LSTM:**

```
Your Window: [100.5, 101.2, 150.0, 101.5, ...]  ← Your NEW data
    ↓
    [LSTM Encoder]
    ↓
Latent Representation: [0.2, -0.1, 0.5, ...]  ← Compressed to 32 numbers
    ↓
    [LSTM Decoder]
    ↓
Reconstructed Window: [100.3, 101.1, 101.0, 101.4, ...]  ← Model's attempt to recreate
```

**Critical Understanding:**

1. **Input**: Your new data window (e.g., `[100.5, 101.2, 150.0, 101.5]`)
2. **Process**: 
   - Model encodes it (compresses to latent space)
   - Model decodes it (reconstructs from latent space)
   - Model outputs: **Reconstruction of the SAME input**
3. **Output**: Reconstructed window (e.g., `[100.3, 101.1, 101.0, 101.4]`)

**The model is NOT predicting the future!** It's trying to recreate what you gave it.

**What the model learned during training:**
- "Normal magnetic field data looks like smooth variations around 100-102"
- "I can reconstruct normal patterns well"
- "I struggle with sudden spikes or unusual patterns"

#### Step 9: Calculate Reconstruction Error
```python
# Line 638: Compare original vs reconstructed
window_rmse = sqrt(mean((original - reconstructed)²))
```

**Example with Normal Data:**
```
Original:      [100.5, 101.2, 101.0, 101.5]
Reconstructed: [100.3, 101.1, 101.0, 101.4]
Difference:    [0.2,   0.1,   0.0,   0.1]
RMSE: 0.12  ← Low error! Model reconstructed well
```

**Example with Anomalous Data:**
```
Original:      [100.5, 101.2, 150.0, 101.5]  ← Sudden spike at position 2!
Reconstructed: [100.3, 101.1, 101.0, 101.4]  ← Model can't reconstruct the spike
Difference:    [0.2,   0.1,   49.0,  0.1]   ← Huge difference!
RMSE: 24.5  ← High error! Model failed to reconstruct
```

**The Logic:**
- **Low RMSE** = Model successfully reconstructed = Data looks normal = ✅ No anomaly
- **High RMSE** = Model failed to reconstruct = Data doesn't match normal patterns = ❌ Anomaly!

#### Step 10: Compare with Threshold
```python
# Lines 639-642: Get threshold from training
threshold = 0.5  # From saved_models/thresholds.json

# Line 644: Flag as anomaly if error > threshold
if RMSE > 0.5:
    Flag as ANOMALY ✅
else:
    Flag as NORMAL ✅
```

**The threshold was computed during training:**
- Based on reconstruction errors from NORMAL training data
- Uses MAD (Median Absolute Deviation) method
- Defines: "What's the maximum normal reconstruction error?"

**Example:**
```
Window 1 RMSE: 0.12 < 0.5 → Normal ✅
Window 2 RMSE: 24.5 > 0.5 → Anomaly! ❌
Window 3 RMSE: 0.15 < 0.5 → Normal ✅
```

#### Step 11: Expand to Full Timeline
```python
# Line 645: Expand window flags to every timestamp
```

**Problem:** We have one flag per window, but need flags for every timestamp.

**Solution:** Mark all timestamps in an anomalous window as anomalous.

**Example:**
```
Timestamps:    [T1, T2, T3, T4, T5, T6, T7, T8]
Windows:       [Window1: T1-T4, Window2: T5-T8]
Window flags:  [Normal, Anomaly]

Expanded flags: [Normal, Normal, Normal, Normal, Anomaly, Anomaly, Anomaly, Anomaly]
                 └─Window1─┘              └────Window2────┘
```

---

### Phase 4: Combining Results (Voting)

#### Step 12: Collect Flags from All Channels
```
After processing all channels:
- SENSOR1__b_x flags: [Normal, Normal, Anomaly, Normal, ...]
- SENSOR1__b_y flags: [Normal, Anomaly, Anomaly, Normal, ...]
- SENSOR1__b_z flags: [Normal, Normal, Normal, Normal, ...]
- SENSOR2__b_x flags: [Normal, Normal, Anomaly, Normal, ...]
- ... (many more channels)
```

#### Step 13: Global Majority Voting
```
For each timestamp, count votes:
Timestamp T1: 3 channels say Anomaly, 15 say Normal → Normal (majority = Normal)
Timestamp T2: 12 channels say Anomaly, 6 say Normal → Anomaly (majority = Anomaly)
Timestamp T3: 8 channels say Anomaly, 10 say Normal → Normal (majority = Normal)
```

**Final Decision:**
- If majority of channels (e.g., 5+ out of 10) flag a timestamp as anomaly → Final flag = Anomaly
- Otherwise → Final flag = Normal

---

## Visual Example: Complete Flow

### Your New CSV Data
```
Time    b_x
10:00   100.5  ← Normal
10:01   101.2  ← Normal
10:02   150.0  ← ANOMALY! (sudden spike)
10:03   101.5  ← Normal
10:04   102.0  ← Normal
```

### What Happens Step-by-Step

#### 1. Load and Normalize
```
Raw:     [100.5, 101.2, 150.0, 101.5, 102.0]
Normalized: [-0.1, 0.0, 5.2, 0.1, 0.2]
```

#### 2. Create Windows (window_size=5 for this example)
```
Window 1: [-0.1, 0.0, 5.2, 0.1, 0.2]
```

#### 3. LSTM Reconstruction
```
Input Window:    [-0.1, 0.0, 5.2, 0.1, 0.2]  ← Your data
    ↓ (LSTM encodes & decodes)
Reconstructed:   [-0.1, 0.0, 0.0, 0.1, 0.2]  ← Model's attempt
                                    ↑
                            Model can't reconstruct the spike!
```

#### 4. Calculate Error
```
Original:      [-0.1, 0.0, 5.2, 0.1, 0.2]
Reconstructed: [-0.1, 0.0, 0.0, 0.1, 0.2]
Difference:    [0.0,  0.0, 5.2, 0.0, 0.0]
RMSE: 2.32  ← High error!
```

#### 5. Compare with Threshold
```
Threshold: 0.5
RMSE: 2.32 > 0.5 → ANOMALY DETECTED! ❌
```

#### 6. Final Result
```
Time    b_x    Flag
10:00   100.5  Normal ✅
10:01   101.2  Normal ✅
10:02   150.0  ANOMALY ❌  ← Correctly detected!
10:03   101.5  Normal ✅
10:04   102.0  Normal ✅
```

---

## Key Concepts Explained Simply

### 1. **Reconstruction vs Prediction**

**Reconstruction (What LSTM Does):**
```
You give: [100, 101, 150, 102]
Model says: "Let me try to recreate this: [100, 101, 101, 102]"
Error: High (because model can't recreate the 150)
→ Anomaly!
```

**Prediction (What LSTM Does NOT Do):**
```
You give: [100, 101, 102]
Model says: "Next value will be 103"
→ This is NOT what happens!
```

### 2. **Why Reconstruction Works for Anomaly Detection**

**During Training:**
- Model sees: Normal patterns only
- Model learns: "I can reconstruct normal patterns well"
- Model never sees: Anomalies

**During Inference:**
- You give: Normal data → Model reconstructs well → Low error → Normal ✅
- You give: Anomalous data → Model can't reconstruct → High error → Anomaly ❌

### 3. **The Threshold**

**What it is:**
- Maximum "normal" reconstruction error
- Computed from training data
- Based on statistics (MAD method)

**How it works:**
- If your data's reconstruction error > threshold → Anomaly
- If your data's reconstruction error ≤ threshold → Normal

### 4. **Why Use the Same Scaler?**

**Training:**
- Data normalized: mean=0, std=1
- Model learns patterns in this normalized space

**Inference:**
- Must use SAME normalization
- Otherwise, model sees data in different scale
- Would break the anomaly detection!

**Example:**
```
Training data: [100, 101, 102] → Normalized: [-1, 0, 1]
Your new data: [100, 101, 150] → Should normalize to: [-1, 0, 50]
If you use different scaler: [0, 1, 2] → Model gets confused!
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ 1. You Load New CSV                                      │
│    timestamp, sensor_id, b_x, b_y, b_z                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. System Loads Pre-Trained Models                      │
│    From: saved_models/ folder                            │
│    - Models (LSTM autoencoders)                          │
│    - Scalers (normalization parameters)                  │
│    - Thresholds (anomaly detection limits)               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Convert CSV to Wide Format                            │
│    SENSOR1__b_x, SENSOR1__b_y, SENSOR1__b_z, ...         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Match Models to Columns                              │
│    Model "SENSOR1__b_x" → Column "SENSOR1__b_x" ✅        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 5. For Each Matched Channel:                            │
│                                                          │
│    a. Extract your new data                             │
│       [100.5, 101.2, 150.0, 101.5, ...]                 │
│                                                          │
│    b. Normalize using training scaler                    │
│       [-0.1, 0.0, 5.2, 0.1, ...]                        │
│                                                          │
│    c. Create windows (64 samples each)                  │
│       Window 1: [-0.1, 0.0, 5.2, ..., 0.2]              │
│                                                          │
│    d. LSTM RECONSTRUCTS each window                     │
│       Input:  [-0.1, 0.0, 5.2, ..., 0.2]               │
│       Output: [-0.1, 0.0, 0.0, ..., 0.2]  ← Recreated  │
│                                                          │
│    e. Calculate reconstruction error (RMSE)             │
│       RMSE = 2.32 (high!)                               │
│                                                          │
│    f. Compare with threshold                            │
│       2.32 > 0.5 → ANOMALY! ❌                           │
│                                                          │
│    g. Expand to full timeline                           │
│       [Normal, Normal, Anomaly, Normal, ...]            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Combine Results from All Channels                    │
│    Global Majority Voting                                │
│    Final Flag: [Normal, Normal, Anomaly, Normal, ...]    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Display Results                                      │
│    - Plots with anomaly regions highlighted              │
│    - Summary statistics                                  │
│    - Download CSV with flags                            │
└─────────────────────────────────────────────────────────┘
```

---

## Answer to Your Main Question



**Detailed Explanation:**

1. **What You Give**: Your new CSV data (e.g., `[100.5, 101.2, 150.0, 101.5]`)

2. **What LSTM Does**: 
   - Takes your data
   - Encodes it (compresses)
   - Decodes it (reconstructs)
   - Returns: **Reconstruction of the SAME data you gave it**

3. **What LSTM Does NOT Do**:
   - Does NOT predict future values
   - Does NOT forecast what comes next
   - Does NOT generate new data

4. **How Anomaly Detection Works**:
   - Compare original vs reconstructed
   - High difference = Model couldn't recreate it = Anomaly
   - Low difference = Model recreated it well = Normal

### **Why Reconstruction, Not Prediction?**

**Reconstruction is perfect for anomaly detection because:**

1. **Model learns normal patterns**: During training, model sees only normal data
2. **Model can reconstruct normal**: If you give normal data, model recreates it well
3. **Model can't reconstruct anomalies**: If you give anomalous data, model struggles
4. **Error reveals anomalies**: High reconstruction error = anomaly detected!

**If it were prediction:**
- Model would try to predict future values
- But we want to detect anomalies in CURRENT data
- Reconstruction allows us to check if current data is normal or not

---

## Real-World Analogy

Think of the LSTM autoencoder like a **skilled artist who learned to draw normal landscapes**:

**During Training:**
- Artist sees: 1000 photos of normal landscapes (mountains, trees, sky)
- Artist learns: "I can recreate these landscapes well"

**During Inference (Your New Data):**
- You show: A photo of a normal landscape
- Artist recreates: Similar landscape
- Result: Looks similar → Normal ✅

- You show: A photo with a UFO in the sky (anomaly!)
- Artist recreates: Normal landscape (no UFO - never saw one!)
- Result: Looks very different → Anomaly detected! ❌

**The "error" is how different the recreation is from the original.**

---

## Summary

1. **You load new CSV** → Your data to check for anomalies

2. **System loads pre-trained models** → Models that learned normal patterns

3. **For each channel:**
   - Extract your data
   - Normalize (same way as training)
   - Create windows
   - **LSTM reconstructs each window** ← Key step!
   - Calculate reconstruction error
   - Compare with threshold
   - Flag as anomaly if error is high

4. **Combine all channels** → Global majority voting

5. **Final result** → Anomaly flags for your data

**The LSTM reconstructs (recreates) your newly loaded data, not predicts future values. High reconstruction error = Anomaly!**

