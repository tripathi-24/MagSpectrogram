MagSpectrogram GUI
==================

Interactive GUI to import magnetic sensor CSVs, visualize time series and spectrograms, and train ML models for object classification.

## Applications

### 1. Data Visualization (`app.py`)
Interactive GUI for magnetic sensor data analysis and visualization.

### 2. ML Classification (`ml_app.py`)
Machine learning application for training models to classify objects based on magnetic signatures.

Setup
-----

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
powershell(Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 1️⃣ Delete the old venv
rm -rf .venv

# 2️⃣ Create a new one (use the same Python version)
python3 -m venv .venv

# 3️⃣ Activate it
source .venv/bin/activate

# 4️⃣ Upgrade pip
python3 -m pip install --upgrade pip

# 5️⃣ Reinstall all requirements
pip install -r requirements.txt

Run
---

### Data Visualization
```bash
streamlit run app_v2_07Nov25.py
```

### ML Classification
```bash
streamlit run ml_app.py
```

Usage
-----

### Data Visualization App
- Click "Import Data" to upload CSVs or check "Load all CSVs from ./Dataset".
- Use sidebar to pick sensor, series (b_x, b_y, b_z, resultant), time window, and spectrogram parameters.
- **Signal Quality Diagnostics**: Real-time analysis of signal characteristics, SNR estimation, and quality warnings.
- **Enhanced Spectrogram Features**:
  - Auto-adjusting window size based on signal characteristics
  - Signal preprocessing (detrending, normalization, enhancement)
  - Multiple window functions (Hann, Hamming, Blackman, Bartlett)
  - Advanced filtering options (band-pass, high-pass, low-pass)
  - **CWT (Continuous Wavelet Transform)** with Morlet wavelets
  - Multitaper analysis and Welch-like smoothing
- **Data Superimposition Analysis**:
  - **Time-based superimposition**: Compare data from same time across different days
  - **Sensor-based superimposition**: Compare data from all sensors at same time
  - **Time Range Selection**: Select start and end times for precise temporal analysis
    - overlay segments across days within the selected time range
    - overlay segments across all sensors within the selected time range
  - **Daily Time Window Overlays (new)**:
    - pick a date range (e.g., Oct 10–15) and a daily window (e.g., 13:30–16:00)
    - choose one sensor or All sensors
    - toggle "Superimpose by time-of-day" to plot all days over the same HH:MM axis, color-coded by date
  - Interactive controls for time range selection and tolerance settings
  - Rich visualizations with actual timestamps or overlaid time-of-day and quality metrics
- **Interactive Visualizations**: Time series and spectrograms with Plotly (pan/zoom).
- **🔗 Correlation Study**: Analyze relationships between sensors and field components with 10 visualization methods (5 Plotly interactive + 5 Seaborn static), statistical analysis (Pearson, Spearman, R², p-values), and session state persistence.
- **Troubleshooting Guide**: Built-in diagnostics for flat spectrograms and signal quality issues.

### ML Classification App
1. **Load & Label Data**: Upload CSV files and label individual records or batch-label by sensor/time range
2. **Train Model**: Configure and train ML models (RandomForest, GradientBoosting, SVM, Neural Network)
3. **Make Predictions**: Use trained models to predict objects with traceable results showing actual sensor IDs and record indices
4. **Model Analysis**: Analyze performance metrics and feature importance

See `ML_GUIDE.md` for detailed ML workflow instructions.

Features
--------

- **Multi-sensor Support**: Handle data from multiple magnetic sensors
- **Interactive Visualization**: Pan, zoom, and explore time series and spectrograms
- **Signal Quality Diagnostics**: Real-time analysis with SNR estimation and quality warnings
- **Advanced Signal Processing**:
  - **CWT (Continuous Wavelet Transform)** with Morlet wavelets for time-scale analysis
  - **Multitaper Analysis** for reduced variance and improved spectral estimation
  - **Adaptive Parameters** that auto-adjust based on signal characteristics
  - **Signal Enhancement** for weak signals with configurable amplification
  - **Advanced Filtering** (band-pass, high-pass, low-pass) with Butterworth filters
- **Data Superimposition Analysis**:
  - **Time-based Superimposition**: Compare data from same time across different days for pattern analysis
  - **Sensor-based Superimposition**: Compare data from all sensors at same time for spatial analysis
  - **Time Range Selection**: Select precise start and end times for temporal analysis
  - **Daily Time Window Overlays (new)**: Date range + daily HH:MM window with optional time-of-day superimposed plot (color-coded by date)
  - **Interactive Controls**: Configurable time range selection, tolerance settings, superimpose-by-time-of-day toggle, and data alignment
  - **Rich Visualizations**: Color-coded plots with actual timestamps or overlaid time-of-day and quality metrics
  - **Data Quality Assessment**: Time alignment tracking and data availability statistics
- **Row-Level Labeling**: Label individual records or batch-label by sensor/time range
- **ML Pipeline**: Complete machine learning workflow from data labeling to prediction
- **Feature Extraction**: Advanced spectrogram and time-domain feature extraction with sliding windows
- **Model Evaluation**: Comprehensive performance metrics and visualization
- **Traceable Predictions**: Results show actual sensor IDs, record indices, and timestamps
- **Real-time Analysis**: Support for large datasets with efficient processing
- **Enhanced Debugging**: Detailed diagnostics and error handling for better user experience
- **Troubleshooting Guide**: Built-in help for common issues like flat spectrograms
- **Correlation Analysis**:
  - **Sensor Comparison**: Compare any two sensors with independent field selection (Bx, By, Bz, or resultant)
  - **Statistical Metrics**: Pearson and Spearman correlation, R², p-values with automatic interpretation
  - **10 Visualization Methods**: Comprehensive Dashboard, Simple X-Y Scatter, Time Series Overlay, Density Heatmap, 3D Scatter Plot (Plotly), plus Seaborn Scatterplot, Heatmap, Correlogram, Bubble Plot, Connected Scatter
  - **Session State**: Correlation data persists when switching visualization methods
  - **Correlation Matrix**: Automatic analysis of all sensor pairs with network graphs and heatmaps

Notes (30 Oct 2025)
-------------------

- Spectrogram scaling behavior:
  - Early high-power segments can make later parts look uniformly purple when global scaling is used.
  - New options mitigate this: per-frequency (row-wise) normalization, stable percentile scaling that ignores edge columns, and safer enhancement to avoid saturation.
- New spectrogram defaults for stability:
  - Normalize per-frequency: ON
  - Stable percentile scaling: ON (trim edge bins = 1)
  - Gamma correction: OFF by default
  - Global z-score: OFF by default
  - Safer enhancement: standardized then capped scaling (lower default factor)
- Y-axis (frequency) limits:
  - Upper bound is Nyquist (fs/2) from the estimated sampling rate.
  - Practical minimum resolvable frequency ≈ 1 / window_seconds; longer windows resolve lower frequencies.


