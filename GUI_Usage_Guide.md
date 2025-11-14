# 🔍 MagSpectrogram GUI - Usage Guide

## 🚀 **How to Run**

### **Main Data Visualization App (Simplified & Cleaned):**
```bash
streamlit run app_12_Nov.py
```

### **Alternative Data Visualization App:**
```bash
streamlit run app_simplified_v2.py
```

### **Sampling Rate Analyzer:**
```bash
python3 analyze_sampling_rate.py
```

## 📋 **Features**

### **Main Data Visualization Interface:**
- **📊 Data Visualization Tab**: Time series plots and spectrograms with **anomaly detection and highlighting**
- **🔗 Correlation Study Tab**: Compare sensors and analyze relationships
- **ℹ️ About Tab**: Application information and help

### **🆕 Anomaly Detection Features (app_12_Nov.py):**
- **🔍 Time Series Anomaly Detection**: Automatically detects and highlights anomalous regions in time series plots
  - **Light red (light coral) shaded regions with red borders** mark anomaly time ranges (highly visible)
  - Uses enhanced multi-method detection (z-score, IQR, rolling window, change points, percentiles)
  - Clean visualization without marker clutter
- **🔍 Spectrogram Anomaly Detection**: Detects anomalies in frequency-time space
  - Red shaded regions highlight anomaly time periods
  - Red 'X' markers show specific anomaly frequency-time points
  - Groups anomalies into regions for better visualization
- **⚙️ Unified Controls**: Single control panel for both time series and spectrogram anomaly detection
  - Enable/disable separately for time series and spectrogram
  - Adjustable z-score threshold (1.0 to 5.0, default: 2.0)
  - Real-time anomaly count and region statistics

### **Correlation Study Features:**
- **🔍 Sensor Comparison**: Select any two sensors for correlation analysis
- **📈 Field Selection**: Choose Bx, By, Bz, or resultant magnetic field
- **📊 Statistical Analysis**: Pearson and Spearman correlation coefficients
- **📉 Comprehensive Visualizations**: Scatter plots, time series, residuals, correlation matrix
- **🎯 Significance Testing**: P-values and statistical interpretation
- **⏰ Time Range Selection**: Focus on specific time periods

### **Sampling Rate Analyzer Interface:**
- **📁 File Selection**: Click "Browse for CSV File" to select your data file
- **🔍 Analysis**: Click "Analyze Sampling Rate" to process the file
- **📊 Results**: View detailed analysis results in the text area
- **💾 Export**: Save results to a text file
- **🗑️ Clear**: Clear results and start over

### **GUI Components:**

1. **File Selection Section**
   - Shows selected file path
   - Browse button to open file dialog
   - Analyze button (enabled after file selection)

2. **Progress Bar**
   - Shows analysis progress
   - Prevents multiple simultaneous analyses

3. **Results Display**
   - Scrollable text area with analysis results
   - Detailed sampling rate information
   - Per-sensor analysis (if multiple sensors)

4. **Control Buttons**
   - Clear Results: Remove current analysis
   - Export Results: Save results to file
   - Exit: Close the application

5. **Status Bar**
   - Shows current operation status
   - File selection confirmations
   - Analysis completion status

## 📊 **What You'll See**

### **Analysis Results Include:**
- Total records and time range
- Calculated sampling rate
- Sampling regularity analysis
- Per-sensor sampling rates
- Nyquist frequency
- Recommendations for spectrogram analysis

### **Example Output:**
```
Analyzing: Data_AllSensors_10Oct_to_15Oct1030h_Downsample_60.csv
============================================================
Total records: 376,355
Time range: 2025-10-10 00:00:00 to 2025-10-15 10:30:00
Duration: 5 days 10:30:00
Unique sensors: 3

Time interval statistics:
  Mean interval: 3.750000 seconds
  Median interval: 3.750000 seconds
  Std deviation: 0.000000 seconds

Calculated sampling rate: 0.266667 Hz
Expected from filename (Downsample_60): 0.016667 Hz

🎯 SUMMARY:
📈 Sampling Rate: 0.266667 Hz
📊 Nyquist Frequency: 0.133333 Hz
```

## 🎯 **File Requirements**

Your CSV file should have these columns:
- **timestamp**: Date/time of measurements
- **b_x, b_y, b_z**: Magnetic field components (nT)
- **sensor_id**: Optional, for multi-sensor analysis

## 💡 **Tips**

1. **File Selection**: Use the browse button to navigate to your CSV file
2. **Large Files**: The analysis runs in a separate thread, so the GUI won't freeze
3. **Export Results**: Save your analysis results for future reference
4. **Multiple Files**: You can analyze different files by selecting new ones
5. **Error Handling**: Clear error messages if something goes wrong

## 🔧 **Troubleshooting**
## 🎛️ New Spectrogram Options (30 Oct 2025)

- Normalize per-frequency (row-wise): ON by default. Stabilizes colors across time and prevents early bright patches from compressing later segments to purple.
- Stable percentile scaling: ON by default. Computes contrast percentiles on the middle of the spectrogram (trimming edge columns) to avoid edge transients dominating. Adjustable “Trim edge time bins”.
- Safer enhancement: The signal is standardized then scaled with a capped factor to avoid overflow/saturation. Default factor reduced.
- Gamma correction: OFF by default.
- Global z-score: OFF by default.

Practical notes:
- If later time looks uniformly purple, keep per-frequency normalization and stable scaling ON, reduce enhancement, and keep gamma/z-score OFF.
- Y-axis frequency limits: you can set max_freq up to Nyquist (fs/2). Minimum resolvable frequency ≈ 1 / window_seconds (use longer windows for μHz–mHz).


### **If GUI doesn't start:**
```bash
# Install tkinter (if needed)
sudo apt-get install python3-tk  # Ubuntu/Debian
brew install python-tk           # macOS with Homebrew
```

### **If analysis fails:**
- Check that your CSV file has the required columns
- Ensure the timestamp column is properly formatted
- Verify the file isn't corrupted

## 🔍 **Anomaly Detection - Step by Step**

### **1. Load Your Data**
1. Run: `streamlit run app_12_Nov.py`
2. Click "📁 Import Data" to upload your CSV files
3. Or check "Load all CSVs from ./Dataset" for quick loading

### **2. Enable Anomaly Detection**
1. Select "Time Series & Spectrogram" visualization mode
2. Expand "🔍 Anomaly Detection" in the sidebar
3. Enable "Time Series Anomaly Detection" (default: ON)
4. Enable "Spectrogram Anomaly Detection" (default: ON)
5. Adjust "Anomaly Threshold (z-score)" as needed (default: 2.0)
   - Lower values (1.0-1.5): More sensitive, detects more anomalies
   - Higher values (3.0-5.0): Less sensitive, only detects extreme anomalies

### **3. View Anomalies**
- **Time Series Plot**: 
  - **Light red (light coral) shaded vertical regions with red borders** indicate anomaly time ranges (highly visible)
  - Clean visualization without marker clutter
  - Regions are placed below the plot layer so they don't obscure the data
- **Spectrogram Plot**:
  - Red shaded regions highlight anomaly time periods
  - Red 'X' markers show anomaly frequency-time coordinates
  - Summary shows total anomaly points and regions found

### **4. Interpret Results**
- **Z-Score > 2.0**: Values more than 2 standard deviations from mean (moderate anomalies)
- **Z-Score > 3.0**: Values more than 3 standard deviations from mean (strong anomalies)
- **Z-Score > 4.0**: Values more than 4 standard deviations from mean (extreme anomalies)

## 🔗 **Correlation Study - Step by Step**

### **1. Load Your Data**
1. Run: `streamlit run app_12_Nov.py` or `streamlit run app_simplified_v2.py`
2. Click "Import Data" to upload your CSV files
3. Or check "Load all CSVs from ./Dataset" for quick loading

### **2. Navigate to Correlation Study**
1. Click on the "🔗 Correlation Study" tab
2. You'll see the correlation analysis interface

### **3. Select Sensors and Field**
1. **Choose First Sensor**: Select from the dropdown menu
2. **Choose Second Sensor**: Select a different sensor from the dropdown
3. **Select Field**: Choose Bx, By, Bz, or Resultant Magnitude
4. **Set Time Range**: Use the slider to select your analysis period

### **4. Run Analysis**
1. Click "🔍 Run Correlation Analysis" button
2. Wait for the analysis to complete (shows spinner)
3. View comprehensive results with multiple visualizations

### **5. Interpret Results**
- **Correlation Coefficients**: Pearson and Spearman values
- **Statistical Significance**: P-values and interpretation
- **Visual Analysis**: Scatter plots, time series, residuals
- **Correlation Matrix**: Heatmap showing relationship strength

### **6. Additional Features**
- **Correlation Matrix for All Sensors**: Check the box to see all sensor pair correlations
- **Detailed Statistics**: Expand the statistics section for more information

## 🚀 **Quick Start - Sampling Rate Analyzer**

1. Run: `python3 analyze_sampling_rate.py`
2. Click "📁 Browse for CSV File"
3. Select your magnetic field data CSV file
4. Click "🔍 Analyze Sampling Rate"
5. View results in the text area
6. Export results if needed

The GUI makes it easy to analyze your magnetic field data sampling rates and correlations with just a few clicks!
