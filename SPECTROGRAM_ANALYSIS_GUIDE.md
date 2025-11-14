# 📊 Spectrogram Analysis Guide: Information Extractable from Magnetic Field Spectrograms

## 🎯 Overview

A spectrogram is a powerful tool for analyzing magnetic field data. It transforms time-domain signals into a frequency-time representation, revealing hidden patterns, anomalies, and characteristics that are not visible in raw time series data.

---

## 🔬 1. Frequency Domain Information

### **1.1 Dominant Frequencies**
- **What it shows**: The most prominent frequency components in your magnetic field data
- **How to identify**: Bright horizontal bands in the spectrogram
- **Applications**:
  - Identify periodic magnetic variations
  - Detect resonant frequencies
  - Find characteristic frequencies of magnetic sources

**Example**: A bright band at 0.001 Hz (1 mHz) indicates a very slow, periodic variation occurring every ~16.7 minutes.

### **1.2 Frequency Bandwidth**
- **What it shows**: The spread of frequency content
- **How to measure**: Width of frequency bands in the spectrogram
- **Applications**:
  - Distinguish between narrowband (focused) and broadband (spread) signals
  - Identify noise characteristics
  - Characterize signal quality

### **1.3 Spectral Centroid**
- **What it shows**: The "center of mass" of the frequency distribution
- **Calculation**: Weighted average frequency
- **Applications**:
  - Characterize overall frequency content
  - Track changes in spectral characteristics over time
  - Compare different sensors or time periods

### **1.4 Harmonic Content**
- **What it shows**: Multiple frequencies related by integer ratios (harmonics)
- **How to identify**: Parallel horizontal bands at regular frequency intervals
- **Applications**:
  - Identify non-linear systems
  - Detect mechanical vibrations affecting sensors
  - Characterize periodic disturbances

---

## ⏱️ 2. Time-Domain Patterns

### **2.1 Temporal Evolution**
- **What it shows**: How frequency content changes over time
- **How to identify**: Vertical patterns or color changes in the spectrogram
- **Applications**:
  - Track magnetic field evolution
  - Identify transient events
  - Monitor long-term trends

**Example**: A vertical bright streak indicates a sudden frequency burst at a specific time.

### **2.2 Periodic Patterns**
- **What it shows**: Repeating patterns in time
- **How to identify**: Diagonal or repeating vertical patterns
- **Applications**:
  - Identify diurnal variations (daily cycles)
  - Detect seasonal patterns
  - Find recurring events

### **2.3 Event Duration**
- **What it shows**: How long specific frequency components persist
- **How to measure**: Width of bright regions along the time axis
- **Applications**:
  - Characterize magnetic storms
  - Identify transient vs. persistent signals
  - Analyze event dynamics

---

## 🚨 3. Anomaly Detection

### **3.1 Anomalous Frequency Bursts**
- **What it shows**: Unexpected frequency components appearing at specific times
- **How to identify**: Isolated bright spots or sudden color changes
- **Applications**:
  - Detect magnetic field disturbances
  - Identify sensor malfunctions
  - Find environmental interference

### **3.2 Frequency Gaps**
- **What it shows**: Missing frequency content where it should exist
- **How to identify**: Dark regions in expected frequency bands
- **Applications**:
  - Identify data quality issues
  - Detect filtering effects
  - Find signal dropouts

### **3.3 Statistical Anomalies**
- **What it shows**: Frequency-time points that deviate significantly from normal patterns
- **How to measure**: Z-score analysis (values > 2-3 standard deviations)
- **Applications**:
  - Automated anomaly detection
  - Quality control
  - Event classification

---

## 📈 4. Statistical Features

### **4.1 Power Spectral Density (PSD)**
- **What it shows**: Power distribution across frequencies
- **Units**: dB/Hz or nT²/Hz
- **Applications**:
  - Quantify signal strength at different frequencies
  - Compare power levels across time
  - Characterize noise floors

### **4.2 Mean Power per Frequency**
- **What it shows**: Average power at each frequency across all time
- **Calculation**: Mean of each frequency row in the spectrogram
- **Applications**:
  - Identify persistent frequency components
  - Characterize background noise
  - Create frequency profiles

### **4.3 Power Variability**
- **What it shows**: How much power fluctuates at each frequency
- **Calculation**: Standard deviation of power across time
- **Applications**:
  - Identify stable vs. variable frequencies
  - Characterize signal stability
  - Detect intermittent signals

### **4.4 Peak Power**
- **What it shows**: Maximum power at each frequency
- **Applications**:
  - Identify maximum signal strength
  - Detect extreme events
  - Characterize dynamic range

---

## 🌍 5. Geomagnetic Applications

### **5.1 Geomagnetic Storms**
- **What to look for**: Broadband increases in power, especially at low frequencies (0.001-0.1 Hz)
- **Characteristics**:
  - Sudden power increases
  - Multiple frequency components activated
  - Duration: hours to days
- **Applications**:
  - Space weather monitoring
  - Solar activity correlation
  - Ionospheric disturbance detection

### **5.2 Diurnal Variations**
- **What to look for**: Regular patterns repeating every ~24 hours
- **Characteristics**:
  - Low frequency components (< 0.00001 Hz)
  - Consistent daily patterns
  - Seasonal variations
- **Applications**:
  - Solar-terrestrial interaction studies
  - Ionospheric current monitoring
  - Long-term trend analysis

### **5.3 Magnetic Pulsations**
- **What to look for**: Narrowband signals at specific frequencies (typically 0.001-1 Hz)
- **Types**:
  - **Pc1** (0.2-5 Hz): Ion cyclotron waves
  - **Pc2** (0.1-0.2 Hz): Fast magnetospheric waves
  - **Pc3** (22-100 mHz): Ultra-low frequency waves
  - **Pc4** (6.7-22 mHz): Magnetospheric oscillations
  - **Pc5** (1.7-6.7 mHz): Field line resonances
- **Applications**:
  - Magnetospheric physics
  - Space weather prediction
  - Ionospheric research

### **5.4 Local Magnetic Anomalies**
- **What to look for**: Persistent frequency patterns at specific locations
- **Characteristics**:
  - Stationary frequency components
  - Sensor-specific patterns
  - Geological correlation
- **Applications**:
  - Mineral exploration
  - Geological mapping
  - Archaeological surveys

---

## 🔧 6. Signal Quality Metrics

### **6.1 Signal-to-Noise Ratio (SNR)**
- **What it shows**: Ratio of signal power to noise power
- **How to calculate**: Compare power in signal bands vs. noise bands
- **Applications**:
  - Assess data quality
  - Optimize sensor placement
  - Validate measurements

### **6.2 Noise Floor**
- **What it shows**: Minimum detectable signal level
- **How to identify**: Baseline power level across all frequencies
- **Applications**:
  - Determine measurement sensitivity
  - Characterize sensor performance
  - Set detection thresholds

### **6.3 Frequency Resolution**
- **What it shows**: Ability to distinguish between close frequencies
- **Determined by**: Window size and sampling rate
- **Applications**:
  - Optimize spectrogram parameters
  - Balance resolution vs. time resolution
  - Characterize analysis capabilities

### **6.4 Time Resolution**
- **What it shows**: Ability to resolve rapid temporal changes
- **Determined by**: Window size and overlap
- **Applications**:
  - Detect transient events
  - Track rapid changes
  - Optimize analysis windows

---

## 🎯 7. Practical Extraction Methods

### **7.1 Feature Extraction for Machine Learning**

From your codebase (`ml_app.py`), you can extract:

1. **Frequency Domain Features**:
   - Mean power per frequency
   - Standard deviation per frequency
   - Maximum power per frequency
   - Dominant frequency
   - Spectral centroid
   - Spectral bandwidth

2. **Time Domain Features**:
   - Mean power per time bin
   - Standard deviation per time bin
   - Maximum power per time bin
   - Temporal trends

3. **Overall Statistics**:
   - Global mean, std, max, min
   - Percentiles (25th, 50th, 75th)
   - Range and variance

### **7.2 Automated Analysis**

```python
# Example: Extract dominant frequency
dominant_freq_idx = np.argmax(freq_means)
dominant_freq = f[dominant_freq_idx]
dominant_power = freq_means[dominant_freq_idx]

# Example: Calculate spectral centroid
spectral_centroid = np.sum(f * freq_means) / np.sum(freq_means)

# Example: Detect anomalies
z_scores = (sxx - mean) / std
anomalies = np.abs(z_scores) > threshold
```

---

## 📊 8. Visualization Insights

### **8.1 Color Patterns**
- **Bright colors (yellow/white)**: High power, strong signals
- **Dark colors (blue/purple)**: Low power, weak signals or noise
- **Color gradients**: Power transitions, signal evolution

### **8.2 Horizontal Patterns**
- **Horizontal bands**: Persistent frequency components
- **Band width**: Frequency bandwidth of signals
- **Band intensity**: Signal strength

### **8.3 Vertical Patterns**
- **Vertical streaks**: Transient events, sudden changes
- **Vertical bands**: Time-localized frequency content
- **Gaps**: Missing data or signal dropouts

### **8.4 Diagonal Patterns**
- **Diagonal lines**: Frequency-modulated signals
- **Chirp signals**: Frequency sweeping over time
- **Doppler effects**: Frequency shifts due to motion

---

## 🔬 9. Advanced Analysis Techniques

### **9.1 Cross-Spectrum Analysis**
- Compare spectrograms from multiple sensors
- Identify correlated frequency components
- Detect spatial patterns

### **9.2 Coherence Analysis**
- Measure frequency-dependent correlation
- Identify common sources
- Characterize sensor relationships

### **9.3 Time-Frequency Localization**
- Identify when specific frequencies occur
- Track frequency evolution
- Characterize transient events

### **9.4 Multi-Scale Analysis**
- Analyze different time scales simultaneously
- Identify patterns at multiple resolutions
- Characterize scale-dependent phenomena

---

## 🎓 10. Interpretation Guidelines

### **10.1 Frequency Ranges for Magnetic Data**

- **Ultra-low (μHz, < 0.000001 Hz)**: 
  - Geomagnetic secular variation
  - Long-term trends
  - Diurnal variations

- **Very low (mHz, 0.001-0.1 Hz)**:
  - Geomagnetic pulsations (Pc3-Pc5)
  - Magnetic storms
  - Ionospheric currents

- **Low (Hz, 0.1-10 Hz)**:
  - Fast pulsations (Pc1-Pc2)
  - Local magnetic anomalies
  - Sensor vibrations

- **Medium (10-100 Hz)**:
  - Electromagnetic interference
  - Power line harmonics
  - Electronic noise

### **10.2 Common Patterns**

1. **Flat Spectrogram**:
   - Constant signal (DC component)
   - Insufficient variation
   - Need signal enhancement

2. **Horizontal Bands**:
   - Persistent frequency components
   - Periodic signals
   - Background sources

3. **Vertical Streaks**:
   - Transient events
   - Sudden changes
   - Anomalies

4. **Diagonal Patterns**:
   - Frequency modulation
   - Chirp signals
   - Time-varying frequencies

---

## 🛠️ 11. Tools and Techniques

### **11.1 Window Functions**
- **Hann/Hamming**: Good general purpose, smooth edges
- **Blackman**: Better frequency resolution
- **Bartlett**: Triangular, simple

### **11.2 Overlap Ratios**
- **0%**: No overlap, fastest but may miss events
- **50%**: Good balance (recommended)
- **75-95%**: High resolution, slower computation

### **11.3 Window Sizes**
- **Small windows**: Better time resolution, lower frequency resolution
- **Large windows**: Better frequency resolution, lower time resolution
- **Adaptive**: Adjust based on signal characteristics

---

## 📝 12. Summary: Key Information Extractable

| Information Type | What It Reveals | Applications |
|-----------------|----------------|--------------|
| **Dominant Frequencies** | Main frequency components | Pattern identification, source characterization |
| **Temporal Evolution** | How frequencies change over time | Event tracking, trend analysis |
| **Anomalies** | Unusual frequency-time patterns | Quality control, event detection |
| **Power Distribution** | Signal strength across frequencies | SNR analysis, noise characterization |
| **Harmonic Content** | Frequency relationships | System identification, vibration analysis |
| **Spectral Statistics** | Quantitative frequency features | Machine learning, classification |
| **Geomagnetic Phenomena** | Space weather, magnetic storms | Scientific research, monitoring |
| **Signal Quality** | Data reliability and sensitivity | Sensor validation, optimization |

---

## 🚀 Next Steps

1. **Automate Feature Extraction**: Use the methods in `ml_app.py` to extract features automatically
2. **Build Classifiers**: Train ML models on extracted features
3. **Correlation Analysis**: Compare spectrograms across sensors
4. **Anomaly Detection**: Implement automated anomaly detection systems
5. **Real-time Monitoring**: Set up continuous spectrogram analysis

---

## 📚 References

- Your codebase: `ml_app.py` (feature extraction)
- Your codebase: `app_12_Nov.py` (spectrogram computation)
- Geomagnetic pulsation classification: Pc1-Pc5 types
- Signal processing: FFT, windowing, overlap

---

*This guide is based on your magnetic field data analysis codebase and standard signal processing techniques.*

