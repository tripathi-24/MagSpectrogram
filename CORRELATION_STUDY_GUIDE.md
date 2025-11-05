# 🔗 Correlation Study Guide

## Overview

The Correlation Study feature allows you to analyze the relationships between different magnetic sensors and field components in your dataset. This powerful tool helps you understand how sensors behave relative to each other and identify patterns in your magnetic field data. The correlation analysis uses advanced statistical methods to quantify relationships between sensors and provides comprehensive visualizations to help interpret the results.

## 🧠 Core Concepts Explained

### What is Correlation Analysis?

**Correlation analysis** is a statistical method that measures how closely two variables (in this case, magnetic field measurements from different sensors) move together. Think of it like asking: "When one sensor shows a high value, does the other sensor also show a high value?"

**Simple Analogy**: Imagine two people walking side by side. If they always walk at the same speed and direction, they have a strong positive correlation. If one walks fast when the other walks slow, they have a negative correlation. If their walking patterns are completely unrelated, they have no correlation.

### Types of Correlation Used

#### 1. **Pearson Correlation (Linear Correlation)**
- **What it measures**: How well two variables follow a straight-line relationship
- **Range**: -1 to +1
- **Interpretation**:
  - +1: Perfect positive linear relationship (as one increases, the other increases proportionally)
  - -1: Perfect negative linear relationship (as one increases, the other decreases proportionally)
  - 0: No linear relationship
- **When to use**: When you expect a linear relationship between sensors

#### 2. **Spearman Correlation (Monotonic Correlation)**
- **What it measures**: How well two variables follow a monotonic relationship (always increasing or always decreasing, but not necessarily linear)
- **Range**: -1 to +1
- **Interpretation**: Same as Pearson, but captures non-linear monotonic relationships
- **When to use**: When the relationship might be curved but still consistently increasing or decreasing

#### 3. **R-squared (Coefficient of Determination)**
- **What it measures**: The proportion of variance in one variable that is explained by the other variable
- **Range**: 0 to 1
- **Interpretation**:
  - 0.8: 80% of the variation in sensor B is explained by sensor A
  - 0.2: Only 20% of the variation is explained
- **Formula**: R² = (Pearson correlation)²

### Statistical Significance (P-values)

**P-value** tells you how confident you can be in your correlation result:
- **p < 0.001**: Highly significant (99.9% confident)
- **p < 0.01**: Very significant (99% confident)
- **p < 0.05**: Significant (95% confident)
- **p ≥ 0.05**: Not significant (less than 95% confident)

**Simple explanation**: If p = 0.03, there's only a 3% chance that this correlation happened by random chance.

## 🔧 Data Processing and Alignment Techniques

### How the App Prepares Data for Correlation Analysis

Before computing correlations, the app performs several sophisticated data processing steps to ensure accurate results:

#### 1. **Data Filtering and Validation**
- **Sensor Data Extraction**: Separates data for each selected sensor
- **Field Value Extraction**: Extracts the specific magnetic field component (Bx, By, Bz, or resultant)
- **Data Validation**: Checks for missing or invalid data points
- **Error Handling**: Provides clear error messages if data is insufficient

#### 2. **Timestamp Alignment**
- **Problem**: Sensors may record data at different times or frequencies
- **Solution**: Creates a common time grid with 1-second resolution
- **Process**:
  1. Finds the overlapping time range between both sensors
  2. Creates a uniform time index (every 1 second)
  3. Interpolates missing values using time-based interpolation
  4. Handles duplicate timestamps by averaging values

#### 3. **Data Interpolation**
- **Method**: Time-based interpolation (`method="time"`)
- **Purpose**: Ensures both sensors have data points at the same time instances
- **How it works**: If sensor A has data at 10:00:00 and 10:00:02, but sensor B only has data at 10:00:01, the app estimates what sensor B's value would be at 10:00:00 and 10:00:02
- **Quality**: Uses linear interpolation which is appropriate for magnetic field data

#### 4. **Duplicate Timestamp Handling**
- **Problem**: Some sensors might record multiple values at the same timestamp
- **Solution**: Groups duplicate timestamps and takes the mean value
- **Example**: If sensor records [10:00:00: 100nT, 10:00:00: 102nT], it becomes [10:00:00: 101nT]

#### 5. **Common Index Finding**
- **Purpose**: Identifies time points where both sensors have valid data
- **Process**: Finds the intersection of time indices after alignment
- **Minimum Requirement**: At least 10 overlapping data points for reliable correlation

### Resultant Magnetic Field Calculation

When analyzing the "resultant" field, the app calculates:
```
Resultant = √(Bx² + By² + Bz²)
```

**Why this matters**: The resultant gives you the total magnetic field strength, which is often more meaningful than individual components for correlation analysis.

## 🚀 Getting Started

### Prerequisites
- Load your magnetic field data using the main application
- Ensure you have at least 2 sensors in your dataset
- Data should include timestamp, sensor_id, and magnetic field components (b_x, b_y, b_z)

### Accessing the Feature
1. Run the main application: `streamlit run app_simplified_v2.py`
2. Load the data using the "Import Data" section
3. Click on the "🔗 Correlation Study" tab

## 📊 Interface Overview

### Main Controls
- **Sensor Selection**: Two dropdown menus to select sensors for comparison
- **Field Selection**: Choose which magnetic field component to analyze
- **Time Range Slider**: Select the time period for analysis
- **Run Analysis Button**: Execute the correlation analysis
- **Correlation Matrix Checkbox**: View all sensor pair correlations

### Available Fields
- **Bx (X-component)**: Horizontal magnetic field component
- **By (Y-component)**: Vertical magnetic field component  
- **Bz (Z-component)**: Depth magnetic field component
- **Resultant Magnitude**: Total magnetic field strength (√(Bx² + By² + Bz²))

## 🔍 Step-by-Step Usage

### 1. Select Sensors
- Choose the first sensor from the "Select First Sensor" dropdown
- Choose the second sensor from the "Select Second Sensor" dropdown
- The second dropdown automatically excludes the first selected sensor

### 2. Choose Field
- Select the magnetic field component you want to analyze
- All sensors will be compared using the same field type

### 3. Set Time Range
- Use the time range slider to focus on specific periods
- The slider shows the full range of your data
- Select start and end times for your analysis

### 4. Run Analysis
- Click "🔍 Run Correlation Analysis" to start the computation
- A spinner will show while the analysis is running
- Results will appear below the controls

## 📈 Understanding Results

### Statistical Metrics
- **Pearson Correlation**: Linear relationship strength (-1 to +1)
- **Spearman Correlation**: Monotonic relationship strength (-1 to +1)
- **R² (R-squared)**: Proportion of variance explained
- **P-values**: Statistical significance of the correlation
- **Data Points**: Number of overlapping measurements used

### Correlation Strength Interpretation
- **|r| > 0.8**: Very strong correlation (sensors behave very similarly)
- **|r| > 0.6**: Strong correlation (sensors show strong relationship)
- **|r| > 0.4**: Moderate correlation (sensors show moderate relationship)
- **|r| > 0.2**: Weak correlation (sensors show weak relationship)
- **|r| ≤ 0.2**: Very weak/no correlation (sensors appear independent)

### Statistical Significance
- **p < 0.001**: Highly significant (very confident in the result)
- **p < 0.01**: Very significant (very confident in the result)
- **p < 0.05**: Significant (confident in the result)
- **p ≥ 0.05**: Not significant (not confident in the result)

## 📊 Comprehensive Visualization Methods

The app provides five different visualization methods to help you understand correlation relationships from different perspectives:

### 1. **Comprehensive Dashboard** (Default)
A 2×2 grid showing four related plots:

#### **Scatter Plot with Regression Line**
- **What it shows**: Data points from both sensors plotted against each other
- **Regression line**: Red line showing the best linear fit
- **Purpose**: Visualize the linear relationship strength and direction
- **How to read**: 
  - Points close to the line = strong linear correlation
  - Steep line = strong relationship
  - Flat line = weak relationship
- **Hover feature**: Shows exact values when you hover over points

#### **Time Series Comparison**
- **What it shows**: Both sensors' data overlaid on the same time axis
- **Purpose**: Identify temporal patterns and synchronization
- **How to read**:
  - Similar patterns = high correlation
  - Opposite patterns = negative correlation
  - No pattern similarity = low correlation
- **Color coding**: Blue for sensor 1, Red for sensor 2

#### **Residuals Plot**
- **What it shows**: Difference between actual values and predicted values from the regression line
- **Purpose**: Assess how well the linear model fits the data
- **How to read**:
  - Points scattered randomly around zero = good linear fit
  - Points forming patterns = non-linear relationship
  - Points far from zero = poor fit

#### **Correlation Matrix Heatmap**
- **What it shows**: 2×2 matrix with correlation values
- **Purpose**: Quick visual summary of correlation strength
- **Color coding**: 
  - Red = negative correlation
  - White = no correlation  
  - Blue = positive correlation
- **Values**: Exact correlation coefficients displayed

### 2. **Simple X-Y Scatter**
Enhanced scatter plot with additional features:

#### **Color Gradient by Time**
- **Feature**: Points colored by their time index
- **Purpose**: See if correlation changes over time
- **Color scale**: Viridis (purple to yellow) showing time progression

#### **Confidence Interval**
- **Feature**: Shaded area around regression line
- **Purpose**: Shows uncertainty in the linear relationship
- **Interpretation**: Wider band = more uncertainty

#### **Enhanced Hover Information**
- **Shows**: X value, Y value, and time index
- **Purpose**: Detailed inspection of individual data points

### 3. **Time Series Overlay**
Dual y-axis time series plot:

#### **Dual Y-Axis Design**
- **Left axis**: First sensor values
- **Right axis**: Second sensor values
- **Purpose**: Compare sensors with different scales

#### **Correlation Annotation**
- **Feature**: Text box showing correlation value
- **Purpose**: Quick reference while viewing time series

### 4. **Density Heatmap**
2D histogram showing data density:

#### **2D Histogram**
- **What it shows**: Where most data points are concentrated
- **Purpose**: Identify data clustering patterns
- **Color intensity**: Darker = more data points in that region

#### **Regression Line Overlay**
- **Feature**: Red line showing linear relationship
- **Purpose**: Compare linear trend with actual data distribution

### 5. **3D Scatter Plot**
Three-dimensional visualization:

#### **3D Axes**
- **X-axis**: First sensor values
- **Y-axis**: Second sensor values  
- **Z-axis**: Time (in seconds from start)

#### **Time-based Coloring**
- **Feature**: Points colored by time progression
- **Purpose**: See how correlation evolves over time
- **Color scale**: Viridis showing time progression

#### **Interactive Rotation**
- **Feature**: Click and drag to rotate the 3D plot
- **Purpose**: View data from different angles

## 🎯 Advanced Visualization Features

### **Correlation Matrix for All Sensors**
When you check "Show Correlation Matrix for All Sensors", you get three visualization types:

#### **1. Heatmap Matrix**
- **What it shows**: Correlation values for all sensor pairs
- **Layout**: Each field (Bx, By, Bz, resultant) gets its own heatmap
- **Color coding**: Red-blue scale with white at zero
- **Purpose**: Quick overview of all sensor relationships

#### **2. Network Graph**
- **What it shows**: Sensors as nodes, correlations as edges
- **Layout**: Circular arrangement of sensors
- **Edge properties**:
  - Thickness = correlation strength
  - Color = correlation strength (red=strong, gray=weak)
- **Purpose**: Visualize sensor network relationships

#### **3. Bar Chart Comparison**
- **What it shows**: Correlation values as bars
- **Color coding**: 
  - Red = strong correlation (|r| > 0.7)
  - Orange = moderate correlation (|r| > 0.3)
  - Light blue = weak correlation
- **Purpose**: Easy comparison of correlation strengths

## 🎯 Advanced Features and Pro Tips

### **Correlation Matrix for All Sensors**

#### **What it does**:
- Automatically computes correlations between ALL sensor pairs
- Analyzes all field types (Bx, By, Bz, resultant) simultaneously
- Provides comprehensive overview of sensor relationships

#### **When to use**:
- **Initial data exploration**: Get overview of all sensor relationships
- **Quality assessment**: Identify problematic sensors quickly
- **Sensor grouping**: Find which sensors behave similarly
- **Research planning**: Identify interesting sensor pairs for detailed analysis

#### **How to interpret results**:
- **High correlations across all fields**: Sensors are very similar (possibly redundant)
- **High correlation in one field only**: Sensors similar for specific magnetic component
- **Low correlations**: Sensors measure different phenomena or have quality issues

### **Detailed Statistics Section**

#### **Data Summary Information**:
- **Time range**: Exact start and end times of analysis
- **Data points**: Number of overlapping measurements used
- **Sensor statistics**: Mean and standard deviation for each sensor
- **Data quality metrics**: Helps assess reliability of results

#### **Correlation Interpretation Guide**:
The app provides automatic interpretation with color-coded indicators:

**🔴 Very Strong Correlation (|r| > 0.8)**:
- Sensors show very similar behavior
- Likely measuring the same phenomena
- High confidence in relationship

**🟠 Strong Correlation (|r| > 0.6)**:
- Sensors show strong relationship
- Good for most analytical purposes
- Reliable for predictions

**🟡 Moderate Correlation (|r| > 0.4)**:
- Sensors show moderate relationship
- Some shared behavior but also differences
- Useful for general trends

**🟢 Weak Correlation (|r| > 0.2)**:
- Sensors show weak relationship
- Limited shared behavior
- May indicate different measurement conditions

**⚪ Very Weak/No Correlation (|r| ≤ 0.2)**:
- Sensors appear independent
- May measure different phenomena
- Check sensor placement and calibration

#### **Statistical Significance Indicators**:
**✅ Highly Significant (p < 0.001)**:
- 99.9% confident in correlation
- Very reliable result
- Strong evidence of relationship

**✅ Very Significant (p < 0.01)**:
- 99% confident in correlation
- Highly reliable result
- Strong evidence of relationship

**✅ Significant (p < 0.05)**:
- 95% confident in correlation
- Reliable result
- Good evidence of relationship

**❌ Not Significant (p ≥ 0.05)**:
- Less than 95% confident
- Result may be due to chance
- Weak evidence of relationship

### **Pro Tips for Effective Analysis**

#### **1. Choosing the Right Field Type**
- **Start with resultant**: Gives overall magnetic field behavior
- **Use individual components**: For directional analysis
- **Compare across fields**: See if correlations are consistent

#### **2. Time Range Selection**
- **Longer periods**: More robust correlations, better statistical power
- **Shorter periods**: Focus on specific events or conditions
- **Avoid gaps**: Ensure continuous data coverage

#### **3. Sensor Selection Strategy**
- **Similar sensors first**: Compare sensors of the same type
- **Different locations**: Study spatial relationships
- **Different orientations**: Understand directional effects

#### **4. Visualization Selection**
- **Comprehensive Dashboard**: Best for detailed analysis
- **Simple Scatter**: Quick overview of relationship
- **Time Series Overlay**: See temporal patterns
- **3D Scatter**: Understand time evolution
- **Density Heatmap**: Identify data clustering

#### **5. Interpreting Results**
- **Consider physical meaning**: What do correlations tell you about the magnetic field?
- **Look at both Pearson and Spearman**: Linear vs. monotonic relationships
- **Check significance**: Don't trust correlations with p > 0.05
- **Use R² for practical significance**: How much variance is explained?

#### **6. Troubleshooting Low Correlations**
- **Check data quality**: Look for outliers or missing data
- **Verify time alignment**: Ensure proper temporal synchronization
- **Consider sensor differences**: Different types, orientations, or locations
- **Analyze time periods**: Correlations may vary over time

#### **7. Advanced Analysis Techniques**
- **Rolling correlations**: Track how correlations change over time
- **Cross-correlation**: Find time delays between sensors
- **Spectral analysis**: Analyze correlations in frequency domain
- **Clustering**: Group sensors based on correlation patterns

### **Common Pitfalls to Avoid**

#### **1. Correlation vs. Causation**
- **Remember**: High correlation doesn't mean one sensor causes the other
- **Reality**: Both sensors may respond to the same external factor
- **Solution**: Consider physical mechanisms and external influences

#### **2. Outlier Effects**
- **Problem**: A few extreme values can skew correlation results
- **Solution**: Check scatter plots for outliers, consider data cleaning
- **Alternative**: Use Spearman correlation (less sensitive to outliers)

#### **3. Non-linear Relationships**
- **Problem**: Pearson correlation only detects linear relationships
- **Solution**: Use Spearman correlation for monotonic relationships
- **Visualization**: Look at scatter plots to identify patterns

#### **4. Sample Size Issues**
- **Problem**: Too few data points lead to unreliable correlations
- **Solution**: Use at least 10-20 data points, preferably more
- **Check**: Look at the "Data Points" metric in results

#### **5. Time Series Pitfalls**
- **Problem**: Time series data may have trends that affect correlations
- **Solution**: Detrend data or use shorter time periods
- **Alternative**: Use first differences instead of raw values

### **Performance Optimization Tips**

#### **1. Data Management**
- **Filter early**: Use time range selection to reduce data size
- **Cache results**: The app caches processed data for faster subsequent analysis
- **Clean data**: Remove obvious outliers before analysis

#### **2. Analysis Efficiency**
- **Start simple**: Begin with basic scatter plots
- **Use presets**: Leverage quick frequency presets for magnetic data
- **Batch analysis**: Use correlation matrix for multiple comparisons

#### **3. Visualization Performance**
- **Choose appropriate plots**: Simple plots load faster than complex 3D visualizations
- **Limit data points**: Very large datasets may slow down interactive plots
- **Use static plots**: For reports, consider static versions of interactive plots

## 💡 Best Practices

### Data Quality
- Ensure sufficient overlapping data points (minimum 10 recommended)
- Check for missing or invalid data in your time range
- Verify that both sensors have data in the selected time period

### Time Range Selection
- Use longer time periods for more robust correlations
- Avoid periods with sensor malfunctions or data gaps
- Consider seasonal or temporal patterns in your data

### Field Selection
- Start with resultant magnitude for overall sensor behavior
- Use individual components (Bx, By, Bz) for specific directional analysis
- Compare the same field type across sensors

### Interpretation
- Consider the physical meaning of correlations in your context
- High correlation might indicate similar environmental conditions
- Low correlation might suggest different sensor locations or sensitivities
- Look at both Pearson (linear) and Spearman (monotonic) correlations

## 🔧 Troubleshooting

### Common Issues

#### "No overlapping time range between sensors"
- **Cause**: Sensors don't have data in the same time period
- **Solution**: Expand your time range or check data availability

#### "Insufficient overlapping data points"
- **Cause**: Too few data points in the selected time range
- **Solution**: Increase the time range or check data quality

#### "One or both sensors have no data"
- **Cause**: Selected sensors don't exist in the dataset
- **Solution**: Check sensor IDs and ensure data is loaded correctly

#### "Correlation computation failed"
- **Cause**: Data processing error
- **Solution**: Check data format and try a different time range

### Data Requirements
- CSV files must have columns: id, b_x, b_y, b_z, timestamp, sensor_id
- Timestamps must be properly formatted
- Magnetic field values must be numeric
- Sensor IDs must be consistent across records

## 📚 Real-World Applications and Use Cases

### **1. Sensor Calibration and Validation**

#### **Purpose**: Ensure sensors are working correctly and consistently
#### **What to look for**:
- **High correlation (r > 0.8)**: Sensors are well-calibrated and measuring the same phenomena
- **Low correlation (r < 0.3)**: Possible calibration drift or sensor malfunction
- **Negative correlation**: One sensor might be inverted or have opposite polarity

#### **Practical Example**:
```
Sensor A: [100, 102, 98, 101, 99] nT
Sensor B: [101, 103, 99, 102, 100] nT
Correlation: r = 0.95 (excellent calibration)
```

#### **Action Items**:
- If correlation drops over time → recalibrate sensors
- If correlation is consistently low → check sensor placement and orientation
- If correlation is negative → check sensor polarity

### **2. Environmental and Spatial Analysis**

#### **Purpose**: Understand how magnetic fields vary across different locations
#### **What to look for**:
- **High correlation**: Similar environmental conditions (same geological features, similar interference)
- **Low correlation**: Different local conditions (different soil types, varying interference sources)
- **Time-varying correlation**: Environmental changes over time

#### **Practical Example**:
```
Urban sensor vs Rural sensor:
- High correlation during quiet periods (natural geomagnetic field)
- Low correlation during rush hour (urban electromagnetic interference)
```

#### **Research Applications**:
- **Geological mapping**: Correlate sensors across different rock types
- **Pollution monitoring**: Detect electromagnetic interference patterns
- **Seismic studies**: Monitor magnetic field changes before earthquakes

### **3. Quality Control and Anomaly Detection**

#### **Purpose**: Monitor data quality and detect unusual events
#### **What to look for**:
- **Sudden correlation changes**: Possible sensor malfunction or environmental event
- **Consistent low correlation**: Data quality issues
- **Correlation patterns**: Identify systematic problems

#### **Monitoring Strategy**:
1. **Daily correlation checks**: Compare sensors daily
2. **Trend analysis**: Track correlation changes over time
3. **Threshold alerts**: Set up alerts for correlation drops below 0.5

#### **Anomaly Examples**:
- **Correlation drops to 0.1**: Sensor malfunction
- **Correlation becomes negative**: Sensor polarity issue
- **Correlation varies with time of day**: Environmental interference

### **4. Scientific Research Applications**

#### **Geomagnetic Studies**:
- **Diurnal variations**: Study daily magnetic field patterns
- **Magnetic storms**: Analyze correlation during solar events
- **Geomagnetic pulsations**: Detect wave-like magnetic field variations

#### **Geological Research**:
- **Magnetic anomalies**: Map subsurface geological features
- **Mineral exploration**: Detect magnetic mineral deposits
- **Tectonic studies**: Monitor magnetic field changes near fault lines

#### **Space Weather Research**:
- **Solar wind effects**: Study how solar activity affects ground magnetic fields
- **Auroral studies**: Correlate ground sensors with auroral activity
- **Magnetospheric studies**: Understand Earth's magnetic field dynamics

### **5. Industrial and Commercial Applications**

#### **Mining and Exploration**:
- **Mineral detection**: Use correlation to identify magnetic ore bodies
- **Survey planning**: Optimize sensor placement for maximum coverage
- **Quality assurance**: Ensure consistent measurements across survey areas

#### **Environmental Monitoring**:
- **Pollution detection**: Monitor electromagnetic pollution from industrial sources
- **Wildlife studies**: Track animal migration using magnetic field changes
- **Climate research**: Study long-term magnetic field variations

#### **Infrastructure Monitoring**:
- **Power line effects**: Monitor magnetic fields near electrical infrastructure
- **Transportation impact**: Study magnetic field changes from vehicles
- **Urban planning**: Assess electromagnetic environment for new developments

### **6. Educational and Training Applications**

#### **Student Projects**:
- **Data analysis skills**: Learn statistical analysis with real data
- **Scientific method**: Practice hypothesis testing and validation
- **Visualization techniques**: Master data visualization and interpretation

#### **Research Training**:
- **Methodology development**: Learn proper experimental design
- **Statistical analysis**: Practice correlation analysis techniques
- **Scientific communication**: Learn to present results clearly

### **7. Troubleshooting and Diagnostics**

#### **Common Problems and Solutions**:

**Problem**: "No overlapping time range between sensors"
- **Cause**: Sensors recorded data at different times
- **Solution**: Check data collection schedules, expand time range

**Problem**: "Insufficient data points for correlation analysis"
- **Cause**: Too few overlapping measurements
- **Solution**: Increase time range, check data quality

**Problem**: "Correlation computation failed"
- **Cause**: Mathematical error or invalid data
- **Solution**: Check data format, verify numeric values

**Problem**: Low correlation despite similar sensors
- **Possible causes**:
  - Different sensor orientations
  - Different local environments
  - Sensor calibration issues
  - Data quality problems
- **Solutions**:
  - Check sensor alignment
  - Verify calibration
  - Clean data of outliers
  - Use longer time periods

### **8. Best Practices for Correlation Analysis**

#### **Data Collection**:
- **Synchronize timestamps**: Ensure accurate time alignment
- **Consistent sampling**: Use similar sampling rates
- **Quality control**: Monitor data quality continuously
- **Documentation**: Record sensor locations and orientations

#### **Analysis Workflow**:
1. **Data validation**: Check for missing or invalid data
2. **Time alignment**: Ensure proper temporal alignment
3. **Correlation calculation**: Use appropriate correlation method
4. **Statistical testing**: Verify significance of results
5. **Visualization**: Create clear, informative plots
6. **Interpretation**: Consider physical meaning of results

#### **Reporting Results**:
- **Include all metrics**: Pearson, Spearman, R², p-values
- **Show visualizations**: Use multiple plot types
- **Discuss limitations**: Acknowledge data quality issues
- **Provide context**: Explain physical meaning of correlations

## 🎓 Technical Details and Mathematical Concepts

### **Correlation Methods - Deep Dive**

#### **Pearson Correlation Coefficient**
**Mathematical Formula**:
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

**What each part means**:
- `xi, yi`: Individual data points from sensors X and Y
- `x̄, ȳ`: Mean values of sensors X and Y
- `(xi - x̄)`: How far each point is from the mean (deviation)
- `Σ`: Sum of all values
- **Result**: A number between -1 and +1

**How it works**:
1. Calculates how much each sensor deviates from its average
2. Multiplies these deviations together
3. If both sensors deviate in the same direction → positive correlation
4. If they deviate in opposite directions → negative correlation
5. Normalizes by the total variation to get a value between -1 and +1

#### **Spearman Rank Correlation**
**Mathematical Process**:
1. **Rank the data**: Convert values to ranks (1st, 2nd, 3rd, etc.)
2. **Apply Pearson formula**: Use the same formula but with ranks instead of values
3. **Result**: Measures monotonic (consistently increasing/decreasing) relationships

**Why use Spearman**:
- Works with non-linear but monotonic relationships
- Less sensitive to outliers
- Doesn't assume linearity

#### **R-squared (Coefficient of Determination)**
**Formula**: `R² = r²` (where r is Pearson correlation)

**What it tells you**:
- If r = 0.8, then R² = 0.64
- This means 64% of the variation in sensor Y is explained by sensor X
- The remaining 36% is due to other factors

### **Data Processing Pipeline**

#### **1. Time Alignment Algorithm**
```python
# Step 1: Find common time range
common_start = max(sensor1_start, sensor2_start)
common_end = min(sensor1_end, sensor2_end)

# Step 2: Create uniform time grid
time_step = 1 second
time_index = [common_start, common_start+1s, common_start+2s, ...]

# Step 3: Interpolate missing values
sensor1_aligned = interpolate(sensor1_data, time_index)
sensor2_aligned = interpolate(sensor2_data, time_index)
```

#### **2. Interpolation Method**
- **Type**: Linear interpolation
- **Formula**: `y = y1 + (y2-y1) × (x-x1)/(x2-x1)`
- **Why linear**: Appropriate for magnetic field data which changes smoothly
- **Quality**: Preserves the general trend while filling gaps

#### **3. Duplicate Handling**
```python
# If multiple values at same timestamp:
duplicate_values = [100, 102, 98]  # nT values
final_value = mean(duplicate_values) = 100  # nT
```

### **Statistical Testing Details**

#### **P-value Calculation**
- **Method**: Two-tailed t-test
- **Null hypothesis**: "There is no correlation" (r = 0)
- **Alternative hypothesis**: "There is a correlation" (r ≠ 0)
- **Test statistic**: `t = r × √[(n-2)/(1-r²)]`
- **Degrees of freedom**: n-2 (where n = number of data points)

#### **Confidence Intervals**
- **Level**: 95% confidence interval around regression line
- **Formula**: `CI = y_pred ± t_critical × SE × √[1 + 1/n + (x-x̄)²/SSx]`
- **Where**:
  - `t_critical`: t-value for 95% confidence
  - `SE`: Standard error of prediction
  - `SSx`: Sum of squared deviations in x

### **Error Handling and Validation**

#### **Data Quality Checks**
1. **Minimum data points**: At least 10 overlapping points required
2. **Time range validation**: Ensures overlapping time periods exist
3. **Numeric validation**: Converts string values to numbers, handles errors
4. **Missing data handling**: Uses interpolation to fill gaps

#### **Common Error Messages and Solutions**
- **"No overlapping time range"**: Sensors don't have data in the same period
- **"Insufficient data points"**: Less than 10 overlapping measurements
- **"Correlation computation failed"**: Mathematical error in calculation
- **"One or both sensors have no data"**: Selected sensors don't exist in dataset

### **Performance Optimizations**

#### **Caching Strategy**
- **Function**: `@st.cache_data(show_spinner=False)`
- **Purpose**: Prevents recomputation of expensive operations
- **When used**: Data reading and processing functions

#### **Memory Management**
- **Data filtering**: Only processes selected time range
- **Efficient indexing**: Uses pandas boolean indexing for fast filtering
- **Garbage collection**: Automatically cleans up unused data

### **Mathematical Libraries Used**
- **scipy.stats**: For correlation calculations and statistical tests
- **numpy**: For mathematical operations and array processing
- **pandas**: For data manipulation and time series operations
- **plotly**: For interactive visualizations

### **Precision and Accuracy**
- **Correlation precision**: 4 decimal places (0.0001)
- **P-value format**: Scientific notation (e.g., 1.23e-05)
- **Time precision**: 1-second resolution
- **Magnetic field units**: Nanotesla (nT) with 2 decimal places

## 🎓 Summary and Key Takeaways

### **What You've Learned**

This comprehensive guide has covered all aspects of the Correlation Study feature in the MagSpectrogram application:

#### **Core Concepts**:
- **Correlation analysis**: Statistical method to measure relationships between sensors
- **Pearson correlation**: Measures linear relationships (-1 to +1)
- **Spearman correlation**: Measures monotonic relationships (less sensitive to outliers)
- **R-squared**: Proportion of variance explained by the relationship
- **P-values**: Statistical significance and confidence levels

#### **Data Processing**:
- **Time alignment**: Synchronizes data from different sensors
- **Interpolation**: Fills gaps in time series data
- **Quality validation**: Ensures data integrity before analysis
- **Resultant calculation**: Computes total magnetic field strength

#### **Visualization Methods**:
- **Comprehensive Dashboard**: 4-panel analysis with scatter, time series, residuals, and correlation matrix
- **Simple Scatter**: Enhanced scatter plot with confidence intervals
- **Time Series Overlay**: Dual-axis comparison over time
- **Density Heatmap**: 2D histogram showing data concentration
- **3D Scatter**: Three-dimensional visualization with time axis

#### **Advanced Features**:
- **Correlation Matrix**: Automatic analysis of all sensor pairs
- **Network Graphs**: Visual representation of sensor relationships
- **Statistical Interpretation**: Automatic correlation strength assessment
- **Error Handling**: Comprehensive validation and error messages

### **Best Practices Summary**

1. **Start with resultant field** for overall magnetic field behavior
2. **Use longer time periods** for more robust correlations
3. **Check both Pearson and Spearman** correlations
4. **Verify statistical significance** (p < 0.05)
5. **Consider physical meaning** of correlations
6. **Use multiple visualization methods** for comprehensive understanding
7. **Document sensor locations and orientations** for proper interpretation

### **Common Applications**

- **Sensor calibration**: Verify sensor accuracy and consistency
- **Quality control**: Monitor data quality and detect anomalies
- **Environmental analysis**: Study spatial and temporal patterns
- **Research**: Investigate magnetic field relationships and phenomena
- **Education**: Learn statistical analysis and data visualization

## 📞 Support and Troubleshooting

### **Getting Help**

If you encounter issues or have questions:

1. **Check this guide**: Review relevant sections for detailed explanations
2. **Verify data format**: Ensure CSV files have required columns
3. **Check data quality**: Look for missing values or outliers
4. **Try different parameters**: Adjust time ranges or sensor combinations
5. **Review error messages**: The app provides specific error descriptions

### **Common Issues and Quick Fixes**

| Problem | Quick Fix |
|---------|-----------|
| "No overlapping time range" | Expand time range or check data availability |
| "Insufficient data points" | Increase time range or check data quality |
| "Correlation computation failed" | Verify data format and numeric values |
| Low correlation despite similar sensors | Check sensor alignment and calibration |
| Flat or uninformative plots | Try different visualization methods |

### **Data Requirements Checklist**

- ✅ CSV files with columns: id, b_x, b_y, b_z, timestamp, sensor_id
- ✅ Properly formatted timestamps
- ✅ Numeric magnetic field values
- ✅ Consistent sensor IDs across records
- ✅ At least 2 sensors in dataset
- ✅ Minimum 10 overlapping data points

### **Performance Tips**

- **Use time filtering**: Select specific time ranges to improve performance
- **Start with simple plots**: Basic visualizations load faster
- **Cache results**: The app automatically caches processed data
- **Clean data**: Remove obvious outliers before analysis

## 🚀 Next Steps

### **For Beginners**
1. Start with the basic correlation analysis using resultant field
2. Try different visualization methods to understand your data
3. Experiment with different time ranges and sensor combinations
4. Use the correlation matrix to explore all sensor relationships

### **For Advanced Users**
1. Implement rolling correlation analysis for time-varying relationships
2. Use cross-correlation to find time delays between sensors
3. Apply spectral analysis to study frequency-domain correlations
4. Develop custom analysis workflows based on your specific needs

### **For Researchers**
1. Document your analysis methodology and parameters
2. Report both Pearson and Spearman correlations with significance levels
3. Consider the physical meaning of correlations in your context
4. Use multiple visualization methods to support your findings

---

**The Correlation Study feature provides powerful insights into your magnetic field data relationships. Use it to understand sensor behavior, validate data quality, and discover patterns in your measurements. With this comprehensive guide, you now have all the knowledge needed to effectively use this advanced analytical tool.**
