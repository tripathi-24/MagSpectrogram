import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from scipy.signal import spectrogram
import joblib

# Import functions from the main app
from app import read_single_csv, read_multiple_csvs, compute_resultant, estimate_sampling_rate

class MagneticSignatureClassifier:
    """Machine Learning classifier for magnetic signature analysis"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.trained = False
        
    def extract_spectrogram_features(self, signal: np.ndarray, fs: float, 
                                   window_seconds: float = 1.0, overlap_ratio: float = 0.5) -> np.ndarray:
        """Extract features from spectrogram"""
        try:
            # Compute spectrogram
            nperseg = max(8, int(window_seconds * fs))
            nperseg = min(nperseg, len(signal) // 2)
            noverlap = int(overlap_ratio * nperseg)
            
            f, t, sxx = spectrogram(
                signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                scaling="density", mode="magnitude", detrend='constant'
            )
            
            # Extract statistical features from spectrogram
            features = []
            
            # 1. Frequency domain features
            freq_means = np.mean(sxx, axis=1)
            freq_stds = np.std(sxx, axis=1)
            freq_maxs = np.max(sxx, axis=1)
            
            # 2. Time domain features from spectrogram
            time_means = np.mean(sxx, axis=0)
            time_stds = np.std(sxx, axis=0)
            time_maxs = np.max(sxx, axis=0)
            
            # 3. Overall statistics
            features.extend([
                np.mean(freq_means), np.std(freq_means), np.max(freq_means),
                np.mean(freq_stds), np.std(freq_stds), np.max(freq_stds),
                np.mean(freq_maxs), np.std(freq_maxs), np.max(freq_maxs),
                np.mean(time_means), np.std(time_means), np.max(time_means),
                np.mean(time_stds), np.std(time_stds), np.max(time_stds),
                np.mean(time_maxs), np.std(time_maxs), np.max(time_maxs),
                np.mean(sxx), np.std(sxx), np.max(sxx), np.min(sxx),
                np.median(sxx), np.percentile(sxx, 25), np.percentile(sxx, 75)
            ])
            
            # 4. Dominant frequency features
            dominant_freq_idx = np.argmax(freq_means)
            dominant_freq = f[dominant_freq_idx] if len(f) > dominant_freq_idx else 0
            dominant_power = freq_means[dominant_freq_idx] if len(freq_means) > dominant_freq_idx else 0
            
            features.extend([dominant_freq, dominant_power])
            
            # 5. Spectral centroid and bandwidth
            spectral_centroid = np.sum(f * freq_means) / np.sum(freq_means) if np.sum(freq_means) > 0 else 0
            spectral_bandwidth = np.sqrt(np.sum(((f - spectral_centroid) ** 2) * freq_means) / np.sum(freq_means)) if np.sum(freq_means) > 0 else 0
            
            features.extend([spectral_centroid, spectral_bandwidth])
            
            return np.array(features)
            
        except Exception as e:
            st.error(f"Feature extraction failed: {e}")
            return np.zeros(29)  # Return zero features if extraction fails
    
    def extract_time_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract time domain features"""
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
            np.median(signal), np.percentile(signal, 25), np.percentile(signal, 75),
            np.var(signal), np.sqrt(np.mean(signal**2))  # RMS
        ])
        
        # Advanced features
        features.extend([
            np.mean(np.abs(np.diff(signal))),  # Mean absolute difference
            np.std(np.diff(signal)),  # Std of differences
            np.sum(np.abs(signal) > np.mean(np.abs(signal))),  # Peak count
            np.corrcoef(signal[:-1], signal[1:])[0, 1] if len(signal) > 1 else 0,  # Autocorrelation
        ])
        
        return np.array(features)
    
    def prepare_dataset(self, df: pd.DataFrame, row_labels: Dict[int, str], 
                       window_seconds: float = 1.0, overlap_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for training using row-level labels"""
        features_list = []
        labels_list = []
        
        # Group by sensor for efficient processing
        for sensor_id, sensor_data in df.groupby('sensor_id'):
            sensor_data = sensor_data.sort_values('timestamp')
            
            # Estimate sampling rate for this sensor
            fs = estimate_sampling_rate(sensor_data['timestamp'])
            if fs <= 0:
                continue
            
            # Process each axis and resultant
            for axis in ['b_x', 'b_y', 'b_z', 'resultant']:
                if axis == 'resultant':
                    values = compute_resultant(sensor_data).values
                else:
                    values = pd.to_numeric(sensor_data[axis], errors="coerce").dropna().values
                
                if len(values) < 50:  # Need sufficient data
                    continue
                
                # Extract features for this axis
                spectro_features = self.extract_spectrogram_features(values, fs, window_seconds, overlap_ratio)
                time_features = self.extract_time_domain_features(values)
                
                # Combine features
                combined_features = np.concatenate([spectro_features, time_features])
                
                # For each row in this sensor's data, check if it has a label
                sensor_indices = sensor_data.index.tolist()
                
                # Create a sliding window approach for feature extraction
                window_size = min(100, len(values))  # Use up to 100 samples for feature extraction
                step_size = max(1, window_size // 10)  # Overlap windows
                
                for start_idx in range(0, len(values) - window_size + 1, step_size):
                    end_idx = start_idx + window_size
                    
                    # Map back to original dataframe index
                    if start_idx < len(sensor_indices):
                        original_idx = sensor_indices[start_idx]
                        
                        # Check if this row has a label
                        if original_idx in row_labels and row_labels[original_idx] != "Unlabeled":
                            # Extract features from this window
                            window_values = values[start_idx:end_idx]
                            
                            if len(window_values) >= 10:  # Minimum window size
                                window_spectro_features = self.extract_spectrogram_features(
                                    window_values, fs, window_seconds, overlap_ratio
                                )
                                window_time_features = self.extract_time_domain_features(window_values)
                                window_combined_features = np.concatenate([window_spectro_features, window_time_features])
                                
                                features_list.append(window_combined_features)
                                labels_list.append(row_labels[original_idx])
        
        if not features_list:
            raise ValueError("No valid features extracted from dataset. Please ensure you have labeled some rows.")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Store feature names for later use (assuming all features have same length)
        if features_list:
            sample_features = features_list[0]
            spectro_feature_count = len(self.extract_spectrogram_features(np.zeros(100), 1.0, 1.0, 0.5))
            time_feature_count = len(self.extract_time_domain_features(np.zeros(100)))
            
            self.feature_names = [
                f"spectro_{i}" for i in range(spectro_feature_count)
            ] + [f"time_{i}" for i in range(time_feature_count)]
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "RandomForest", 
                   test_size: float = 0.2, cv_folds: int = 5):
        """Train the ML model"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "SVM":
            self.model = SVC(kernel='rbf', random_state=42)
        elif model_type == "NeuralNetwork":
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        self.trained = True
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.model.feature_importances_ if hasattr(self.model, 'feature_importances_') else None
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        # Decode predictions
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'trained': self.trained
        }
        
        joblib.dump(model_data, filepath)
        st.success(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.trained = model_data['trained']
        
        st.success(f"Model loaded from {filepath}")


def create_label_interface(df: pd.DataFrame) -> Dict[int, str]:
    """Create interface for labeling individual rows/records"""
    st.subheader("📝 Label Your Data")
    st.markdown("Label each individual record/row with the object type present during that measurement.")
    
    # Common object types
    common_objects = [
        "None", "Metal Object", "Electronic Device", "Vehicle", "Building", 
        "Machinery", "Power Lines", "Underground Cable", "Water Pipe", "Other"
    ]
    
    # Add custom object option
    if "Custom" not in common_objects:
        common_objects.append("Custom")
    
    st.markdown("### Labeling Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        batch_label_sensor = st.selectbox(
            "Batch label by sensor",
            options=["Select sensor..."] + sorted(df['sensor_id'].dropna().unique().tolist()),
            key="batch_sensor_selector"
        )
        
        batch_object_type = st.selectbox(
            "Object type for batch",
            options=common_objects,
            key="batch_object_type"
        )
        
        if st.button("Apply to Sensor") and batch_label_sensor != "Select sensor...":
            st.session_state.batch_label_sensor = batch_label_sensor
            st.session_state.batch_object_type = batch_object_type
            st.rerun()
    
    with col2:
        batch_label_time = st.checkbox("Batch label by time range")
        if batch_label_time:
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                start_time = st.time_input("Start time", key="batch_start_time")
            with time_col2:
                end_time = st.time_input("End time", key="batch_end_time")
            
            time_object_type = st.selectbox(
                "Object type for time range",
                options=common_objects,
                key="time_object_type"
            )
            
            if st.button("Apply to Time Range"):
                st.session_state.batch_label_time_flag = True
                st.session_state.time_range_data = (start_time, end_time)
                st.session_state.time_object_type_data = time_object_type
                st.rerun()
    
    with col3:
        st.markdown("**Quick Actions**")
        if st.button("Clear All Labels"):
            if 'row_labels' in st.session_state:
                del st.session_state.row_labels
            st.rerun()
        
        if st.button("Load Sample Labels"):
            # Auto-label some sample data for demonstration
            sample_labels = {}
            for i, row in df.head(100).iterrows():
                if i % 10 == 0:
                    sample_labels[i] = "Metal Object"
                elif i % 10 == 1:
                    sample_labels[i] = "Electronic Device"
                else:
                    sample_labels[i] = "None"
            st.session_state.row_labels = sample_labels
            st.rerun()
    
    # Initialize row labels in session state
    if 'row_labels' not in st.session_state:
        st.session_state.row_labels = {}
    
    # Handle batch labeling
    if 'batch_label_sensor' in st.session_state:
        sensor_id = st.session_state.batch_label_sensor
        object_type = st.session_state.batch_object_type
        sensor_indices = df[df['sensor_id'] == sensor_id].index.tolist()
        
        for idx in sensor_indices:
            st.session_state.row_labels[idx] = object_type
        
        del st.session_state.batch_label_sensor
        del st.session_state.batch_object_type
        st.success(f"Labeled {len(sensor_indices)} records for sensor {sensor_id} as '{object_type}'")
    
    if 'batch_label_time_flag' in st.session_state:
        start_time, end_time = st.session_state.time_range_data
        object_type = st.session_state.time_object_type_data
        
        # Convert time inputs to datetime for comparison
        time_mask = (
            (df['timestamp'].dt.time >= start_time) & 
            (df['timestamp'].dt.time <= end_time)
        )
        time_indices = df[time_mask].index.tolist()
        
        for idx in time_indices:
            st.session_state.row_labels[idx] = object_type
        
        del st.session_state.batch_label_time_flag
        del st.session_state.time_range_data
        del st.session_state.time_object_type_data
        st.success(f"Labeled {len(time_indices)} records in time range as '{object_type}'")
    
    # Show labeling progress
    total_rows = len(df)
    labeled_rows = len(st.session_state.row_labels)
    progress = labeled_rows / total_rows if total_rows > 0 else 0
    
    st.progress(progress)
    st.caption(f"Labeled {labeled_rows}/{total_rows} records ({progress:.1%})")
    
    # Individual row labeling interface
    st.markdown("### Individual Row Labeling")
    
    # Pagination controls
    rows_per_page = st.slider("Rows per page", min_value=10, max_value=100, value=20, step=10)
    page = st.number_input("Page", min_value=0, max_value=total_rows // rows_per_page, value=0)
    
    start_idx = page * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    # Show current page of data
    current_page_df = df.iloc[start_idx:end_idx].copy()
    
    # Add current labels to the dataframe
    current_page_df['Current_Label'] = current_page_df.index.map(
        lambda x: st.session_state.row_labels.get(x, "Unlabeled")
    )
    
    # Display the data with labeling controls
    for idx, row in current_page_df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp']) else "N/A"
            st.write(f"**Row {idx}**: {timestamp_str} | Sensor: {row['sensor_id']}")
        
        with col2:
            st.write(f"Bx: {row['b_x']:.1f}")
        
        with col3:
            st.write(f"By: {row['b_y']:.1f}")
        
        with col4:
            st.write(f"Bz: {row['b_z']:.1f}")
        
        with col5:
            current_label = st.session_state.row_labels.get(idx, "Unlabeled")
            new_label = st.selectbox(
                f"Label {idx}",
                options=["Unlabeled"] + common_objects,
                index=common_objects.index(current_label) + 1 if current_label in common_objects else 0,
                key=f"label_row_{idx}"
            )
            
            if new_label != current_label and new_label != "Unlabeled":
                st.session_state.row_labels[idx] = new_label
                st.rerun()
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 0:
            if st.button("← Previous"):
                st.session_state.current_page = page - 1
                st.rerun()
    
    with col2:
        st.write(f"Page {page + 1} of {(total_rows // rows_per_page) + 1}")
    
    with col3:
        if end_idx < total_rows:
            if st.button("Next →"):
                st.session_state.current_page = page + 1
                st.rerun()
    
    # Summary statistics
    st.markdown("### Labeling Summary")
    if st.session_state.row_labels:
        label_counts = pd.Series(st.session_state.row_labels).value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Label Distribution:**")
            for label, count in label_counts.items():
                st.write(f"- {label}: {count} records")
        
        with col2:
            # Visualize label distribution
            fig = px.bar(x=label_counts.index, y=label_counts.values, 
                        title="Label Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    return st.session_state.row_labels


def visualize_model_performance(results: Dict, classifier):
    """Visualize model performance metrics"""
    st.subheader("📊 Model Performance")
    
    # Accuracy metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{results['accuracy']:.3f}")
    with col2:
        st.metric("CV Mean", f"{results['cv_mean']:.3f}")
    with col3:
        st.metric("CV Std", f"{results['cv_std']:.3f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = results['confusion_matrix']
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=classifier.label_encoder.classes_,
                    y=classifier.label_encoder.classes_,
                    color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("Classification Report")
    st.text(results['classification_report'])
    
    # Feature Importance (if available)
    if results['feature_importance'] is not None and classifier.feature_names is not None:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'feature': classifier.feature_names,
            'importance': results['feature_importance']
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(importance_df.head(20), x='importance', y='feature', 
                     orientation='h', title="Top 20 Most Important Features")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Magnetic Signature ML", layout="wide")
    st.title("🧲 Magnetic Signature Classification")
    st.markdown("Train ML models to classify objects based on magnetic field signatures from spectrograms")
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = MagneticSignatureClassifier()
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'dataset' not in st.session_state:
        st.session_state.dataset = pd.DataFrame()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Navigate", [
        "📊 Load & Label Data", 
        "🤖 Train Model", 
        "🔮 Make Predictions", 
        "📈 Model Analysis"
    ])
    
    if page == "📊 Load & Label Data":
        st.header("Data Loading and Labeling")
        
        # File upload section
        with st.expander("📁 Load Dataset", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload CSV files", type=["csv"], accept_multiple_files=True
            )
            
            # Quick load from Dataset directory
            dataset_dir = os.path.join(os.getcwd(), "Dataset")
            quick_load = st.checkbox("Load all CSVs from ./Dataset", value=True)
            
            files_from_dir = []
            if quick_load and os.path.isdir(dataset_dir):
                for name in os.listdir(dataset_dir):
                    if name.lower().endswith(".csv"):
                        files_from_dir.append(os.path.join(dataset_dir, name))
            
            if st.button("Load Data") or st.session_state.dataset_loaded:
                if uploaded_files or files_from_dir:
                    with st.spinner("Loading data..."):
                        df = read_multiple_csvs((uploaded_files or []) + files_from_dir)
                        st.session_state.dataset = df
                        st.session_state.dataset_loaded = True
                    
                    st.success(f"Loaded {len(df)} rows from {len((uploaded_files or [])) + len(files_from_dir)} files")
                    
                    # Show dataset info
                    st.subheader("Dataset Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Unique Sensors", df['sensor_id'].nunique())
                    with col3:
                        if not df.empty and df['timestamp'].notna().any():
                            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                            st.metric("Duration (hours)", f"{duration:.1f}")
                    
                    # Create labeling interface
                    if not df.empty:
                        row_labels = create_label_interface(df)
                        st.session_state.row_labels = row_labels
                        
                        if st.button("Save Labels"):
                            st.success("Labels saved! Proceed to training.")
                else:
                    st.warning("Please upload CSV files or enable quick load from Dataset folder.")
        
        # Show current dataset
        if not st.session_state.dataset.empty:
            st.subheader("Current Dataset Preview")
            st.dataframe(st.session_state.dataset.head(10))
    
    elif page == "🤖 Train Model":
        st.header("Model Training")
        
        if st.session_state.dataset.empty:
            st.warning("Please load data first in the 'Load & Label Data' section.")
            return
        
        if 'row_labels' not in st.session_state or not st.session_state.row_labels:
            st.warning("Please label your data first in the 'Load & Label Data' section.")
            return
        
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["RandomForest", "GradientBoosting", "SVM", "NeuralNetwork"]
            )
            
            window_seconds = st.slider(
                "Spectrogram Window (seconds)", 
                min_value=0.1, max_value=10.0, value=1.0, step=0.1
            )
        
        with col2:
            test_size = st.slider(
                "Test Size", 
                min_value=0.1, max_value=0.5, value=0.2, step=0.05
            )
            
            overlap_ratio = st.slider(
                "Overlap Ratio", 
                min_value=0.0, max_value=0.9, value=0.5, step=0.05
            )
        
        if st.button("🚀 Start Training"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Check if we have labeled data
                    if not st.session_state.row_labels:
                        st.error("No labeled data found. Please label some rows in the 'Load & Label Data' section.")
                        return
                    
                    labeled_count = len([label for label in st.session_state.row_labels.values() if label != "Unlabeled"])
                    if labeled_count < 10:
                        st.warning(f"Only {labeled_count} labeled records found. Consider labeling more data for better results.")
                    
                    st.info(f"Found {labeled_count} labeled records for training")
                    
                    # Prepare dataset
                    X, y = st.session_state.classifier.prepare_dataset(
                        st.session_state.dataset, 
                        st.session_state.row_labels,
                        window_seconds, 
                        overlap_ratio
                    )
                    
                    st.info(f"Extracted {X.shape[0]} samples with {X.shape[1]} features each")
                    
                    # Check for sufficient data
                    if X.shape[0] < 5:
                        st.error(f"Not enough training samples ({X.shape[0]}). Please label more data.")
                        return
                    
                    # Check for class balance
                    unique_classes, counts = np.unique(y, return_counts=True)
                    st.info(f"Class distribution: {dict(zip(unique_classes, counts))}")
                    
                    # Train model
                    results = st.session_state.classifier.train_model(
                        X, y, model_type, test_size
                    )
                    
                    st.session_state.training_results = results
                    st.session_state.training_results['classifier'] = st.session_state.classifier  # Store classifier reference
                    st.success("Model training completed!")
                    
                    # Show results
                    visualize_model_performance(results, st.session_state.classifier)
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    import traceback
                    st.error(f"Full error details: {traceback.format_exc()}")
        
        # Model saving
        if st.session_state.classifier.trained:
            st.subheader("💾 Save Model")
            model_name = st.text_input("Model name", value="magnetic_signature_model")
            
            if st.button("Save Model"):
                model_path = f"{model_name}.pkl"
                st.session_state.classifier.save_model(model_path)
    
    elif page == "🔮 Make Predictions":
        st.header("Make Predictions")
        
        # Load model section
        if not st.session_state.classifier.trained:
            st.subheader("Load Trained Model")
            uploaded_model = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])
            
            if uploaded_model:
                try:
                    st.session_state.classifier.load_model(uploaded_model)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    return
        else:
            st.success("✅ Model is ready for predictions")
        
        # Prediction interface
        if st.session_state.classifier.trained:
            st.subheader("Predict on New Data")
            
            # Upload new data for prediction
            prediction_files = st.file_uploader(
                "Upload CSV files for prediction", type=["csv"], accept_multiple_files=True
            )
            
            if prediction_files:
                with st.spinner("Processing prediction data..."):
                    pred_df = read_multiple_csvs(prediction_files)
                    
                    # Use same parameters as training
                    window_seconds = 1.0  # Default
                    overlap_ratio = 0.5   # Default
                    
                    # Create dummy labels for feature extraction (using row indices)
                    dummy_labels = {idx: "Unknown" for idx in pred_df.index}
                    
                    try:
                        X_pred, _ = st.session_state.classifier.prepare_dataset(
                            pred_df, dummy_labels, window_seconds, overlap_ratio
                        )
                        
                        # Make predictions
                        predictions, probabilities = st.session_state.classifier.predict(X_pred)
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        # Create results with proper identifiers
                        # Get the original record indices that were used for prediction
                        prediction_indices = []
                        prediction_sensors = []
                        
                        # Re-extract the mapping between prediction samples and original data
                        for sensor_id, sensor_data in pred_df.groupby('sensor_id'):
                            sensor_data = sensor_data.sort_values('timestamp')
                            fs = estimate_sampling_rate(sensor_data['timestamp'])
                            if fs <= 0:
                                continue
                                
                            # Process each axis
                            for axis in ['b_x', 'b_y', 'b_z', 'resultant']:
                                if axis == 'resultant':
                                    values = compute_resultant(sensor_data).values
                                else:
                                    values = pd.to_numeric(sensor_data[axis], errors="coerce").dropna().values
                                
                                if len(values) < 50:
                                    continue
                                
                                sensor_indices = sensor_data.index.tolist()
                                window_size = min(100, len(values))
                                step_size = max(1, window_size // 10)
                                
                                for start_idx in range(0, len(values) - window_size + 1, step_size):
                                    if start_idx < len(sensor_indices):
                                        prediction_indices.append(sensor_indices[start_idx])
                                        prediction_sensors.append(sensor_id)
                        
                        # Limit to actual predictions made
                        prediction_indices = prediction_indices[:len(predictions)]
                        prediction_sensors = prediction_sensors[:len(predictions)]
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Record_ID': prediction_indices,
                            'Sensor_ID': prediction_sensors,
                            'Predicted_Object': predictions,
                            'Confidence': np.max(probabilities, axis=1) if probabilities is not None else [1.0] * len(predictions)
                        })
                        
                        # Add timestamp information if available
                        if not pred_df.empty and 'timestamp' in pred_df.columns:
                            timestamp_map = pred_df.set_index(pred_df.index)['timestamp'].to_dict()
                            results_df['Timestamp'] = results_df['Record_ID'].map(timestamp_map)
                        
                        st.dataframe(results_df)
                        
                        # Visualize predictions
                        # Group by sensor and show average confidence
                        sensor_summary = results_df.groupby('Sensor_ID').agg({
                            'Predicted_Object': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common prediction
                            'Confidence': 'mean'
                        }).reset_index()
                        
                        fig = px.bar(sensor_summary, x='Sensor_ID', y='Confidence', 
                                   color='Predicted_Object', 
                                   title="Average Prediction Confidence by Sensor",
                                   labels={'Confidence': 'Average Confidence', 'Sensor_ID': 'Sensor ID'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction distribution
                        st.subheader("Prediction Distribution")
                        prediction_counts = results_df['Predicted_Object'].value_counts()
                        fig2 = px.pie(values=prediction_counts.values, names=prediction_counts.index,
                                    title="Distribution of Predicted Objects")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    
    elif page == "📈 Model Analysis":
        st.header("Model Analysis")
        
        if 'training_results' not in st.session_state:
            st.warning("Please train a model first in the 'Train Model' section.")
            return
        
        visualize_model_performance(st.session_state.training_results, st.session_state.classifier)


if __name__ == "__main__":
    main()
