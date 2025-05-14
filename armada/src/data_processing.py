"""
Data Processing for Warehouse Robotics Maintenance

This module contains data processing functions for warehouse robotics maintenance optimization:
1. Data loading and preprocessing
2. Feature engineering for predictive maintenance
3. Time series processing for sensor data
4. Data augmentation techniques for limited datasets
5. Data quality assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from scipy import signal
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoboticsDataProcessor:
    """
    Data processor for warehouse robotics sensor data
    
    Provides functions for loading, cleaning, and preprocessing
    multimodal sensor data from warehouse robots.
    """
    
    def __init__(self, scaler_type='robust', imputer_type='knn', random_state=None):
        """
        Initialize the data processor.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
            imputer_type: Type of imputer to use ('simple', 'knn')
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = self._get_scaler(scaler_type)
        self.imputer = self._get_imputer(imputer_type)
        self.feature_names = None
        self.sensor_groups = {}
        self.time_features = []
        self.categorical_features = []
        self.numerical_features = []
        
    def _get_scaler(self, scaler_type):
        """Initialize the appropriate scaler"""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
            
    def _get_imputer(self, imputer_type):
        """Initialize the appropriate imputer"""
        if imputer_type == 'simple':
            return SimpleImputer(strategy='mean')
        elif imputer_type == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unsupported imputer type: {imputer_type}")
    
    def load_data(self, data_path, file_format='csv'):
        """
        Load data from file.
        
        Args:
            data_path: Path to data file or directory
            file_format: Format of data files ('csv', 'parquet', 'json')
            
        Returns:
            Pandas DataFrame with loaded data
        """
        if os.path.isfile(data_path):
            # Load single file
            return self._load_file(data_path, file_format)
        elif os.path.isdir(data_path):
            # Load multiple files from directory
            return self._load_directory(data_path, file_format)
        else:
            raise ValueError(f"Data path does not exist: {data_path}")
            
    def _load_file(self, file_path, file_format):
        """Load a single data file"""
        logger.info(f"Loading file: {file_path}")
        
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_format == 'json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def _load_directory(self, dir_path, file_format):
        """Load multiple files from directory and concatenate"""
        all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                     if f.endswith(f'.{file_format}')]
        
        dfs = []
        for file in all_files:
            try:
                df = self._load_file(file, file_format)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading file {file}: {str(e)}")
                
        if not dfs:
            raise ValueError(f"No valid data files found in {dir_path}")
            
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(all_files)} files into dataset with {len(combined_df)} records")
        
        return combined_df
    
    def preprocess(self, df, target_col=None, categorical_cols=None, time_cols=None):
        """
        Preprocess the data for machine learning.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column (if any)
            categorical_cols: List of categorical columns
            time_cols: List of time-related columns
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle target column
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(columns=[target_col])
            
        # Process categorical features
        self.categorical_features = categorical_cols or []
        if self.categorical_features:
            df = self._encode_categorical(df, self.categorical_features)
            
        # Process time features
        self.time_features = time_cols or []
        if self.time_features:
            df = self._process_time_features(df, self.time_features)
            
        # Identify numerical features
        self.numerical_features = [c for c in df.columns 
                               if c not in self.categorical_features + self.time_features]
        
        # Group features by sensor type
        self._group_features_by_sensor(df.columns)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Handle missing values
        df = self._impute_missing(df)
        
        # Scale numerical features
        df = self._scale_features(df)
        
        # Return processed data (and target if provided)
        if y is not None:
            return df, y
        else:
            return df
    
    def _impute_missing(self, df):
        """Impute missing values in the DataFrame"""
        logger.info("Imputing missing values")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) == 0:
            logger.info("No missing values found")
            return df
            
        logger.info(f"Found {len(missing_cols)} columns with missing values")
        for col, count in missing_cols.items():
            logger.info(f"  {col}: {count} missing values ({count/len(df):.2%})")
            
        # Apply imputer
        imputed_data = self.imputer.fit_transform(df)
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
        
        logger.info("Missing value imputation completed")
        return imputed_df
    
    def _scale_features(self, df):
        """Scale numerical features"""
        logger.info("Scaling numerical features")
        
        # Only scale numerical features
        num_df = df[self.numerical_features]
        
        # Apply scaler
        scaled_data = self.scaler.fit_transform(num_df)
        scaled_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)
        
        # Combine with non-numerical features
        non_num_features = [f for f in df.columns if f not in self.numerical_features]
        if non_num_features:
            return pd.concat([scaled_df, df[non_num_features]], axis=1)
        else:
            return scaled_df
    
    def _encode_categorical(self, df, categorical_cols):
        """Encode categorical features"""
        logger.info("Encoding categorical features")
        
        for col in categorical_cols:
            if col in df.columns:
                # Use one-hot encoding
                one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
                
                # Drop the original column and join the one-hot encoded columns
                df = df.drop(columns=[col])
                df = df.join(one_hot)
                
        return df
    
    def _process_time_features(self, df, time_cols):
        """Process time-related features"""
        logger.info("Processing time features")
        
        for col in time_cols:
            if col in df.columns:
                try:
                    # Convert to datetime
                    df[col] = pd.to_datetime(df[col])
                    
                    # Extract useful components
                    df[f"{col}_hour"] = df[col].dt.hour
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    
                    # Drop original column
                    df = df.drop(columns=[col])
                    
                except Exception as e:
                    logger.error(f"Error processing time column {col}: {str(e)}")
                    
        return df
    
    def _group_features_by_sensor(self, columns):
        """Group features by sensor type based on column names"""
        self.sensor_groups = {}
        
        # Try to identify sensor types from column prefixes
        for col in columns:
            # Extract sensor type from column name (e.g., "temp_" from "temp_motor")
            parts = col.split('_')
            if len(parts) > 1:
                sensor_type = parts[0]
                
                if sensor_type not in self.sensor_groups:
                    self.sensor_groups[sensor_type] = []
                    
                self.sensor_groups[sensor_type].append(col)
            else:
                # For columns without clear prefix, group as "other"
                if "other" not in self.sensor_groups:
                    self.sensor_groups["other"] = []
                self.sensor_groups["other"].append(col)
                
        # Log sensor groups
        for group, cols in self.sensor_groups.items():
            logger.info(f"Sensor group '{group}': {len(cols)} features")
            
    def create_sliding_windows(self, df, window_size=10, stride=1, include_target=None):
        """
        Create sliding windows for time series prediction.
        
        Args:
            df: Input DataFrame
            window_size: Size of each window
            stride: Stride between windows
            include_target: Optional target column name for supervised learning
            
        Returns:
            Array of windowed data (and targets if provided)
        """
        logger.info(f"Creating sliding windows with size={window_size}, stride={stride}")
        
        data = df.values
        n_samples, n_features = data.shape
        
        # Calculate number of windows
        n_windows = (n_samples - window_size) // stride + 1
        
        # Initialize array for windows
        windows = np.zeros((n_windows, window_size, n_features))
        
        # Create windows
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows[i] = data[start_idx:end_idx]
            
        logger.info(f"Created {n_windows} windows from {n_samples} samples")
        
        # Handle target for supervised learning
        if include_target and include_target in df.columns:
            feature_idx = df.columns.get_loc(include_target)
            X = np.delete(windows, feature_idx, axis=2)
            y = windows[:, -1, feature_idx]  # Use last timestep's value as target
            return X, y
        else:
            return windows
    
    def denoise_signal(self, data, method='lowpass', **kwargs):
        """
        Apply noise filtering to sensor data.
        
        Args:
            data: Input data, can be array-like or DataFrame column
            method: Filtering method ('lowpass', 'bandpass', 'savgol')
            **kwargs: Additional parameters for the filter
            
        Returns:
            Filtered data
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.array(data)
            
        # Default filter parameters
        fs = kwargs.get('fs', 100)  # Sampling frequency
        cutoff = kwargs.get('cutoff', 10)  # Cutoff frequency
        order = kwargs.get('order', 5)  # Filter order
        window_length = kwargs.get('window_length', 11)  # Savitzky-Golay window length
        polyorder = kwargs.get('polyorder', 3)  # Savitzky-Golay polynomial order
        
        if method == 'lowpass':
            # Butterworth lowpass filter
            b, a = signal.butter(order, cutoff / (0.5 * fs), btype='lowpass')
            filtered_data = signal.filtfilt(b, a, values)
            
        elif method == 'bandpass':
            # Butterworth bandpass filter
            low = kwargs.get('low_cutoff', 5)
            high = kwargs.get('high_cutoff', 20)
            b, a = signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
            filtered_data = signal.filtfilt(b, a, values)
            
        elif method == 'savgol':
            # Savitzky-Golay filter
            filtered_data = signal.savgol_filter(values, window_length, polyorder)
            
        else:
            raise ValueError(f"Unsupported filter method: {method}")
            
        # Return in same format as input
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(filtered_data, index=data.index, columns=data.columns)
        elif isinstance(data, pd.Series):
            return pd.Series(filtered_data, index=data.index, name=data.name)
        else:
            return filtered_data
    
    def augment_data(self, X, y=None, techniques=None, factor=2):
        """
        Augment data for better model training.
        
        Args:
            X: Input features
            y: Optional target values
            techniques: List of augmentation techniques to use
            factor: Multiplication factor for dataset size
            
        Returns:
            Augmented dataset (and targets if provided)
        """
        logger.info(f"Augmenting data with factor {factor}")
        
        # Default augmentation techniques
        if techniques is None:
            techniques = ['jitter', 'scaling', 'time_warp']
            
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        X_aug = [X_values]  # Start with original data
        
        # Apply each technique
        for technique in techniques:
            if technique == 'jitter':
                # Add random noise
                noise_level = 0.01  # 1% of the signal amplitude
                for _ in range(factor - 1):
                    noise = np.random.normal(0, noise_level, X_values.shape)
                    X_aug.append(X_values + noise)
                    
            elif technique == 'scaling':
                # Scale by random factors
                for _ in range(factor - 1):
                    # Scale each feature by a random factor between 0.9 and 1.1
                    scales = np.random.uniform(0.9, 1.1, size=X_values.shape[1])
                    X_scaled = X_values * scales[np.newaxis, :]
                    X_aug.append(X_scaled)
                    
            elif technique == 'time_warp':
                # Time warping (simplified version)
                if len(X_values.shape) >= 3:  # For time series data
                    for _ in range(factor - 1):
                        warped = np.copy(X_values)
                        for i in range(X_values.shape[0]):
                            # Stretch or compress the time dimension slightly
                            time_stretch = np.random.uniform(0.9, 1.1)
                            orig_len = X_values.shape[1]
                            new_len = int(orig_len * time_stretch)
                            if new_len > orig_len:
                                # Truncate if longer
                                new_len = orig_len
                                
                            # Interpolate
                            warped[i] = signal.resample(X_values[i], new_len)
                            
                            # Pad if shorter
                            if new_len < orig_len:
                                pad_width = orig_len - new_len
                                warped[i] = np.pad(warped[i], ((0, pad_width), (0, 0)),
                                                 mode='constant', constant_values=0)
                        X_aug.append(warped)
        
        # Combine all augmented data
        X_augmented = np.vstack(X_aug)
        
        # Handle target values if provided
        if y is not None:
            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y
                
            # Duplicate y to match X
            y_augmented = np.tile(y_values, factor)
            
            logger.info(f"Augmented dataset size: {X_augmented.shape}")
            return X_augmented, y_augmented
        else:
            logger.info(f"Augmented dataset size: {X_augmented.shape}")
            return X_augmented
    
    def assess_data_quality(self, df):
        """
        Assess the quality of the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Assessing data quality")
        
        # Initialize metrics dictionary
        metrics = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_values': {},
            'outliers': {},
            'feature_correlations': {},
            'class_balance': None
        }
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_pcts = missing_counts / len(df) * 100
        for col in df.columns:
            if missing_counts[col] > 0:
                metrics['missing_values'][col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_pcts[col])
                }
                
        # Outliers (using IQR method)
        for col in self.numerical_features:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                metrics['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'min': float(outliers.min()),
                    'max': float(outliers.max())
                }
                
        # Feature correlations
        corr_matrix = df[self.numerical_features].corr()
        high_corr_pairs = []
        
        for i in range(len(self.numerical_features)):
            for j in range(i+1, len(self.numerical_features)):
                col1 = self.numerical_features[i]
                col2 = self.numerical_features[j]
                corr = corr_matrix.loc[col1, col2]
                
                if abs(corr) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': float(corr)
                    })
                    
        metrics['feature_correlations'] = high_corr_pairs
        
        # Class balance (if binary target column present)
        target_cols = [col for col in df.columns if col.startswith('target_') or col.endswith('_target')]
        if target_cols:
            target_col = target_cols[0]
            if df[target_col].nunique() == 2:
                value_counts = df[target_col].value_counts()
                metrics['class_balance'] = {
                    str(value): int(count) for value, count in value_counts.items()
                }
                
                # Class imbalance ratio
                if len(value_counts) > 1:
                    max_count = value_counts.max()
                    min_count = value_counts.min()
                    metrics['class_balance']['imbalance_ratio'] = float(max_count / min_count)
                
        logger.info("Data quality assessment completed")
        return metrics
    
    def plot_data_quality(self, df, output_path=None):
        """
        Create visualizations of data quality.
        
        Args:
            df: Input DataFrame
            output_path: Optional path to save plots
            
        Returns:
            Dictionary with plot figures
        """
        logger.info("Creating data quality visualizations")
        
        plots = {}
        
        # Missing values chart
        plt.figure(figsize=(12, 6))
        missing_data = df.isnull().mean().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', figsize=(12, 6))
            plt.title('Percentage of Missing Values by Feature')
            plt.ylabel('Percentage Missing')
            plt.xlabel('Features')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(os.path.join(output_path, 'missing_values.png'))
                
            plots['missing_values'] = plt.gcf()
            
        # Feature correlation heatmap
        plt.figure(figsize=(14, 10))
        corr_matrix = df[self.numerical_features].corr()
        plt.matshow(corr_matrix, fignum=1, cmap='viridis')
        plt.colorbar()
        plt.title('Feature Correlation Matrix')
        plt.xticks(range(len(self.numerical_features)), self.numerical_features, rotation=90)
        plt.yticks(range(len(self.numerical_features)), self.numerical_features)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'correlation_matrix.png'))
            
        plots['correlation_matrix'] = plt.gcf()
        
        # Class distribution (if applicable)
        target_cols = [col for col in df.columns if col.startswith('target_') or col.endswith('_target')]
        if target_cols:
            target_col = target_cols[0]
            plt.figure(figsize=(8, 6))
            df[target_col].value_counts().plot(kind='bar')
            plt.title('Class Distribution')
            plt.ylabel('Count')
            plt.xlabel('Class')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(os.path.join(output_path, 'class_distribution.png'))
                
            plots['class_distribution'] = plt.gcf()
            
        # Feature distributions
        n_features = min(6, len(self.numerical_features))  # Plot up to 6 features
        if n_features > 0:
            fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
            
            if n_features == 1:
                axes = [axes]
                
            for i, feature in enumerate(self.numerical_features[:n_features]):
                axes[i].hist(df[feature].dropna(), bins=30)
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_ylabel('Frequency')
                
            plt.tight_layout()
            
            if output_path:
                plt.savefig(os.path.join(output_path, 'feature_distributions.png'))
                
            plots['feature_distributions'] = plt.gcf()
            
        logger.info("Created data quality visualizations")
        return plots

def generate_synthetic_data(n_samples=1000, n_features=20, noise_level=0.05,
                           n_robots=10, time_periods=100, random_state=None):
    """
    Generate synthetic robotics sensor data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (sensors)
        noise_level: Level of noise to add
        n_robots: Number of robot IDs to simulate
        time_periods: Number of time periods to simulate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    logger.info(f"Generating synthetic data with {n_samples} samples, {n_features} features")
    
    # Create feature names
    sensor_types = ['temp', 'vibration', 'pressure', 'current', 'voltage']
    feature_names = []
    
    for i in range(n_features):
        sensor_type = sensor_types[i % len(sensor_types)]
        feature_names.append(f"{sensor_type}_sensor_{i//len(sensor_types) + 1}")
    
    # Generate data
    X = np.zeros((n_samples, n_features))
    
    # Generate robot IDs and timestamps
    robot_ids = np.random.choice(range(1, n_robots+1), size=n_samples)
    
    # Create timestamps
    base_date = datetime(2023, 1, 1)
    timestamps = []
    for i in range(n_samples):
        # Assign sequential timestamps, but ensure some overlap between robots
        robot_id = robot_ids[i]
        time_idx = (i // n_robots) % time_periods
        hours_offset = time_idx * 24 + robot_id  # Different offset for each robot
        timestamps.append(base_date + timedelta(hours=int(hours_offset)))
    
    # Generate feature data with time dependencies
    for i in range(n_samples):
        robot_id = robot_ids[i]
        time_idx = timestamps[i].day * 24 + timestamps[i].hour
        
        # Base patterns for each sensor type
        for j in range(n_features):
            sensor_type = sensor_types[j % len(sensor_types)]
            
            # Different patterns for different sensor types
            if sensor_type == 'temp':
                # Temperature increases over time with daily cycles
                base_value = 60 + 0.01 * time_idx + 5 * np.sin(time_idx / 24 * 2 * np.pi)
                # Different robots have different baseline temperatures
                base_value += robot_id * 2
            
            elif sensor_type == 'vibration':
                # Vibration spikes occasionally
                base_value = 0.5 + 0.3 * np.sin(time_idx / 8 * 2 * np.pi)
                # Add occasional spikes
                if time_idx % 50 < 2:
                    base_value += 1.0
            
            elif sensor_type == 'pressure':
                # Pressure varies by robot but is mostly stable
                base_value = 100 + robot_id * 5 + 2 * np.sin(time_idx / 12 * 2 * np.pi)
            
            elif sensor_type == 'current':
                # Current depends on robot workload
                hour_of_day = time_idx % 24
                # Higher during work hours
                if 8 <= hour_of_day <= 18:
                    base_value = 10 + 2 * np.sin(hour_of_day / 6 * np.pi)
                else:
                    base_value = 5 + np.random.random()
            
            else:  # voltage
                # Voltage is mostly stable with occasional dips
                base_value = 220 + np.random.random() * 5
                # Add occasional dips
                if time_idx % 80 < 3:
                    base_value -= 10
            
            # Add noise
            X[i, j] = base_value + noise_level * np.random.randn()
    
    # Create target variable (maintenance needed)
    # Higher temperatures and vibrations over extended periods indicate maintenance needs
    rolling_temp = np.zeros(n_samples)
    rolling_vibration = np.zeros(n_samples)
    
    # Group by robot and calculate rolling averages
    for robot in range(1, n_robots+1):
        robot_mask = robot_ids == robot
        
        # Get temperature and vibration sensors
        temp_indices = [j for j, name in enumerate(feature_names) if 'temp' in name]
        vib_indices = [j for j, name in enumerate(feature_names) if 'vibration' in name]
        
        # Calculate mean values for this robot's sensors
        if len(temp_indices) > 0:
            robot_temps = X[robot_mask][:, temp_indices].mean(axis=1)
            # Sort by timestamp
            robot_timestamps = [timestamps[i] for i in range(n_samples) if robot_mask[i]]
            sorted_indices = np.argsort(robot_timestamps)
            sorted_temps = robot_temps[sorted_indices]
            
            # Calculate rolling average
            window_size = min(5, len(sorted_temps))
            for i in range(len(sorted_temps)):
                window_start = max(0, i - window_size + 1)
                rolling_temp[robot_mask][sorted_indices[i]] = sorted_temps[window_start:i+1].mean()
                
        if len(vib_indices) > 0:
            robot_vibs = X[robot_mask][:, vib_indices].mean(axis=1)
            # Sort by timestamp
            robot_timestamps = [timestamps[i] for i in range(n_samples) if robot_mask[i]]
            sorted_indices = np.argsort(robot_timestamps)
            sorted_vibs = robot_vibs[sorted_indices]
            
            # Calculate rolling average
            window_size = min(5, len(sorted_vibs))
            for i in range(len(sorted_vibs)):
                window_start = max(0, i - window_size + 1)
                rolling_vibration[robot_mask][sorted_indices[i]] = sorted_vibs[window_start:i+1].mean()
    
    # Determine maintenance needed based on thresholds
    # High temperature and vibration indicate maintenance needs
    temp_threshold = np.percentile(rolling_temp, 80)  # Top 20% temperatures
    vib_threshold = np.percentile(rolling_vibration, 80)  # Top 20% vibrations
    
    maintenance_needed = ((rolling_temp > temp_threshold) & 
                          (rolling_vibration > vib_threshold)).astype(int)
    
    # Create a maintenance history (some robots received maintenance)
    maintenance_history = np.zeros(n_samples)
    for robot in range(1, n_robots+1):
        robot_mask = robot_ids == robot
        
        # Sort by timestamp
        robot_indices = np.where(robot_mask)[0]
        robot_timestamps = [timestamps[i] for i in robot_indices]
        sorted_indices = [robot_indices[i] for i in np.argsort(robot_timestamps)]
        
        # Schedule maintenance every ~30 days
        for i in range(len(sorted_indices)):
            if i > 0 and i % 30 == 0:
                maintenance_history[sorted_indices[i]] = 1
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['robot_id'] = robot_ids
    df['timestamp'] = timestamps
    df['maintenance_needed'] = maintenance_needed
    df['maintenance_performed'] = maintenance_history
    
    # Derive days since last maintenance
    df['days_since_maintenance'] = 0
    for robot in range(1, n_robots+1):
        robot_df = df[df['robot_id'] == robot].sort_values('timestamp')
        days = 0
        for idx, row in robot_df.iterrows():
            if row['maintenance_performed'] == 1:
                days = 0
            df.loc[idx, 'days_since_maintenance'] = days
            days += 1
    
    logger.info(f"Generated synthetic dataset with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Maintenance needed: {df['maintenance_needed'].sum()} instances ({df['maintenance_needed'].mean():.2%})")
    
    return df

# Example usage if run directly
if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(n_samples=5000, random_state=42)
    
    # Initialize data processor
    processor = RoboticsDataProcessor(scaler_type='robust', imputer_type='knn')
    
    # Define feature types
    categorical_features = ['robot_id']
    time_features = ['timestamp']
    
    # Preprocess data
    preprocessed_data, maintenance_labels = processor.preprocess(
        synthetic_data, 
        target_col='maintenance_needed',
        categorical_cols=categorical_features,
        time_cols=time_features
    )
    
    # Assess data quality
    quality_metrics = processor.assess_data_quality(preprocessed_data)
    
    print("Data processing completed:")
    print(f"Original shape: {synthetic_data.shape}")
    print(f"Processed shape: {preprocessed_data.shape}")
    print(f"Target distribution: {np.bincount(maintenance_labels.astype(int))}")
    print("\nFeature groups detected:")
    for group, features in processor.sensor_groups.items():
        print(f"- {group}: {len(features)} features") 