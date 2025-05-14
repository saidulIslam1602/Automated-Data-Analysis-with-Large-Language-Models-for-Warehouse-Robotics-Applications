"""
Hybrid Anomaly Detection System

This module implements a hybrid anomaly detection system that combines:
1. Traditional ML models for efficient anomaly detection
2. Large Language Models for context-aware explanation and refinement
"""

import os
import json
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
from sklearn.feature_selection import VarianceThreshold
import joblib
import scipy.stats as stats
import scipy.signal as signal
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import TensorFlow for autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Autoencoder will not be used.")
    TENSORFLOW_AVAILABLE = False

class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detector that learns normal patterns."""
    
    def __init__(self, contamination=0.05, random_state=None, encoding_dim=None):
        """Initialize the autoencoder model.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
            encoding_dim: Dimension of the encoded representation (default: auto)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
        
    def build_model(self, input_dim):
        """Build the autoencoder model architecture.
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Autoencoder model
        """
        # Set encoding dimension if not specified
        if self.encoding_dim is None:
            self.encoding_dim = max(min(input_dim // 2, 10), 2)
            
        # Determine layer sizes for bottleneck architecture
        layer_sizes = [input_dim]
        current_size = input_dim
        while current_size > self.encoding_dim:
            current_size = max(current_size // 2, self.encoding_dim)
            layer_sizes.append(current_size)
            
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for size in layer_sizes[1:]:
            encoded = Dense(size, activation='relu')(encoded)
            if size > 5:  # Only add dropout to larger layers
                encoded = Dropout(0.1)(encoded)
        
        # Decoder (mirror of encoder)
        decoded = encoded
        for size in reversed(layer_sizes[:-1]):
            decoded = Dense(size, activation='relu')(decoded)
            
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def fit(self, X, y=None):
        """Fit the autoencoder to the data.
        
        Args:
            X: Training data
            y: Not used, included for API compatibility
            
        Returns:
            self
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Autoencoder will not be trained.")
            return self
            
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build the model
        self.model = self.build_model(X_scaled.shape[1])
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        logger.info("Training autoencoder...")
        self.history = self.model.fit(
            X_scaled, X_scaled,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate reconstruction errors
        reconstruction_errors = self._get_reconstruction_errors(X_scaled)
        
        # Set threshold for anomaly detection
        self.threshold = np.percentile(
            reconstruction_errors, 
            100 * (1 - self.contamination)
        )
        
        logger.info(f"Autoencoder trained. Threshold: {self.threshold:.4f}")
        return self
    
    def _get_reconstruction_errors(self, X_scaled):
        """Calculate reconstruction errors for scaled data.
        
        Args:
            X_scaled: Scaled input data
            
        Returns:
            array: Reconstruction errors
        """
        if self.model is None:
            return np.zeros(len(X_scaled))
            
        # Get reconstructions
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate MSE for each sample
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        return reconstruction_errors
    
    def decision_function(self, X):
        """Calculate anomaly scores as reconstruction errors.
        
        Args:
            X: Input data
            
        Returns:
            array: Anomaly scores
        """
        if self.model is None:
            return np.zeros(len(X))
            
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Calculate reconstruction errors
        return self._get_reconstruction_errors(X_scaled)
    
    def predict(self, X):
        """Predict if samples are anomalies.
        
        Args:
            X: Input data
            
        Returns:
            array: -1 for anomalies, 1 for normal samples
        """
        if self.model is None:
            return np.ones(len(X))
            
        # Get anomaly scores
        scores = self.decision_function(X)
        
        # Apply threshold
        return np.where(scores > self.threshold, -1, 1)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator
                and contained subobjects
                
        Returns:
            dict: Parameter names mapped to their values
        """
        return {
            "contamination": self.contamination,
            "random_state": self.random_state,
            "encoding_dim": self.encoding_dim
        }

class HybridAnomalyDetector:
    """
    Enhanced Hybrid Anomaly Detection System combining ML, deep learning, and LLM approaches.
    
    This system uses multiple detection techniques for efficient anomaly detection,
    advanced feature engineering, and large language models for explanation.
    """
    
    def __init__(
        self, 
        base_detector='iforest', 
        contamination=0.05, 
        random_state=None, 
        api_key=None,
        llm_model="gpt-3.5-turbo",
        domain_knowledge=None,
        threshold_config=None,
        use_deep_learning=True,
        feature_engineering_level='advanced',
        ensemble_size=5,
        use_time_features=True,
        use_autoencoder=True
    ):
        """
        Initialize the enhanced hybrid anomaly detector.
        
        Args:
            base_detector (str): Base ML detector type ('iforest', 'ocsvm', 'lof')
            contamination (float): Expected proportion of anomalies
            random_state (int): Random seed for reproducibility
            api_key (str, optional): OpenAI API key for LLM integration
            llm_model (str): LLM model to use
            domain_knowledge (dict, optional): Domain knowledge for the LLM
            threshold_config (dict, optional): Configuration of sensor thresholds
            use_deep_learning (bool): Whether to use deep learning components
            feature_engineering_level (str): Level of feature engineering ('basic', 'advanced')
            ensemble_size (int): Number of models in the ensemble
            use_time_features (bool): Whether to generate time-domain features
            use_autoencoder (bool): Whether to use autoencoder for anomaly detection
        """
        self.base_detector_type = base_detector
        self.contamination = contamination
        self.random_state = random_state
        self.ml_detector = None
        self.scaler = StandardScaler()
        self.is_ml_fitted = False
        self.column_names = None
        self.sensor_thresholds = threshold_config or {}
        self.domain_knowledge = domain_knowledge or {}
        self.use_deep_learning = use_deep_learning
        self.feature_engineering_level = feature_engineering_level
        self.ensemble_size = max(3, min(ensemble_size, 10))  # Keep between 3-10
        self.use_time_features = use_time_features
        self.use_autoencoder = use_autoencoder
        self.autoencoder = None
        self.engineered_feature_names = []
        
        # Initialize LLM client if API key is provided
        self.llm_model = llm_model
        self.has_llm = False
        self.setup_llm(api_key)
        
        # Cache for LLM responses
        self.response_cache = {}
        
        # Enhanced performance metrics
        self.performance_metrics = {
            "ml_only": {},
            "hybrid": {},
            "deep_learning": {}
        }
        
        # Create anomaly score history for adaptive thresholding
        self.score_history = []
        self.detection_threshold = None
        
        # Autoencoder component if enabled and available
        if self.use_autoencoder and TENSORFLOW_AVAILABLE:
            self.autoencoder = AutoencoderAnomalyDetector(
                contamination=contamination,
                random_state=random_state
            )
        
        logger.info(f"Initialized Enhanced Hybrid Anomaly Detector with base detector: {base_detector}")
        logger.info(f"Feature engineering level: {feature_engineering_level}, Ensemble size: {self.ensemble_size}")
        if self.use_deep_learning:
            logger.info(f"Deep learning components enabled")
    
    def setup_llm(self, api_key=None):
        """Set up the LLM client if API key is available."""
        try:
            from openai import OpenAI
            
            # Try different sources for API key
            if api_key is None:
                # Environment variable
                api_key = os.environ.get("OPENAI_API_KEY")
                
                # File
                if api_key is None and os.path.exists("openai_api_key.txt"):
                    with open("openai_api_key.txt", "r") as f:
                        api_key = f.read().strip()
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.has_llm = True
                logger.info(f"LLM integration enabled using model: {self.llm_model}")
            else:
                logger.warning("No OpenAI API key provided. LLM integration disabled.")
                self.has_llm = False
                
        except ImportError:
            logger.warning("OpenAI package not installed. LLM integration disabled.")
            self.has_llm = False
    
    def _create_ml_detector(self):
        """Create the machine learning anomaly detector."""
        if self.base_detector_type == 'ocsvm':
            from sklearn.svm import OneClassSVM
            # Increased gamma for better regularization
            return OneClassSVM(
                nu=self.contamination,
                gamma='auto',  # Use 'auto' instead of 'scale' for better generalization
                kernel='rbf',
                tol=1e-3  # More tolerant convergence to avoid overfitting
            )
        elif self.base_detector_type == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            # Increase n_neighbors for better regularization
            return LocalOutlierFactor(
                n_neighbors=30,  # Increased from 20
                contamination=self.contamination, 
                novelty=True,
                leaf_size=40  # Increased leaf size for better generalization
            )
        else:  # Default to Isolation Forest
            # Add regularization parameters to reduce overfitting
            return IsolationForest(
                contamination=self.contamination, 
                random_state=self.random_state,
                n_estimators=200,  # Increased from 100
                max_samples='auto',
                bootstrap=True,  # Enable bootstrapping for better generalization
                max_features=0.8  # Use subset of features to reduce overfitting
            )
    
    def _engineer_features(self, X, is_training=False):
        """
        Apply feature engineering to enhance model performance.
        
        Args:
            X: Input data (DataFrame or numpy array)
            is_training: Whether this is during training phase
            
        Returns:
            DataFrame: Dataframe with engineered features
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.column_names)
        else:
            X_df = X.copy()
            
        # Skip if basic feature engineering requested
        if self.feature_engineering_level == 'basic':
            return X_df
            
        try:
            # Convert any datetime columns to numeric timestamps - more thoroughly
            # First identify any columns that might be dates
            datetime_cols = []
            date_pattern = re.compile(r'(date|time|timestamp|day|month|year)', re.IGNORECASE)
            
            for col in X_df.columns:
                # Check for explicit datetime dtypes
                if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                    datetime_cols.append(col)
                    # Convert to Unix timestamp (seconds since epoch)
                    X_df[f"{col}_timestamp"] = X_df[col].astype(np.int64) // 10**9
                # Check for column names that sound like dates
                elif date_pattern.search(str(col)) and not isinstance(X_df[col].iloc[0], (int, float)):
                    try:
                        dt_series = pd.to_datetime(X_df[col], errors='coerce')
                        # Only convert if it actually parsed as dates (not all NaT)
                        if not dt_series.isna().all():
                            X_df[f"{col}_timestamp"] = dt_series.astype(np.int64) // 10**9
                            datetime_cols.append(col)
                    except:
                        pass
                # Check for object columns that might be dates
                elif X_df[col].dtype == 'object':
                    try:
                        # Try to convert to datetime
                        dt_series = pd.to_datetime(X_df[col], errors='coerce')
                        # If at least 90% parsed successfully, treat as datetime
                        if dt_series.isna().mean() < 0.1:
                            X_df[f"{col}_timestamp"] = dt_series.astype(np.int64) // 10**9
                            datetime_cols.append(col)
                    except:
                        pass
                        
            # Drop original datetime columns to prevent dtype issues
            if datetime_cols:
                X_df = X_df.drop(columns=datetime_cols)
                
            # Create feature names list during training
            if is_training:
                self.engineered_feature_names = []
                
            all_features = []
            new_feature_names = []
            
            # Process numerical columns only
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns
            X_numeric = X_df[numeric_cols]
            
            # 1. Statistical features
            # Z-scores
            for col in numeric_cols:
                if f"{col}_zscore" not in X_df.columns:
                    X_df[f"{col}_zscore"] = stats.zscore(X_df[col].values, nan_policy='omit')
                    new_feature_names.append(f"{col}_zscore")
            
            # 2. Interaction features for sensor relationships
            if len(numeric_cols) >= 2:
                # Get important pairs of features (skip if too many to avoid explosion)
                if len(numeric_cols) < 10:
                    important_pairs = []
                    for i, col1 in enumerate(numeric_cols):
                        for col2 in numeric_cols[i+1:]:
                            # Create ratios for physically related quantities
                            ratio_name = f"{col1}_to_{col2}_ratio"
                            if ratio_name not in X_df.columns:
                                # Avoid division by zero
                                X_df[ratio_name] = X_df[col1] / (X_df[col2] + 1e-8)
                                new_feature_names.append(ratio_name)
                                # Use basic statistical properties for the ratios
                                X_df[f"{ratio_name}_zscore"] = stats.zscore(
                                    X_df[ratio_name].values, nan_policy='omit')
                                new_feature_names.append(f"{ratio_name}_zscore")
            
            # 3. Frequency domain features for key columns (detect oscillations)
            if len(X_df) > 10:  # Only if enough samples
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns to avoid explosion
                    # Simple FFT-based features
                    values = X_df[col].values
                    if len(values) >= 10:  # Need enough samples for FFT
                        # Get magnitude spectrum
                        fft_vals = np.abs(fft(values))
                        # Use first few frequency components
                        n_components = min(3, len(fft_vals)//2)
                        fft_features = fft_vals[1:n_components+1]
                        for i, val in enumerate(fft_features):
                            feature_name = f"{col}_fft_{i+1}"
                            X_df[feature_name] = val
                            new_feature_names.append(feature_name)
            
            # 4. Time domain features if enabled and timestamp column exists
            if self.use_time_features:
                timestamp_cols = [col for col in X_df.columns if any(
                    time_kw in col.lower() for time_kw in ['time', 'date', 'timestamp'])]
                
                if timestamp_cols:
                    # Use first timestamp column found
                    time_col = timestamp_cols[0]
                    try:
                        # Convert to datetime if not already
                        if not pd.api.types.is_datetime64_any_dtype(X_df[time_col]):
                            X_df[time_col] = pd.to_datetime(X_df[time_col], errors='coerce')
                            
                        # Extract hour of day
                        X_df['hour_of_day'] = X_df[time_col].dt.hour
                        new_feature_names.append('hour_of_day')
                        
                        # Is weekend
                        X_df['is_weekend'] = X_df[time_col].dt.dayofweek >= 5
                        X_df['is_weekend'] = X_df['is_weekend'].astype(int)
                        new_feature_names.append('is_weekend')
                    except Exception as e:
                        logger.warning(f"Error creating time features: {str(e)}")
            
            # Store feature names during training
            if is_training:
                self.engineered_feature_names = new_feature_names
            
            # Drop NaN values that might have been introduced
            X_df = X_df.fillna(0)
            
            return X_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}. Using original features.")
            return X_df
    
    def _create_autoencoder(self, input_dim):
        """Create a simple autoencoder for anomaly detection."""
        try:
            from tensorflow.keras.models import Sequential, Model
            from tensorflow.keras.layers import Dense, Input
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras import regularizers
            
            # Define autoencoder architecture
            input_layer = Input(shape=(input_dim,))
            
            # Encoder
            encoded = Dense(input_dim, activation='relu')(input_layer)
            encoded = Dense(max(input_dim // 2, 2), activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(encoded)
            
            # Bottleneck
            bottleneck = Dense(max(input_dim // 4, 2), activation='relu',
                              kernel_regularizer=regularizers.l2(0.001))(encoded)
            
            # Decoder
            decoded = Dense(max(input_dim // 2, 2), activation='relu')(bottleneck)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)
            
            # Create autoencoder model
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            return autoencoder
            
        except ImportError:
            logger.warning("TensorFlow not available. Disabling deep learning components.")
            self.use_deep_learning = False
            return None
    
    def fit(self, X, column_names=None):
        """
        Fit the hybrid model to the training data.
        
        Args:
            X: The training data
            column_names: Optional list of column names
            
        Returns:
            self: The fitted model
        """
        start_time = time.time()
        
        # Store column names
        if column_names is None and isinstance(X, pd.DataFrame):
            self.column_names = X.columns.tolist()
        else:
            self.column_names = column_names or [f"feature_{i}" for i in range(X.shape[1])]
            
        # First, preprocess the data by removing datetime columns and converting to numerical
        if isinstance(X, pd.DataFrame):
            X_processed = X.copy()
            # Handle datetime columns
            for col in X_processed.columns:
                if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
                    # Extract useful features from datetime
                    col_name = str(col)
                    X_processed[f"{col_name}_hour"] = X_processed[col].dt.hour
                    X_processed[f"{col_name}_day"] = X_processed[col].dt.day
                    X_processed[f"{col_name}_month"] = X_processed[col].dt.month
                    X_processed[f"{col_name}_dayofweek"] = X_processed[col].dt.dayofweek
                    # Convert to timestamp for numerical representation
                    X_processed[f"{col_name}_timestamp"] = X_processed[col].astype(np.int64) // 10**9
                    # Drop the original datetime column
                    X_processed = X_processed.drop(columns=[col])
            
            # Convert to numpy array for processing
            if len(X_processed) > 0:
                X_numeric = X_processed.select_dtypes(include=['number']).to_numpy()
            else:
                # Fallback if no numeric columns
                X_numeric = np.array(X_processed)
        else:
            X_numeric = np.array(X)
        
        # Apply feature engineering if enabled
        if self.feature_engineering_level != 'none':
            X_engineered = self._engineer_features(X, is_training=True)
            if isinstance(X_engineered, pd.DataFrame):
                X_numeric = X_engineered.select_dtypes(include=['number']).to_numpy()
            else:
                X_numeric = np.array(X_engineered)
                
        # Scale the data
        self.scaler.fit(X_numeric)
        X_scaled = self.scaler.transform(X_numeric)
        
        # Create and fit base ML detector
        self.ml_detector = self._create_ml_detector()
        self.ml_detector.fit(X_scaled)
        self.is_ml_fitted = True
        
        # Create deep learning detector if enabled
        if self.use_autoencoder and TENSORFLOW_AVAILABLE and X_scaled.shape[0] > 50:
            input_dim = X_scaled.shape[1]
            self.autoencoder = self._create_autoencoder(input_dim)
            try:
                # Use validation split for early stopping
                self.autoencoder.fit(
                    X_scaled, 
                    X_scaled,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    ],
                    verbose=0
                )
            except Exception as e:
                warnings.warn(f"Error training autoencoder: {str(e)}. Continuing without autoencoder.")
                self.autoencoder = None
        
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X, return_details=False, use_llm=True, batch_size=25, high_anomaly_only=True):
        """
        Predict anomalies using the hybrid approach.
        
        Args:
            X: Data to predict on, either numpy array or pandas DataFrame
            return_details: If True, return detailed analysis
            use_llm: Whether to use LLM for refinement
            batch_size: Number of samples to process in each LLM batch
            high_anomaly_only: If True, only use LLM for high anomaly score samples
            
        Returns:
            If return_details is False:
                numpy array with -1 for anomalies, 1 for normal data
            If return_details is True:
                tuple of (predictions, details)
                where details is a list of dicts with detailed analysis
        """
        if not self.is_ml_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.column_names)
        else:
            X_df = X.copy()
        
        # Apply feature engineering
        X_engineered = self._engineer_features(X_df)
            
        # Apply same feature selection as during training
        if hasattr(self, 'selected_features'):
            # Get available features (some engineered features might not be present)
            available_features = [col for col in self.selected_features if col in X_engineered.columns]
            if len(available_features) >= 3:  # Only if we have enough features
                X_engineered = X_engineered[available_features]
                logger.info(f"Using {len(available_features)} selected features for prediction")
        
        # Scale the data
        X_scaled = self.scaler.transform(X_engineered)
        
        # Get ensemble predictions from multiple ML models
        ensemble_preds = np.zeros((len(X_df), len(self.ml_detectors)))
        ensemble_scores = np.zeros((len(X_df), len(self.ml_detectors)))
        
        for i, model in enumerate(self.ml_detectors):
            ensemble_preds[:, i] = model.predict(X_scaled)
            
            # Get scores for each model
            if hasattr(model, "decision_function"):
                ensemble_scores[:, i] = -model.decision_function(X_scaled)
            elif hasattr(model, "score_samples"):
                ensemble_scores[:, i] = -model.score_samples(X_scaled)
            else:
                ensemble_scores[:, i] = np.zeros(len(X_df))
        
        # Majority voting for predictions
        ml_predictions = np.zeros(len(X_df))
        for i in range(len(X_df)):
            # Count number of models predicting anomaly (-1)
            anomaly_votes = sum(ensemble_preds[i, :] == -1)
            # Majority rule (at least half of models must agree)
            ml_predictions[i] = -1 if anomaly_votes >= len(self.ml_detectors) / 2 else 1
        
        # Average anomaly scores across models
        anomaly_scores = np.mean(ensemble_scores, axis=1)
        
        # If deep learning is enabled, incorporate autoencoder predictions
        if self.use_deep_learning and self.autoencoder is not None:
            try:
                # Get reconstruction error
                reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
                mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                
                # Convert to anomaly scores (higher = more anomalous)
                dl_scores = mse / self.reconstruction_threshold
                
                # Identify autoencoder anomalies
                dl_predictions = np.ones(len(X_df))
                dl_predictions[dl_scores > 1.0] = -1  # Mark as anomaly
                
                # Combine with ensemble scores using weighted average
                combined_scores = 0.7 * anomaly_scores + 0.3 * dl_scores
                
                # Get combined predictions
                # Higher weight on traditional models as they tend to be more stable
                combined_predictions = np.ones(len(X_df))
                for i in range(len(X_df)):
                    # If both agree it's an anomaly, mark as anomaly
                    if ml_predictions[i] == -1 and dl_predictions[i] == -1:
                        combined_predictions[i] = -1
                    # If one method has high confidence, use its prediction
                    elif ml_predictions[i] == -1 and anomaly_scores[i] > np.percentile(anomaly_scores, 90):
                        combined_predictions[i] = -1
                    elif dl_predictions[i] == -1 and dl_scores[i] > np.percentile(dl_scores, 95):
                        combined_predictions[i] = -1
                
                # Use combined predictions and scores
                ml_predictions = combined_predictions
                anomaly_scores = combined_scores
                
                logger.info(f"Incorporated deep learning predictions")
            except Exception as e:
                logger.warning(f"Error using deep learning predictions: {str(e)}")
        
        # Store anomaly scores for adaptive thresholding
        self.score_history.extend(anomaly_scores.tolist())
        
        # Apply adaptive thresholding if we have enough history
        if len(self.score_history) > 100:
            # Keep only the most recent 1000 scores
            self.score_history = self.score_history[-1000:]
            # Set threshold at 95th percentile (assuming 5% anomaly rate)
            self.detection_threshold = np.percentile(self.score_history, (1 - self.contamination) * 100)
            
            # Apply adaptive threshold
            for i in range(len(X_df)):
                if anomaly_scores[i] > self.detection_threshold:
                    ml_predictions[i] = -1
                else:
                    ml_predictions[i] = 1
        
        # Prepare hybrid predictions (default to ML predictions)
        hybrid_predictions = ml_predictions.copy()
        
        # Prepare details
        details = []
        
        # Refine with LLM if available and requested
        if self.has_llm and use_llm:
            # Only process potential anomalies or high score data points to reduce API costs
            if high_anomaly_only:
                # Select top 25% highest anomaly scores or actual ML-flagged anomalies
                high_score_threshold = np.percentile(anomaly_scores, 75)
                indices_to_process = np.where(
                    (ml_predictions == -1) | (anomaly_scores > high_score_threshold)
                )[0]
            else:
                # Process all data points (expensive and usually unnecessary)
                indices_to_process = np.arange(len(X_df))
            
            # Process in batches
            for batch_start in range(0, len(indices_to_process), batch_size):
                batch_indices = indices_to_process[batch_start:batch_start + batch_size]
                
                # Process this batch of samples
                batch_inputs = []
                batch_scores = []
                batch_rows = []
                
                for i in batch_indices:
                    row_data = X_df.iloc[i]
                    batch_inputs.append(row_data)
                    batch_scores.append(anomaly_scores[i])
                    batch_rows.append(i)
                
                # Analyze batch with LLM
                batch_results = self._analyze_batch_with_llm(batch_inputs, batch_scores)
                
                # Process batch results
                for j, (i, llm_result) in enumerate(zip(batch_rows, batch_results)):
                    # Refine prediction based on LLM analysis
                    refined_prediction = self._refine_prediction(
                        ml_prediction=ml_predictions[i],
                        anomaly_score=anomaly_scores[i],
                        llm_result=llm_result,
                        data_point=X_df.iloc[i]
                    )
                    
                    # Update hybrid prediction
                    hybrid_predictions[i] = refined_prediction
                    
                    # Store details
                    details.append({
                        "index": int(i),
                        "ml_prediction": int(ml_predictions[i]),
                        "anomaly_score": float(anomaly_scores[i]),
                        "hybrid_prediction": int(refined_prediction),
                        "llm_analysis": llm_result
                    })
            
            # For non-processed indices, just use ML prediction and store basic details
            for i in range(len(X_df)):
                if i not in indices_to_process:
                    details.append({
                        "index": int(i),
                        "ml_prediction": int(ml_predictions[i]),
                        "anomaly_score": float(anomaly_scores[i]),
                        "hybrid_prediction": int(ml_predictions[i]),
                        "llm_analysis": None
                    })
        else:
            # If LLM is not available or not requested
            hybrid_predictions = ml_predictions
            
            # Still populate basic details
            for i in range(len(X_df)):
                details.append({
                    "index": int(i),
                    "ml_prediction": int(ml_predictions[i]),
                    "anomaly_score": float(anomaly_scores[i]),
                    "hybrid_prediction": int(ml_predictions[i]),
                    "llm_analysis": None
                })
        
        # Sort details by index for consistency
        details.sort(key=lambda x: x["index"])
        
        if return_details:
            return hybrid_predictions, details
        else:
            return hybrid_predictions
    
    def _analyze_batch_with_llm(self, data_points, anomaly_scores):
        """
        Analyze multiple data points in a batch with LLM.
        
        Args:
            data_points: List of data points as pandas Series
            anomaly_scores: List of anomaly scores from ML model
            
        Returns:
            List of LLM analysis results
        """
        if not self.has_llm:
            return [{"error": "LLM integration not available"} for _ in data_points]
        
        # Create a combined prompt for all points in batch
        batch_size = len(data_points)
        
        # For each data point, check cache first
        results = []
        uncached_indices = []
        uncached_data_points = []
        uncached_scores = []
        
        for i, (data_point, score) in enumerate(zip(data_points, anomaly_scores)):
            # Create a cache key from the data point and score
            cache_key = hash(tuple(data_point.values)) + hash(float(score))
            
            if cache_key in self.response_cache:
                # Use cached result
                results.append(self.response_cache[cache_key])
            else:
                # Need to query LLM
                results.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_data_points.append(data_point)
                uncached_scores.append(score)
        
        # If all results were cached, return immediately
        if not uncached_indices:
            return results
        
        # Process uncached data points in a single batch
        # This could be further optimized by using OpenAI's function calling API
        # which would allow for more structured responses
        
        batch_prompt = (
            "Analyze the following data points to determine if they represent anomalous states "
            "for industrial robots. For each data point, provide a structured analysis.\n\n"
        )
        
        for i, (data_point, score) in enumerate(zip(uncached_data_points, uncached_scores)):
            # Format the data point
            point_data = "\n".join([f"{col}: {val}" for col, val in data_point.items()])
            
            # Identify potentially anomalous readings
            anomalous_readings = []
            for col, val in data_point.items():
                if col in self.sensor_thresholds:
                    threshold = self.sensor_thresholds[col]
                    if val < threshold.get("min", float("-inf")) or val > threshold.get("max", float("inf")):
                        anomalous_readings.append(f"{col}: {val} (outside range [{threshold.get('min', 'N/A')}, {threshold.get('max', 'N/A')}])")
            
            batch_prompt += f"""
            DATA POINT {i+1}:
            {point_data}
            
            Machine Learning Anomaly Score: {score:.4f} (higher = more anomalous)
            
            Potentially anomalous readings:
            {os.linesep.join(anomalous_readings) if anomalous_readings else "None identified based on thresholds."}
            
            """
        
        batch_prompt += """
        Domain knowledge:
        - Industrial robots typically operate in temperature ranges of 5-80°C
        - Vibration levels above 0.8 usually indicate mechanical issues
        - Battery levels below 20% are critical
        - Error rates above 2% need investigation
        
        Provide your analysis for each data point using the following format:
        
        DATA POINT X:
        1. Is this an anomaly? (yes/no)
        2. Confidence in assessment (0-100%)
        3. Reasoning for this conclusion
        4. Potential false positive/negative factors
        5. Recommended actions
        
        Provide your responses for each data point, using this exact format.
        """
        
        try:
            # Call LLM with batch prompt
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert in industrial robotics anomaly detection."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,
                max_tokens=1000 + (500 * len(uncached_data_points))  # Scale tokens with batch size
            )
            
            llm_response = response.choices[0].message.content
            
            # Split response by data points
            data_point_responses = []
            
            # Split by "DATA POINT X:" headers
            parts = re.split(r"DATA POINT \d+:", llm_response)
            
            # Skip the first part if it's empty (usually is)
            if parts and not parts[0].strip():
                parts = parts[1:]
            
            # Parse each part
            for i, part in enumerate(parts):
                if i < len(uncached_indices):  # Ensure we don't go out of bounds
                    # Parse structured response
                    is_anomaly_match = re.search(r"1\.\s*Is this an anomaly\?\s*\(yes/no\)\s*:?\s*(yes|no)", part, re.IGNORECASE)
                    is_anomaly = is_anomaly_match.group(1).lower() == "yes" if is_anomaly_match else None
                    
                    confidence_match = re.search(r"2\.\s*Confidence[^:]*:?\s*(\d+)", part)
                    confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
                    
                    reasoning_match = re.search(r"3\.\s*Reasoning[^:]*:?\s*(.*?)(?=4\.|\n\s*4\.|\Z)", part, re.DOTALL)
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                    
                    fp_fn_match = re.search(r"4\.\s*Potential false[^:]*:?\s*(.*?)(?=5\.|\n\s*5\.|\Z)", part, re.DOTALL)
                    fp_fn_factors = fp_fn_match.group(1).strip() if fp_fn_match else ""
                    
                    actions_match = re.search(r"5\.\s*Recommended actions[^:]*:?\s*(.*?)(?=DATA POINT|\Z)", part, re.DOTALL)
                    actions = actions_match.group(1).strip() if actions_match else ""
                    
                    result = {
                        "is_anomaly": is_anomaly,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "false_positive_negative_factors": fp_fn_factors,
                        "recommended_actions": actions
                    }
                    
                    # Store in cache
                    cache_key = hash(tuple(uncached_data_points[i].values)) + hash(float(uncached_scores[i]))
                    self.response_cache[cache_key] = result
                    
                    # Update the results list
                    original_idx = uncached_indices[i]
                    results[original_idx] = result
            
            # Handle any missing responses with a default
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = {"error": "Failed to parse LLM response"}
            
            return results
            
        except Exception as e:
            logger.error(f"Error calling LLM for batch: {str(e)}")
            
            # Fill in errors for all uncached data points
            for i in uncached_indices:
                results[i] = {"error": str(e)}
            
            return results
    
    def _analyze_with_llm(self, data_point, anomaly_score):
        """
        Analyze a data point with LLM to determine if it's truly anomalous.
        
        Args:
            data_point: Single data point as pandas Series
            anomaly_score: Anomaly score from ML model
            
        Returns:
            dict: LLM analysis results
        """
        if not self.has_llm:
            return {"error": "LLM integration not available"}
        
        # Create a cache key from the data point and score
        cache_key = hash(tuple(data_point.values)) + hash(float(anomaly_score))
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Format the data point
        point_data = "\n".join([f"{col}: {val}" for col, val in data_point.items()])
        
        # Identify potentially anomalous readings
        anomalous_readings = []
        for col, val in data_point.items():
            if col in self.sensor_thresholds:
                threshold = self.sensor_thresholds[col]
                if val < threshold.get("min", float("-inf")) or val > threshold.get("max", float("inf")):
                    anomalous_readings.append(f"{col}: {val} (outside range [{threshold.get('min', 'N/A')}, {threshold.get('max', 'N/A')}])")
        
        # Create the prompt
        prompt = f"""
        Analyze this data point to determine if it represents an anomalous state for an industrial robot:
        
        Data Point:
        {point_data}
        
        Machine Learning Anomaly Score: {anomaly_score:.4f} (higher = more anomalous)
        
        Potentially anomalous readings:
        {os.linesep.join(anomalous_readings) if anomalous_readings else "None identified based on thresholds."}
        
        Domain knowledge:
        - Industrial robots typically operate in temperature ranges of 5-80°C
        - Vibration levels above 0.8 usually indicate mechanical issues
        - Battery levels below 20% are critical
        - Error rates above 2% need investigation
        
        Please analyze and respond in the following structured format:
        
        1. Is this an anomaly? (yes/no)
        2. Confidence in assessment (0-100%)
        3. Reasoning for this conclusion
        4. Potential false positive/negative factors
        5. Recommended actions
        """
        
        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert in industrial robotics anomaly detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse structured response
            is_anomaly_match = re.search(r"1\.\s*Is this an anomaly\?\s*\(yes/no\)\s*:?\s*(yes|no)", llm_response, re.IGNORECASE)
            is_anomaly = is_anomaly_match.group(1).lower() == "yes" if is_anomaly_match else None
            
            confidence_match = re.search(r"2\.\s*Confidence[^:]*:?\s*(\d+)", llm_response)
            confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
            
            reasoning_match = re.search(r"3\.\s*Reasoning[^:]*:?\s*(.*?)(?=4\.|\n\s*4\.|\Z)", llm_response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            fp_fn_match = re.search(r"4\.\s*Potential false[^:]*:?\s*(.*?)(?=5\.|\n\s*5\.|\Z)", llm_response, re.DOTALL)
            fp_fn_factors = fp_fn_match.group(1).strip() if fp_fn_match else ""
            
            actions_match = re.search(r"5\.\s*Recommended actions[^:]*:?\s*(.*?)(?=\Z)", llm_response, re.DOTALL)
            actions = actions_match.group(1).strip() if actions_match else ""
            
            result = {
                "is_anomaly": is_anomaly,
                "confidence": confidence,
                "reasoning": reasoning,
                "false_positive_negative_factors": fp_fn_factors,
                "recommended_actions": actions,
                "raw_response": llm_response
            }
            
            # Cache the result
            self.response_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return {"error": str(e)}
    
    def _refine_prediction(self, ml_prediction, anomaly_score, llm_result, data_point):
        """
        Refine the ML prediction using LLM analysis.
        
        Args:
            ml_prediction: ML model prediction (-1 for anomaly, 1 for normal)
            anomaly_score: Anomaly score from ML model
            llm_result: Results from LLM analysis
            data_point: The data point being analyzed
            
        Returns:
            int: Refined prediction (-1 for anomaly, 1 for normal)
        """
        # Default to ML prediction if there's an error or no LLM result
        if not llm_result or "error" in llm_result or llm_result.get("is_anomaly") is None:
            return ml_prediction
        
        llm_is_anomaly = llm_result.get("is_anomaly")
        llm_confidence = llm_result.get("confidence", 0.5)
        
        # Use a weighted decision approach combining ML and LLM results
        # 1. If both agree, keep the prediction
        if (ml_prediction == -1 and llm_is_anomaly) or (ml_prediction == 1 and not llm_is_anomaly):
            return ml_prediction
        
        # 2. If they disagree, consider confidence and anomaly score
        if llm_confidence > 0.8:
            # High LLM confidence can override ML
            return -1 if llm_is_anomaly else 1
        elif llm_confidence > 0.6:
            # Medium confidence: consider anomaly score
            if anomaly_score > 0.7 and llm_is_anomaly:
                # Both somewhat indicate anomaly
                return -1
            elif anomaly_score < 0.3 and not llm_is_anomaly:
                # Both somewhat indicate normal
                return 1
            else:
                # Conflicting signals, fall back to ML
                return ml_prediction
        else:
            # Low LLM confidence, stick with ML prediction
            return ml_prediction
    
    def evaluate(self, X, y_true, use_llm=True):
        """
        Evaluate the model performance.
        
        Args:
            X: Test data
            y_true: True labels (1 for anomaly, 0 for normal)
            use_llm: Whether to use LLM for refinement
            
        Returns:
            dict: Performance metrics for both ML and hybrid approaches
        """
        # Ensure y_true is in the right format (0 for normal, 1 for anomaly)
        # but our predict method returns 1 for normal and -1 for anomaly (sklearn convention)
        y_true_binary = y_true.copy()
        
        # Convert X to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.column_names)
        else:
            X_df = X.copy()
        
        # Get ML-only predictions
        y_pred_ml = self.predict(X_df, use_llm=False)
        y_pred_ml_binary = (y_pred_ml == -1).astype(int)  # Convert to binary (0=normal, 1=anomaly)
        
        # Calculate ML-only metrics
        ml_metrics = {
            "accuracy": accuracy_score(y_true_binary, y_pred_ml_binary),
            "precision": precision_score(y_true_binary, y_pred_ml_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_ml_binary, zero_division=0),
            "f1_score": f1_score(y_true_binary, y_pred_ml_binary, zero_division=0)
        }
        
        # Get hybrid predictions if LLM is available
        if self.has_llm and use_llm:
            y_pred_hybrid, details = self.predict(X_df, return_details=True, use_llm=True)
            y_pred_hybrid_binary = (y_pred_hybrid == -1).astype(int)
            
            # Calculate hybrid metrics
            hybrid_metrics = {
                "accuracy": accuracy_score(y_true_binary, y_pred_hybrid_binary),
                "precision": precision_score(y_true_binary, y_pred_hybrid_binary, zero_division=0),
                "recall": recall_score(y_true_binary, y_pred_hybrid_binary, zero_division=0),
                "f1_score": f1_score(y_true_binary, y_pred_hybrid_binary, zero_division=0)
            }
            
            # Calculate changes
            differences = {
                "accuracy_delta": hybrid_metrics["accuracy"] - ml_metrics["accuracy"],
                "precision_delta": hybrid_metrics["precision"] - ml_metrics["precision"],
                "recall_delta": hybrid_metrics["recall"] - ml_metrics["recall"],
                "f1_score_delta": hybrid_metrics["f1_score"] - ml_metrics["f1_score"]
            }
            
            # Store for later reference
            self.performance_metrics["ml_only"] = ml_metrics
            self.performance_metrics["hybrid"] = hybrid_metrics
            
            # Get correction statistics
            corrections = self._analyze_corrections(y_true_binary, y_pred_ml_binary, y_pred_hybrid_binary)
            
            return {
                "ml_only": ml_metrics,
                "hybrid": hybrid_metrics,
                "differences": differences,
                "corrections": corrections
            }
        else:
            # Only ML metrics available
            self.performance_metrics["ml_only"] = ml_metrics
            return {
                "ml_only": ml_metrics
            }
    
    def _analyze_corrections(self, y_true, y_pred_ml, y_pred_hybrid):
        """Analyze how the LLM corrections affected predictions."""
        total_samples = len(y_true)
        total_changed = sum(y_pred_ml != y_pred_hybrid)
        
        # Count corrections by type
        false_positives_fixed = sum((y_pred_ml == 1) & (y_pred_hybrid == 0) & (y_true == 0))
        false_negatives_fixed = sum((y_pred_ml == 0) & (y_pred_hybrid == 1) & (y_true == 1))
        
        correct_to_incorrect = sum((y_pred_ml == y_true) & (y_pred_hybrid != y_true))
        
        return {
            "total_samples": int(total_samples),
            "total_changed": int(total_changed),
            "percent_changed": float(total_changed / total_samples * 100 if total_samples > 0 else 0),
            "false_positives_fixed": int(false_positives_fixed),
            "false_negatives_fixed": int(false_negatives_fixed),
            "correct_to_incorrect": int(correct_to_incorrect),
            "net_improvement": int((false_positives_fixed + false_negatives_fixed) - correct_to_incorrect)
        }
    
    def explain_prediction(self, data_point, return_raw=False):
        """
        Generate a human-readable explanation for a prediction.
        
        Args:
            data_point: Single data point to explain
            return_raw: Whether to return raw LLM response
            
        Returns:
            str or dict: Explanation of the prediction
        """
        if not self.has_llm:
            return "LLM integration not available for explanations."
        
        # Convert to DataFrame if necessary
        if isinstance(data_point, np.ndarray):
            if len(data_point.shape) == 1:
                data_point = pd.Series(data_point, index=self.column_names)
            else:
                data_point = pd.DataFrame(data_point, columns=self.column_names).iloc[0]
        
        # Scale the data point
        data_point_values = data_point.values.reshape(1, -1)
        scaled_values = self.scaler.transform(data_point_values).flatten()
        
        # Get ML prediction
        ml_prediction = self.ml_detector.predict(scaled_values.reshape(1, -1))[0]
        
        # Get anomaly score
        if hasattr(self.ml_detector, "decision_function"):
            anomaly_score = -self.ml_detector.decision_function(scaled_values.reshape(1, -1))[0]
        elif hasattr(self.ml_detector, "score_samples"):
            anomaly_score = -self.ml_detector.score_samples(scaled_values.reshape(1, -1))[0]
        else:
            anomaly_score = 0
        
        # Get LLM analysis
        llm_result = self._analyze_with_llm(data_point, anomaly_score)
        
        if return_raw:
            return {
                "ml_prediction": "Anomaly" if ml_prediction == -1 else "Normal",
                "anomaly_score": float(anomaly_score),
                "llm_analysis": llm_result
            }
        
        # Format a human-readable explanation
        if "error" in llm_result:
            explanation = f"The machine learning model detected this data point as {'anomalous' if ml_prediction == -1 else 'normal'} with an anomaly score of {anomaly_score:.4f}."
        else:
            explanation = f"The AI system {'detected' if ml_prediction == -1 or llm_result.get('is_anomaly', False) else 'did not detect'} an anomaly in this data point.\n\n"
            
            if llm_result.get("reasoning"):
                explanation += f"Reasoning: {llm_result['reasoning']}\n\n"
                
            if llm_result.get("recommended_actions"):
                explanation += f"Recommended actions: {llm_result['recommended_actions']}"
        
        return explanation
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: Whether the save was successful
        """
        import joblib
        
        try:
            # Create a dict with all serializable components
            model_data = {
                "base_detector_type": self.base_detector_type,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "ml_detector": self.ml_detector,
                "scaler": self.scaler,
                "is_ml_fitted": self.is_ml_fitted,
                "column_names": self.column_names,
                "sensor_thresholds": self.sensor_thresholds,
                "domain_knowledge": self.domain_knowledge,
                "llm_model": self.llm_model,
                "has_llm": self.has_llm,
                "performance_metrics": self.performance_metrics
            }
            
            # Save to file
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filepath, api_key=None):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            api_key: Optional API key for LLM
            
        Returns:
            HybridAnomalyDetector: Loaded model
        """
        import joblib
        
        try:
            # Load data from file
            model_data = joblib.load(filepath)
            
            # Create new instance
            model = cls(
                base_detector=model_data["base_detector_type"],
                contamination=model_data["contamination"],
                random_state=model_data["random_state"],
                api_key=api_key,
                llm_model=model_data["llm_model"],
                domain_knowledge=model_data["domain_knowledge"],
                threshold_config=model_data["sensor_thresholds"]
            )
            
            # Restore state
            model.ml_detector = model_data["ml_detector"]
            model.scaler = model_data["scaler"]
            model.is_ml_fitted = model_data["is_ml_fitted"]
            model.column_names = model_data["column_names"]
            model.performance_metrics = model_data["performance_metrics"]
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# Example usage when run as a script
if __name__ == "__main__":
    # Example threshold configuration
    thresholds = {
        "temperature": {"min": 0, "max": 95},
        "vibration": {"min": 0, "max": 0.8},
        "power": {"min": 10, "max": 1000},
        "humidity": {"min": 10, "max": 90},
        "pressure": {"min": 0.8, "max": 1.2}
    }
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Normal data
    X_normal = np.random.normal(loc=0.5, scale=0.15, size=(int(n_samples * 0.9), n_features))
    
    # Anomalies
    X_anomalies = np.random.normal(loc=0.8, scale=0.2, size=(int(n_samples * 0.1), n_features))
    
    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    y = np.zeros(n_samples)
    y[int(n_samples * 0.9):] = 1  # Mark anomalies
    
    # Create column names
    columns = ["temperature", "vibration", "power", "humidity", "pressure"]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=columns)
    
    # Scale to realistic values
    df["temperature"] = df["temperature"] * 100  # 0-100°C
    df["vibration"] = df["vibration"]  # 0-1 scale
    df["power"] = df["power"] * 1000  # 0-1000W
    df["humidity"] = df["humidity"] * 100  # 0-100%
    df["pressure"] = df["pressure"]  # 0-2 scale
    
    # Create detector
    try:
        # Try to read API key from file
        api_key = None
        if os.path.exists("openai_api_key.txt"):
            with open("openai_api_key.txt", "r") as f:
                api_key = f.read().strip()
        
        detector = HybridAnomalyDetector(
            base_detector="iforest",
            contamination=0.1,
            random_state=42,
            api_key=api_key,
            threshold_config=thresholds
        )
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
        
        # Train model
        detector.fit(X_train)
        
        # Evaluate
        print("Evaluating model...")
        results = detector.evaluate(X_test, y_test, use_llm=detector.has_llm)
        
        print("\nEvaluation Results:")
        print("ML-only metrics:", results["ml_only"])
        
        if "hybrid" in results:
            print("Hybrid metrics:", results["hybrid"])
            print("Differences:", results["differences"])
            print("Corrections:", results["corrections"])
        
        # Example explanation
        if detector.has_llm:
            print("\nExample Explanation:")
            explanation = detector.explain_prediction(X_test.iloc[0])
            print(explanation)
        
    except Exception as e:
        print(f"Error in example: {str(e)}") 