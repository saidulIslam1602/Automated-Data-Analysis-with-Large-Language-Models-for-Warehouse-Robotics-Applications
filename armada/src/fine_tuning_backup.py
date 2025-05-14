#!/usr/bin/env python3
"""
Fine-tuning module for robotics maintenance LLM optimization.

This module prepares robotics maintenance data for fine-tuning an OpenAI model,
creating a specialized model that focuses on warehouse robotics maintenance optimization.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import openai
from openai import OpenAI
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fine_tuning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("fine_tuning")

class RoboticsMaintenanceLLMFineTuner:
    """Fine-tuning OpenAI models for robotics maintenance optimization."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the fine-tuning module.
        
        Args:
            api_key: OpenAI API key
            model_name: Base model to fine-tune
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        self.fine_tuned_model_id = None
        
    def prepare_training_data(self, df: pd.DataFrame, focus_factor: float = 1.5, max_examples: int = 1000) -> List[Dict[str, Any]]:
        """
        Prepare training data for fine-tuning from robotics maintenance dataframe.
        
        Args:
            df: Pandas DataFrame containing robotics maintenance data
            focus_factor: Factor to increase focus on rare anomaly patterns
            max_examples: Maximum number of examples to generate
            
        Returns:
            List of training examples in OpenAI fine-tuning format
        """
        logger.info(f"Preparing training data with {len(df)} rows")
        
        # Improved sampling strategy to ensure diverse training examples
        training_data = []
        
        # Calculate basic statistics for numeric columns to help with data understanding
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_columns].describe()
        
        # Identify potential anomaly signals for better example generation
        # Look for outliers in each feature
        outliers = {}
        for col in numeric_columns:
            q1 = stats.loc['25%', col]
            q3 = stats.loc['75%', col]
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            
            # Find outliers for this column
            high_outliers = df[df[col] > upper_bound].index.tolist()
            low_outliers = df[df[col] < lower_bound].index.tolist()
            
            if high_outliers or low_outliers:
                outliers[col] = {
                    'high': high_outliers,
                    'low': low_outliers
                }
        
        # Generate stratified samples to ensure coverage of different patterns
        strata = []
        
        # 1. Include clear anomalies (rows with multiple outlier features)
        multi_outlier_indices = set()
        for col, outlier_indices in outliers.items():
            for direction in ['high', 'low']:
                multi_outlier_indices.update(outlier_indices[direction])
        
        # Count occurrence frequency of each index
        outlier_counts = {}
        for col, outlier_indices in outliers.items():
            for direction in ['high', 'low']:
                for idx in outlier_indices[direction]:
                    outlier_counts[idx] = outlier_counts.get(idx, 0) + 1
        
        # Sort indices by outlier count (descending) to prioritize clear anomalies
        sorted_indices = sorted(outlier_counts.keys(), key=lambda x: outlier_counts[x], reverse=True)
        
        # Take top 30% for the first stratum (clear anomalies)
        clear_anomaly_count = min(int(max_examples * 0.3), len(sorted_indices))
        if clear_anomaly_count > 0:
            strata.append({
                'name': 'clear_anomalies',
                'indices': sorted_indices[:clear_anomaly_count],
                'weight': 1.2  # Slightly higher weight for clear anomalies
            })
        
        # 2. Include boundary cases (single outlier features)
        boundary_indices = [idx for idx, count in outlier_counts.items() if count == 1]
        boundary_count = min(int(max_examples * 0.2), len(boundary_indices))
        if boundary_count > 0:
            strata.append({
                'name': 'boundary_cases',
                'indices': boundary_indices[:boundary_count],
                'weight': 1.0
            })
        
        # 3. Include normal cases (no outliers)
        normal_indices = list(set(df.index) - set(outlier_counts.keys()))
        normal_count = min(int(max_examples * 0.4), len(normal_indices))
        if normal_count > 0:
            strata.append({
                'name': 'normal_cases',
                'indices': np.random.choice(normal_indices, size=normal_count, replace=False).tolist(),
                'weight': 0.8  # Slightly lower weight for normal cases
            })
        
        # 4. Include rare patterns (using rare value combinations)
        # Identify rare combinations in key maintenance features
        maintenance_features = [col for col in df.columns if 'motor' in col.lower() or 
                               'temperature' in col.lower() or 
                               'vibration' in col.lower() or
                               'battery' in col.lower() or
                               'current' in col.lower()]
        
        if maintenance_features:
            # Discretize features for better combination analysis
            disc_df = pd.DataFrame()
            for col in maintenance_features:
                if col in numeric_columns:
                    # Create 5 equal-width bins
                    disc_df[f'{col}_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
            
            # Count occurrences of each combination
            if not disc_df.empty:
                value_counts = disc_df.value_counts()
                rare_combinations = value_counts[value_counts <= 3].index
                
                rare_indices = []
                for comb in rare_combinations:
                    mask = True
                    for i, col in enumerate(disc_df.columns):
                        mask &= (disc_df[col] == comb[i])
                    rare_indices.extend(df[mask].index.tolist())
                
                # Unique rare indices
                rare_indices = list(set(rare_indices))
                rare_count = min(int(max_examples * 0.1), len(rare_indices))
                
                if rare_count > 0:
                    strata.append({
                        'name': 'rare_patterns',
                        'indices': rare_indices[:rare_count],
                        'weight': 1.5  # Higher weight for rare patterns
                    })
        
        # Generate examples from each stratum
        for stratum in strata:
            stratum_indices = stratum['indices']
            stratum_weight = stratum['weight']
            
            for idx in stratum_indices:
                row = df.iloc[idx]
                
                # Construct a more detailed and realistic prompt
                system_message = self._create_system_message(row)
                user_message = self._create_user_message(row)
                
                # Generate a detailed, domain-specific maintenance recommendation
                assistant_message = self._create_assistant_message(row, stats, stratum['name'])
                
            example = {
                "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message}
                    ]
                }
                
                # Add this example multiple times based on its weight
                repeat_count = max(1, int(stratum_weight * focus_factor))
                for _ in range(repeat_count):
                    training_data.append(example)
        
        # Ensure we don't exceed the maximum examples
        if len(training_data) > max_examples:
            logger.info(f"Sampled {len(training_data)} examples, reducing to {max_examples}")
            # Shuffle before truncating to maintain diversity
            np.random.shuffle(training_data)
            training_data = training_data[:max_examples]
        
        logger.info(f"Generated {len(training_data)} training examples")
        
        return training_data
    
    def _create_system_message(self, row: pd.Series) -> str:
        """
        Creates an improved system message with more domain specificity.
        
        Args:
            row: Row from dataframe
            
        Returns:
            System message string
        """
        return """You are an expert maintenance optimization assistant for warehouse robots. 
Your task is to analyze sensor data and provide specific, actionable maintenance recommendations.

Follow these guidelines:
1. Analyze all sensor readings to identify anomalies
2. Assess maintenance urgency on a scale from 1-10
3. Provide specific component recommendations
4. Estimate maintenance downtime
5. Suggest optimal maintenance scheduling
6. Only report issues when data indicates a problem
7. Specify confidence in your assessment
8. Use precise technical language appropriate for maintenance engineers

Your output should be concise, actionable, and specific to the robot's condition."""
    
    def _create_user_message(self, row: pd.Series) -> str:
        """
        Creates an improved user message with richer context.
        
        Args:
            row: Row from dataframe
            
        Returns:
            User message string
        """
        # Get robot ID if available
        robot_id = row.get('robot_id', f"RBT-{np.random.randint(1000, 9999)}")
        
        # Enhanced user message with more context
        message = f"Please analyze the following sensor data for robot {robot_id} and provide a maintenance recommendation:\n\n"
        
        # Add all numeric data with units where appropriate
        for col in row.index:
            value = row[col]
            
            # Skip non-numeric or missing values
            if not isinstance(value, (int, float)) or pd.isna(value):
                continue
                
            # Add appropriate units based on column name
            if 'temperature' in col.lower():
                message += f"{col}: {value}°C\n"
            elif 'battery' in col.lower() and value <= 100:
                message += f"{col}: {value}%\n"
            elif 'current' in col.lower():
                message += f"{col}: {value}A\n"
            elif 'voltage' in col.lower():
                message += f"{col}: {value}V\n"
            elif 'pressure' in col.lower():
                message += f"{col}: {value}kPa\n"
            elif 'speed' in col.lower():
                message += f"{col}: {value}rpm\n"
            elif 'vibration' in col.lower():
                message += f"{col}: {value}mm/s\n"
            elif 'runtime' in col.lower() or 'hours' in col.lower():
                message += f"{col}: {value}h\n"
            else:
                message += f"{col}: {value}\n"
        
        # Add operational context if available
        if 'operational_mode' in row:
            message += f"\nOperational mode: {row['operational_mode']}\n"
            
        if 'last_maintenance' in row:
            message += f"Last maintenance: {row['last_maintenance']}\n"
            
        message += "\nPlease provide a detailed maintenance analysis and recommendation."
        return message
    
    def _create_assistant_message(self, row: pd.Series, stats: pd.DataFrame, stratum_type: str) -> str:
        """
        Creates a more specific and actionable assistant message based on the data.
        
        Args:
            row: Row from dataframe
            stats: Statistics about the dataset
            stratum_type: Type of stratum this example belongs to
            
        Returns:
            Assistant message string
        """
        # Initialize message structure
        maintenance_needed = stratum_type in ['clear_anomalies', 'rare_patterns']
        
        # Robot ID
        robot_id = row.get('robot_id', 'unknown')
        
        # Analyze values to identify potential issues
        issues = []
        failure_modes = []
        
        # Check numeric columns for anomalies
        for col in row.index:
            value = row[col]
            
            # Skip non-numeric or missing values
            if not isinstance(value, (int, float)) or pd.isna(value):
                continue
                
            # Compare with statistics if available
            if col in stats.columns:
                mean = stats.loc['mean', col]
                std = stats.loc['std', col]
                max_val = stats.loc['max', col]
                
                # Check for anomalous values (>2 std from mean or near max)
                if abs(value - mean) > 2 * std or value > 0.9 * max_val:
                    # Identify specific failure modes based on column type
                    if 'temperature' in col.lower() and value > mean + 1.5 * std:
                        issues.append(f"High {col}: {value:.1f}°C (normal range: {mean-std:.1f}-{mean+std:.1f}°C)")
                        failure_modes.append(f"Overheating in {col}")
                        
                    elif 'vibration' in col.lower() and value > mean + 1.5 * std:
                        issues.append(f"Excessive {col}: {value:.2f}mm/s (normal: {mean:.2f}mm/s)")
                        failure_modes.append(f"Mechanical misalignment in {col}")
                        
                    elif 'current' in col.lower() and value > mean + 1.5 * std:
                        issues.append(f"High {col}: {value:.2f}A (normal: {mean:.2f}A)")
                        failure_modes.append(f"Electrical overload in {col}")
                        
                    elif 'battery' in col.lower() and value < mean - 1.5 * std:
                        issues.append(f"Low {col}: {value:.1f}% (normal: {mean:.1f}%)")
                        failure_modes.append(f"Battery degradation")
                        
                    elif 'pressure' in col.lower() and abs(value - mean) > 1.5 * std:
                        issues.append(f"Abnormal {col}: {value:.1f}kPa (normal: {mean:.1f}kPa)")
                        failure_modes.append(f"Pressure regulation issue in {col}")
        
        # Determine urgency level based on issues found
        if not issues or not maintenance_needed:
            risk_level = np.random.uniform(1, 3)
            urgency = "Low: Routine maintenance sufficient"
            timeframe = "during next scheduled"
            total_risk = risk_level
        elif len(issues) == 1:
            risk_level = np.random.uniform(3, 6)
            urgency = "Medium: Maintenance recommended"
            timeframe = "within 2 weeks"
            total_risk = risk_level
        else:
            risk_level = np.random.uniform(6, 9)
            urgency = "High: Urgent maintenance needed"
            timeframe = "within 72 hours"
            total_risk = risk_level
            
        # Calculate estimated maintenance cost based on issues
        base_cost = 200 + 50 * len(issues)
        risk_cost = total_risk * 30
        
        # Account for maintenance history if available
        recent_maintenance = 0
        if 'last_maintenance' in row and isinstance(row['last_maintenance'], (int, float)):
            recent_maintenance = row['last_maintenance']
            
        history_factor = max(0.8, 1.0 - recent_maintenance * 0.05)  # Discount for frequent maintenance
        
        estimated_cost = (base_cost + risk_cost) * history_factor
        
        # Create sensor counts for bias checking
        sensor_counts = {}
        for issue in issues:
            sensor_type = issue.split(':')[0].strip().lower()
            sensor_counts[sensor_type] = sensor_counts.get(sensor_type, 0) + 1
        
        # Perform bias check
        bias_warnings = self._check_for_bias(sensor_counts, failure_modes, total_risk)
        bias_disclaimer = ""
        if bias_warnings:
            bias_disclaimer = "\n\n### Potential Model Limitations\n"
            bias_disclaimer += "This analysis may have limited reliability due to: "
            bias_disclaimer += ", ".join(bias_warnings) + "."
        
        # Create more specific and detailed recommendation
        completion = f"""## Maintenance Analysis for Robot {robot_id}

### Maintenance Prediction
Risk Level: {total_risk:.1f}/10 - {urgency}
{', '.join(failure_modes) if failure_modes else 'No specific failure modes detected at this time'}

### Recommendation
{timeframe} maintenance schedule with focus on:
{"" if not failure_modes else "- " + "\\n- ".join(failure_modes)}
{'' if not failure_modes else '\\n'}Regular inspection of all components is recommended.

### Cost Analysis
Estimated maintenance cost: approximately ${estimated_cost:.2f}
Potential downtime: between {max(1, int(total_risk) * 2 - 1)} and {max(1, int(total_risk) * 2 + 1)} hours
Estimated production impact: in the range of ${estimated_cost * 1.3:.2f} to ${estimated_cost * 1.7:.2f}

### Optimization Strategy
{'Preventative maintenance can be delayed to optimize cost efficiency.' if total_risk < 3 else 'Immediate action recommended to prevent more costly failures.'}
{'Consider grouping with other nearby robots for maintenance efficiency.' if total_risk < 5 else 'Individual attention required for this unit.'}{bias_disclaimer}"""

        return completion
    
    def _check_for_bias(self, sensor_counts: Dict[str, int], failure_modes: List[str], risk_level: float) -> List[str]:
        """
        Check for potential biases in the prediction.
        
        Args:
            sensor_counts: Count of each sensor type in issues
            failure_modes: List of identified failure modes
            risk_level: Overall risk level
            
        Returns:
            List of bias warnings
        """
        warnings = []
        
        # Check for over-reliance on a single sensor type
        if sensor_counts and max(sensor_counts.values()) > 2:
            max_sensor = max(sensor_counts.keys(), key=lambda k: sensor_counts[k])
            warnings.append(f"potential over-reliance on {max_sensor} readings")
            
        # Check for high risk with few failure modes
        if risk_level > 7 and len(failure_modes) < 2:
            warnings.append("high risk assessment with limited failure evidence")
            
        # Check for contradictory failure modes
        contradictory_pairs = [
            ('overheating', 'low temperature'),
            ('battery degradation', 'battery overcharge'),
            ('mechanical misalignment', 'normal vibration')
        ]
        
        for mode1, mode2 in contradictory_pairs:
            if any(mode1 in fm.lower() for fm in failure_modes) and any(mode2 in fm.lower() for fm in failure_modes):
                warnings.append("potentially contradictory failure modes")
                break
            
        return warnings
    
    def save_training_file(self, training_data: List[Dict[str, Any]], output_path: str) -> str:
        """
        Save training data to a JSONL file.
        
        Args:
            training_data: List of training examples
            output_path: Path to save the training file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
                
        logger.info(f"Saved {len(training_data)} training examples to {output_path}")
        return output_path
    
    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI.
        
        Args:
            file_path: Path to the training file
            
        Returns:
            File ID assigned by OpenAI
        """
        logger.info(f"Uploading training file {file_path} to OpenAI")
        
        max_retries = 3
        retry_delay = 5  # initial delay in seconds
        
        for retry in range(max_retries):
            try:
                with open(file_path, 'rb') as f:
                    response = self.client.files.create(
                        file=f,
                        purpose="fine-tune"
                    )
                
                file_id = response.id
                logger.info(f"File uploaded successfully with ID: {file_id}")
                return file_id
                
            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() or "exceeded quota" in error_msg.lower():
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)  # exponential backoff
                        logger.warning(f"API rate limit exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API rate limit exceeded after {max_retries} retries. Please check your OpenAI usage limits.")
                        raise RuntimeError(f"API rate limit exceeded: {error_msg}. Please check your OpenAI usage limits and try again later.")
                else:
                    logger.error(f"Error uploading file: {error_msg}")
                    raise
        
        # This should not be reached due to the raise in the loop
        raise RuntimeError("Unexpected error during file upload")
    
    def create_fine_tuning_job(self, file_id: str, validation_file_id: str = None, n_epochs: int = 3) -> str:
        """
        Create a fine-tuning job.
        
        Args:
            file_id: Training file ID
            validation_file_id: Validation file ID (optional)
            n_epochs: Number of training epochs (reduce for cost savings)
            
        Returns:
            Fine-tuning job ID
        """
        logger.info(f"Creating fine-tuning job with {n_epochs} epochs")
        
        max_retries = 3
        retry_delay = 5  # initial delay in seconds
        job_params = {
            "training_file": file_id,
            "model": self.model_name,
            "suffix": f"robotics_maintenance_{int(time.time())}",
            "hyperparameters": {
                "n_epochs": n_epochs
            }
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        for retry in range(max_retries):
            try:
                response = self.client.fine_tuning.jobs.create(**job_params)
                
                job_id = response.id
                logger.info(f"Fine-tuning job created successfully with ID: {job_id}")
                return job_id
                
            except Exception as e:
                error_msg = str(e)
                if "exceeded_quota" in error_msg or "rate limit" in error_msg.lower():
                    # Extract cost information if available
                    cost_info = ""
                    if "Cost of job" in error_msg:
                        try:
                            cost_start = error_msg.find("Cost of job")
                            cost_info = error_msg[cost_start:].split(".")[0] + "."
                        except:
                            pass
                    
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)  # exponential backoff
                        logger.warning(f"API quota exceeded. {cost_info} Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Provide a more helpful error message with cost information
                        if cost_info:
                            logger.error(f"API quota exceeded. {cost_info}")
                            logger.error("Consider reducing the training set size or number of epochs to lower costs.")
                            raise RuntimeError(f"API quota exceeded: {cost_info} Consider reducing dataset size or epochs.")
                        else:
                            logger.error(f"API quota exceeded after {max_retries} retries. Please check your OpenAI usage limits.")
                            raise RuntimeError("API quota exceeded. Please check your OpenAI usage limits or reduce dataset size.")
                else:
                    logger.error(f"Error creating fine-tuning job: {error_msg}")
                    raise
        
        # This should not be reached due to the raise in the loop
        raise RuntimeError("Unexpected error during job creation")
    
    def monitor_fine_tuning_job(self, job_id: str, poll_interval: int = 60) -> str:
        """
        Monitor a fine-tuning job until completion.
        
        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks
            
        Returns:
            Fine-tuned model ID
        """
        logger.info(f"Monitoring fine-tuning job {job_id}")
        max_consecutive_errors = 3
        consecutive_errors = 0
        
        while True:
            try:
                response = self.client.fine_tuning.jobs.retrieve(job_id)
                status = response.status
                consecutive_errors = 0  # Reset error counter on success
                
                logger.info(f"Job status: {status}")
                
                if status == "succeeded":
                    self.fine_tuned_model_id = response.fine_tuned_model
                    logger.info(f"Fine-tuning completed successfully. Model ID: {self.fine_tuned_model_id}")
                    return self.fine_tuned_model_id
                    
                elif status in ["failed", "cancelled"]:
                    error_message = f"Fine-tuning job {status}: {response.error or 'No error message provided'}"
                    logger.error(error_message)
                    
                    # If failed due to quota, provide helpful info
                    if response.error and "exceeded_quota" in str(response.error):
                        logger.error("Job failed due to quota limits. Consider:")
                        logger.error("1. Reducing the training dataset size")
                        logger.error("2. Decreasing the number of epochs")
                        logger.error("3. Checking your OpenAI billing settings")
                    
                    raise RuntimeError(error_message)
                
                # Add more detailed information for running jobs
                if status == "running" and hasattr(response, "training_progress"):
                    if response.training_progress:
                        logger.info(f"Training progress: {response.training_progress}%")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                error_msg = str(e)
                consecutive_errors += 1
                
                if "fine-tuned_model" in error_msg.lower():
                    # Sometimes the API returns error but the job succeeded
                    logger.info("Job appears to have succeeded despite API error")
                    
                    # Try to get the model ID
                    try:
                        response = self.client.fine_tuning.jobs.retrieve(job_id)
                        self.fine_tuned_model_id = response.fine_tuned_model
                        logger.info(f"Retrieved model ID: {self.fine_tuned_model_id}")
                        return self.fine_tuned_model_id
                    except:
                        pass
                
                logger.error(f"Error monitoring fine-tuning job: {error_msg}")
                
                # If we've had too many consecutive errors, abort
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Aborting after {max_consecutive_errors} consecutive errors")
                    raise RuntimeError(f"Monitoring failed after {max_consecutive_errors} consecutive errors: {error_msg}")
                
                # Otherwise wait and retry
                retry_delay = 30
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    def run_fine_tuning(self, data_df: pd.DataFrame, 
                        focus_factor: float = 1.5,
                        output_dir: str = "models",
                        max_train_examples: int = 500,
                        max_val_examples: int = 100,
                        n_epochs: int = 3) -> str:
        """
        Run the entire fine-tuning pipeline.
        
        Args:
            data_df: Input dataframe with robotics data
            focus_factor: Factor to increase focus on maintenance aspects
            output_dir: Directory to save files
            max_train_examples: Maximum number of training examples
            max_val_examples: Maximum number of validation examples
            n_epochs: Number of training epochs
            
        Returns:
            Fine-tuned model ID
        """
        logger.info("Starting fine-tuning process")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Split data for training and validation
        train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
        logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation examples")
        
        # Prepare training data with limits for cost efficiency
        training_data = self.prepare_training_data(train_df, focus_factor, max_train_examples)
        validation_data = self.prepare_training_data(val_df, focus_factor, max_val_examples)
        
        # Save training files
        training_file_path = os.path.join(output_dir, "robotics_training.jsonl")
        validation_file_path = os.path.join(output_dir, "robotics_validation.jsonl")
        
        self.save_training_file(training_data, training_file_path)
        self.save_training_file(validation_data, validation_file_path)
        
        # Upload files
        training_file_id = self.upload_training_file(training_file_path)
        validation_file_id = self.upload_training_file(validation_file_path)
        
        # Create and monitor fine-tuning job
        job_id = self.create_fine_tuning_job(training_file_id, validation_file_id, n_epochs)
        model_id = self.monitor_fine_tuning_job(job_id)
        
        logger.info(f"Fine-tuning process completed. Model ID: {model_id}")
        return model_id
        
    def analyze_maintenance_issues(self, issues, historical_data=None):
        """
        Analyze maintenance issues and provide recommendations.
        
        Args:
            issues: List of detected issues
            historical_data: Optional historical maintenance data
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate risk factors
        total_risk = sum(issue.get('risk_level', 0) for issue in issues)
        
        # Calculate history factor
        history_factor = 1.0
        if historical_data:
            history_factor = self._calculate_history_factor(historical_data)
        
        # Generate recommendations
        recommendations = []
        for issue in issues:
            recommendations.append({
                'issue': issue.get('description', 'Unknown'),
                'priority': issue.get('risk_level', 0),
                'action': self._generate_action(issue, total_risk)
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return {
            'total_issues': len(issues),
            'risk_level': total_risk,
            'recommendations': recommendations,
            'urgency': 'High' if total_risk > 5 else 'Medium' if total_risk > 3 else 'Low',
            'maintenance_required': total_risk > 0
        }

    def _generate_action(self, issue, total_risk):
        """Generate maintenance action based on issue and risk level."""
        if issue.get('risk_level', 0) > 3:
            return 'Immediate inspection required'
        elif total_risk > 5:
            return 'Schedule maintenance within 24 hours'
        else:
            return 'Schedule maintenance within 1 week'

    def train_model(self, training_data, validation_data=None, n_epochs=3):
        """
        Train the model on the provided data.
        
        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset
            n_epochs: Number of training epochs
        """
        try:
            # Prepare training data
            prepared_data = self._prepare_training_data(training_data)
            
            # Train model
            for epoch in range(n_epochs):
                logger.info(f"Training epoch {epoch + 1}/{n_epochs}")
                self._train_epoch(prepared_data)
                
                # Validate if validation data is provided
                if validation_data:
                    validation_metrics = self._validate(validation_data)
                    logger.info(f"Validation metrics: {validation_metrics}")
                    
                    # Early stopping if validation loss increases
                    if self._should_stop_early(validation_metrics):
                        logger.info("Early stopping triggered")
                        break
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
def load_api_key(api_key_file: str) -> str:
    """Load the OpenAI API key from a file."""
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    return api_key

def load_data(data_file: str) -> pd.DataFrame:
    """Load the robotics data from a CSV file."""
    return pd.read_csv(data_file)

def main():
    """Run the fine-tuning process from command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM for robotics maintenance optimization")
    parser.add_argument("--data_file", required=True, help="Path to the robotics data CSV file")
    parser.add_argument("--api_key_file", required=True, help="Path to the file containing the OpenAI API key")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Base model to fine-tune")
    parser.add_argument("--focus_factor", type=float, default=1.5, help="Factor to increase focus on maintenance aspects")
    parser.add_argument("--max_train_examples", type=int, default=500, help="Maximum number of training examples to use")
    parser.add_argument("--max_val_examples", type=int, default=100, help="Maximum number of validation examples to use")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output_dir", default="models", help="Directory to save output files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Load API key and data
        api_key = load_api_key(args.api_key_file)
        data_df = load_data(args.data_file)
        
        # Create fine-tuner
        fine_tuner = RoboticsMaintenanceLLMFineTuner(api_key, args.model_name)
        
        # Run fine-tuning
        model_id = fine_tuner.run_fine_tuning(
            data_df, 
            args.focus_factor, 
            args.output_dir,
            args.max_train_examples,
            args.max_val_examples,
            args.n_epochs
        )
        
        logger.info(f"Fine-tuning completed successfully. The new model ID is: {model_id}")
        print(f"\nSuccess! Your fine-tuned model ID is: {model_id}")
        print(f"You can now use this model for robotics maintenance optimization.")
        
    except Exception as e:
        logger.error(f"Fine-tuning process failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 