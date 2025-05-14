#!/usr/bin/env python3
"""
LLM Robotics Enhancer module for using fine-tuned models on robotics maintenance.

This module provides functionalities to utilize fine-tuned LLMs for robotics maintenance
optimization and predictive recommendations.
"""

import os
import sys
import json
import re
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
from openai import OpenAI
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/llm_enhancer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("llm_enhancer")

# Import utilities
from utils.llm_utils import call_llm, format_sensor_data, with_retry
from utils.config import get_system_prompt, is_sensor_anomalous, is_sensor_critical
from utils.data_processing import detect_sensor_anomalies, extract_sensor_stats

class RoboticsMaintenanceLLMEnhancer:
    """Enhance robotics maintenance with fine-tuned LLMs."""
    
    def __init__(self, api_key: str = None, model_id: str = None):
        """
        Initialize the LLM enhancer.
        
        Args:
            api_key: OpenAI API key
            model_id: ID of the fine-tuned model (if None, uses gpt-3.5-turbo)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model_id = model_id or os.environ.get("OPENAI_MODEL_ID") or "ft:gpt-3.5-turbo-0125:personal:robotics-maintenance"
        self.cache = {}  # Simple cache to avoid redundant API calls
        self.is_available = True
        logger.info(f"Initialized with model: {self.model_id}")
        
        # Default confidence thresholds
        self.min_confidence_threshold = 0.7
        self.high_confidence_threshold = 0.85
        
        # Tracking for analysis
        self.last_query = None
        self.last_response = None
        self.performance_metrics = []
        
    def analyze_robot_data(self, robot_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze robot sensor data and provide maintenance recommendations.
        
        Args:
            robot_data: Pandas Series with robot sensor data
            
        Returns:
            Dictionary with analysis results
        """
        # Generate prompt from robot data
        prompt = self.generate_prompt(robot_data)
        
        # Store the query for later analysis
        self.last_query = prompt
        
        # Call the LLM with the prompt
        system_message = get_system_prompt('maintenance')
        response = call_llm(
            client=self.client,
            prompt=prompt,
            model_id=self.model_id,
            system_message=system_message,
            temperature=0.2,
            max_tokens=800
        )
        
        # Store the response for later analysis
        self.last_response = response
        
        # Parse the response
        result = self._parse_llm_response(response)
        
        # Apply confidence and safety checks
        result = self._apply_confidence_checks(result, robot_data)
        
        # Log the result for debugging
        logger.debug(f"Result after confidence fix: {result}")
        
        # Update performance metrics
        self._update_performance_metrics(result)
        
        return result
    
    def generate_prompt(self, robot_data: pd.Series) -> str:
        """
        Generate a prompt for the LLM based on robot data.
        
        Args:
            robot_data: Pandas Series with robot sensor data
            
        Returns:
            Formatted prompt string
        """
        prompt = "Analyze this robot's sensor data for maintenance issues:\n\n"
        
        # Convert Series to dictionary for formatting
        data_dict = robot_data.to_dict()
        prompt += format_sensor_data(data_dict, include_units=True)
            
        # Add context about normal/abnormal readings
        anomalies = []
        for col, value in data_dict.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                if is_sensor_anomalous(value, col):
                    anomalies.append(col)
                    if is_sensor_critical(value, col):
                        prompt += f"\nNOTE: The {col} reading of {value} is at a CRITICAL level.\n"
        
        if anomalies:
            prompt += f"\nThe following sensors show anomalous readings: {', '.join(anomalies)}\n"
            
        prompt += "\nBased on this data, determine:\n"
        prompt += "1. Is maintenance required? (True/False)\n"
        prompt += "2. What is the maintenance urgency? (High/Medium/Low)\n"
        prompt += "3. What is the root cause of any issues?\n"
        prompt += "4. What specific maintenance actions do you recommend?\n"
        prompt += "5. How confident are you in this assessment? (0-100%)\n"
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary with parsed results
        """
        result = {
            "maintenance_required": None,
            "urgency": None,
            "root_cause": None, 
            "recommendations": None,
            "confidence": None,
            "raw_response": response
        }
        
        # Extract maintenance required (True/False)
        maintenance_match = re.search(r"maintenance required\?.*?(true|false)", response, re.IGNORECASE)
        if maintenance_match:
            result["maintenance_required"] = maintenance_match.group(1).lower() == "true"
        
        # Extract urgency (High/Medium/Low)
        urgency_match = re.search(r"urgency\?.*?(high|medium|low)", response, re.IGNORECASE)
        if urgency_match:
            result["urgency"] = urgency_match.group(1).title()
        
        # Extract root cause
        root_cause_match = re.search(r"root cause.*?:(.+?)(?=\n\d|\n\nWhat|$)", response, re.IGNORECASE | re.DOTALL)
        if root_cause_match:
            result["root_cause"] = root_cause_match.group(1).strip()
        
        # Extract recommendations
        recommendations_match = re.search(r"recommend.*?:(.+?)(?=\n\d|\n\nHow|$)", response, re.IGNORECASE | re.DOTALL)
        if recommendations_match:
            # Split into list if separated by numbers, bullet points or newlines
            recommendations_text = recommendations_match.group(1).strip()
            recommendations = []
            
            # Check if there are numbered points
            numbered_items = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|$)', recommendations_text, re.DOTALL)
            if numbered_items:
                recommendations = [item.strip() for item in numbered_items]
            else:
                # Check for bullet points
                bullet_items = re.findall(r'[-•*]\s+(.+?)(?=\n[-•*]|\n\n|$)', recommendations_text, re.DOTALL)
                if bullet_items:
                    recommendations = [item.strip() for item in bullet_items]
                else:
                    # Just split by newlines
                    recommendations = [item.strip() for item in recommendations_text.split('\n') if item.strip()]
            
            result["recommendations"] = recommendations
        
        # Extract confidence
        confidence_match = re.search(r"confidence.*?(\d+)%", response, re.IGNORECASE)
        if confidence_match:
            try:
                result["confidence"] = int(confidence_match.group(1)) / 100.0
            except ValueError:
                result["confidence"] = None
        
        return result
    
    def _apply_confidence_checks(self, result: Dict[str, Any], robot_data: pd.Series) -> Dict[str, Any]:
        """
        Apply confidence and safety checks to the LLM result.
        
        Args:
            result: Initial parsed result from LLM
            robot_data: Original robot sensor data
            
        Returns:
            Updated result with confidence adjustments
        """
        # Make a copy to avoid modifying the original
        updated_result = result.copy()
        
        # Check for missing fields
        missing_fields = []
        for field in ["maintenance_required", "urgency", "root_cause", "recommendations", "confidence"]:
            if result[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"Missing fields in LLM response: {missing_fields}")
            # Reduce confidence if fields are missing
            if result["confidence"] is not None:
                updated_result["confidence"] = max(0.1, result["confidence"] - 0.1 * len(missing_fields))
        
        # Apply statistical override for critical sensors
        critical_sensors = []
        for col, value in robot_data.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                if is_sensor_critical(value, col):
                    critical_sensors.append(col)
        
        # If critical sensors are detected, but LLM says no maintenance required
        if critical_sensors and result["maintenance_required"] is False:
            logger.warning(f"Critical sensors detected but LLM says no maintenance required: {critical_sensors}")
            updated_result["maintenance_required"] = True
            updated_result["urgency"] = "High"
            
            # Update root cause if empty
            if not updated_result["root_cause"]:
                updated_result["root_cause"] = f"Critical sensor readings: {', '.join(critical_sensors)}"
                
            # Update confidence
            updated_result["confidence"] = min(0.7, updated_result.get("confidence", 0.5))
        
        # Check confidence threshold
        if updated_result["confidence"] is not None and updated_result["confidence"] < self.min_confidence_threshold:
            logger.warning(f"Low confidence result: {updated_result['confidence']}")
            
            # For low confidence cases, fill in sensible defaults
            if updated_result["maintenance_required"] is None:
                # Default to True if we have any anomalous readings
                anomalies = [col for col, value in robot_data.items() 
                           if isinstance(value, (int, float)) and not pd.isna(value) and is_sensor_anomalous(value, col)]
                updated_result["maintenance_required"] = len(anomalies) > 0
            
            if updated_result["urgency"] is None:
                # Default to Medium unless we have critical sensors
                updated_result["urgency"] = "High" if critical_sensors else "Medium"
                
            if not updated_result["recommendations"]:
                updated_result["recommendations"] = ["Investigate anomalous sensor readings."]
        
        return updated_result
    
    def _update_performance_metrics(self, result: Dict[str, Any]) -> None:
        """
        Track performance metrics for model evaluation.
        
        Args:
            result: Analysis result to track
        """
        # Only track complete results
        if None in [result.get("maintenance_required"), result.get("urgency"), result.get("confidence")]:
            return
            
        # Add to metrics history
        self.performance_metrics.append({
            "timestamp": time.time(),
            "maintenance_required": result["maintenance_required"],
            "urgency": result["urgency"],
            "confidence": result["confidence"],
            "completeness": sum(1 for v in result.values() if v is not None) / len(result)
        })
        
        # Trim history if needed
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of model performance.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_metrics:
            return {"error": "No performance data available"}
            
        # Extract metrics as lists
        confidences = [m["confidence"] for m in self.performance_metrics if m["confidence"] is not None]
        completeness = [m["completeness"] for m in self.performance_metrics]
        maintenance_required = [m["maintenance_required"] for m in self.performance_metrics]
        
        # Calculate statistics
        return {
            "total_analyses": len(self.performance_metrics),
            "avg_confidence": np.mean(confidences) if confidences else None,
            "median_confidence": np.median(confidences) if confidences else None,
            "avg_completeness": np.mean(completeness) if completeness else None,
            "maintenance_required_pct": sum(maintenance_required) / len(maintenance_required) * 100 if maintenance_required else None,
            "high_urgency_pct": sum(1 for m in self.performance_metrics if m["urgency"] == "High") / len(self.performance_metrics) * 100,
            "medium_urgency_pct": sum(1 for m in self.performance_metrics if m["urgency"] == "Medium") / len(self.performance_metrics) * 100,
            "low_urgency_pct": sum(1 for m in self.performance_metrics if m["urgency"] == "Low") / len(self.performance_metrics) * 100
        }

def load_api_key(api_key_file: str) -> str:
    """Load the OpenAI API key from a file."""
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    return api_key

def main():
    """Run the LLM enhancer from command line arguments."""
    parser = argparse.ArgumentParser(description="Use fine-tuned LLM for robotics maintenance optimization")
    parser.add_argument("--data_file", required=True, help="Path to the robotics data CSV file")
    parser.add_argument("--api_key_file", required=True, help="Path to the file containing the OpenAI API key")
    parser.add_argument("--model_id", help="Fine-tuned model ID (if not provided, uses gpt-3.5-turbo)")
    parser.add_argument("--output_file", help="Path to save the analysis results")
    parser.add_argument("--max_rows", type=int, default=10, help="Maximum number of rows to analyze")
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Load API key
        api_key = load_api_key(args.api_key_file)
        
        # Create enhancer
        enhancer = RoboticsMaintenanceLLMEnhancer(api_key, args.model_id)
        
        # Read CSV data
        df = pd.read_csv(args.data_file)
        if len(df) > args.max_rows:
            logger.info(f"Limiting analysis to {args.max_rows} rows")
            df = df.sample(args.max_rows, random_state=42).reset_index(drop=True)
        
        # Analyze the data
        data_list = df.to_dict(orient='records')
        results = enhancer.batch_analyze(data_list)
        
        # Save the results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved analysis results to {args.output_file}")
            
        # Print a summary
        print(f"\nAnalyzed {len(results)} robot records:")
        for i, result in enumerate(results):
            print(f"\nRobot {result['robot_id']}:")
            maintenance_pred = result.get('maintenance_prediction', '')
            if maintenance_pred:
                print(f"  Maintenance prediction: {maintenance_pred[:100]}...")
            risk = result.get('risk_assessment', '')
            if risk:
                print(f"  Risk assessment: {risk[:100]}...")
        
    except Exception as e:
        logger.error(f"LLM enhancer process failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 