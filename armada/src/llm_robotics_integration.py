"""
LLM Robotics Integration Module

This module provides integration between large language models and robotics systems
for enhanced decision making, anomaly explanation, and maintenance optimization.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRoboticsEnhancer:
    """
    Enhances robotics analysis using large language models to solve complex challenges
    that traditional machine learning methods struggle with.
    """
    
    def __init__(self, api_key: Optional[str] = None, knowledge_base=None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM Robotics Enhancer.
        
        Args:
            api_key: OpenAI API key (optional, will look in .env if not provided)
            knowledge_base: KnowledgeBase object for domain knowledge
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self._setup_api_client(api_key)
        self.cache = {}  # Simple cache for LLM responses
        
        logger.info(f"Initialized LLM Robotics Enhancer with model: {model_name}")
    
    def _setup_api_client(self, api_key: Optional[str] = None):
        """Set up the OpenAI API client."""
        try:
            from openai import OpenAI
            
            # Try to get API key from different sources
            if api_key is None:
                # Try from environment variable
                api_key = os.getenv("OPENAI_API_KEY")
                
                # Try from file if still None
                if api_key is None and os.path.exists("openai_api_key.txt"):
                    with open("openai_api_key.txt", "r") as f:
                        api_key = f.read().strip()
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.is_available = True
                logger.info("OpenAI API client initialized successfully.")
            else:
                logger.warning("No OpenAI API key found. LLM functionality will be disabled.")
                self.is_available = False
                
        except ImportError:
            logger.warning("OpenAI package not installed. LLM functionality will be disabled.")
            self.is_available = False
    
    def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: Prompt for the LLM
            temperature: Creativity parameter (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response
        """
        # Return from cache if already queried
        cache_key = hash(prompt + str(temperature) + str(max_tokens))
        if cache_key in self.cache:
            logger.info("Returning cached LLM response")
            return self.cache[cache_key]
        
        if not self.is_available:
            return "LLM functionality is not available. Please ensure the OpenAI package is installed and a valid API key is provided."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in robotics and warehouse automation systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            self.cache[cache_key] = result  # Cache the result
            return result
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error generating analysis: {str(e)}"
    
    def explain_anomaly(self, robot_state: Dict[str, float], thresholds: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Provide an explanation for anomalies in robot state.
        
        Args:
            robot_state: Dictionary of robot sensor readings
            thresholds: Dictionary of normal operating thresholds
            
        Returns:
            Dictionary with anomaly explanation and recommendations
        """
        # Identify which values are outside thresholds
        anomalies = {}
        for key, value in robot_state.items():
            if key in thresholds:
                if value < thresholds[key].get('min', float('-inf')) or value > thresholds[key].get('max', float('inf')):
                    anomalies[key] = {
                        'value': value,
                        'threshold': thresholds[key],
                        'deviation': min(
                            abs(value - thresholds[key].get('min', value)),
                            abs(value - thresholds[key].get('max', value))
                        )
                    }
        
        if not anomalies:
            return {
                "has_anomaly": False,
                "explanation": "No anomalies detected. All values within normal operating thresholds.",
                "recommendations": []
            }
        
        # Prepare prompt for LLM
        anomaly_info = "\n".join([
            f"- {key}: Current value = {data['value']}, Normal range = {data['threshold'].get('min', 'None')} to {data['threshold'].get('max', 'None')}"
            for key, data in anomalies.items()
        ])
        
        sensor_context = "\n".join([
            f"- {key}: {value}" for key, value in robot_state.items() if key not in anomalies
        ])
        
        prompt = f"""
        Analyze the following robotic sensor anomalies:
        {anomaly_info}
        
        Additional sensor context:
        {sensor_context}
        
        Provide:
        1. A detailed explanation of possible causes for these anomalies
        2. How the anomalies might be related to each other
        3. Recommended actions in order of priority
        4. Potential maintenance implications
        """
        
        # Get LLM response
        llm_response = self._call_llm(prompt)
        
        # Parse and structure the response
        sections = llm_response.split("\n\n")
        explanation = sections[0] if len(sections) > 0 else ""
        
        # Extract recommendations (lines starting with numbers or dashes in later sections)
        recommendations = []
        for section in sections[1:]:
            for line in section.split("\n"):
                if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-")):
                    recommendations.append(line.strip())
        
        return {
            "has_anomaly": True,
            "anomalies": anomalies,
            "explanation": explanation,
            "recommendations": recommendations,
            "full_analysis": llm_response,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_incomplete_data(self, robot_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Make intelligent inferences when sensor data is missing or incomplete.
        
        Args:
            robot_state: Dictionary of available robot sensor readings
            
        Returns:
            Dictionary with analysis and inferred values
        """
        # Define expected sensors for a complete state
        expected_sensors = {
                "temperature", "battery_level", "motor_current", "vibration_level",
            "humidity", "ambient_temperature", "throughput_rate", "error_rate"
        }
        
        # Identify missing sensors
        available_sensors = set(robot_state.keys())
        missing_sensors = expected_sensors - available_sensors
        
        if not missing_sensors:
            return {
                "complete_data": True,
                "analysis": "Data is complete. All expected sensors are present.",
                "robot_state": robot_state
            }
        
        # Prepare prompt for LLM
        prompt = f"""
        The following robot sensor data is incomplete:
        {json.dumps(robot_state, indent=2)}
        
        Missing sensors: {', '.join(missing_sensors)}
        
        Based on the available sensor data:
        1. What are the likely values or ranges for the missing sensors?
        2. What can be inferred about the robot's overall state?
        3. Is there enough information to make a reliable assessment of the robot's condition?
        4. What additional data would be most valuable to collect?
        """
        
        # Get LLM response
        llm_response = self._call_llm(prompt)
        
        # Create inferred state (this would be more sophisticated in a real implementation)
        inferred_state = robot_state.copy()
        
        # Extract inferred values from LLM response (simplified implementation)
        # A real implementation would use more sophisticated parsing
        for sensor in missing_sensors:
            # Look for patterns like "temperature: 35-40°C" or "temperature is likely around 37°C"
            for line in llm_response.split("\n"):
                if sensor in line.lower():
                    # Very simple extraction - would be more robust in production
                    numbers = [float(s) for s in line.replace('°C', '').replace('%', '').split() if s.replace('.', '', 1).isdigit()]
                    if numbers:
                        inferred_state[sensor] = {"inferred_value": sum(numbers) / len(numbers), "confidence": "medium"}
                        break
        
        return {
            "complete_data": False,
            "missing_sensors": list(missing_sensors),
            "inferred_state": inferred_state,
            "analysis": llm_response,
            "timestamp": datetime.now().isoformat()
        }
    
    def optimize_maintenance_decisions(self, robot_states: Dict[str, Dict], 
                                   maintenance_history: List[Dict], 
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize maintenance decisions using LLM reasoning.
        
        Args:
            robot_states: Dictionary of robot states by robot_id
            maintenance_history: List of past maintenance events
            constraints: Operational constraints (e.g., budget, max_downtime)
            
        Returns:
            Dictionary with maintenance recommendations
        """
        # Prepare maintenance history summary
        history_summary = self._summarize_maintenance_history(maintenance_history)
        
        # Prepare current state summary
        state_summary = []
        critical_issues = []
        
        for robot_id, state in robot_states.items():
            # Calculate a simple health score (would be more sophisticated in production)
            health_indicators = ['temperature', 'vibration_level', 'error_rate', 'battery_health']
            available_indicators = [h for h in health_indicators if h in state]
            
            health_score = 0
            if available_indicators:
                # Simplified scoring logic
                if 'temperature' in state and state['temperature'] > 80:
                    health_score -= 0.2
                    critical_issues.append(f"Robot {robot_id}: High temperature ({state['temperature']})")
                    
                if 'vibration_level' in state and state['vibration_level'] > 0.7:
                    health_score -= 0.3
                    critical_issues.append(f"Robot {robot_id}: High vibration ({state['vibration_level']})")
                    
                if 'error_rate' in state and state['error_rate'] > 0.05:
                    health_score -= 0.2
                    critical_issues.append(f"Robot {robot_id}: High error rate ({state['error_rate']})")
                    
                if 'battery_health' in state and state['battery_health'] < 0.6:
                    health_score -= 0.2
                    critical_issues.append(f"Robot {robot_id}: Poor battery health ({state['battery_health']})")
            
            days_since_maintenance = 365  # Default to a high number
            for event in maintenance_history:
                if event.get('robot_id') == robot_id:
                    days_ago = (datetime.now() - datetime.fromisoformat(event['timestamp'])).days
                    days_since_maintenance = min(days_since_maintenance, days_ago)
            
            state_summary.append(f"Robot {robot_id}: Health score: {health_score:.2f}, Days since maintenance: {days_since_maintenance}")
        
        # Prepare prompt
        prompt = f"""
        I need to optimize the maintenance schedule for a warehouse robot fleet with the following constraints:
        - Budget: {constraints.get('budget', 'unlimited')}
        - Maximum allowed downtime: {constraints.get('max_downtime', 'N/A')} hours
        - Available maintenance staff: {constraints.get('staff_available', 'N/A')}
        - Maintenance window: {constraints.get('maintenance_window', 'anytime')}
        
        Current robot states:
        {os.linesep.join(state_summary)}
        
        Critical issues detected:
        {os.linesep.join(critical_issues) if critical_issues else "No critical issues detected."}
        
        Maintenance history summary:
        {history_summary}
        
        Please provide:
        1. A prioritized list of robots that need maintenance
        2. Recommended maintenance actions for each robot
        3. A suggested maintenance schedule that minimizes impact on operations
        4. Any additional considerations for long-term maintenance planning
        """
        
        # Get LLM response
        llm_response = self._call_llm(prompt, max_tokens=800)
        
        # Extract prioritized robots (simplified implementation)
        prioritized_robots = []
        for line in llm_response.split("\n"):
            if "robot" in line.lower() and ":" in line and any(d.isdigit() for d in line):
                # Extract robot IDs from lines like "1. Robot 3: Requires immediate maintenance"
                for part in line.split():
                    if part.isdigit() or (len(part) > 0 and part[0] == "#" and part[1:].isdigit()):
                        robot_id = part.replace("#", "")
                        if robot_id in robot_states:
                            prioritized_robots.append(robot_id)
                            break
        
        return {
            "maintenance_recommendations": llm_response,
            "prioritized_robots": prioritized_robots,
            "critical_issues": critical_issues,
            "timestamp": datetime.now().isoformat()
        }
    
    def _summarize_maintenance_history(self, maintenance_history: List[Dict]) -> str:
        """Create a summary of maintenance history."""
        if not maintenance_history:
            return "No maintenance history available."
        
        # Group by robot
        robots = {}
        for event in maintenance_history:
            robot_id = event.get('robot_id', 'unknown')
            if robot_id not in robots:
                robots[robot_id] = []
            robots[robot_id].append(event)
        
        summary = []
        for robot_id, events in robots.items():
            # Sort by timestamp
            events = sorted(events, key=lambda x: x.get('timestamp', ''))
            
            # Get latest event
            latest = events[-1] if events else None
            
            if latest:
                days_ago = (datetime.now() - datetime.fromisoformat(latest['timestamp'])).days
                summary.append(f"Robot {robot_id}: Last maintenance {days_ago} days ago. Type: {latest.get('maintenance_type', 'unknown')}")
        
        return "\n".join(summary)

    def explain_model_decision(self, model_prediction: Dict[str, Any], 
                           input_features: Dict[str, float],
                           model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate human-understandable explanations for model predictions.
        
        Args:
            model_prediction: The model's prediction including confidence
            input_features: Features that went into the model
            model_info: Information about the model (type, important features, etc.)
            
        Returns:
            Dictionary with explanation of the model decision
        """
        # Format the input features and prediction
        feature_str = "\n".join([f"- {k}: {v}" for k, v in input_features.items()])
        
        prediction_str = "No prediction data available"
        if isinstance(model_prediction, dict):
            prediction_str = "\n".join([f"- {k}: {v}" for k, v in model_prediction.items()])
        else:
            prediction_str = f"Prediction: {model_prediction}"
        
        # Prepare most important features if available 
        important_features = model_info.get('important_features', {})
        if important_features:
            important_features_str = "\n".join([f"- {k}: {v}" for k, v in important_features.items()])
        else:
            important_features_str = "No feature importance information available"
        
        # Prepare prompt
        prompt = f"""
        Please explain the following model prediction in simple, human-readable terms:
        
        MODEL INFORMATION:
        Type: {model_info.get('model_type', 'Unknown')}
        Purpose: {model_info.get('purpose', 'Prediction')}
        
        INPUT FEATURES:
        {feature_str}
        
        MODEL PREDICTION:
        {prediction_str}
        
        FEATURE IMPORTANCE:
        {important_features_str}
        
        Explain:
        1. What factors most influenced this prediction?
        2. Are there any unusual or noteworthy values in the input?
        3. How confident can we be in this prediction?
        4. What actions should be considered based on this prediction?
        
        Please avoid technical jargon and explain as if to a warehouse manager without data science background.
        """
        
        # Get LLM response
        explanation = self._call_llm(prompt)
        
        return {
            "model_prediction": model_prediction,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_maintenance_report(self, robot_data: Dict[str, Any], 
                                   maintenance_actions: List[Dict],
                                   audience: str = "technical") -> Dict[str, Any]:
        """
        Generate a maintenance report tailored to different audiences.
        
        Args:
            robot_data: Data about the robot's condition
            maintenance_actions: List of maintenance actions performed
            audience: Intended audience (technical, management, operator)
            
        Returns:
            Dictionary with report sections
        """
        # Prepare data summaries
        state_summary = "\n".join([f"- {k}: {v}" for k, v in robot_data.get('state', {}).items()])
        
        actions_summary = "No maintenance actions recorded"
        if maintenance_actions:
            actions_summary = "\n".join([
                f"- {action.get('timestamp', 'N/A')}: {action.get('action', 'Unknown')} " + 
                f"(Parts: {', '.join(action.get('parts', []))})"
                for action in maintenance_actions
            ])
        
        # Tailor prompt based on audience
        audience_guidance = ""
        if audience == "technical":
            audience_guidance = "The audience is technical maintenance staff. Include detailed technical information."
        elif audience == "management":
            audience_guidance = "The audience is management. Focus on costs, efficiency, and business impact."
        elif audience == "operator":
            audience_guidance = "The audience is robot operators. Focus on operational implications and safety."
        
        prompt = f"""
        Generate a maintenance report based on the following information:
        
        ROBOT DATA:
        Robot ID: {robot_data.get('robot_id', 'Unknown')}
        Type: {robot_data.get('type', 'Unknown')}
        Age: {robot_data.get('age', 'Unknown')} days
        
        CURRENT STATE:
        {state_summary}
        
        MAINTENANCE ACTIONS PERFORMED:
        {actions_summary}
        
        ADDITIONAL CONTEXT:
        {robot_data.get('notes', 'No additional notes')}
        
        {audience_guidance}
        
        The report should include:
        1. An executive summary
        2. Current state assessment
        3. Maintenance actions summary
        4. Recommendations for future maintenance
        5. Estimated impact on operations
        """
        
        # Get LLM response
        report_text = self._call_llm(prompt, max_tokens=1000)
        
        # Simple section extraction (would be more robust in production)
        sections = {}
        current_section = "unsorted"
        sections[current_section] = []
        
        for line in report_text.split("\n"):
            # Check for section headers (uppercase words followed by colon)
            if line.strip().isupper() and ":" in line:
                current_section = line.strip().rstrip(":")
                sections[current_section] = []
            # Or check for numbered section headers
            elif any(line.strip().startswith(prefix) for prefix in ["1.", "2.", "3.", "4.", "5."]):
                # Extract section name after the number
                parts = line.strip().split(".", 1)
                if len(parts) > 1:
                    current_section = parts[1].strip()
                    sections[current_section] = []
                    sections[current_section].append(line.strip())
            else:
                sections[current_section].append(line.strip())
        
        # Convert section lists to strings
        for section, lines in sections.items():
            sections[section] = "\n".join(lines)
        
        # Ensure key sections exist
        required_sections = ["Executive Summary", "Current State", "Maintenance Actions", "Recommendations"]
        for section in required_sections:
            if not any(section.lower() in k.lower() for k in sections):
                sections[section] = "No information available"
        
        return {
            "report": report_text,
            "sections": sections,
            "target_audience": audience,
            "timestamp": datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Initialize the LLM enhancer
    enhancer = LLMRoboticsEnhancer()
    
    # Example robot state with an anomaly
    robot_state = {
        "temperature": 85.2,
        "vibration_level": 0.62,
        "battery_level": 0.45,
        "motor_current": 15.3,
        "throughput_rate": 120,
        "error_rate": 0.02
    }
    
    # Example thresholds
    thresholds = {
        "temperature": {"min": 20, "max": 75},
        "vibration_level": {"min": 0, "max": 0.5},
        "battery_level": {"min": 0.2, "max": 1.0},
        "motor_current": {"min": 5, "max": 20},
        "throughput_rate": {"min": 80, "max": 150},
        "error_rate": {"min": 0, "max": 0.05}
    }
    
    # Get anomaly explanation
    if enhancer.is_available:
        explanation = enhancer.explain_anomaly(robot_state, thresholds)
        print("Anomaly Analysis:")
        print(explanation["explanation"])
        print("\nRecommendations:")
        for rec in explanation["recommendations"]:
            print(f"- {rec}")
    else:
        print("LLM functionality is not available. Please ensure you have an OpenAI API key.") 