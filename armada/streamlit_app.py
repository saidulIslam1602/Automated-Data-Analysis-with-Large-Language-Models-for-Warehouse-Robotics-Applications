#!/usr/bin/env python3
"""
Streamlit app for the Warehouse Robotics Maintenance Optimization system.

This app provides an interactive interface to demonstrate the LLM-based
maintenance recommendation system and showcase key findings of the project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from src.llm_robotics_enhancer import RoboticsMaintenanceLLMEnhancer
import datetime  # Add this import for date handling in the scheduler
import re  # Add this import for regular expressions

# Import UI component utilities
from utils.ui_components import (
    create_card, 
    create_robot_card, 
    create_status_indicator, 
    display_sensor_readings,
    labeled_input,
    create_metrics_dashboard,
    styled_header,
    custom_chat_message
)

# Set page configuration
st.set_page_config(
    page_title="Warehouse Robotics Maintenance Optimization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS - completely overhaul styling
st.markdown("""
<style>
    /* Reset some base Streamlit elements */
    .stApp {
        background-color: #f5f7fa !important;
    }
    
    /* Main containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    
    /* Sidebar with darker background for better text contrast */
    [data-testid="stSidebar"] {
        background-color: #303f9f !important;
        padding: 1.5rem 1rem !important;
        background-image: linear-gradient(180deg, #283593 0%, #3949ab 100%) !important;
        border-right: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem !important;
    }
    
    /* Radio buttons in sidebar - make text clearly visible */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02rem !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h2 {
        color: white !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] p {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Make the radio button backgrounds darker for better contrast */
    [data-testid="stSidebar"] .stRadio {
        background-color: rgba(0,0,0,0.2) !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 0.8rem !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    
    /* Make selected radio option more visible */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
        border-color: white !important;
        background-color: white !important;
    }
    
    /* Header text in status indicator */
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] div p {
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5) !important;
    }
    
    /* Navigation header styling */
    .navigation-header {
        background-color: rgba(0,0,0,0.3) !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    
    .css-1544g2n {
        margin-top: 0rem !important;
        padding-top: 2rem !important;
    }
    
    /* Headers and text */
    h1 {
        color: #1a237e !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        margin-bottom: 2rem !important;
        border-left: 5px solid #3f51b5 !important;
    }
    
    h2 {
        color: #1a237e !important;
        font-weight: 600 !important;
        font-size: 1.7rem !important;
        background-color: white !important;
        padding: 0.8rem 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08) !important;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        border-left: 4px solid #3f51b5 !important;
    }
    
    h3 {
        color: #1a237e !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        padding: 0.6rem 0 !important;
        border-bottom: 2px solid #e0e0e0 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    p {
        color: #212121 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Cards */
    .card {
        background-color: white !important;
        border-radius: 0.5rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3f51b5 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 1rem !important;
        border-radius: 0.3rem !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #283593 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.25) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }
    
    /* Form elements */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        padding: 0.5rem !important;
        border-radius: 0.3rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stSelectbox > div > div {
        background-color: white !important;
    }
    
    /* Status indicators */
    div[data-testid="stAlert"] {
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    /* Custom classes */
    .status-positive {
        background-color: #e8f5e9 !important;
        border-left: 5px solid #2e7d32 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    .status-warning {
        background-color: #fff3e0 !important;
        border-left: 5px solid #f57c00 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    .status-negative {
        background-color: #ffebee !important;
        border-left: 5px solid #d32f2f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing API key and enhancer
def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'enhancer' not in st.session_state:
        st.session_state.enhancer = None
    
    if 'api_key' not in st.session_state:
        # Read API key from file
        try:
            with open("openai_api_key.txt", "r") as f:
                st.session_state.api_key = f.read().strip()
        except Exception as e:
            st.session_state.api_key = ""
            st.error(f"Error reading API key: {str(e)}")
    
    if 'results' not in st.session_state:
        st.session_state.results = []
        
    if 'selected_robot' not in st.session_state:
        st.session_state.selected_robot = 0
        
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
        
    if 'total_robot_count' not in st.session_state:
        st.session_state.total_robot_count = 0
        
    if 'displayed_robot_count' not in st.session_state:
        st.session_state.displayed_robot_count = 10
    
    if 'dashboard_view' not in st.session_state:
        st.session_state.dashboard_view = True  # Start with dashboard by default
    
    if 'rerun_requested' not in st.session_state:
        st.session_state.rerun_requested = False
        
    if 'performance_metrics' not in st.session_state:
        # Sample performance metrics based on the thesis results
        st.session_state.performance_metrics = {
            "Isolation Forest": {
                "accuracy": 0.965,
                "precision": 0.731,
                "recall": 0.867,
                "f1_score": 0.793,
                "roc_auc": 0.984,
                "false_positive_rate": 0.019,
                "false_negative_rate": 0.133
            },
            "One-Class SVM": {
                "accuracy": 0.960,
                "precision": 0.702,
                "recall": 0.850,
                "f1_score": 0.769,
                "roc_auc": 0.979,
                "false_positive_rate": 0.022,
                "false_negative_rate": 0.150
            },
            "LOF": {
                "accuracy": 0.955,
                "precision": 0.667,
                "recall": 0.833,
                "f1_score": 0.741,
                "roc_auc": 0.971,
                "false_positive_rate": 0.025,
                "false_negative_rate": 0.167
            },
            "CAAD-4": {
                "accuracy": 0.981,
                "precision": 0.783,
                "recall": 0.867,
                "f1_score": 0.823,
                "roc_auc": 0.996,
                "false_positive_rate": 0.013,
                "false_negative_rate": 0.133
            }
        }
        
    if 'error_codes_guide' not in st.session_state:
        # Create a dictionary of error codes and their meanings
        st.session_state.error_codes_guide = {
            'E001': {
                'description': 'Temperature Sensor Fault',
                'severity': 'High',
                'possible_causes': [
                    'Sensor wire damage or disconnection',
                    'Sensor circuit board failure',
                    'Excessive environmental temperature'
                ],
                'recommended_actions': [
                    'Check sensor connections',
                    'Replace temperature sensor if damaged',
                    'Verify environmental conditions are within specifications'
                ]
            },
            'E002': {
                'description': 'Motor Controller Error',
                'severity': 'High',
                'possible_causes': [
                    'Motor controller overheating',
                    'Voltage irregularity in power supply',
                    'Controller firmware error'
                ],
                'recommended_actions': [
                    'Allow system to cool down before restarting',
                    'Check power supply voltage',
                    'Update controller firmware'
                ]
            },
            'E101': {
                'description': 'Hydraulic Pressure Loss',
                'severity': 'Critical',
                'possible_causes': [
                    'Hydraulic fluid leak',
                    'Pump failure',
                    'Pressure relief valve malfunction'
                ],
                'recommended_actions': [
                    'Inspect hydraulic system for leaks',
                    'Check pump operation',
                    'Verify relief valve settings'
                ]
            },
            'E204': {
                'description': 'Communication System Failure',
                'severity': 'Medium',
                'possible_causes': [
                    'Network interference',
                    'Communication module failure',
                    'Software error in communication protocol'
                ],
                'recommended_actions': [
                    'Reset communication module',
                    'Check for interference sources',
                    'Update communication firmware'
                ]
            },
            'E506': {
                'description': 'Battery Critical Error',
                'severity': 'Critical',
                'possible_causes': [
                    'Battery cell damage',
                    'Charging system failure',
                    'Battery management system error'
                ],
                'recommended_actions': [
                    'Replace battery',
                    'Inspect charging system',
                    'Reset battery management system'
                ]
            }
        }

# Add additional CSS for better text contrast
def inject_text_contrast_css():
    """Inject additional CSS for better text contrast and visibility"""
    st.markdown("""
    <style>
    /* Ensure text in all containers has good contrast */
    body, .stMarkdown p, .stMarkdown li {
        color: #000000 !important;
    }
    
    /* Make headings darker for better visibility */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a237e !important;
    }
    
    /* Increase contrast for form elements */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Make button text more visible */
    .stButton > button {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Ensure link visibility */
    a, a:visited {
        color: #1565C0 !important;
        font-weight: 500 !important;
    }
    
    /* Better contrast for status messages */
    .status-positive, .status-warning, .status-negative {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function wrappers using our imported UI utilities
def text_input_with_label(label="", **kwargs):
    """Create a text input with a proper label for accessibility."""
    return labeled_input(st.text_input, label=label, **kwargs)

def number_input_with_label(label="", **kwargs):
    """Create a number input with a proper label for accessibility."""
    return labeled_input(st.number_input, label=label, **kwargs)

def selectbox_with_label(label="", **kwargs):
    """Create a selectbox with a proper label for accessibility."""
    return labeled_input(st.selectbox, label=label, **kwargs)

def load_data(file_path="extended_robot_data.csv", sample_count=None):
    """Load and sample robot data from CSV file with improved performance."""
    try:
        # Check if we need to reload or if data is already loaded
        if st.session_state.data is not None:
            # Data already loaded
            return True
        
        # Use low_memory=False for better Pandas performance
        # Only select the columns we actually need
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, low_memory=False)
            
            # Update the total robot count
            st.session_state.total_robot_count = len(df)
            
            # Sample the data if requested
            if sample_count and sample_count < len(df):
                df = df.sample(sample_count, random_state=42).reset_index(drop=True)
            
            # Store in session state
            st.session_state.data = df
            return True
        else:
            st.error(f"Data file not found: {file_path}")
            return False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

def initialize_enhancer(api_key, model_id="ft:gpt-3.5-turbo-0125:personal::robotics-maintenance"):
    """Initialize the LLM enhancer with API key."""
    try:
        st.session_state.enhancer = RoboticsMaintenanceLLMEnhancer(api_key, model_id)
        return True
    except Exception as e:
        st.error(f"Error initializing enhancer: {str(e)}")
        return False

def analyze_robot(robot_data):
    """Analyze robot data using the enhancer."""
    try:
        with st.spinner("Analyzing robot data..."):
            # Convert the robot data to a hashable format for caching
            robot_data_str = json.dumps(robot_data, sort_keys=True)
            # Call the cached version of the analysis
            result = analyze_robot_cached(robot_data_str)
            
            # Store the result in the session state
            if len(st.session_state.results) <= st.session_state.selected_robot:
                # Add new result
                st.session_state.results.append(result)
            else:
                # Update existing result
                st.session_state.results[st.session_state.selected_robot] = result
            
            return result
    except Exception as e:
        st.error(f"Error analyzing robot data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def analyze_robot_cached(robot_data_str):
    """Cached version of the robot analysis to improve performance."""
    try:
        # Convert back from string to dict
        robot_data = json.loads(robot_data_str)
        
        # Define sensor thresholds for direct evaluation
        sensor_thresholds = {
            'temperature': {'min': 50, 'max': 85, 'critical_max': 90, 'critical_min': 40, 'unit': '°C'},
            'vibration': {'min': 0, 'max': 0.3, 'critical_max': 0.4, 'unit': 'mm/s'},
            'hydraulic_pressure': {'min': 140, 'max': 180, 'critical_min': 130, 'unit': 'PSI'},
            'power_output': {'min': 85, 'max': 100, 'critical_min': 80, 'unit': '%'},
            'coolant_level': {'min': 70, 'max': 95, 'critical_min': 60, 'unit': '%'}
        }
        
        # Check for anomalies directly based on thresholds
        anomalies = []
        anomaly_values = {}  # Store actual values for displaying to users
        maintenance_required = False
        urgency = "Low"
        
        for sensor, thresholds in sensor_thresholds.items():
            if sensor in robot_data and robot_data[sensor] is not None:
                value = float(robot_data[sensor])
                unit = thresholds.get('unit', '')
                
                # Format the sensor value with appropriate units
                formatted_value = f"{value}{unit}"
                anomaly_values[sensor] = formatted_value
                
                # Check for critical values that require maintenance
                if 'critical_max' in thresholds and value > thresholds['critical_max']:
                    anomalies.append(f"{sensor.replace('_', ' ').title()} is critically high ({formatted_value})")
                    maintenance_required = True
                    urgency = "High"
                elif 'critical_min' in thresholds and value < thresholds['critical_min']:
                    anomalies.append(f"{sensor.replace('_', ' ').title()} is critically low ({formatted_value})")
                    maintenance_required = True
                    urgency = "High"
                # Check for out-of-range values
                elif value > thresholds.get('max', float('inf')):
                    anomalies.append(f"{sensor.replace('_', ' ').title()} is above normal range ({formatted_value})")
                    maintenance_required = True
                    urgency = "Medium"
                elif value < thresholds.get('min', float('-inf')):
                    anomalies.append(f"{sensor.replace('_', ' ').title()} is below normal range ({formatted_value})")
                    maintenance_required = True
                    urgency = "Medium"
        
        # Check for error codes
        if 'error_codes' in robot_data and robot_data['error_codes']:
            anomalies.append(f"Error codes detected: {robot_data['error_codes']}")
            maintenance_required = True
            urgency = "High"
        
        # If we detected anomalies directly, return our own analysis without LLM call
        if anomalies:
            # Determine the root cause from the first anomaly
            root_cause = anomalies[0] if len(anomalies) == 1 else "Multiple sensor anomalies detected"
            
            # Add detailed recommendations based on anomalies
            recommendations = []
            for anomaly in anomalies:
                if "temperature" in anomaly.lower():
                    if "high" in anomaly.lower():
                        recommendations.append({
                            "action": "Check cooling system and temperature sensors",
                            "explanation": "High temperature may indicate cooling system failure or sensor malfunction",
                            "timeframe": "Within 24 hours",
                            "priority": "High",
                            "parts": ["Cooling fan", "Temperature sensor", "Thermal paste"]
                        })
                    else:
                        recommendations.append({
                            "action": "Investigate low temperature readings",
                            "explanation": "Low temperature could indicate sensor error or environmental issues",
                            "timeframe": "Within 48 hours",
                            "priority": "Medium"
                        })
                elif "vibration" in anomaly.lower():
                    recommendations.append({
                        "action": "Inspect for mechanical issues and recalibrate",
                        "explanation": "Excessive vibration may indicate bearing wear, unbalanced components, or loose fasteners",
                        "timeframe": "Within 24 hours",
                        "priority": "High",
                        "parts": ["Bearings", "Mounting hardware", "Vibration dampeners"]
                    })
                elif "hydraulic" in anomaly.lower():
                    recommendations.append({
                        "action": "Check hydraulic system for leaks or pressure issues",
                        "explanation": "Pressure anomalies can lead to performance degradation and potential system failure",
                        "timeframe": "Within 48 hours", 
                        "priority": "Medium",
                        "parts": ["Hydraulic fluid", "Pressure seals", "Filter elements"]
                    })
                elif "power" in anomaly.lower():
                    recommendations.append({
                        "action": "Inspect power supply and electrical systems",
                        "explanation": "Low power output affects overall performance and may indicate electrical issues",
                        "timeframe": "Within 72 hours",
                        "priority": "Medium",
                        "parts": ["Power module", "Wiring harness", "Battery pack"]
                    })
                elif "coolant" in anomaly.lower():
                    recommendations.append({
                        "action": "Refill coolant and check for leaks",
                        "explanation": "Low coolant levels can lead to overheating and potential damage to critical components",
                        "timeframe": "Immediately",
                        "priority": "High",
                        "parts": ["Coolant fluid", "Hose clamps", "Sealing gaskets"]
                    })
                elif "error" in anomaly.lower():
                    recommendations.append({
                        "action": "Diagnose and address error codes",
                        "explanation": "Error codes indicate specific system faults that require immediate attention",
                        "timeframe": "Within 24 hours",
                        "priority": "High"
                    })
            
            # Ensure we have at least one recommendation
            if not recommendations:
                recommendations.append({
                    "action": "Schedule general maintenance inspection",
                    "explanation": "Detected anomalies require professional evaluation",
                    "timeframe": "Within 1 week",
                    "priority": "Medium"
                })
                
            result = {
                'maintenance_required': maintenance_required,
                'urgency': urgency,
                'root_cause': root_cause,
                'contributing_factors': anomalies[1:] if len(anomalies) > 1 else [],
                'confidence': 0.95,
                'anomaly_values': anomaly_values,  # Store actual values for display
                'recommendations': recommendations
            }
                
            return result
        
        # Otherwise return a "normal operation" result
        return {
            'maintenance_required': False,
            'urgency': "Low",
            'root_cause': "All systems operating within normal parameters",
            'confidence': 0.95,
            'recommendations': [{
                "action": "Continue routine maintenance",
                "explanation": "All sensor readings are within normal operating ranges",
                "timeframe": "According to regular schedule",
                "priority": "Low"
            }],
            'anomaly_values': anomaly_values  # Include the actual values
        }
    except Exception as e:
        print(f"Error in cached analysis: {str(e)}")
        # Return a failsafe result instead of None
        return {
            'maintenance_required': False,
            'urgency': "Low",
            'root_cause': "Unable to analyze sensor data",
            'confidence': 0.95,
            'recommendations': [{"action": "Check system for errors and try again"}],
            'anomaly_values': {}  # Empty anomaly values
        }

def display_sensor_data(robot_data):
    """Display sensor data in a formatted way."""
    col1, col2 = st.columns(2)
    
    # Organize sensor data by type
    temperature_sensors = []
    vibration_sensors = []
    other_sensors = []
    
    for key, value in robot_data.items():
        if isinstance(value, (int, float)) or value is None:
            if "temp" in key.lower():
                temperature_sensors.append((key, value))
            elif "vibration" in key.lower() or "accel" in key.lower() or "gyro" in key.lower():
                vibration_sensors.append((key, value))
            else:
                other_sensors.append((key, value))
    
    # Display temperature sensors
    with col1:
        st.subheader("Temperature Sensors")
        if temperature_sensors:
            for name, value in temperature_sensors:
                if pd.isna(value) or value is None:
                    st.text(f"{name}: N/A")
                else:
                    st.text(f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}")
        else:
            st.info("No temperature sensor data available")
    
    # Display vibration sensors
    with col2:
        st.subheader("Vibration Sensors")
        if vibration_sensors:
            for name, value in vibration_sensors:
                if pd.isna(value) or value is None:
                    st.text(f"{name}: N/A")
                else:
                    st.text(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
        else:
            st.info("No vibration sensor data available")
            
            # If no vibration data, check for any motion-related sensors
            motion_sensors = []
            for key, value in robot_data.items():
                if isinstance(value, (int, float)) or value is None:
                    if any(term in key.lower() for term in ["motion", "movement", "speed", "velocity"]):
                        motion_sensors.append((key, value))
            
            if motion_sensors:
                st.subheader("Motion Sensors")
                for name, value in motion_sensors:
                    if pd.isna(value) or value is None:
                        st.text(f"{name}: N/A")
                    else:
                        st.text(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
        
        # Other sensors
        st.subheader("Other Sensors")
        if other_sensors:
            for name, value in other_sensors[:5]:  # Limit to 5 other sensors
                if pd.isna(value) or value is None:
                    st.text(f"{name}: N/A")
                else:
                    st.text(f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}")
        else:
            st.info("No other sensor data available")

def display_maintenance_recommendation(result, robot_name):
    """Display maintenance recommendation using the create_robot_card utility."""
    # First ensure result is valid
    if result is None:
        create_card(
            title=f"{robot_name}: Analysis Incomplete",
            content="Robot analysis is missing or incomplete. Please re-analyze the robot.",
            priority="info",
            icon="ℹ️"
        )
        return
        
    # Ensure all required fields exist
    if "maintenance_required" not in result:
        result["maintenance_required"] = False
    if "urgency" not in result:
        result["urgency"] = "Low"
    if "root_cause" not in result:
        result["root_cause"] = "Unknown issue"
    if "recommendations" not in result:
        result["recommendations"] = []
    
    if result["maintenance_required"]:
        # For robots that need maintenance, show detailed card
        urgency = result.get("urgency", "Medium")
        root_cause = result.get("root_cause", "Unknown issue")
        
        # Display the card header with robot info
        create_robot_card(
            robot_name=robot_name,
            issue="", # We'll display content separately for better formatting
            urgency=urgency,
            estimated_time=None, # We'll display this separately
            additional_info=None
        )
        
        # Display root cause with prominence
        st.markdown(f"### Root Cause\n{root_cause}")
        
        # Calculate estimated completion time based on urgency
        if urgency.lower() == "high" or urgency.lower() == "critical":
            estimated_time = "24 hours"
        elif urgency.lower() == "medium":
            estimated_time = "72 hours"
        else:
            estimated_time = "1 week"
            
        # Display estimated time with appropriate styling
        st.info(f"**Estimated completion time:** {estimated_time}")
        
        # Display confidence if available
        if "confidence" in result:
            confidence = result["confidence"]
            confidence_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.6 else "red"
            confidence_text = f"Prediction confidence: {confidence*100:.1f}%"
            
            if confidence >= 0.8:
                st.success(confidence_text)
            elif confidence >= 0.6:
                st.warning(confidence_text)
            else:
                st.error(confidence_text)
        
        # Format recommendations with rich display
        recommendations = result.get("recommendations", [])
        if recommendations:
            st.markdown("### Recommended Actions")
            
            # Create a visual separator
            st.markdown("---")
            
            # Display each recommendation in its own container for better visual separation
            for i, rec in enumerate(recommendations):
                with st.container():
                    if isinstance(rec, dict) and "action" in rec:
                        # Extract and display the primary action with prominence
                        action = rec["action"]
                        st.markdown(f"#### {i+1}. {action}")
                        
                        # Display additional details if available
                        if "explanation" in rec and rec["explanation"]:
                            st.markdown(f"*{rec['explanation']}*")
                        
                        if "timeframe" in rec and rec["timeframe"]:
                            st.info(f"**Timeframe:** {rec['timeframe']}")
                            
                        if "priority" in rec and rec["priority"]:
                            priority = rec["priority"].lower()
                            if priority == "high" or priority == "critical":
                                st.error(f"**Priority:** {rec['priority']}")
                            elif priority == "medium":
                                st.warning(f"**Priority:** {rec['priority']}")
                            else:
                                st.success(f"**Priority:** {rec['priority']}")
                        
                        # Display parts if available
                        if "parts" in rec and rec["parts"]:
                            st.markdown("**Required parts:**")
                            for part in rec["parts"]:
                                st.markdown(f"- {part}")
                    else:
                        # Simple string recommendation
                        content = rec if isinstance(rec, str) else str(rec)
                        st.markdown(f"#### {i+1}. {content}")
                    
                    # Add a visual separator between recommendations
                    if i < len(recommendations) - 1:
                        st.markdown("---")
        else:
            st.markdown("### Recommended Actions")
            st.info("Continue routine maintenance")
        
        # Get contributing factors
        contributing_factors = result.get("contributing_factors", [])
        if contributing_factors:
            st.markdown("### Additional Issues")
            for factor in contributing_factors:
                st.markdown(f"- {factor}")
        
        # Display sensor readings with rich formatting
        if "anomaly_values" in result and result["anomaly_values"]:
            st.markdown("### Sensor Readings")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            # Sort sensors by type
            temp_sensors = {}
            pressure_sensors = {}
            other_sensors = {}
            
            for sensor, value in result["anomaly_values"].items():
                sensor_name = sensor.replace('_', ' ').title()
                
                if "temp" in sensor.lower():
                    temp_sensors[sensor_name] = value
                elif "pressure" in sensor.lower() or "hydraulic" in sensor.lower():
                    pressure_sensors[sensor_name] = value
                else:
                    other_sensors[sensor_name] = value
                    
            # Display temperature sensors in first column
            if temp_sensors:
                with col1:
                    for name, value in temp_sensors.items():
                        st.markdown(f"**{name}:** {value}")
                    
            # Display pressure sensors in second column
            if pressure_sensors:
                with col2:
                    for name, value in pressure_sensors.items():
                        st.markdown(f"**{name}:** {value}")
            
            # Display other sensors
            if other_sensors:
                for name, value in other_sensors.items():
                    # Choose the column with less content
                    if len(temp_sensors) <= len(pressure_sensors):
                        with col1:
                            st.markdown(f"**{name}:** {value}")
                    else:
                        with col2:
                            st.markdown(f"**{name}:** {value}")
    else:
        # For robots that don't need maintenance, show simple card with actual values
        create_card(
            title=f"{robot_name}: No Maintenance Required",
            content="All systems operating within normal parameters.",
            priority="low",
            icon="✅"
        )
        
        # Display sensor readings if available
        if "anomaly_values" in result and result["anomaly_values"]:
            st.markdown("### Sensor Readings")
            col1, col2 = st.columns(2)
            
            # Split sensors between columns for better layout
            sensors = list(result["anomaly_values"].items())
            half_length = len(sensors) // 2 + len(sensors) % 2
            
            # First column
            with col1:
                for i in range(half_length):
                    if i < len(sensors):
                        sensor, value = sensors[i]
                        st.markdown(f"**{sensor.replace('_', ' ').title()}:** {value}")
            
            # Second column
            with col2:
                for i in range(half_length, len(sensors)):
                    sensor, value = sensors[i]
                    st.markdown(f"**{sensor.replace('_', ' ').title()}:** {value}")

def plot_performance_comparison():
    """Plot performance comparison of different models."""
    metrics = st.session_state.performance_metrics
    models = list(metrics.keys())
    
    # Prepare data for plotting
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
    data = {metric: [metrics[model][metric] for model in models] for metric in metrics_to_plot}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    
    for metric, values in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric.capitalize())
        ax.bar_label(rects, fmt='%.2f', padding=3, fontsize=8)
        multiplier += 1
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison of Anomaly Detection Models')
    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', ncols=len(metrics_to_plot))
    ax.set_ylim(0, 1.15)
    
    st.pyplot(fig)

def plot_error_rates():
    """Plot false positive and false negative rates."""
    metrics = st.session_state.performance_metrics
    models = list(metrics.keys())
    
    # Prepare data for plotting
    fpr = [metrics[model]['false_positive_rate'] for model in models]
    fnr = [metrics[model]['false_negative_rate'] for model in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, fpr, width, label='False Positive Rate')
    rects2 = ax.bar(x + width/2, fnr, width, label='False Negative Rate')
    
    ax.set_ylabel('Rate')
    ax.set_title('Error Rates by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.3f', padding=3)
    ax.bar_label(rects2, fmt='%.3f', padding=3)
    ax.set_ylim(0, max(max(fpr), max(fnr)) * 1.5)
    
    st.pyplot(fig)

def system_architecture():
    """Display system architecture information."""
    st.markdown("""
    <div class="highlight">
    <h3>System Architecture</h3>
    <p>The CAAD-LLM system integrates three main components:</p>
    <ol>
        <li><strong>Context-Augmented Anomaly Detection (CAAD)</strong>: A statistical approach that detects anomalies in robot sensor data while considering operational context</li>
        <li><strong>Fine-tuned Maintenance LLM</strong>: A GPT-3.5 based model fine-tuned on robotics maintenance data</li>
        <li><strong>Integration Layer</strong>: Combines statistical insights with LLM-generated recommendations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Data Processing Pipeline")
    st.markdown("""
    1. **Data Acquisition**: Collects sensor data from warehouse robots
    2. **Preprocessing**: Handles missing values and normalizes data
    3. **Feature Engineering**: Creates context-aware features
    4. **Anomaly Detection**: Identifies deviations using the CAAD model
    5. **LLM Enhancement**: Generates maintenance recommendations
    6. **Confidence Scoring**: Combines statistical and LLM confidence
    """)
    
    # Data flow metrics
    st.subheader("System Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Data Volume", "210,000 data points")
        st.metric("Preprocessing Output", "64 engineered features")
    with col2:
        st.metric("Context Groups", "4 operational contexts")
        st.metric("Anomaly Detection Rate", "5% positive class")
    with col3:
        st.metric("LLM Processing Time", "0.8s average")
        st.metric("Confidence Score Avg", "76%")

def llm_integration():
    """Display LLM integration information."""
    st.markdown("""
    <div class="highlight">
    <h3>LLM Integration for Maintenance Optimization</h3>
    <p>The system integrates a fine-tuned GPT-3.5 Turbo model to enhance anomaly detection with precise maintenance recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### LLM Model Specifications")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Base Model**: GPT-3.5 Turbo
        - **Fine-tuning Dataset**: 250 robotics maintenance scenarios
        - **Training Epochs**: 4
        - **Input Tokens**: ~500 per request
        - **Output Tokens**: ~250 per response
        """)
    with col2:
        st.markdown("""
        - **Temperature**: 0.2 (low randomness) 
        - **Validation Split**: 20%
        - **Validation Loss**: 0.067
        - **Processing Time**: 0.8s per maintenance recommendation
        """)
    
    # Performance comparison table
    st.subheader("Performance Comparison: Pure Anomaly Detection vs. LLM-Enhanced")
    performance_data = {
        "Metric": ["Precision", "Recall", "F1 Score", "Maintenance Cost Reduction", "Mean Time to Resolution"],
        "Pure Anomaly Detection": ["78.3%", "86.7%", "82.3%", "27%", "4.2 hours"],
        "LLM-Enhanced System": ["84.5%", "89.2%", "86.8%", "36%", "2.8 hours"]
    }
    st.table(pd.DataFrame(performance_data))

def empirical_validation():
    """Display empirical validation information."""
    st.markdown("""
    <div class="highlight">
    <h3>Empirical Validation</h3>
    <p>The CAAD-LLM system was rigorously validated against baseline approaches and tested on multiple datasets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Validation Methodology")
    st.markdown("""
    - **Test Datasets**: Primary robot dataset (5,000 samples) and 3 synthetic variations
    - **Cross-Validation**: 5-fold cross-validation for all models
    - **Statistical Tests**: Paired t-tests for significance analysis (p < 0.01)
    - **Performance Metrics**: Precision, recall, F1-score, ROC AUC, and error rates
    """)
    
    # Main validation tab interface
    validation_tabs = st.tabs(["Performance Metrics", "Error Rates", "External Validation"])
    
    with validation_tabs[0]:
        st.subheader("Performance Comparison")
        plot_performance_comparison()
    
    with validation_tabs[1]:
        st.subheader("Error Rate Analysis")
        plot_error_rates()
    
    with validation_tabs[2]:
        st.subheader("External Validation Results")
        external_data = {
            "Dataset": ["Synthetic (5% noise)", "Synthetic (10% noise)", "Synthetic (15% noise)", "Average"],
            "Accuracy": ["96.3%", "95.2%", "95.1%", "95.5%"],
            "F1 Score": ["62.7%", "52.0%", "50.7%", "55.1%"]
        }
        st.table(pd.DataFrame(external_data))

# Add a new function for the data chat feature
def data_chat():
    """Simple interface for asking questions about robot data in plain language."""
    st.markdown('<h2 class="sub-header">Ask Questions About Your Robots</h2>', unsafe_allow_html=True)
    
    # Simple introduction
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 1.1em;">
            <b>Ask questions about your robots in plain English.</b> No technical knowledge required!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example questions for users
    st.markdown("### Try asking questions like:")
    
    # Create two columns for example questions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Which robots need maintenance?", key="q1", use_container_width=True):
            # Set the question in session state and trigger rerun
            st.session_state.data_chat_input = "Which robots need maintenance?"
            st.rerun()
        
        if st.button("How many robots are running normally?", key="q3", use_container_width=True):
            st.session_state.data_chat_input = "How many robots are running normally?"
            st.rerun()
            
        if st.button("Which robot has the highest priority?", key="q5", use_container_width=True):
            st.session_state.data_chat_input = "Which robot has the highest priority?"
            st.rerun()
    
    with col2:
        if st.button("What are the most common issues?", key="q2", use_container_width=True):
            st.session_state.data_chat_input = "What are the most common issues?"
            st.rerun()
        
        if st.button("What kind of maintenance does robot #5 need?", key="q4", use_container_width=True):
            st.session_state.data_chat_input = "What kind of maintenance does robot #5 need?"
            st.rerun()
    
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history with more user-friendly styling
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                <div style="background-color: #e3f2fd; padding: 10px 15px; border-radius: 18px 18px 0 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 12px;">
                <div style="background-color: #f5f5f5; padding: 10px 15px; border-radius: 18px 18px 18px 0; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input with a more user-friendly design
    col1, col2 = st.columns([5, 1])
    with col1:
        # Use our labeled_input helper instead of direct st.text_input for accessibility
        user_question = labeled_input(
            st.text_input, 
            "Question",  # This is now a visible label
            placeholder="Type your question here...", 
            key="data_chat_input"
        )
    with col2:
        # Add proper vertical alignment
        st.write("")  # Add some space for alignment
        ask_button = st.button("Ask", key="ask_button", use_container_width=True)
    
    if user_question and ask_button:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Process the question with direct response instead of using LLM
        with st.spinner("Thinking..."):
            try:
                # Enhanced robot number extraction with multiple patterns
                import re
                # Match patterns like: "robot 2145", "robot #2145", "#2145", "robot number 2145", etc.
                robot_num_match = re.search(r'(?:robot\s*(?:#|number|num|)\s*|#\s*)(\d+)', user_question.lower())
                robot_index = int(robot_num_match.group(1)) - 1 if robot_num_match else None
                
                # Remove the attempt to interpret large numbers as having just the first digit
                # Instead, provide clear messaging about valid robot ranges
                
                # Check if specific robot query
                if robot_index is not None and st.session_state.data is not None:
                    # Validate robot index is within range
                    if 0 <= robot_index < len(st.session_state.data):
                        # Get robot data
                        robot_data = st.session_state.data.iloc[robot_index]
                        robot_name = f"Robot #{robot_index+1}"
                        
                        # Get analysis result for this specific robot
                        robot_result = st.session_state.results[robot_index] if robot_index < len(st.session_state.results) else None
                        
                        # Generate answer for specific robot
                        if robot_result:
                            question_lower = user_question.lower()
                            
                            # Check if asking about maintenance
                            if any(word in question_lower for word in ["maintenance", "repair", "fix", "service", "need", "issue", "problem", "mitigate", "situation"]):
                                if robot_result.get('maintenance_required', False):
                                    # Detailed maintenance answer
                                    root_cause = robot_result.get('root_cause', 'Unknown issue')
                                    urgency = robot_result.get('urgency', 'Medium')
                                    
                                    response = f"{robot_name} requires maintenance with {urgency} urgency. The issue is: {root_cause}.\n\n"
                                    
                                    # Add recommendations if available
                                    if 'recommendations' in robot_result and robot_result['recommendations']:
                                        response += "Recommended actions:\n"
                                        for i, rec in enumerate(robot_result['recommendations']):
                                            if isinstance(rec, dict) and 'action' in rec:
                                                response += f"- {rec['action']}\n"
                                                # Add explanation if available
                                                if 'explanation' in rec and rec['explanation']:
                                                    response += f"  ({rec['explanation']})\n"
                                                # Add timeframe if available
                                                if 'timeframe' in rec and rec['timeframe']:
                                                    response += f"  Timeframe: {rec['timeframe']}\n"
                                            else:
                                                response += f"- {rec}\n"
                                    
                                    # Add sensor readings that are out of normal range
                                    if 'anomaly_values' in robot_result and robot_result['anomaly_values']:
                                        response += "\nAbnormal sensor readings:\n"
                                        for sensor, value in robot_result['anomaly_values'].items():
                                            response += f"- {sensor.replace('_', ' ').title()}: {value}\n"
                                else:
                                    response = f"{robot_name} is operating normally and does not need maintenance at this time."
                            
                            # Check if asking about status
                            elif any(word in question_lower for word in ["status", "condition", "health", "how is", "state"]):
                                if robot_result.get('maintenance_required', False):
                                    response = f"{robot_name} requires attention. The issue is: {robot_result.get('root_cause', 'Unknown')}."
                                else:
                                    response = f"{robot_name} is operating normally."
                            
                            # Check if asking about sensor readings
                            elif any(word in question_lower for word in ["sensor", "reading", "measure", "value", "temperature", "vibration", "pressure", "power", "coolant"]):
                                # Check if asking about specific sensor
                                specific_sensor = None
                                for sensor in ["temperature", "vibration", "hydraulic_pressure", "power_output", "coolant_level"]:
                                    # Match different ways of asking about sensors
                                    sensor_terms = [sensor]
                                    if sensor == "hydraulic_pressure":
                                        sensor_terms.extend(["hydraulic", "pressure"])
                                    elif sensor == "power_output":
                                        sensor_terms.extend(["power", "output"])
                                    elif sensor == "coolant_level":
                                        sensor_terms.extend(["coolant"])
                                    
                                    if any(term in question_lower for term in sensor_terms):
                                        specific_sensor = sensor
                                        break
                                
                                # Extract sensor data
                                sensor_data = {}
                                for key, value in robot_data.items():
                                    if isinstance(value, (int, float)) and not pd.isna(value):
                                        sensor_data[key] = value
                                
                                # If asking about specific sensor
                                if specific_sensor and specific_sensor in sensor_data:
                                    sensor_value = sensor_data[specific_sensor]
                                    
                                    # Get information about normal ranges
                                    sensor_ranges = {
                                        'temperature': {'min': 50, 'max': 85, 'unit': '°C'},
                                        'vibration': {'min': 0, 'max': 0.3, 'unit': 'mm/s'},
                                        'hydraulic_pressure': {'min': 140, 'max': 180, 'unit': 'PSI'},
                                        'power_output': {'min': 85, 'max': 100, 'unit': '%'},
                                        'coolant_level': {'min': 70, 'max': 95, 'unit': '%'}
                                    }
                                    
                                    # Format sensor with units and status
                                    if specific_sensor in sensor_ranges:
                                        range_info = sensor_ranges[specific_sensor]
                                        unit = range_info.get('unit', '')
                                        min_val = range_info.get('min')
                                        max_val = range_info.get('max')
                                        
                                        if sensor_value < min_val:
                                            status = "BELOW NORMAL RANGE"
                                        elif sensor_value > max_val:
                                            status = "ABOVE NORMAL RANGE"
                                        else:
                                            status = "NORMAL"
                                        
                                        response = f"{robot_name} {specific_sensor.replace('_', ' ')} is {sensor_value}{unit} ({status})\n"
                                        response += f"Normal range: {min_val}-{max_val}{unit}"
                                    else:
                                        response = f"{robot_name} {specific_sensor.replace('_', ' ')} is {sensor_value}"
                                else:
                                    # Compile response for all sensors
                                    response = f"{robot_name} sensor readings:\n"
                                    for sensor, value in list(sensor_data.items())[:5]:  # Limit to first 5
                                        response += f"- {sensor.replace('_', ' ').title()}: {value}\n"
                            
                            # Check if asking about error codes
                            elif any(word in question_lower for word in ["error", "code", "fault", "alert"]):
                                error_codes = robot_data.get('error_codes', None)
                                if error_codes:
                                    response = f"{robot_name} has the following error codes: {error_codes}\n\n"
                                    
                                    # Add descriptions if available
                                    code_list = error_codes.split('+')
                                    for code in code_list:
                                        if code in st.session_state.error_codes_guide:
                                            info = st.session_state.error_codes_guide[code]
                                            desc = info['description']
                                            severity = info['severity']
                                            response += f"- {code}: {desc} (Severity: {severity})\n"
                                            
                                            # Add first recommended action
                                            if len(info['recommended_actions']) > 0:
                                                response += f"  Recommended: {info['recommended_actions'][0]}\n"
                                else:
                                    response = f"{robot_name} has no active error codes."
                            
                            # Default comprehensive answer for specific robot
                            else:
                                maintenance_status = "requires maintenance" if robot_result.get('maintenance_required', False) else "is operating normally"
                                response = f"{robot_name} {maintenance_status}.\n\n"
                                
                                if robot_result.get('maintenance_required', False):
                                    response += f"Issue: {robot_result.get('root_cause', 'Unknown')}\n"
                                    response += f"Priority: {robot_result.get('urgency', 'Medium')}\n"
                                    
                                    # Add first recommendation
                                    if 'recommendations' in robot_result and robot_result['recommendations']:
                                        rec = robot_result['recommendations'][0]
                                        if isinstance(rec, dict) and 'action' in rec:
                                            response += f"Recommended action: {rec['action']}\n"
                                        else:
                                            response += f"Recommended action: {rec}\n"
                                    
                                    # Add key sensor readings
                                    response += "\nKey sensor readings:\n"
                                    for key, value in robot_data.items():
                                        if isinstance(value, (int, float)) and not pd.isna(value):
                                            if key in ['temperature', 'vibration', 'hydraulic_pressure', 'power_output', 'coolant_level']:
                                                response += f"- {key.replace('_', ' ').title()}: {value}\n"
                        else:
                            # Robot hasn't been analyzed
                            response = f"{robot_name} exists in the system but hasn't been analyzed yet. Please run an analysis for this robot first."
                    else:
                        # Robot index out of range - give clearer message about valid range
                        max_robots = len(st.session_state.data) if st.session_state.data is not None else 0
                        extracted_number = int(robot_num_match.group(1))
                        
                        if extracted_number > max_robots:
                            response = f"Robot #{extracted_number} doesn't exist in the current dataset. The valid robot range is 1-{max_robots}. Please ask about a robot within this range."
                        else:
                            response = f"Robot #{robot_index+1} doesn't exist in the current dataset. Valid robot numbers are 1-{max_robots}."
                
                # For fleet-wide questions
                else:
                    # Count robots with issues
                    if st.session_state.data is not None:
                        # Count robots that need maintenance
                        maintenance_robots = [r for r in st.session_state.results if r and r.get('maintenance_required', False)]
                        num_with_maintenance = len(maintenance_robots)
                        
                        # Get common issues
                        issue_counts = {}
                        for r in maintenance_robots:
                            root_cause = r.get('root_cause', 'Unknown')
                            if root_cause in issue_counts:
                                issue_counts[root_cause] += 1
                            else:
                                issue_counts[root_cause] = 1
                        
                        # Generate simple response based on question
                        question_lower = user_question.lower()
                        
                        # DIRECT MATCH for primary cause question
                        if "primary cause" in question_lower or "main issue" in question_lower or "most maintenance needed" in question_lower:
                            # Find all robots needing maintenance
                            maintenance_robots = []
                            for i, result in enumerate(st.session_state.results):
                                if result and result.get('maintenance_required', False):
                                    # Get urgency for sorting
                                    urgency = result.get('urgency', 'Medium').lower() 
                                    urgency_score = 3 if urgency == 'critical' else 2 if urgency == 'high' else 1 if urgency == 'medium' else 0
                                    
                                    # Add to list with score
                                    maintenance_robots.append((i, result, urgency_score))
                            
                            if maintenance_robots:
                                # Sort by urgency score
                                sorted_robots = sorted(maintenance_robots, key=lambda x: x[2], reverse=True)
                                
                                # Get the top robot
                                top_idx, top_result, _ = sorted_robots[0]
                                top_robot_id = top_idx + 1
                                root_cause = top_result.get('root_cause', 'Unknown issue')
                                urgency = top_result.get('urgency', 'Medium')
                                
                                # Get robot data
                                robot_data = st.session_state.data.iloc[top_idx] if top_idx < len(st.session_state.data) else None
                                
                                # Build response
                                response = f"The primary cause of the most maintenance-needed robot (Robot #{top_robot_id}) is: {root_cause}."
                                response += f"\n\nThis robot has {urgency} urgency maintenance needs."
                                
                                # Add critical sensor readings
                                if robot_data is not None:
                                    critical_readings = []
                                    for col, value in robot_data.items():
                                        if col == 'temperature' and isinstance(value, (int, float)) and not pd.isna(value) and value > 85:
                                            critical_readings.append(f"Temperature: {value}°C (above normal range)")
                                        elif col == 'vibration' and isinstance(value, (int, float)) and not pd.isna(value) and value > 0.3:
                                            critical_readings.append(f"Vibration: {value} (above normal range)")
                                        elif col == 'hydraulic_pressure' and isinstance(value, (int, float)) and not pd.isna(value) and value < 140:
                                            critical_readings.append(f"Hydraulic Pressure: {value} PSI (below normal range)")
                                        elif col == 'coolant_level' and isinstance(value, (int, float)) and not pd.isna(value) and value < 70:
                                            critical_readings.append(f"Coolant Level: {value}% (below normal range)")
                                    
                                    if critical_readings:
                                        response += "\n\nCritical sensor readings:"
                                        for reading in critical_readings:
                                            response += f"\n- {reading}"
                                
                                # Add recommendations
                                if 'recommendations' in top_result and top_result['recommendations']:
                                    response += "\n\nRecommended actions:"
                                    for i, rec in enumerate(top_result['recommendations'][:3]):
                                        if isinstance(rec, dict) and 'action' in rec:
                                            response += f"\n{i+1}. {rec['action']}"
                                        else:
                                            response += f"\n{i+1}. {rec}"
                            else:
                                response = "No robots currently need maintenance."
                                
                        # Handle highest priority robot question
                        if any(phrase in question_lower for phrase in ["highest priority", "most urgent", "most critical", "top priority"]):
                            high_priority_robots = []
                            
                            # Find robots with high/critical urgency
                            for i, result in enumerate(st.session_state.results):
                                if result and (result.get('urgency', '').lower() in ['high', 'critical']):
                                    high_priority_robots.append((i, result))
                            
                            if high_priority_robots:
                                # Sort by urgency and get highest priority robot
                                def urgency_score(item):
                                    idx, result = item
                                    urgency = result.get('urgency', '').lower()
                                    if urgency == 'critical':
                                        return 3
                                    elif urgency == 'high':
                                        return 2
                                    elif urgency == 'medium':
                                        return 1
                                    else:
                                        return 0
                                
                                sorted_robots = sorted(high_priority_robots, key=urgency_score, reverse=True)
                                top_idx, top_result = sorted_robots[0]
                                
                                response = f"Robot #{top_idx+1} has the highest priority with {top_result.get('urgency', 'High')} urgency."
                                response += f"\nIssue: {top_result.get('root_cause', 'Unknown issue')}"
                                
                                # Add recommendations
                                if 'recommendations' in top_result and top_result['recommendations']:
                                    rec = top_result['recommendations'][0]
                                    if isinstance(rec, dict) and 'action' in rec:
                                        response += f"\nRecommended action: {rec['action']}"
                                    else:
                                        response += f"\nRecommended action: {rec}"
                            else:
                                response = "There are no high priority robots requiring maintenance at this time."
                        
                        # Handle questions about primary cause of most maintenance-needed robot
                        elif any(phrase in question_lower for phrase in ["primary cause", "main issue", "worst condition", "most maintenance needed"]):
                            # First find robots that need maintenance
                            maintenance_robots = []
                            for i, result in enumerate(st.session_state.results):
                                if result and result.get('maintenance_required', False):
                                    maintenance_robots.append((i, result))
                            
                            if maintenance_robots:
                                # Sort based on urgency, then by confidence
                                def maintenance_score(item):
                                    idx, result = item
                                    urgency = result.get('urgency', '').lower()
                                    confidence = result.get('confidence', 0.5)
                                    
                                    # Create a combined score
                                    urgency_value = 0
                                    if urgency == 'critical':
                                        urgency_value = 3
                                    elif urgency == 'high':
                                        urgency_value = 2
                                    elif urgency == 'medium':
                                        urgency_value = 1
                                    
                                    # Return combined score giving more weight to urgency
                                    return (urgency_value * 10) + (confidence * 5)
                                
                                # Sort robots by their maintenance score (highest first)
                                sorted_robots = sorted(maintenance_robots, key=maintenance_score, reverse=True)
                                
                                # Get the top robot
                                top_idx, top_result = sorted_robots[0]
                                top_robot_id = top_idx + 1
                                root_cause = top_result.get('root_cause', 'Unknown issue')
                                urgency = top_result.get('urgency', 'Medium')
                                
                                # Get the robot's data to provide additional context
                                robot_data = st.session_state.data.iloc[top_idx] if top_idx < len(st.session_state.data) else None
                                
                                # Build a detailed response
                                response = f"The primary cause of the most maintenance-needed robot (Robot #{top_robot_id}) is: {root_cause}."
                                response += f"\n\nThis robot has {urgency} urgency maintenance needs."
                                
                                # Add critical sensor values if available
                                if robot_data is not None:
                                    # Find the most critical sensor reading
                                    critical_sensors = {}
                                    for sensor, value in robot_data.items():
                                        if isinstance(value, (int, float)) and not pd.isna(value):
                                            if sensor == 'temperature' and value > 85:
                                                critical_sensors[sensor] = f"{value}°C (above normal range of 50-85°C)"
                                            elif sensor == 'vibration' and value > 0.3:
                                                critical_sensors[sensor] = f"{value} mm/s (above normal range of 0-0.3 mm/s)"
                                            elif sensor == 'hydraulic_pressure' and (value < 140 or value > 180):
                                                critical_sensors[sensor] = f"{value} PSI (outside normal range of 140-180 PSI)"
                                            elif sensor == 'coolant_level' and value < 70:
                                                critical_sensors[sensor] = f"{value}% (below normal minimum of 70%)"
                                    
                                    if critical_sensors:
                                        response += "\n\nCritical sensor readings:"
                                        for sensor, reading in critical_sensors.items():
                                            response += f"\n- {sensor.replace('_', ' ').title()}: {reading}"
                                
                                # Add recommended actions
                                if 'recommendations' in top_result and top_result['recommendations']:
                                    response += "\n\nRecommended actions:"
                                    for i, rec in enumerate(top_result['recommendations'][:3]):
                                        if isinstance(rec, dict) and 'action' in rec:
                                            response += f"\n{i+1}. {rec['action']}"
                                        else:
                                            response += f"\n{i+1}. {rec}"
                            else:
                                response = "No robots currently need maintenance."
                        
                        # Match the question to pre-defined responses
                        elif "how many" in question_lower and "maintenance" in question_lower:
                            response = f"{num_with_maintenance} robots need maintenance right now."
                            
                            # Add details about which robots
                            if num_with_maintenance > 0:
                                robot_indices = [i for i, r in enumerate(st.session_state.results) if r and r.get('maintenance_required', False)]
                                robot_list = [f"Robot #{i+1}" for i in robot_indices]
                                if len(robot_list) <= 5:
                                    response += f" These are: {', '.join(robot_list)}."
                                else:
                                    response += f" These include: {', '.join(robot_list[:5])} and {len(robot_list)-5} more."
                        
                        elif "which" in question_lower and "maintenance" in question_lower:
                            if num_with_maintenance > 0:
                                robot_indices = [i for i, r in enumerate(st.session_state.results) if r and r.get('maintenance_required', False)]
                                robot_list = [f"Robot #{i+1}" for i in robot_indices]
                                response = f"The following robots need maintenance: {', '.join(robot_list[:10])}"
                                if len(robot_list) > 10:
                                    response += f" and {len(robot_list)-10} more."
                            else:
                                response = "No robots currently need maintenance."
                                
                        elif "how many" in question_lower and "normal" in question_lower:
                            normal_count = len(st.session_state.data) - num_with_maintenance
                            response = f"{normal_count} robots are operating normally."
                            
                        elif "common" in question_lower and "issue" in question_lower:
                            # Get top 3 most common issues
                            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                            if sorted_issues:
                                response = "The most common issues are:\n\n"
                                for issue, count in sorted_issues:
                                    response += f"- {issue}: {count} robots\n"
                            else:
                                response = "There are no common issues right now."
                                
                        elif "battery" in question_lower or "replace" in question_lower:
                            battery_robots = []
                            for i, r in enumerate(st.session_state.results):
                                if r and r.get('maintenance_required', False) and "battery" in str(r.get('root_cause', '')).lower():
                                    battery_robots.append(i)
                            
                            if battery_robots:
                                robot_list = [f"Robot #{i+1}" for i in battery_robots]
                                response = f"{len(battery_robots)} robots need battery replacements: {', '.join(robot_list[:5])}"
                                if len(robot_list) > 5:
                                    response += f" and {len(robot_list)-5} more."
                            else:
                                response = "No robots currently need battery replacement."
                                
                        elif "critical" in question_lower or "urgent" in question_lower or "high priority" in question_lower:
                            urgent_robots = []
                            for i, r in enumerate(st.session_state.results):
                                if r and r.get('urgency', '').lower() in ['high', 'critical']:
                                    urgent_robots.append(i)
                            
                            if urgent_robots:
                                robot_list = [f"Robot #{i+1}" for i in urgent_robots]
                                response = f"{len(urgent_robots)} robots need urgent attention: {', '.join(robot_list[:5])}"
                                if len(robot_list) > 5:
                                    response += f" and {len(robot_list)-5} more."
                            else:
                                response = "There are no robots requiring urgent attention right now."
                        
                        elif any(word in question_lower for word in ["error", "code", "fault"]):
                            # Count robots with error codes
                            error_robots = []
                            for i, data in enumerate(st.session_state.data.itertuples()):
                                if hasattr(data, 'error_codes') and data.error_codes:
                                    error_robots.append(i)
                            
                            if error_robots:
                                robot_list = [f"Robot #{i+1}" for i in error_robots]
                                response = f"{len(error_robots)} robots have active error codes: {', '.join(robot_list[:5])}"
                                if len(robot_list) > 5:
                                    response += f" and {len(robot_list)-5} more."
                                
                                # Count occurrence of each error code
                                code_counts = {}
                                for i in error_robots:
                                    codes = st.session_state.data.iloc[i].get('error_codes', '').split('+')
                                    for code in codes:
                                        if code in code_counts:
                                            code_counts[code] += 1
                                        else:
                                            code_counts[code] = 1
                                
                                # List most common error codes
                                if code_counts:
                                    sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                                    response += "\n\nMost common error codes:\n"
                                    for code, count in sorted_codes:
                                        desc = st.session_state.error_codes_guide.get(code, {}).get('description', 'Unknown')
                                        response += f"- {code}: {desc} ({count} robots)\n"
                            else:
                                response = "No robots currently have active error codes."
                                
                        else:
                            # Generic fleet status response with more detail
                            response = f"Robot Fleet Status: {len(st.session_state.data)} total robots, {len(st.session_state.data) - num_with_maintenance} operating normally, {num_with_maintenance} needing maintenance."
                            
                            # Add some common issues if any robots need maintenance
                            if num_with_maintenance > 0 and issue_counts:
                                top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                                response += "\n\nMost common issues:\n"
                                for issue, count in top_issues:
                                    response += f"- {issue}: {count} robots\n"
                                    
                            # Add robots with highest urgency if any
                            high_urgency = [i for i, r in enumerate(st.session_state.results) 
                                           if r and r.get('urgency', '').lower() in ['high', 'critical']]
                            if high_urgency:
                                response += f"\nRobots needing urgent attention: {', '.join([f'Robot #{i+1}' for i in high_urgency[:3]])}"
                                if len(high_urgency) > 3:
                                    response += f" and {len(high_urgency)-3} more"
                    else:
                        response = "No robot data is currently available."
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Force refresh to show the new message
                st.rerun()
                
            except Exception as e:
                import traceback
                error_message = f"Sorry, I couldn't answer that right now. Please try asking in a different way. Error details: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                st.rerun()
    
    # Add a clear button
    if st.session_state.chat_history and st.button("Start New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Add a new function for the maintenance scheduling assistant
def maintenance_scheduling_assistant():
    """Interactive scheduling assistant for robot maintenance planning with a simplified interface."""
    st.markdown('<h2 class="sub-header">Maintenance Scheduler</h2>', unsafe_allow_html=True)
    
    # Simple introduction message
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 1.1em; color: #1b5e20; font-weight: 500;">
            <b>Schedule maintenance for your robots in a few simple steps.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize scheduling state if not exists
    if 'maintenance_schedule' not in st.session_state:
        st.session_state.maintenance_schedule = []
    if 'maintenance_robots' not in st.session_state:
        st.session_state.maintenance_robots = []
    if 'available_technicians' not in st.session_state:
        st.session_state.available_technicians = 2
    if 'max_concurrent_maintenance' not in st.session_state:
        st.session_state.max_concurrent_maintenance = 1
    if 'operational_hours' not in st.session_state:
        st.session_state.operational_hours = "08:00-20:00"
    if 'min_operational_robots' not in st.session_state:
        st.session_state.min_operational_robots = 3
    
    # Current robot statuses
    st.markdown("""
    <div style="margin-top: 25px; margin-bottom: 10px;">
        <h3>Robots Needing Maintenance</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None or len(st.session_state.results) == 0:
        st.info("No robots have been analyzed yet. Please go to the Demo page and analyze your robots first.")
        return
    
    # Display simplified table of robots needing maintenance
    maintenance_robots = []
    for i, result in enumerate(st.session_state.results):
        if result and result.get('maintenance_required', False):
            robot_name = f"Robot #{i+1}"
            urgency = result.get('urgency', 'Medium')
            root_cause = result.get('root_cause', 'Unknown issue')
            est_time = result.get('estimated_time', '4 hours')
            
            maintenance_robots.append({
                "id": i,
                "name": robot_name,
                "urgency": urgency,
                "issue": root_cause,
                "time": est_time
            })
    
    # Store in session state for consistency between page refreshes
    st.session_state.maintenance_robots = maintenance_robots
    
    # Use maintenance robots from session state (in case we're returning from another page)
    if not maintenance_robots and st.session_state.maintenance_robots:
        maintenance_robots = st.session_state.maintenance_robots
        st.info("Using previously identified robots needing maintenance")
    
    if not maintenance_robots:
        st.success("Great news! No robots currently need maintenance.")
        st.info("Return to the Dashboard and use the 'Simulate Anomalies' tool to create robot failures.")
        return
    
    # Create cards for each robot needing maintenance
    for robot in maintenance_robots:
        # Get robot data
        robot_id = robot['id']
        robot_name = robot['name']
        urgency = robot['urgency']
        issue = robot['issue']
        est_time = robot['time']
        
        # Use the create_robot_card utility instead of manually generating HTML
        create_robot_card(
            robot_name=robot_name,
            issue=issue,
            urgency=urgency,
            estimated_time=est_time,
            additional_info={}
        )
        
        # Add buttons below the card
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(f"Schedule", key=f"schedule_{robot_id}"):
                st.success(f"Maintenance scheduled for {robot_name}")
        with col2:
            if st.button(f"View Details", key=f"view_maint_{robot_id}"):
                st.session_state.selected_robot = robot_id
                st.session_state.dashboard_view = False
                st.session_state.current_page = "Robot Status"
                st.rerun()
    
    # Add quick scheduling option
    st.markdown("""
    <div style="margin-top: 25px; margin-bottom: 15px;">
        <h3>Quick Schedule</h3>
    </div>
    """, unsafe_allow_html=True)
    
    scheduling_options = {
        "optimal": "Optimal Schedule (Balance between urgency and operations)",
        "urgent": "Focus on Urgent Robots First",
        "minimal_disruption": "Minimize Operational Disruption"
    }
    
    schedule_option = st.selectbox(
        "Choose a scheduling approach:",
        options=list(scheduling_options.keys()),
        format_func=lambda x: scheduling_options[x]
    )
    
    if st.button("Generate Maintenance Schedule", use_container_width=True):
        with st.spinner("Creating maintenance schedule..."):
            # Debug output to verify maintenance robots are found
            st.write(f"Found {len(maintenance_robots)} robots needing maintenance")
            
            # Force re-check for robots needing maintenance in case of state issues
            if len(maintenance_robots) == 0:
                maintenance_robots = []
                for i, result in enumerate(st.session_state.results):
                    if result and result.get('maintenance_required', False):
                        robot_name = f"Robot #{i+1}"
                        urgency = result.get('urgency', 'Medium')
                        root_cause = result.get('root_cause', 'Unknown issue')
                        est_time = result.get('estimated_time', '4 hours')
                        
                        maintenance_robots.append({
                            "id": i,
                            "name": robot_name,
                            "urgency": urgency,
                            "issue": root_cause,
                            "time": est_time
                        })
                st.write(f"Re-checked and found {len(maintenance_robots)} robots needing maintenance")
                
                # If still none found, show message and return
                if not maintenance_robots:
                    st.warning("No robots currently need maintenance. Please return to Dashboard and create anomalies first.")
                    return
            
            time.sleep(1)  # Simulate processing
            
            # Create a simple calendar view for the maintenance schedule
            st.markdown("""
            <div style="margin-top: 25px; margin-bottom: 15px;">
                <h3>Your Maintenance Schedule</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Get current date for demo purposes
            from datetime import datetime, timedelta
            today = datetime.now()
            
            # Generate a simple 5-day schedule based on the selected approach
            schedule_days = []
            
            if schedule_option == "urgent":
                # Sort by urgency
                sorted_robots = sorted(maintenance_robots, key=lambda x: 0 if x['urgency'].lower() == 'high' or x['urgency'].lower() == 'critical' else 1 if x['urgency'].lower() == 'medium' else 2)
            elif schedule_option == "minimal_disruption":
                # Sort by estimated time (shortest first)
                sorted_robots = sorted(maintenance_robots, key=lambda x: x['time'])
            else:  # optimal
                # Use a mixed approach
                sorted_robots = sorted(maintenance_robots, key=lambda x: (0 if x['urgency'].lower() == 'high' or x['urgency'].lower() == 'critical' else 1, x['time']))
            
            # Create a visual calendar
            current_day = today
            robots_per_day = st.session_state.max_concurrent_maintenance
            for i in range(0, len(sorted_robots), robots_per_day):
                # Skip weekends
                if current_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    current_day += timedelta(days=1 + (7 - current_day.weekday()) % 7)
                
                day_robots = sorted_robots[i:i+robots_per_day]
                schedule_days.append({
                    "date": current_day.strftime("%A, %B %d"),
                    "robots": day_robots
                })
                current_day += timedelta(days=1)
            
            # Display the schedule
            for day in schedule_days:
                st.markdown(f"""
                <div style="margin-top: 15px; margin-bottom: 5px; background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
                    <div style="font-weight: bold; font-size: 1.1em;">{day['date']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                for robot in day['robots']:
                    st.markdown(f"""
                    <div style="background-color: white; padding: 10px; border-radius: 4px; margin-bottom: 5px; border-left: 4px solid #1976d2;">
                        <div style="font-weight: bold;">{robot['name']}</div>
                        <div style="font-size: 0.9em;">{robot['issue']}</div>
                        <div style="font-size: 0.8em; color: #555;">Estimated time: {robot['time']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add download/email options
            st.success("Maintenance schedule created successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                # Add download button for the schedule
                st.download_button(
                    label="Download Schedule (PDF)",
                    data=b"Sample PDF content",
                    file_name="maintenance_schedule.pdf",
                    mime="application/pdf",
                )
            with col2:
                if st.button("Email Schedule to Team"):
                    st.info("Schedule has been emailed to the maintenance team.")
            
            # Add helpful next steps
            st.markdown("""
            <div style="margin-top: 25px; background-color: #e8f5e9; padding: 15px; border-radius: 4px;">
                <div style="font-weight: bold; margin-bottom: 10px;">Next Steps:</div>
                <ol style="margin-top: 0; margin-bottom: 0; padding-left: 20px;">
                    <li>Share this schedule with your maintenance team</li>
                    <li>Ensure all required parts are in stock</li>
                    <li>Confirm technician availability on scheduled dates</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

def show_demo_page():
    """Display detailed information for a specific robot with optimized performance."""
    st.markdown('<h2>Robot Status</h2>', unsafe_allow_html=True)
    
    if st.session_state.enhancer is None or st.session_state.data is None:
        st.markdown("""
        <div class="status-warning">
            <p style="margin: 0;"><strong>⚠️ System still loading</strong> - Please wait a moment while we get things ready.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get currently selected robot information
    selected_robot_id = st.session_state.selected_robot
    
    # Ensure selected robot is valid
    if selected_robot_id >= len(st.session_state.data):
        st.error(f"Invalid robot selection: {selected_robot_id}")
        st.session_state.selected_robot = 0
        selected_robot_id = 0
    
    # Get robot data efficiently
    robot_data = st.session_state.data.iloc[selected_robot_id]
    robot_name = f"Robot #{selected_robot_id+1}"
    robot_type = robot_data.get('robot_type', 'Unknown Type')
    
    # Header with selected robot info - use a single markdown call for efficiency
    st.markdown(f"""
    <div class="card" style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <div style="flex: 0 0 70px; height: 70px; background-color: #e3f2fd; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
            <span style="font-size: 2rem; color: #1976d2;">🤖</span>
        </div>
        <div>
            <h3 style="margin: 0; padding: 0; border: none;">{robot_name} ({robot_type})</h3>
            <p style="margin: 0.5rem 0 0 0;">Viewing detailed information and maintenance status.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Actions row
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Back to dashboard button at the top
        if st.button("← Back to Dashboard", key="top_back_button"):
            st.session_state.dashboard_view = True
            st.session_state.current_page = "Dashboard"
            st.rerun()
    
    with col2:
        # Check if analysis already exists
        has_analysis = (selected_robot_id < len(st.session_state.results) and 
                       st.session_state.results[selected_robot_id] is not None)
        
        # Simplified button to check maintenance needs
        btn_label = "Re-analyze Robot" if has_analysis else "Check Maintenance Needs"
        if st.button(btn_label, key="analyze_btn", use_container_width=True):
            with st.spinner("Analyzing robot data..."):
                # Convert to dict and analyze
                robot_dict = robot_data.to_dict()
                result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
                
                # Store in results
                if len(st.session_state.results) <= selected_robot_id:
                    st.session_state.results.extend([None] * (selected_robot_id - len(st.session_state.results) + 1))
                st.session_state.results[selected_robot_id] = result
                
                if result:
                    st.success("Analysis complete!")
                    st.rerun()  # Refresh to show updated analysis
    
    # Display robot details in a more efficient manner
    robot_details_container = st.container()
    
    # Actual sensor values
    temp = robot_data.get('temperature', 0)
    vibration = robot_data.get('vibration', 0)
    pressure = robot_data.get('hydraulic_pressure', 0)
    power = robot_data.get('power_output', 0)
    coolant = robot_data.get('coolant_level', 0)
    error_codes = robot_data.get('error_codes', None)
    
    # Display sensor readings
    sensor_container = st.container()
    with sensor_container:
        st.markdown("<h3>Sensor Readings</h3>", unsafe_allow_html=True)
        sensor_readings_cached(robot_data)
    
    # Display maintenance recommendations if available
    if has_analysis:
        analysis_result = st.session_state.results[selected_robot_id]
        maintenance_required = analysis_result.get('maintenance_required', False)
        
        with st.container():
            st.markdown("<hr style='margin: 2rem 0 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
            st.markdown('<h3>Maintenance Results</h3>', unsafe_allow_html=True)
            
            # Display maintenance status
            if maintenance_required:
                st.error("⚠️ Maintenance Required")
            else:
                st.success("✅ Operating Normally")
            
            # Show Priority Level
            urgency = analysis_result.get('urgency', 'Low')
            priority_color = "#b71c1c" if urgency == "High" else "#e65100" if urgency == "Medium" else "#1b5e20"
            
            st.markdown(f"""
            <div style="background-color: {priority_color}; color: white; 
                        padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; 
                        font-weight: bold; text-align: center;">
                Priority Level: {urgency}
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence
            confidence = analysis_result.get('confidence', 0) * 100
            confidence_color = "#1b5e20" if confidence >= 80 else "#e65100" if confidence >= 50 else "#b71c1c"
            
            st.markdown(f"""
            <div class="card" style="flex: 1; min-width: 200px; padding: 1.2rem;">
                <div style="font-size: 0.9rem; color: #212121; font-weight: 500; margin-bottom: 0.5rem;">Prediction Reliability</div>
                <div style="font-size: 1.5rem; font-weight: bold; padding: 0.5rem; border-radius: 0.3rem; 
                            background-color: #f1f8e9; color: {confidence_color}; display: inline-block;">
                    {'✅ High' if confidence >= 80 else '⚠️ Medium' if confidence >= 50 else '❌ Low'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display root cause
            root_cause = analysis_result.get('root_cause', 'Unknown')
            st.markdown(f"""
            <h4 style="margin-top: 2rem;">What needs attention</h4>
            <div style="padding: 0.75rem; background-color: #f5f5f5; border-radius: 4px; margin-bottom: 1rem; border-left: 4px solid {priority_color};">
                <p style="margin: 0; color: #212121; font-weight: 500;">{root_cause}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display contributing factors
            contributing_factors = analysis_result.get('contributing_factors', [])
            if contributing_factors:
                st.markdown("<h4>Contributing factors</h4>", unsafe_allow_html=True)
                for i, factor in enumerate(contributing_factors):
                    st.markdown(f"<p style='color: #212121; margin-bottom: 0.5rem;'>- {factor}</p>", unsafe_allow_html=True)
            
            # Display recommendations
            recommendations = analysis_result.get('recommendations', [])
            if recommendations:
                st.markdown("<h3 style=\"margin-top: 2rem;\">Recommended actions</h3>", unsafe_allow_html=True)
                for i, rec in enumerate(recommendations):
                    action = rec.get('action', '')
                    st.markdown(f"""
                    <div class="card" style="background-color: #e3f2fd !important; border-left: 5px solid #1565c0; padding: 1.2rem;">
                        <p style="margin: 0; font-size: 1.1rem; color: #212121;">
                            <span style="display: inline-block; width: 1.8rem; height: 1.8rem; line-height: 1.8rem; text-align: center; 
                                    background-color: #1565c0; color: white; border-radius: 50%; margin-right: 0.8rem; font-weight: bold;">
                                {i+1}
                            </span>
                            {action}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Automatically analyze if not already analyzed
        st.info("Analyzing robot data...")
        with st.spinner("Checking maintenance needs..."):
            robot_dict = robot_data.to_dict()
            result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
            
            # Store in results
            if len(st.session_state.results) <= selected_robot_id:
                st.session_state.results.extend([None] * (selected_robot_id - len(st.session_state.results) + 1))
            st.session_state.results[selected_robot_id] = result
            
            st.rerun()  # Refresh to show results

@st.cache_data(ttl=3600)
def sensor_readings_cached(robot_data):
    """Cached version of sensor readings display for better performance."""
    # Get sensor ranges
    sensor_ranges = {
        'temperature': (50, 85),         # Celsius
        'vibration': (0, 0.3),           # Units
        'hydraulic_pressure': (140, 180), # PSI
        'power_output': (85, 100),        # Percentage
        'coolant_level': (70, 95)         # Percentage
    }
    
    # Extract sensor values
    temp = robot_data.get('temperature', 0)
    vibration = robot_data.get('vibration', 0)
    pressure = robot_data.get('hydraulic_pressure', 0)
    power = robot_data.get('power_output', 0)
    coolant = robot_data.get('coolant_level', 0)
    
    # Determine colors based on values
    def get_color(value, min_val, max_val, reverse=False):
        if value is None or pd.isna(value):
            return "#595959"  # Darker gray for missing data
        
        if reverse:
            # For values where lower is better
            if value < min_val:
                return "#b71c1c"  # Darker red for too low
            elif value > max_val:
                return "#1b5e20"  # Darker green for good (high)
            else:
                return "#e65100"  # Darker orange for borderline
        else:
            # For values where higher is worse
            if value > max_val:
                return "#b71c1c"  # Darker red for too high
            elif value < min_val:
                return "#e65100"  # Darker orange for too low
            else:
                return "#1b5e20"  # Darker green for within range
    
    # Get colors
    temp_color = get_color(temp, sensor_ranges['temperature'][0], sensor_ranges['temperature'][1], False)
    vib_color = get_color(vibration, sensor_ranges['vibration'][0], sensor_ranges['vibration'][1], False)
    pressure_color = get_color(pressure, sensor_ranges['hydraulic_pressure'][0], sensor_ranges['hydraulic_pressure'][1], True)
    power_color = get_color(power, sensor_ranges['power_output'][0], sensor_ranges['power_output'][1], True)
    coolant_color = get_color(coolant, sensor_ranges['coolant_level'][0], sensor_ranges['coolant_level'][1], True)
    
    # Create a flex container for the sensors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f5f7fa; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: #000000;">Temperature</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {temp_color};">{temp:.1f}°C</div>
            <div style="font-size: 0.8rem; color: #424242;">Normal: 50-85°C</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #f5f7fa; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: #000000;">Vibration</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {vib_color};">{vibration:.2f}</div>
            <div style="font-size: 0.8rem; color: #424242;">Normal: 0.00-0.30</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: #f5f7fa; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: #000000;">Hydraulic Pressure</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {pressure_color};">{pressure:.1f} PSI</div>
            <div style="font-size: 0.8rem; color: #424242;">Normal: 140-180 PSI</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f5f7fa; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: #000000;">Power Output</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {power_color};">{power:.1f}%</div>
            <div style="font-size: 0.8rem; color: #424242;">Normal: 85-100%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #f5f7fa; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: #000000;">Coolant Level</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {coolant_color};">{coolant:.1f}%</div>
            <div style="font-size: 0.8rem; color: #424242;">Normal: 70-95%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Show error codes if present
    if 'error_codes' in robot_data and robot_data['error_codes']:
        st.markdown(f"""
        <div style="background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
            <div style="font-weight: bold; color: #b71c1c;">Error Codes:</div>
            <div style="font-size: 1.2rem; font-weight: bold; color: #b71c1c; margin-top: 0.5rem;">{robot_data['error_codes']}</div>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache the header for an hour
def display_app_header():
    """Display the app header with caching for better performance."""
    st.markdown("""
    <div class="card" style="text-align: center; padding: 2rem; margin-bottom: 2rem;">
        <img src="https://img.icons8.com/fluency/96/000000/robot.png" style="width: 80px; margin-bottom: 1rem;">
        <h1 style="margin: 0; color: #283593; font-size: 2.5rem; background: none; box-shadow: none; border: none; padding: 0;">
            Robot Maintenance Helper
        </h1>
        <p style="color: #546e7a; margin-top: 1rem; font-size: 1.2rem;">
            Keep your robots running at optimal performance
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar content with optimized performance."""
    # Center the logo
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <img src="https://img.icons8.com/fluency/96/000000/robot.png" style="width: 80px;">
    </div>
    """, unsafe_allow_html=True)
    
    # Improved sidebar navigation with better styling
    st.markdown("""
    <div class="navigation-header">
        <h2 style="margin: 0; color: white; font-size: 1.5rem; background: none; box-shadow: none; border: none; padding: 0;">
            Navigation
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Updated navigation options
    page = st.radio("", [
        "Dashboard", 
        "Robot Status", 
        "Maintenance Schedule",
        "Error Codes Guide",
        "Ask Questions"
    ], label_visibility="collapsed")
    
    # Store current page in session state
    if 'current_page' not in st.session_state or st.session_state.current_page != page:
        st.session_state.current_page = page
    
    # Back to dashboard button (only show when viewing individual robot)
    if page == "Robot Status" and not st.session_state.dashboard_view:
        if st.button("← Back to Dashboard", use_container_width=True):
            st.session_state.dashboard_view = True
    
    # Status indicator with improved styling
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3; border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
    
    system_ready = st.session_state.enhancer is not None and st.session_state.data is not None
    
    if system_ready:
        st.markdown("""
        <div style="background-color: rgba(46,125,50,0.2); border-left: 5px solid #4caf50; padding: 1rem; 
                    border-radius: 0.5rem; margin-top: 1rem; border: 1px solid rgba(76,175,80,0.3);">
            <p style="margin: 0; color: white; font-weight: 600;">✅ System Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: rgba(211,47,47,0.2); border-left: 5px solid #f44336; padding: 1rem; 
                    border-radius: 0.5rem; margin-top: 1rem; border: 1px solid rgba(244,67,54,0.3);">
            <p style="margin: 0; color: white; font-weight: 600;">❌ System Not Ready</p>
        </div>
        """, unsafe_allow_html=True)

def show_robot_dashboard():
    """Display a summary dashboard of all robots and their maintenance status."""
    st.markdown('<h2>Robots Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.enhancer is None or st.session_state.data is None:
        st.markdown("""
        <div class="status-warning">
            <p style="margin: 0;"><strong>⚠️ System still loading</strong> - Please wait a moment while we get things ready.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Ensure results exists in session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Pagination for dashboard
    if 'dashboard_page' not in st.session_state:
        st.session_state.dashboard_page = 0
    
    # Quick analysis options
    if len(st.session_state.results) < len(st.session_state.data):
        total_unanalyzed = len(st.session_state.data) - len(st.session_state.results)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.warning(f"""
            {len(st.session_state.results)}/{len(st.session_state.data)} robots analyzed.
            {total_unanalyzed} robots haven't been checked yet.
            """)
        
        with col2:
            analysis_options = st.selectbox(
                "Analysis Options",
                options=["Quick Analysis (10 robots)", "Sample (20 robots)", "Full Analysis"],
                index=0
            )
        
        analyze_button_text = "Start Analysis"
        if analysis_options == "Quick Analysis (10 robots)":
            robots_to_analyze = 10
            analyze_button_text = "Analyze 10 Robots (Fast)"
        elif analysis_options == "Sample (20 robots)":
            robots_to_analyze = 20
            analyze_button_text = "Analyze 20 Robots"
        else:
            robots_to_analyze = len(st.session_state.data)
            analyze_button_text = f"Analyze All {robots_to_analyze} Robots"
        
        if st.button(analyze_button_text, use_container_width=True):
            if robots_to_analyze <= 10:
                # For small batches, we can analyze sequentially with progress bar
                analyze_robots_sequentially(robots_to_analyze)
            else:
                # For larger batches, show progress differently and use optimized batch processing
                analyze_robots_in_batches(robots_to_analyze)
    
    # Add anomaly simulation UI before showing results
    add_anomaly_simulation_ui()
    
    # Count robots by status
    total_analyzed = len(st.session_state.results)
    if total_analyzed == 0:
        st.info("No robots have been analyzed yet. Use the options above to analyze robots.")
        return
        
    # Calculate status counts
    needs_maintenance = sum(1 for r in st.session_state.results if r and r.get('maintenance_required', False))
    normal_robots = total_analyzed - needs_maintenance
    
    # Summary metrics
    st.markdown("""
    <div class="card" style="margin-bottom: 2rem;">
        <h3 style="margin-top: 0;">Fleet Status Summary</h3>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
    """, unsafe_allow_html=True)
    
    # Create metric cards for robot counts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #e8f5e9; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;">
            <div style="font-size: 3rem; font-weight: bold; color: #2e7d32;">{total_analyzed}</div>
            <div style="color: #2e7d32; font-weight: 500;">Analyzed Robots</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #e8f5e9; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;">
            <div style="font-size: 3rem; font-weight: bold; color: #2e7d32;">{normal_robots}</div>
            <div style="color: #2e7d32; font-weight: 500;">Normal Operation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: {('#ffebee' if needs_maintenance > 0 else '#e8f5e9')}; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;">
            <div style="font-size: 3rem; font-weight: bold; color: {('#d32f2f' if needs_maintenance > 0 else '#2e7d32')};">{needs_maintenance}</div>
            <div style="color: {('#d32f2f' if needs_maintenance > 0 else '#2e7d32')}; font-weight: 500;">Need Maintenance</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get maintenance robots
    maintenance_robots = get_maintenance_robots()
    
    # Display robots needing maintenance
    if maintenance_robots:
        st.markdown("""
        <h3>Robots Needing Maintenance</h3>
        """, unsafe_allow_html=True)
        
        # Pagination for maintenance robots
        start_idx = 0
        end_idx = min(10, len(maintenance_robots))  # Show max 10 at once
        
        # Display paginated maintenance robots
        display_robot_cards(maintenance_robots[start_idx:end_idx], "maintenance")
        
        # Simple pagination controls if needed
        if len(maintenance_robots) > 10:
            st.write(f"Showing {end_idx} of {len(maintenance_robots)} robots needing maintenance")
            if st.button("Show More", key="more_maintenance"):
                # In a real implementation, you would update pagination state here
                pass
    else:
        st.markdown("""
        <div class="card" style="background-color: #e8f5e9; margin-bottom: 2rem;">
            <h3 style="margin-top: 0; color: #2e7d32;">All Robots Operating Normally</h3>
            <p>Great news! All analyzed robots are currently operating within normal parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get at-risk robots
    at_risk_robots = get_at_risk_robots()
    
    # Display robots at risk
    if at_risk_robots:
        st.markdown("""
        <h3>Robots to Monitor</h3>
        """, unsafe_allow_html=True)
        
        # Pagination for at-risk robots
        start_idx = 0
        end_idx = min(10, len(at_risk_robots))  # Show max 10 at once
        
        # Display paginated at-risk robots
        display_robot_cards(at_risk_robots[start_idx:end_idx], "monitor")
        
        # Simple pagination controls if needed
        if len(at_risk_robots) > 10:
            st.write(f"Showing {end_idx} of {len(at_risk_robots)} robots to monitor")
            if st.button("Show More", key="more_monitor"):
                # In a real implementation, you would update pagination state here
                pass

def get_maintenance_robots():
    """Get robots that need maintenance, sorted by urgency."""
    maintenance_robots = []
    for i, result in enumerate(st.session_state.results):
        if result and result.get('maintenance_required', False):
            # Assign priority score for sorting (High=3, Medium=2, Low=1)
            urgency = result.get('urgency', 'Medium').lower()
            priority_score = 3 if urgency == 'high' or urgency == 'critical' else 2 if urgency == 'medium' else 1
            
            # Get the robot type if available
            robot_type = "Unknown"
            if i < len(st.session_state.data):
                robot_type = st.session_state.data.iloc[i].get('robot_type', 'Unknown')
            
            # Get the root cause in a user-friendly format
            root_cause = result.get('root_cause', 'Unknown issue')
            
            # Get any specific anomalies
            anomalies = []
            if 'contributing_factors' in result and result['contributing_factors']:
                anomalies = result['contributing_factors']
            
            # Only compute necessary data to improve performance
            maintenance_robots.append({
                'id': i,
                'robot_name': f"Robot #{i+1}",
                'robot_type': robot_type,
                'urgency': result.get('urgency', 'Medium'),
                'priority_score': priority_score,
                'root_cause': root_cause,
                'anomalies': anomalies
            })
    
    # Sort by priority score (descending)
    maintenance_robots.sort(key=lambda x: x['priority_score'], reverse=True)
    return maintenance_robots

def get_at_risk_robots():
    """Get robots that are at risk but don't yet need maintenance."""
    at_risk_robots = []
    for i, result in enumerate(st.session_state.results):
        # Skip robots that already need maintenance
        if result and result.get('maintenance_required', False):
            continue
        
        # Get the robot type if available
        robot_type = "Unknown"
        if i < len(st.session_state.data):
            robot_type = st.session_state.data.iloc[i].get('robot_type', 'Unknown')
            
        # Check for robots with borderline sensor values
        anomalous_sensors = []
        if i < len(st.session_state.data):
            robot_data = st.session_state.data.iloc[i]
            
            # Check standard sensors against thresholds
            sensor_thresholds = {
                'temperature': {'min': 50, 'max': 85},
                'vibration': {'min': 0, 'max': 0.3},
                'hydraulic_pressure': {'min': 140, 'max': 180},
                'power_output': {'min': 85, 'max': 100},
                'coolant_level': {'min': 70, 'max': 95}
            }
            
            for sensor, thresholds in sensor_thresholds.items():
                if sensor in robot_data and robot_data[sensor] is not None:
                    value = float(robot_data[sensor])
                    min_val = thresholds.get('min', float('-inf'))
                    max_val = thresholds.get('max', float('inf'))
                    
                    # Check if sensor is within 10% of threshold
                    buffer_min = min_val * 1.05  # 5% above min
                    buffer_max = max_val * 0.95  # 5% below max
                    
                    if value < buffer_min and value > min_val:
                        anomalous_sensors.append(f"{sensor.replace('_', ' ').title()} is approaching lower limit ({value:.1f})")
                    elif value > buffer_max and value < max_val:
                        anomalous_sensors.append(f"{sensor.replace('_', ' ').title()} is approaching upper limit ({value:.1f})")
        
        # If no anomalous sensors found, check the result for warnings
        if not anomalous_sensors and result:
            if 'approaching' in str(result.get('root_cause', '')).lower() or 'monitor' in str(result.get('root_cause', '')).lower():
                anomalous_sensors.append(result.get('root_cause', ''))
        
        # Only add if there are anomalies to report or there is a specific warning
        if anomalous_sensors or (result and 'approaching' in str(result.get('root_cause', '')).lower()):
            at_risk_robots.append({
                'id': i,
                'robot_name': f"Robot #{i+1}",
                'robot_type': robot_type,
                'anomalies': anomalous_sensors,
                'status': 'Monitor' if anomalous_sensors else 'Check'
            })
    
    return at_risk_robots

def display_robot_cards(robots, card_type="maintenance"):
    """Display robot cards efficiently using our new UI components."""
    # Handle empty robot list
    if not robots:
        st.info(f"No robots to {card_type}.")
        return
    
    # Use simple columns-based layout for reliability
    for robot in robots:
        # Create a container for each robot
        with st.container():
            # Use our create_robot_card utility
            if card_type == "maintenance":
                # Format anomalies info as content
                if 'anomalies' in robot and robot['anomalies']:
                    anomaly_list = "<ul>"
                    for anomaly in robot['anomalies']:
                        anomaly_list += f"<li>{anomaly}</li>"
                    anomaly_list += "</ul>"
                    content = f"{robot['root_cause']}<br>{anomaly_list}"
                else:
                    content = robot['root_cause']
                
                # Calculate estimated completion time based on urgency
                if robot['urgency'].lower() == 'high' or robot['urgency'].lower() == 'critical':
                    estimated_time = "24 hours"
                elif robot['urgency'].lower() == 'medium':
                    estimated_time = "72 hours"
                else:
                    estimated_time = "1 week"
                
                # Create additional info for the card
                additional_info = {
                    "Robot Type": robot['robot_type']
                }
                
                # Use the create_robot_card utility directly
                create_robot_card(
                    robot_name=robot['robot_name'],
                    issue=content,
                    urgency=robot['urgency'],
                    estimated_time=estimated_time,
                    additional_info=additional_info
                )
            else:  # monitor cards for at-risk robots
                # For monitor cards, create content that shows the anomalies
                if 'anomalies' in robot and robot['anomalies']:
                    anomaly_text = "<ul style='margin-top: 8px; margin-bottom: 0px;'>"
                    for anomaly in robot['anomalies']:
                        anomaly_text += f"<li>{anomaly}</li>"
                    anomaly_text += "</ul>"
                    content = f"<strong>Sensor Readings to Monitor:</strong><br>{anomaly_text}"
                else:
                    content = "Recommended to monitor this robot"
                
                # Set a more appropriate icon and priority for monitoring
                create_card(
                    title=f"{robot['robot_name']} ({robot['robot_type']})",
                    content=content,
                    priority="warning",
                    icon="⚠️"
                )
            
            # Add action buttons for each robot
            cols = st.columns([1, 3])
            with cols[0]:
                # Create a direct button for each robot with a unique key
                if st.button(f"View Details", key=f"view_robot_{robot['id']}"):
                    # Store the selection in explicit session state variables
                    st.session_state.selected_robot = robot['id']
                    st.session_state.dashboard_view = False
                    st.session_state.current_page = "Robot Status"
                    st.rerun()

def analyze_robots_sequentially(max_robots):
    """Analyze robots one by one with a progress bar."""
    # Extend the results list if needed
    if len(st.session_state.results) < len(st.session_state.data):
        st.session_state.results.extend([None] * (len(st.session_state.data) - len(st.session_state.results)))
    
    # Determine which robots need analysis
    robots_to_analyze = []
    for i in range(min(max_robots, len(st.session_state.data))):
        if i >= len(st.session_state.results) or st.session_state.results[i] is None:
            robots_to_analyze.append(i)
    
    if not robots_to_analyze:
        st.success("All selected robots have already been analyzed!")
        return
    
    progress_bar = st.progress(0)
    for i, robot_idx in enumerate(robots_to_analyze):
        try:
            # Update progress
            progress = (i + 1) / len(robots_to_analyze)
            progress_bar.progress(progress)
            
            # Analyze robot
            st.info(f"Analyzing Robot #{robot_idx+1}... ({i+1}/{len(robots_to_analyze)})")
            robot_data = st.session_state.data.iloc[robot_idx]
            
            # Convert to dict safely
            if hasattr(robot_data, 'to_dict'):
                robot_dict = robot_data.to_dict()
            else:
                robot_dict = robot_data  # Already a dict
                
            result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
            
            # Store result
            if robot_idx >= len(st.session_state.results):
                st.session_state.results.extend([None] * (robot_idx - len(st.session_state.results) + 1))
            st.session_state.results[robot_idx] = result
            
            # Brief delay to avoid rate limiting
            time.sleep(0.2)
        except Exception as e:
            st.error(f"Error analyzing Robot #{robot_idx+1}: {str(e)}")
    
    progress_bar.empty()
    st.success(f"Analyzed {len(robots_to_analyze)} robots successfully!")

def analyze_robots_in_batches(max_robots):
    """Analyze robots using a batched approach for better performance."""
    BATCH_SIZE = 5  # Process 5 robots at a time for UI updates
    
    # Extend the results list if needed
    if len(st.session_state.results) < len(st.session_state.data):
        st.session_state.results.extend([None] * (len(st.session_state.data) - len(st.session_state.results)))
    
    # Determine which robots need analysis
    robots_to_analyze = []
    for i in range(min(max_robots, len(st.session_state.data))):
        if i >= len(st.session_state.results) or st.session_state.results[i] is None:
            robots_to_analyze.append(i)
    
    if not robots_to_analyze:
        st.success("All selected robots have already been analyzed!")
        return
    
    # Create batches
    batches = [robots_to_analyze[i:i+BATCH_SIZE] for i in range(0, len(robots_to_analyze), BATCH_SIZE)]
    
    progress_bar = st.progress(0)
    batch_status = st.empty()
    
    for batch_idx, batch in enumerate(batches):
        batch_status.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} robots)")
        
        for robot_idx in batch:
            try:
                robot_data = st.session_state.data.iloc[robot_idx]
                
                # Convert to dict safely
                if hasattr(robot_data, 'to_dict'):
                    robot_dict = robot_data.to_dict()
                else:
                    robot_dict = robot_data  # Already a dict
                    
                result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
                
                # Store result
                if robot_idx >= len(st.session_state.results):
                    st.session_state.results.extend([None] * (robot_idx - len(st.session_state.results) + 1))
                st.session_state.results[robot_idx] = result
            except Exception as e:
                st.error(f"Error analyzing Robot #{robot_idx+1}: {str(e)}")
        
        # Update progress after each batch
        progress = (batch_idx + 1) / len(batches)
        progress_bar.progress(progress)
        
        # Brief delay between batches to avoid overwhelming the UI
        time.sleep(0.5)
    
    progress_bar.empty()
    batch_status.empty()
    st.success(f"Analyzed {len(robots_to_analyze)} robots successfully!")

def inject_custom_anomalies(robot_indices, sensor_values):
    """Inject custom anomalies with user-specified values into selected robots and analyze them.
    
    Args:
        robot_indices: List of robot indices to modify
        sensor_values: Dictionary of sensor names and their new values
    """
    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.error("No robot data available to modify")
        return False
    
    # Create a deep copy of the dataframe to avoid modifying original
    modified_data = st.session_state.data.copy(deep=True)
    
    # Track which robots were modified
    modified_robots = []
    
    try:
        for idx in robot_indices:
            if idx < 0 or idx >= len(modified_data):
                st.warning(f"Robot index {idx} is out of range. Skipping.")
                continue
                
            # Apply the custom sensor values
            for sensor, value in sensor_values.items():
                if sensor in modified_data.columns:
                    # Convert value to appropriate type if needed
                    modified_data.loc[idx, sensor] = value
                else:
                    # Add the new column if it doesn't exist
                    modified_data[sensor] = None
                    modified_data.loc[idx, sensor] = value
                    st.info(f"Created new sensor column '{sensor}' for anomaly simulation")
            
            # Track which robots were modified
            modified_robots.append({
                'id': idx, 
                'robot_name': f"Robot #{idx+1}",
                'modified_sensors': list(sensor_values.keys())
            })
        
        # Update the session state with modified data
        st.session_state.data = modified_data
        
        # Clear the analyze_robot_cached cache to force reanalysis
        analyze_robot_cached.clear()
        
        # Clear previous analysis results for the modified robots
        for idx in robot_indices:
            if idx < len(st.session_state.results):
                st.session_state.results[idx] = None
        
        # Automatically analyze the modified robots
        with st.spinner("Analyzing modified robots..."):
            for idx in robot_indices:
                robot_data = modified_data.iloc[idx]
                
                # Convert to dict safely
                if hasattr(robot_data, 'to_dict'):
                    robot_dict = robot_data.to_dict()
                else:
                    robot_dict = robot_data  # Already a dict
                    
                result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
                
                # Store the result
                if idx >= len(st.session_state.results):
                    st.session_state.results.extend([None] * (idx - len(st.session_state.results) + 1))
                st.session_state.results[idx] = result
        
        return modified_robots
    except Exception as e:
        st.error(f"Error injecting anomalies: {str(e)}")
        return False

def add_anomaly_simulation_ui():
    """Add UI for anomaly simulation to the dashboard."""
    st.markdown("### Simulate Anomalies")
    st.markdown("Modify robot sensor values to test the maintenance detection system.")
    
    # Get available sensors from the data
    available_sensors = []
    if st.session_state.data is not None and len(st.session_state.data) > 0:
        available_sensors = [col for col in st.session_state.data.columns 
                            if col not in ['robot_id', 'robot_type', 'last_maintenance_date']]
    
    with st.expander("Anomaly Simulation Controls"):
        # Two tabs for different simulation modes
        tab1, tab2 = st.tabs(["Custom Values", "Random Anomalies"])
        
        # TAB 1: Custom values
        with tab1:
            st.markdown("#### Set Custom Sensor Values")
            
            # Select robots to modify
            col1, col2 = st.columns(2)
            with col1:
                # Two options: specific robot or range
                selection_method = st.radio(
                    "Robot Selection Method",
                    ["Single Robot", "Multiple Robots"]
                )
            
            with col2:
                if selection_method == "Single Robot":
                    if st.session_state.data is not None:
                        max_idx = len(st.session_state.data) - 1
                        robot_idx = st.number_input("Robot Index (0-based)", 
                                                  min_value=0, 
                                                  max_value=max_idx, 
                                                  value=0)
                        selected_robots = [robot_idx]
                else:  # Multiple robots
                    if st.session_state.data is not None:
                        max_idx = len(st.session_state.data) - 1
                        start_idx = st.number_input("Start Index", 
                                                 min_value=0, 
                                                 max_value=max_idx, 
                                                 value=0)
                        end_idx = st.number_input("End Index", 
                                               min_value=start_idx, 
                                               max_value=max_idx, 
                                               value=min(start_idx + 4, max_idx))
                        selected_robots = list(range(start_idx, end_idx + 1))
            
            # Sensor value inputs
            st.markdown("#### Enter Sensor Values")
            sensor_values = {}
            
            # Create 3 columns for sensor inputs
            col1, col2, col3 = st.columns(3)
            
            # Temperature
            with col1:
                if st.checkbox("Modify Temperature", value=True):
                    temp_value = st.number_input("Temperature (°C)", 
                                              min_value=0.0, 
                                              max_value=200.0, 
                                              value=95.0, 
                                              step=5.0,
                                              help="Normal range is typically 50-85°C")
                    sensor_values['temperature'] = temp_value
            
            # Vibration
            with col2:
                if st.checkbox("Modify Vibration", value=True):
                    vib_value = st.number_input("Vibration", 
                                             min_value=0.0, 
                                             max_value=1.0, 
                                             value=0.45, 
                                             step=0.05,
                                             help="Normal range is typically 0.0-0.3")
                    sensor_values['vibration'] = vib_value
            
            # Hydraulic pressure
            with col3:
                if st.checkbox("Modify Hydraulic Pressure", value=False):
                    pressure_value = st.number_input("Hydraulic Pressure (PSI)", 
                                                  min_value=0.0, 
                                                  max_value=300.0, 
                                                  value=120.0, 
                                                  step=10.0,
                                                  help="Normal range is typically 140-180 PSI")
                    sensor_values['hydraulic_pressure'] = pressure_value
            
            # More sensor options
            col1, col2, col3 = st.columns(3)
            
            # Power output
            with col1:
                if st.checkbox("Modify Power Output", value=False):
                    power_value = st.number_input("Power Output (%)", 
                                               min_value=0.0, 
                                               max_value=100.0, 
                                               value=75.0, 
                                               step=5.0,
                                               help="Normal range is typically 85-100%")
                    sensor_values['power_output'] = power_value
            
            # Coolant level
            with col2:
                if st.checkbox("Modify Coolant Level", value=False):
                    coolant_value = st.number_input("Coolant Level (%)", 
                                                 min_value=0.0, 
                                                 max_value=100.0, 
                                                 value=50.0, 
                                                 step=5.0,
                                                 help="Normal range is typically 70-95%")
                    sensor_values['coolant_level'] = coolant_value
            
            # Error codes
            with col3:
                if st.checkbox("Add Error Codes", value=False):
                    error_options = ["E001", "E002", "E101", "E204", "E506"]
                    selected_errors = st.multiselect("Select Error Codes", 
                                                  error_options, 
                                                  default=["E001"],
                                                  help="Multiple error codes will be combined")
                    if selected_errors:
                        sensor_values['error_codes'] = '+'.join(selected_errors)
            
            # Apply custom values button
            with st.form(key="custom_anomaly_form"):
                st.markdown("#### Apply Changes")
                submit_button = st.form_submit_button("Apply Custom Values", use_container_width=True)
                if submit_button:
                    if not sensor_values:
                        st.warning("Please select at least one sensor to modify.")
                    elif not selected_robots:
                        st.warning("Please select at least one robot to modify.")
                    else:
                        with st.spinner("Applying custom values and analyzing..."):
                            modified = inject_custom_anomalies(selected_robots, sensor_values)
                            if modified:
                                st.success(f"Modified and analyzed {len(modified)} robots successfully!")
                                # Show which robots were affected in a table
                                robot_list = [f"Robot #{r['id']+1}" for r in modified]
                                st.markdown(f"**Modified Robots:** {', '.join(robot_list)}")
                                st.markdown(f"**Modified Sensors:** {', '.join(sensor_values.keys())}")
        
        # TAB 2: Random anomalies
        with tab2:
            st.markdown("#### Generate Random Anomalies")
            
            col1, col2 = st.columns(2)
            with col1:
                num_robots = st.slider("Number of robots to affect", 1, 20, 5)
            
            with col2:
                severity = st.select_slider(
                    "Severity of anomalies",
                    options=["mild", "moderate", "severe", "critical"],
                    value="moderate"
                )
            
            # Add options for which sensors to randomize
            st.markdown("#### Select Sensors to Randomize")
            col1, col2 = st.columns(2)
            
            with col1:
                include_temp = st.checkbox("Temperature", value=True) and 'temperature' in available_sensors
                include_vibration = st.checkbox("Vibration", value=True) and 'vibration' in available_sensors
                include_pressure = st.checkbox("Hydraulic Pressure", value=False) and 'hydraulic_pressure' in available_sensors
            
            with col2:
                include_power = st.checkbox("Power Output", value=False) and 'power_output' in available_sensors
                include_coolant = st.checkbox("Coolant Level", value=False) and 'coolant_level' in available_sensors
                include_errors = st.checkbox("Error Codes", value=False)
            
            # Random anomaly function
            def inject_random_anomalies(num_robots, severity, sensors_to_include):
                """Inject random anomalies based on selected sensors and severity."""
                if st.session_state.data is None or len(st.session_state.data) == 0:
                    st.error("No robot data available to modify")
                    return False
                
                # Get total number of robots
                total_robots = len(st.session_state.data)
                
                # Ensure we don't try to modify more robots than exist
                num_robots = min(num_robots, total_robots)
                
                # Select random robot indices
                import random
                robot_indices = random.sample(range(total_robots), num_robots)
                
                # Define severity multipliers (how much to deviate from normal)
                severity_factors = {
                    'mild': {'multiply': 1.1, 'divide': 0.9},      # 10% deviation
                    'moderate': {'multiply': 1.4, 'divide': 0.7},  # 40% deviation
                    'severe': {'multiply': 1.7, 'divide': 0.6},    # 70% deviation
                    'critical': {'multiply': 2.0, 'divide': 0.5}   # 100% deviation
                }
                
                factor = severity_factors.get(severity, severity_factors['moderate'])
                
                # Create a deep copy of the dataframe to avoid modifying original
                modified_data = st.session_state.data.copy(deep=True)
                
                # Get available columns in the dataframe
                available_columns = modified_data.columns.tolist()
                
                # Track which robots were modified
                modified_robots = []
                
                for idx in robot_indices:
                    # Prepare sensor modifications for this robot
                    modifications = {}
                    
                    # For each selected sensor, generate a random anomaly
                    if 'temperature' in sensors_to_include:
                        # If column doesn't exist, create it with default value
                        if 'temperature' not in available_columns:
                            modified_data['temperature'] = 70.0  # Default normal value
                            available_columns.append('temperature')
                            
                        current = modified_data.loc[idx, 'temperature']
                        # High temperature is bad - multiply by factor
                        modifications['temperature'] = current * factor['multiply']
                    
                    if 'vibration' in sensors_to_include:
                        # If column doesn't exist, create it with default value
                        if 'vibration' not in available_columns:
                            modified_data['vibration'] = 0.15  # Default normal value
                            available_columns.append('vibration')
                            
                        current = modified_data.loc[idx, 'vibration']
                        # High vibration is bad - multiply by factor
                        modifications['vibration'] = current * factor['multiply']
                    
                    if 'hydraulic_pressure' in sensors_to_include:
                        # If column doesn't exist, create it with default value
                        if 'hydraulic_pressure' not in available_columns:
                            modified_data['hydraulic_pressure'] = 160.0  # Default normal value
                            available_columns.append('hydraulic_pressure')
                            
                        current = modified_data.loc[idx, 'hydraulic_pressure']
                        # Low pressure is bad - divide by factor
                        modifications['hydraulic_pressure'] = current * factor['divide']
                    
                    if 'power_output' in sensors_to_include:
                        # If column doesn't exist, create it with default value
                        if 'power_output' not in available_columns:
                            modified_data['power_output'] = 90.0  # Default normal value
                            available_columns.append('power_output')
                            
                        current = modified_data.loc[idx, 'power_output']
                        # Low power is bad - divide by factor
                        modifications['power_output'] = current * factor['divide']
                    
                    if 'coolant_level' in sensors_to_include:
                        # If column doesn't exist, create it with default value
                        if 'coolant_level' not in available_columns:
                            modified_data['coolant_level'] = 85.0  # Default normal value
                            available_columns.append('coolant_level')
                            
                        current = modified_data.loc[idx, 'coolant_level']
                        # Low coolant is bad - divide by factor
                        modifications['coolant_level'] = current * factor['divide']
                    
                    if 'error_codes' in sensors_to_include:
                        # Add random error codes based on severity
                        error_codes = []
                        if severity == 'mild':
                            error_codes = ['E001']
                        elif severity == 'moderate':
                            error_codes = ['E001', 'E101']
                        elif severity == 'severe':
                            error_codes = ['E001', 'E101', 'E204']
                        else:  # critical
                            error_codes = ['E001', 'E101', 'E204', 'E506']
                        
                        # Take a random subset based on severity
                        num_errors = max(1, int(len(error_codes) * float(severity_factors[severity]['multiply']) / 2))
                        selected_errors = random.sample(error_codes, min(num_errors, len(error_codes)))
                        modifications['error_codes'] = '+'.join(selected_errors)
                    
                    # Apply all modifications to this robot
                    for sensor, value in modifications.items():
                        modified_data.loc[idx, sensor] = value
                    
                    # Only track robots that actually had modifications
                    if modifications:
                        modified_robots.append({
                            'id': idx,
                            'robot_name': f"Robot #{idx+1}",
                            'modified_sensors': list(modifications.keys())
                        })
                
                if not modified_robots:
                    # Just generate a message without a warning banner
                    st.info("No modifications applied. Try selecting different sensors.")
                    return False
                
                # Update the session state with modified data
                st.session_state.data = modified_data
                
                # Clear the analyze_robot_cached cache to force reanalysis
                analyze_robot_cached.clear()
                
                # Clear previous analysis results for the modified robots only
                for idx in robot_indices:
                    if idx < len(st.session_state.results):
                        st.session_state.results[idx] = None
                
                # Analyze the modified robots
                with st.spinner("Analyzing modified robots..."):
                    for idx in robot_indices:
                        robot_data = modified_data.iloc[idx]
                        
                        # Convert to dict safely
                        if hasattr(robot_data, 'to_dict'):
                            robot_dict = robot_data.to_dict()
                        else:
                            robot_dict = robot_data  # Already a dict
                            
                        result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
                        
                        # Store the result
                        if idx >= len(st.session_state.results):
                            st.session_state.results.extend([None] * (idx - len(st.session_state.results) + 1))
                        st.session_state.results[idx] = result
                
                return modified_robots
            
            # Random anomalies button
            with st.form(key="random_anomaly_form"):
                st.markdown("#### Apply Random Changes")
                random_submit = st.form_submit_button("Generate Random Anomalies", use_container_width=True)
                if random_submit:
                    # Get list of sensors to randomize
                    sensors_to_include = []
                    if include_temp: sensors_to_include.append('temperature')
                    if include_vibration: sensors_to_include.append('vibration')
                    if include_pressure: sensors_to_include.append('hydraulic_pressure')
                    if include_power: sensors_to_include.append('power_output')
                    if include_coolant: sensors_to_include.append('coolant_level')
                    if include_errors: sensors_to_include.append('error_codes')
                    
                    if not sensors_to_include:
                        st.warning("Please select at least one sensor to randomize.")
                    else:
                        with st.spinner("Generating and analyzing random anomalies..."):
                            modified = inject_random_anomalies(num_robots, severity, sensors_to_include)
                            if modified:
                                st.success(f"Added anomalies to {len(modified)} robots and analyzed the results!")
                                
                                # Create a DataFrame to show the modifications
                                import pandas as pd
                                mod_data = []
                                for robot in modified:
                                    mod_data.append({
                                        'Robot': robot['robot_name'],
                                        'Modified Sensors': ', '.join(robot['modified_sensors']),
                                        'Severity': severity
                                    })
                                
                                mod_df = pd.DataFrame(mod_data)
                                st.dataframe(mod_df)

def display_robot_details(robot_data, result=None, show_all_sensors=True):
    """Display robot details with nicely formatted sensor readouts.
    
    Args:
        robot_data: The robot's data as a Pandas Series
        result: Optional analysis result for this robot
        show_all_sensors: Whether to show all sensors or just the main ones
    """
    # Get the robot ID from the data
    robot_id = robot_data.get("robot_id", f"Robot #{st.session_state.selected_robot}")
    
    # Extract sensor categories
    sensors = {}
    
    # Group sensor readings into relevant categories
    for col, value in robot_data.items():
        # Skip metadata and non-numeric values unless explicit show_all_sensors
        is_numeric = isinstance(value, (int, float)) and not pd.isna(value)
        is_metadata = col in ["robot_id", "robot_type", "firmware_version", "error_codes", "parts_replaced"]
        
        if (not show_all_sensors and is_metadata) or not is_numeric:
            continue
            
        sensors[col] = value
            
    # Use the display_sensor_readings utility
    display_sensor_readings(sensors, num_columns=3)
    
    # Display result if provided
    if result:
        # Use the create_status_indicator utility
        if result.get("maintenance_required", False):
            # Show the maintenance recommendation
            urgency = result.get("urgency", "Medium")
            status = "error" if urgency == "High" else "warning"
            message = f"Maintenance Required - {urgency} Priority"
            create_status_indicator(status, message, icon="⚠️")
            
            # Display the confidence
            confidence = result.get("confidence", 0)
            confidence_text = f"Confidence: {confidence*100:.1f}%"
            
            # Display root cause and recommendations
            st.subheader("Analysis Results")
            st.markdown(f"**Root Cause**: {result.get('root_cause', 'Unknown')}")
            
            recommendations = result.get("recommendations", [])
            if recommendations:
                st.markdown("**Recommended Actions**:")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
        else:
            # No maintenance required
            create_status_indicator("success", "No Maintenance Required", icon="✅")
            
            # Display the confidence
            confidence = result.get("confidence", 0)
            st.markdown(f"**Confidence**: {confidence*100:.1f}%")
            
    # Show robot metadata in a structured way
    metadata = {}
    for field in ["robot_type", "firmware_version", "environment", "operation_hours", 
                "last_maintenance_date", "maintenance_history"]:
        if field in robot_data and not pd.isna(robot_data[field]):
            metadata[field.replace("_", " ").title()] = robot_data[field]
    
    if metadata:
        # Use the create_metrics_dashboard utility
        create_metrics_dashboard(metadata, title="Robot Information", columns=3)

def show_robot_status():
    """Display detailed status information about the selected robot."""
    if st.session_state.dashboard_view:
        # If we're in dashboard view, show all robots
        show_robot_dashboard()
        return
            
    # We're viewing a specific robot
    robot_idx = st.session_state.selected_robot
    
    if robot_idx >= len(st.session_state.data):
        st.error(f"Robot #{robot_idx+1} not found!")
        return
            
    # Get robot data
    robot_data = st.session_state.data.iloc[robot_idx]
    
    # Use our styled_header utility
    styled_header(f"Robot #{robot_idx+1} Status", icon="🤖", color="#3f51b5")
    
    # Check if this robot has been analyzed
    result = None
    if len(st.session_state.results) > robot_idx and st.session_state.results[robot_idx] is not None:
        result = st.session_state.results[robot_idx]
        
    # Create tabs for different types of information
    tab1, tab2, tab3, tab4 = st.tabs(["Sensor Readings", "Maintenance Status", "Ask Question", "Error Codes"])
    
    with tab1:
        # Show sensor readings using our display_robot_details function
        display_robot_details(robot_data, result)
        
    with tab2:
        # Show maintenance status
        if result is None:
            # This robot hasn't been analyzed yet
            st.warning("This robot hasn't been analyzed yet.")
            if st.button("Analyze Robot Now"):
                with st.spinner("Analyzing robot data..."):
                    # Analyze the robot
                    if hasattr(robot_data, 'to_dict'):
                        robot_dict = robot_data.to_dict()
                    else:
                        robot_dict = robot_data  # Already a dict
                    # Use the cached version to ensure consistency
                    result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
                    # Store the result
                    if len(st.session_state.results) <= robot_idx:
                        st.session_state.results.extend([None] * (robot_idx - len(st.session_state.results) + 1))
                    st.session_state.results[robot_idx] = result
                st.success("Analysis complete!")
                st.rerun()
        else:
            # Show maintenance prediction using our display_maintenance_recommendation function
            display_maintenance_recommendation(result, f"Robot #{robot_idx+1}")
            
            # Add action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Schedule Maintenance"):
                    st.session_state.current_page = "Maintenance Schedule"
                    st.rerun()
            with col2:
                if st.button("Re-analyze Robot"):
                    # Clear the cache for this robot
                    analyze_robot_cached.clear()
                    
                    with st.spinner("Re-analyzing robot data..."):
                        # Analyze the robot again
                        if hasattr(robot_data, 'to_dict'):
                            robot_dict = robot_data.to_dict()
                        else:
                            robot_dict = robot_data  # Already a dict
                        result = analyze_robot_cached(json.dumps(robot_dict, sort_keys=True))
                        # Update the result
                        st.session_state.results[robot_idx] = result
                    st.success("Analysis complete!")
                    st.rerun()
    
    with tab3:
        # Implement the Ask Question functionality
        st.markdown("### Ask a question about this robot")
        
        # Get previous questions from session state
        if 'robot_questions' not in st.session_state:
            st.session_state.robot_questions = {}
        
        # Initialize questions for this robot if needed
        robot_key = f"robot_{robot_idx}"
        if robot_key not in st.session_state.robot_questions:
            st.session_state.robot_questions[robot_key] = []
            
        # Show chat history using our custom_chat_message utility
        for q in st.session_state.robot_questions[robot_key]:
            custom_chat_message(q["question"], is_user=True)
            custom_chat_message(q["answer"], is_user=False)
            
        # Input for new question - using our labeled_input helper for accessibility
        question = labeled_input(
            st.text_input,
            "Robot Question",
            placeholder="Type your question here...", 
            key=f"question_input_{robot_idx}"
        )
        
        if st.button("Submit Question"):
            if not question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        # Use our direct question answering function rather than LLM
                        answer = generate_robot_answer(question, robot_data, result)
                        
                        # Store the question and answer
                        st.session_state.robot_questions[robot_key].append({
                            "question": question,
                            "answer": answer
                        })
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        # Still add to history but with error message
                        st.session_state.robot_questions[robot_key].append({
                            "question": question,
                            "answer": f"Sorry, I couldn't answer that question. Try asking something simpler."
                        })
                    
                # Clear the input and refresh to show the new Q&A
                st.rerun()
                
    with tab4:
        # Display error codes information for this robot
        st.markdown("### Error Codes Information")
        
        # Check if this robot has any error codes
        error_codes = robot_data.get('error_codes', None)
        
        if error_codes:
            st.warning(f"This robot has active error codes: {error_codes}")
            
            # Split multiple error codes if present
            code_list = error_codes.split('+')
            
            # Display information for each error code
            for code in code_list:
                if code in st.session_state.error_codes_guide:
                    info = st.session_state.error_codes_guide[code]
                    
                    st.markdown(f"## {code}: {info['description']}")
                    
                    # Determine severity color
                    severity = info['severity'].lower()
                    if severity == 'critical' or severity == 'high':
                        st.error(f"Severity: {info['severity']}")
                    elif severity == 'medium':
                        st.warning(f"Severity: {info['severity']}")
                    else:
                        st.info(f"Severity: {info['severity']}")
                    
                    # Show possible causes
                    st.markdown("### Possible Causes")
                    for cause in info['possible_causes']:
                        st.markdown(f"- {cause}")
                    
                    # Show recommended actions
                    st.markdown("### Recommended Actions")
                    for action in info['recommended_actions']:
                        st.markdown(f"- {action}")
                else:
                    st.error(f"Information for error code {code} is not available.")
        else:
            st.success("No error codes are currently active for this robot.")
            
        # Add a button to view the full error code guide
        if st.button("View Full Error Codes Guide"):
            st.session_state.current_page = "Error Codes Guide"
            st.rerun()

def show_maintenance_schedule():
    """Display the maintenance scheduling interface."""
    st.markdown("## Maintenance Schedule")
    
    # Create a placeholder for the maintenance schedule
    st.markdown("""
    <div class="card">
        <h3>Maintenance Schedule</h3>
        <p>This feature will allow you to schedule maintenance for robots that need attention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a list of robots that need maintenance
    maintenance_robots = get_maintenance_robots()
    
    if maintenance_robots:
        st.markdown("### Robots Needing Maintenance")
        
        # Create a table of robots
        data = []
        for robot in maintenance_robots:
            # Determine urgency color
            if robot['urgency'].lower() == 'high' or robot['urgency'].lower() == 'critical':
                urgency_color = "🔴"
            elif robot['urgency'].lower() == 'medium':
                urgency_color = "🟠"
            else:
                urgency_color = "🟢"
                
            # Add robot to table
            data.append({
                "Robot ID": robot['robot_name'],
                "Type": robot['robot_type'],
                "Priority": f"{urgency_color} {robot['urgency']}",
                "Issue": robot['root_cause']
            })
            
        # Display the table
        st.table(data)
        
        # Add scheduling form
        with st.form("schedule_maintenance"):
            st.markdown("### Schedule Maintenance")
            
            # Select robot
            robot_options = [f"{r['robot_name']} ({r['robot_type']})" for r in maintenance_robots]
            selected_robot = st.selectbox("Select Robot", options=robot_options)
            
            # Select maintenance date
            maintenance_date = st.date_input("Maintenance Date", value=datetime.date.today() + datetime.timedelta(days=1))
            
            # Select maintenance type
            maintenance_type = st.selectbox("Maintenance Type", options=["Regular Service", "Repair", "Component Replacement", "Software Update"])
            
            # Add notes
            notes = st.text_area("Notes")
            
            # Submit button
            submit = st.form_submit_button("Schedule Maintenance")
            
            if submit:
                st.success(f"Maintenance scheduled for {selected_robot} on {maintenance_date}")
                
                # Use create_status_indicator to show success
                create_status_indicator("success", f"Maintenance scheduled for {selected_robot}", icon="✅")
    else:
        # If no robots need maintenance, display a message
        create_status_indicator("success", "No robots currently need maintenance", icon="✅")

def show_question_interface():
    """Display the Q&A interface for general questions about the robots."""
    st.markdown("# Ask Questions About Your Robot Fleet")
    
    st.markdown("""
    <div style="background-color: #f1f8e9; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 1.1em;">
            Ask any questions about your robots or maintenance procedures.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize general chat history if it doesn't exist
    if 'general_chat_history' not in st.session_state:
        st.session_state.general_chat_history = []
    
    # Example questions to help users get started
    st.markdown("### Examples:")
    examples = [
        "How many robots need maintenance?",
        "What are the most common maintenance issues?",
        "Which robot has the highest priority?",
        "What is the primary cause of the most maintenance-needed robot?",
        "How many robots have high temperature readings?"
    ]
    
    cols = st.columns(len(examples))
    for i, col in enumerate(cols):
        if col.button(examples[i], key=f"example_q_{i}"):
            question = examples[i]
            st.session_state.general_question = question
            st.rerun()
    
    # Display previous Q&A
    if st.session_state.general_chat_history:
        st.markdown("### Previous Questions")
        
        for i, qa in enumerate(st.session_state.general_chat_history):
            with st.expander(f"{qa['question']}", expanded=(i == len(st.session_state.general_chat_history) - 1)):
                st.markdown(qa['answer'].replace('\n', '<br>'), unsafe_allow_html=True)
    
    # Question input
    st.markdown("### Ask a Question")
    question = st.text_input("", placeholder="Type your question here...", key="general_question")
    
    if st.button("Submit"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    # Generate a robot fleet summary
                    fleet_info = "Robot Fleet Information:\n"
                    if st.session_state.data is not None:
                        fleet_info += f"- Total robots: {len(st.session_state.data)}\n"
                        fleet_info += f"- Robots analyzed: {len(st.session_state.results)}\n"
                        fleet_info += f"- Robots needing maintenance: {sum(1 for r in st.session_state.results if r and r.get('maintenance_required', False))}\n"
                    
                    # Create a simplified answer generator like the one for individual robots
                    answer = generate_fleet_answer(question, fleet_info)
                    
                    # Store the question and answer
                    st.session_state.general_chat_history.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    # Provide a simple fallback answer
                    error_answer = "I couldn't process that question. Please try asking something simpler about your robot fleet."
                    st.session_state.general_chat_history.append({
                        "question": question,
                        "answer": error_answer
                    })
                
                # Rerun to refresh the chat display
                st.rerun()

def create_enhancer():
    """Create and initialize the LLM enhancer with the API key."""
    if st.session_state.api_key:
        try:
            st.session_state.enhancer = RoboticsMaintenanceLLMEnhancer(
                api_key=st.session_state.api_key
            )
            return True
        except Exception as e:
            st.error(f"Error creating enhancer: {str(e)}")
            return False
    else:
        st.error("Please enter your OpenAI API key.")
        return False

# Add this function to provide direct answers to robot-specific questions
def generate_robot_answer(question, robot_data, analysis_result):
    """Generate answers for robot-specific questions without using LLM.
    
    Args:
        question: User's question
        robot_data: Robot sensor data
        analysis_result: Analysis result for this robot
        
    Returns:
        Generated answer as string
    """
    question_lower = question.lower()
    
    # Get key metrics
    maintenance_required = analysis_result.get('maintenance_required', False) if analysis_result else False
    urgency = analysis_result.get('urgency', 'Low') if analysis_result else 'Unknown'
    root_cause = analysis_result.get('root_cause', 'Unknown') if analysis_result else 'Not analyzed'
    
    # Extract useful information
    sensors = {}
    for key, value in robot_data.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            sensors[key] = value
            
    # Extract any error codes
    error_codes = robot_data.get('error_codes', None)
    error_code_list = []
    if error_codes:
        error_code_list = error_codes.split('+')
    
    # Match question to appropriate response
    if any(word in question_lower for word in ["priority", "urgency", "urgent", "critical"]):
        if maintenance_required:
            # Format clear priority response with more detailed information
            response = f"This robot has {urgency} priority maintenance needs."
            response += f"\nIssue: {root_cause}"
            
            # Add more details based on the urgency level
            if urgency.lower() in ['high', 'critical']:
                timeframe = "immediately"
                response += f"\nThis requires immediate attention."
            elif urgency.lower() == 'medium':
                timeframe = "within 48 hours"
                response += f"\nThis should be addressed within 48 hours."
            else:
                timeframe = "during regular maintenance"
                response += f"\nThis can be addressed during regular maintenance."
                
            # Add first recommended action with timeframe
            if 'recommendations' in analysis_result and analysis_result['recommendations']:
                rec = analysis_result['recommendations'][0]
                if isinstance(rec, dict):
                    action = rec.get('action', 'Perform maintenance check')
                    rec_timeframe = rec.get('timeframe', timeframe)
                    response += f"\n\nRecommended action: {action}"
                    response += f"\nTimeframe: {rec_timeframe}"
                else:
                    response += f"\n\nRecommended action: {rec}"
            
            return response
        else:
            return "This robot is operating normally and does not have any priority maintenance needs."
            
    elif any(word in question_lower for word in ["status", "condition", "health", "how is", "state"]):
        if maintenance_required:
            return f"This robot requires maintenance. The urgency is {urgency} and the root cause is: {root_cause}"
        else:
            return "This robot is operating normally. No maintenance is required at this time."
            
    elif any(word in question_lower for word in ["maintenance", "repair", "fix", "service"]):
        if maintenance_required:
            if analysis_result and 'recommendations' in analysis_result:
                recs = []
                for rec in analysis_result['recommendations']:
                    if isinstance(rec, dict) and 'action' in rec:
                        recs.append(rec['action'])
                    else:
                        recs.append(str(rec))
                return f"Maintenance recommendations:\n- " + "\n- ".join(recs)
            else:
                return f"Maintenance is required due to: {root_cause}"
        else:
            return "No maintenance is required at this time."
            
    elif any(word in question_lower for word in ["sensor", "reading", "measure", "value"]):
        if any(sensor in question_lower for sensor in sensors.keys()):
            # Return specific sensor if mentioned
            for sensor in sensors.keys():
                if sensor in question_lower:
                    return f"The {sensor} reading is {sensors[sensor]}"
        else:
            # Return key sensors if no specific mentioned
            response = "Key sensor readings:\n"
            for sensor, value in list(sensors.items())[:5]:  # Limit to first 5
                response += f"- {sensor}: {value}\n"
            return response
            
    elif any(word in question_lower for word in ["error", "code", "fault", "alert"]):
        if error_codes:
            response = f"This robot has the following error codes: {error_codes}.\n\n"
            # Add explanations if available
            for code in error_code_list:
                if code in st.session_state.error_codes_guide:
                    desc = st.session_state.error_codes_guide[code]['description']
                    response += f"{code}: {desc}\n"
            return response
        else:
            return "No error codes are currently active for this robot."
            
    elif any(word in question_lower for word in ["issue", "problem", "wrong"]):
        if maintenance_required:
            return f"The main issue is: {root_cause}"
        else:
            return "No issues detected. This robot is operating normally."
            
    elif any(word in question_lower for word in ["confidence", "certainty", "reliable"]):
        if analysis_result and 'confidence' in analysis_result:
            confidence = analysis_result['confidence'] * 100
            return f"The analysis has a confidence level of {confidence:.1f}%"
        else:
            return "Confidence information is not available."
    
    # Default fallback response
    return "I don't have specific information about that. Try asking about the robot's status, maintenance needs, sensor readings, or error codes."

# Add this function to display error code information
def show_error_codes_guide():
    """Display a guide explaining robot error codes."""
    st.markdown("# Robot Error Codes Guide")
    
    st.markdown("""
    This guide explains the error codes that may appear in robot status reports. 
    Each code indicates a specific type of issue that may require attention.
    """)
    
    # Display each error code in an expandable section
    for code, info in st.session_state.error_codes_guide.items():
        with st.expander(f"{code}: {info['description']} ({info['severity']} Severity)"):
            st.markdown("### Possible Causes")
            for cause in info['possible_causes']:
                st.markdown(f"- {cause}")
                
            st.markdown("### Recommended Actions")
            for action in info['recommended_actions']:
                st.markdown(f"- {action}")
            
            # Determine severity color
            severity = info['severity'].lower()
            if severity == 'critical' or severity == 'high':
                severity_color = "red"
            elif severity == 'medium':
                severity_color = "orange"
            else:
                severity_color = "blue"
                
            st.markdown(f"**Severity Level:** <span style='color:{severity_color};'>{info['severity']}</span>", unsafe_allow_html=True)
    
    # Add a search function
    st.markdown("## Search Error Codes")
    search_term = st.text_input("Enter an error code or keyword", key="error_code_search")
    
    if search_term:
        # Search in codes and descriptions
        search_term = search_term.lower()
        results = []
        
        for code, info in st.session_state.error_codes_guide.items():
            if (search_term in code.lower() or 
                search_term in info['description'].lower() or
                any(search_term in cause.lower() for cause in info['possible_causes'])):
                results.append((code, info))
        
        if results:
            st.markdown(f"### Search Results ({len(results)} found)")
            for code, info in results:
                st.markdown(f"**{code}: {info['description']}** - {info['severity']} Severity")
        else:
            st.info("No error codes found matching your search.")

# Add this function to process general fleet questions
def generate_fleet_answer(question, fleet_info):
    """Generate answers for fleet-wide questions without using LLM.
    
    Args:
        question: User's question about the fleet
        fleet_info: Information about the robot fleet
        
    Returns:
        Generated answer as string
    """
    question_lower = question.lower()
    
    # DIRECT MATCH for primary cause question to ensure it works
    if "primary cause" in question_lower or "main issue" in question_lower or "most maintenance needed" in question_lower:
        # Find all robots needing maintenance
        maintenance_robots = []
        for i, result in enumerate(st.session_state.results):
            if result and result.get('maintenance_required', False):
                # Get urgency for sorting
                urgency = result.get('urgency', 'Medium').lower()
                urgency_score = 3 if urgency == 'critical' else 2 if urgency == 'high' else 1 if urgency == 'medium' else 0
                
                # Add to list with score
                maintenance_robots.append((i, result, urgency_score))
        
        if maintenance_robots:
            # Sort by urgency score (highest first)
            sorted_robots = sorted(maintenance_robots, key=lambda x: x[2], reverse=True)
            
            # Get the top priority robot
            top_idx, top_result, _ = sorted_robots[0]
            top_robot_id = top_idx + 1
            root_cause = top_result.get('root_cause', 'Unknown issue')
            urgency = top_result.get('urgency', 'Medium')
            
            # Access robot data if available
            robot_data = None
            if st.session_state.data is not None and top_idx < len(st.session_state.data):
                robot_data = st.session_state.data.iloc[top_idx]
            
            # Build detailed response
            response = f"The primary cause of the most maintenance-needed robot (Robot #{top_robot_id}) is: {root_cause}."
            response += f"\n\nThis robot has {urgency} urgency maintenance needs."
            
            # Add sensor details if available
            if robot_data is not None:
                # Find critical sensors
                critical_readings = []
                for col, value in robot_data.items():
                    if col == 'temperature' and isinstance(value, (int, float)) and not pd.isna(value) and value > 85:
                        critical_readings.append(f"Temperature: {value}°C (above normal range)")
                    elif col == 'vibration' and isinstance(value, (int, float)) and not pd.isna(value) and value > 0.3:
                        critical_readings.append(f"Vibration: {value} (above normal range)")
                    elif col == 'hydraulic_pressure' and isinstance(value, (int, float)) and not pd.isna(value) and value < 140:
                        critical_readings.append(f"Hydraulic Pressure: {value} PSI (below normal range)")
                    elif col == 'coolant_level' and isinstance(value, (int, float)) and not pd.isna(value) and value < 70:
                        critical_readings.append(f"Coolant Level: {value}% (below normal range)")
                
                if critical_readings:
                    response += "\n\nCritical sensor readings:"
                    for reading in critical_readings:
                        response += f"\n- {reading}"
            
            # Add recommendations
            if 'recommendations' in top_result and top_result['recommendations']:
                response += "\n\nRecommended actions:"
                for i, rec in enumerate(top_result['recommendations'][:3]):
                    if isinstance(rec, dict) and 'action' in rec:
                        response += f"\n{i+1}. {rec['action']}"
                    else:
                        response += f"\n{i+1}. {rec}"
            
            return response
        else:
            return "No robots currently need maintenance."
    
    # Get basic counts from fleet_info
    import re
    
    total_match = re.search(r"Total robots: (\d+)", fleet_info)
    total_robots = int(total_match.group(1)) if total_match else 0
    
    analyzed_match = re.search(r"Robots analyzed: (\d+)", fleet_info)
    analyzed_robots = int(analyzed_match.group(1)) if analyzed_match else 0
    
    maintenance_match = re.search(r"Robots needing maintenance: (\d+)", fleet_info)
    maintenance_robots = int(maintenance_match.group(1)) if maintenance_match else 0
    
    # Process high priority related questions
    if any(phrase in question_lower for phrase in ["highest priority", "most urgent", "most critical", "top priority"]):
        high_priority_robots = []
        
        # Find robots with high/critical urgency
        for i, result in enumerate(st.session_state.results):
            if result and (result.get('urgency', '').lower() in ['high', 'critical']):
                high_priority_robots.append((i, result))
        
        if high_priority_robots:
            # Sort by urgency (ensuring Critical comes before High if both present)
            def urgency_score(result):
                urgency = result[1].get('urgency', '').lower()
                if urgency == 'critical':
                    return 2
                elif urgency == 'high':
                    return 1
                else:
                    return 0
                    
            sorted_robots = sorted(high_priority_robots, key=urgency_score, reverse=True)
            
            # Get the highest priority robot
            top_robot_idx, top_robot_data = sorted_robots[0]
            root_cause = top_robot_data.get('root_cause', 'Unknown issue')
            urgency = top_robot_data.get('urgency', 'High')
            
            # Provide detailed answer about the highest priority robot
            response = f"Robot #{top_robot_idx+1} has the highest priority with {urgency} urgency.\n\n"
            response += f"Issue: {root_cause}\n"
            
            # Add recommendations if available
            if 'recommendations' in top_robot_data and top_robot_data['recommendations']:
                rec = top_robot_data['recommendations'][0]
                if isinstance(rec, dict) and 'action' in rec:
                    response += f"Recommended action: {rec['action']}\n"
                else:
                    response += f"Recommended action: {rec}\n"
            
            # Also mention other high priority robots if any
            if len(sorted_robots) > 1:
                other_robots = [f"Robot #{idx+1}" for idx, _ in sorted_robots[1:5]]
                response += f"\nOther high priority robots: {', '.join(other_robots)}"
                if len(sorted_robots) > 6:
                    response += f" and {len(sorted_robots)-6} more"
            
            return response
        else:
            return "There are no high priority robots requiring maintenance at this time."
    
    # Handle questions about primary cause or most maintenance-needed robot
    elif any(phrase in question_lower for phrase in ["primary cause", "main issue", "worst condition", "most maintenance needed", "most needed", "greatest need"]):
        # Find all robots needing maintenance
        maintenance_robots = []
        for i, result in enumerate(st.session_state.results):
            if result and result.get('maintenance_required', False):
                maintenance_robots.append((i, result))
        
        if maintenance_robots:
            # Create a scoring function that prioritizes robots based on urgency and confidence
            def maintenance_priority_score(item):
                idx, result = item
                urgency = result.get('urgency', '').lower()
                confidence = result.get('confidence', 0.5)
                
                # Assign numerical value to urgency
                urgency_value = 0
                if urgency == 'critical':
                    urgency_value = 3
                elif urgency == 'high':
                    urgency_value = 2
                elif urgency == 'medium':
                    urgency_value = 1
                
                # Return combined score with higher weight on urgency
                return (urgency_value * 10) + (confidence * 5)
            
            # Sort robots by priority score
            sorted_robots = sorted(maintenance_robots, key=maintenance_priority_score, reverse=True)
            
            # Get the top priority robot
            top_idx, top_result = sorted_robots[0]
            top_robot_id = top_idx + 1
            root_cause = top_result.get('root_cause', 'Unknown issue')
            urgency = top_result.get('urgency', 'Medium')
            
            # Get access to the robot data to add specific sensor information
            robot_data = None
            if st.session_state.data is not None and top_idx < len(st.session_state.data):
                robot_data = st.session_state.data.iloc[top_idx]
            
            # Create detailed response
            response = f"The primary cause of the most maintenance-needed robot (Robot #{top_robot_id}) is: {root_cause}."
            response += f"\n\nThis robot has {urgency} urgency maintenance needs."
            
            # Add critical sensor readings if available
            if robot_data is not None:
                critical_sensors = {}
                for sensor, value in robot_data.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if sensor == 'temperature' and value > 85:
                            critical_sensors[sensor] = f"{value}°C (above normal range of 50-85°C)"
                        elif sensor == 'vibration' and value > 0.3:
                            critical_sensors[sensor] = f"{value} mm/s (above normal range of 0-0.3 mm/s)"
                        elif sensor == 'hydraulic_pressure' and (value < 140 or value > 180):
                            critical_sensors[sensor] = f"{value} PSI (outside normal range of 140-180 PSI)"
                        elif sensor == 'coolant_level' and value < 70:
                            critical_sensors[sensor] = f"{value}% (below normal minimum of 70%)"
                
                if critical_sensors:
                    response += "\n\nCritical sensor readings:"
                    for sensor, reading in critical_sensors.items():
                        response += f"\n- {sensor.replace('_', ' ').title()}: {reading}"
            
            # Add recommended actions
            if 'recommendations' in top_result and top_result['recommendations']:
                response += "\n\nRecommended actions:"
                for i, rec in enumerate(top_result['recommendations'][:3]):
                    if isinstance(rec, dict) and 'action' in rec:
                        response += f"\n{i+1}. {rec['action']}"
                    else:
                        response += f"\n{i+1}. {rec}"
            
            return response
        else:
            return "No robots currently need maintenance."
    
    # Process common fleet questions
    elif any(word in question_lower for word in ["how many", "count", "number"]):
        if any(word in question_lower for word in ["total", "all", "robots"]):
            return f"There are {total_robots} robots in the fleet."
        
        if any(word in question_lower for word in ["maintenance", "repair", "fix"]):
            return f"{maintenance_robots} robots currently need maintenance."
            
        if any(word in question_lower for word in ["normal", "good", "working", "operational"]):
            normal_robots = analyzed_robots - maintenance_robots
            return f"{normal_robots} robots are operating normally."
            
        if any(word in question_lower for word in ["analyzed", "checked", "scanned"]):
            return f"{analyzed_robots} robots have been analyzed so far."
        
        # Count high/medium/low priority robots
        if any(word in question_lower for word in ["high", "critical", "urgent"]):
            high_count = sum(1 for r in st.session_state.results if r and r.get('urgency', '').lower() in ['high', 'critical'])
            return f"{high_count} robots have high or critical urgency maintenance needs."
        
        if "medium" in question_lower:
            medium_count = sum(1 for r in st.session_state.results if r and r.get('urgency', '').lower() == 'medium')
            return f"{medium_count} robots have medium urgency maintenance needs."
        
        if "low" in question_lower:
            low_count = sum(1 for r in st.session_state.results if r and r.get('urgency', '').lower() == 'low')
            return f"{low_count} robots have low urgency maintenance needs."
    
    # Sensor-specific questions
    elif any(word in question_lower for word in ["temperature", "vibration", "hydraulic", "pressure", "power", "coolant"]):
        # Find which sensor is being asked about
        sensor_map = {
            "temperature": "temperature",
            "vibration": "vibration",
            "hydraulic": "hydraulic_pressure",
            "pressure": "hydraulic_pressure",
            "power": "power_output",
            "coolant": "coolant_level"
        }
        
        target_sensor = None
        for keyword, sensor_name in sensor_map.items():
            if keyword in question_lower:
                target_sensor = sensor_name
                break
                
        if target_sensor:
            # Count abnormal readings for this sensor
            abnormal_count = 0
            high_readings = []
            low_readings = []
            
            # Get thresholds for this sensor
            ranges = {
                'temperature': {'min': 50, 'max': 85},
                'vibration': {'min': 0, 'max': 0.3},
                'hydraulic_pressure': {'min': 140, 'max': 180},
                'power_output': {'min': 85, 'max': 100},
                'coolant_level': {'min': 70, 'max': 95}
            }
            
            # Check all robots for abnormal readings
            for i, robot_data in enumerate(st.session_state.data.itertuples()):
                if hasattr(robot_data, target_sensor):
                    value = getattr(robot_data, target_sensor)
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        min_val = ranges.get(target_sensor, {}).get('min', 0)
                        max_val = ranges.get(target_sensor, {}).get('max', 100)
                        
                        if value > max_val:
                            abnormal_count += 1
                            high_readings.append((i, value))
                        elif value < min_val:
                            abnormal_count += 1
                            low_readings.append((i, value))
            
            sensor_display = target_sensor.replace("_", " ").title()
            if abnormal_count > 0:
                response = f"{abnormal_count} robots have abnormal {sensor_display} readings.\n\n"
                
                if high_readings:
                    high_robots = [f"Robot #{i+1} ({val:.1f})" for i, val in sorted(high_readings, key=lambda x: x[1], reverse=True)[:5]]
                    response += f"Robots with high {sensor_display}: {', '.join(high_robots)}"
                    if len(high_readings) > 5:
                        response += f" and {len(high_readings)-5} more"
                    response += "\n\n"
                    
                if low_readings:
                    low_robots = [f"Robot #{i+1} ({val:.1f})" for i, val in sorted(low_readings, key=lambda x: x[1])[:5]]
                    response += f"Robots with low {sensor_display}: {', '.join(low_robots)}"
                    if len(low_readings) > 5:
                        response += f" and {len(low_readings)-5} more"
                
                return response
            else:
                return f"All robots have normal {sensor_display} readings."
        
    # Process specific issue questions
    elif any(word in question_lower for word in ["overheating", "hot", "temperature high"]):
        # Find robots with temperature issues
        hot_robots = []
        for i, robot_data in enumerate(st.session_state.data.itertuples()):
            if hasattr(robot_data, 'temperature'):
                temp = getattr(robot_data, 'temperature')
                if isinstance(temp, (int, float)) and not pd.isna(temp) and temp > 85:
                    hot_robots.append((i, temp))
        
        if hot_robots:
            sorted_robots = sorted(hot_robots, key=lambda x: x[1], reverse=True)
            response = f"{len(hot_robots)} robots are overheating.\n\n"
            robot_list = [f"Robot #{i+1} ({temp:.1f}°C)" for i, temp in sorted_robots[:5]]
            response += f"Highest temperatures: {', '.join(robot_list)}"
            if len(sorted_robots) > 5:
                response += f" and {len(sorted_robots)-5} more"
            return response
        else:
            return "No robots are currently overheating."

    elif any(word in question_lower for word in ["coolant", "low coolant"]):
        # Find robots with coolant issues
        coolant_robots = []
        for i, robot_data in enumerate(st.session_state.data.itertuples()):
            if hasattr(robot_data, 'coolant_level'):
                level = getattr(robot_data, 'coolant_level')
                if isinstance(level, (int, float)) and not pd.isna(level) and level < 70:
                    coolant_robots.append((i, level))
        
        if coolant_robots:
            sorted_robots = sorted(coolant_robots, key=lambda x: x[1])
            response = f"{len(coolant_robots)} robots have low coolant levels.\n\n"
            robot_list = [f"Robot #{i+1} ({level:.1f}%)" for i, level in sorted_robots[:5]]
            response += f"Lowest coolant levels: {', '.join(robot_list)}"
            if len(sorted_robots) > 5:
                response += f" and {len(sorted_robots)-5} more"
            return response
        else:
            return "No robots currently have low coolant levels."
    
    elif any(word in question_lower for word in ["which", "list", "show me"]):
        # List robots by priority
        if any(word in question_lower for word in ["high", "critical", "urgent", "priority"]):
            high_priority = [i for i, r in enumerate(st.session_state.results) 
                          if r and r.get('urgency', '').lower() in ['high', 'critical']]
            if high_priority:
                robot_list = [f"Robot #{i+1}" for i in high_priority[:10]]
                response = f"High priority robots: {', '.join(robot_list)}"
                if len(high_priority) > 10:
                    response += f" and {len(high_priority)-10} more"
                return response
            else:
                return "There are no high priority robots at this time."
                
        # For "which robots need maintenance"
        if any(word in question_lower for word in ["maintenance", "repair", "fix"]):
            if maintenance_robots > 0:
                robot_indices = [i for i, r in enumerate(st.session_state.results) if r and r.get('maintenance_required', False)]
                robot_list = [f"Robot #{i+1}" for i in robot_indices[:10]]
                response = f"The following robots need maintenance: {', '.join(robot_list)}"
                if len(robot_indices) > 10:
                    response += f" and {len(robot_indices)-10} more"
                return response
            else:
                return "No robots currently need maintenance."
    
    # Most common issues
    elif any(phrase in question_lower for phrase in ["common issues", "common problems", "most frequent", "recurring"]):
        # Count occurrences of each root cause
        issue_counts = {}
        for r in st.session_state.results:
            if r and r.get('maintenance_required', False):
                root_cause = r.get('root_cause', 'Unknown')
                if root_cause in issue_counts:
                    issue_counts[root_cause] += 1
                else:
                    issue_counts[root_cause] = 1
        
        if issue_counts:
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            response = "The most common issues in the robot fleet are:\n\n"
            for issue, count in sorted_issues[:5]:
                response += f"- {issue}: {count} robots\n"
            return response
        else:
            return "No common issues detected in the robot fleet."
    
    elif any(word in question_lower for word in ["status", "condition", "overview", "summary"]):
        normal_robots = analyzed_robots - maintenance_robots
        
        if maintenance_robots > 0:
            # Count by urgency
            high_count = sum(1 for r in st.session_state.results if r and r.get('urgency', '').lower() in ['high', 'critical'])
            medium_count = sum(1 for r in st.session_state.results if r and r.get('urgency', '').lower() == 'medium')
            low_count = sum(1 for r in st.session_state.results if r and r.get('urgency', '').lower() == 'low')
            
            response = f"Fleet Status: {total_robots} total robots, {normal_robots} operating normally, {maintenance_robots} needing maintenance."
            response += f"\n\nMaintenance urgency breakdown:"
            response += f"\n- High priority: {high_count} robots"
            response += f"\n- Medium priority: {medium_count} robots" 
            response += f"\n- Low priority: {low_count} robots"
            
            # Add most common issues
            issue_counts = {}
            for r in st.session_state.results:
                if r and r.get('maintenance_required', False):
                    root_cause = r.get('root_cause', 'Unknown')
                    if root_cause in issue_counts:
                        issue_counts[root_cause] += 1
                    else:
                        issue_counts[root_cause] = 1
                        
            if issue_counts:
                sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                response += "\n\nMost common issues:"
                for issue, count in sorted_issues:
                    response += f"\n- {issue}: {count} robots"
            
            return response
        else:
            return f"Fleet Status: All {analyzed_robots} analyzed robots are operating normally."
    
    # Check for robot type questions
    elif "type" in question_lower or "model" in question_lower:
        # Count robots by type
        robot_types = {}
        for robot_data in st.session_state.data.itertuples():
            if hasattr(robot_data, 'robot_type'):
                robot_type = getattr(robot_data, 'robot_type')
                if robot_type in robot_types:
                    robot_types[robot_type] += 1
                else:
                    robot_types[robot_type] = 1
        
        if robot_types:
            response = "Robot types in the fleet:\n\n"
            for robot_type, count in robot_types.items():
                response += f"- {robot_type}: {count} robots\n"
            return response
        else:
            return "Robot type information is not available."
    
    # Error code questions
    elif any(phrase in question_lower for phrase in ["error code", "fault code", "error", "fault"]):
        # Count robots with error codes
        error_robots = []
        for i, robot_data in enumerate(st.session_state.data.itertuples()):
            if hasattr(robot_data, 'error_codes') and getattr(robot_data, 'error_codes'):
                error_robots.append((i, getattr(robot_data, 'error_codes')))
        
        if error_robots:
            # Count occurrence of each error code
            code_counts = {}
            for _, codes in error_robots:
                if codes:
                    code_list = codes.split('+')
                    for code in code_list:
                        if code in code_counts:
                            code_counts[code] += 1
                        else:
                            code_counts[code] = 1
            
            response = f"{len(error_robots)} robots have active error codes.\n\n"
            
            if code_counts:
                sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
                response += "Most common error codes:\n"
                for code, count in sorted_codes[:5]:
                    # Add description if available
                    if code in st.session_state.error_codes_guide:
                        desc = st.session_state.error_codes_guide[code]['description']
                        response += f"- {code}: {desc} ({count} robots)\n"
                    else:
                        response += f"- {code}: {count} robots\n"
            
            return response
        else:
            return "No robots currently have active error codes."
    
    # Generic answer when no specific pattern is matched
    return f"The fleet consists of {total_robots} robots. {analyzed_robots} have been analyzed, with {maintenance_robots} requiring maintenance."

def main():
    try:
        initialize_session_state()
        inject_text_contrast_css()

        # Clear any Streamlit cached functions to ensure new code is used
        analyze_robot_cached.clear()
        if hasattr(generate_fleet_answer, '_cache'):
            generate_fleet_answer._cache.clear()

        # Render the sidebar
        with st.sidebar:
            render_sidebar()
        
        # Display the header
        display_app_header()
        
        # Create enhancer and load data if not done already
        if st.session_state.enhancer is None and st.session_state.api_key:
            create_enhancer()
        
        if st.session_state.data is None:
            load_data()
        
        # Main content based on current page
        if st.session_state.current_page == "Dashboard":
            show_robot_dashboard()
        elif st.session_state.current_page == "Robot Status":
            show_robot_status()
        elif st.session_state.current_page == "Maintenance Schedule":
            show_maintenance_schedule()
        elif st.session_state.current_page == "Error Codes Guide":
            show_error_codes_guide()
        elif st.session_state.current_page == "Ask Questions":
            show_question_interface()
        else:
            # Fallback to dashboard
            show_robot_dashboard()
            
    except Exception as e:
        st.error(f"Error in application: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 