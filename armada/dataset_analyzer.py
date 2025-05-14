#!/usr/bin/env python3
"""
Dataset Analyzer for Warehouse Robotics

This script performs comprehensive analysis of all columns in the robotics dataset,
using the fine-tuned LLM to extract insights beyond just maintenance recommendations.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import colorama
from colorama import Fore, Style
import logging
import time
from openai import OpenAI

# Import utility modules
from utils.llm_utils import call_llm, with_retry
from utils.config import get_system_prompt
from utils.data_processing import preprocess_sensor_data, extract_sensor_stats, find_correlation_patterns
from utils.visualization import save_plot, plot_sensor_over_time, plot_correlation_matrix

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dataset_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dataset_analyzer")

class DatasetAnalyzer:
    """Analyzes all columns in the robotics dataset using LLM and statistical methods."""
    
    def __init__(self, api_key: str, model_id: str = None):
        """
        Initialize the dataset analyzer.
        
        Args:
            api_key: OpenAI API key
            model_id: ID of the fine-tuned model (if None, uses gpt-3.5-turbo)
        """
        self.api_key = api_key
        self.model_id = model_id if model_id else "gpt-3.5-turbo"
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized with model: {self.model_id}")
        
    def load_data(self, data_file: str) -> pd.DataFrame:
        """
        Load data from a CSV file and perform initial processing.
        
        Args:
            data_file: Path to the CSV file
            
        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Use the preprocess_sensor_data utility instead of custom code
        df = preprocess_sensor_data(df, dropna=False, fill_method='mean')
        
        return df
    
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the dataset structure and generate summary statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset structure analysis
        """
        logger.info("Analyzing dataset structure")
        result = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_types": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "boolean_columns": [],
            "datetime_columns": [],
            "other_columns": [],
            "missing_values": {},
            "column_statistics": {}
        }
        
        # Categorize columns by type
        for col in df.columns:
            dtype = str(df[col].dtype)
            result["column_types"][col] = dtype
            
            # Categorize by type
            if np.issubdtype(df[col].dtype, np.number):
                result["numeric_columns"].append(col)
            elif df[col].nunique() < 10 and df[col].dtype == 'object':
                result["categorical_columns"].append(col)
            elif df[col].dtype == 'bool':
                result["boolean_columns"].append(col)
            elif 'datetime' in dtype:
                result["datetime_columns"].append(col)
            else:
                result["other_columns"].append(col)
                
            # Count missing values
            result["missing_values"][col] = int(df[col].isna().sum())
        
        # Use extract_sensor_stats utility for numeric columns
        sensor_stats = extract_sensor_stats(df, exclude_cols=result["datetime_columns"])
        for col, stats in sensor_stats.items():
            if col in result["numeric_columns"]:
                result["column_statistics"][col] = stats
                
        # Calculate statistics for categorical columns
        for col in result["categorical_columns"]:
            value_counts = df[col].value_counts().head(5).to_dict()
            result["column_statistics"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
        
        return result
    
    def identify_column_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify logical column groupings based on column names and patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with column groups
        """
        logger.info("Identifying column groups")
        
        # Define common prefixes/patterns for grouping
        patterns = {
            "robot_sensors": ["robot_", "sensor_", "reading_", "temperature", "pressure", "voltage", "vibration", "current"],
            "environment": ["environment_", "ambient_", "warehouse_", "facility_", "external_"],
            "maintenance": ["maintenance_", "repair_", "service_", "downtime_", "fix_"],
            "performance": ["performance_", "efficiency_", "productivity_", "success_", "delivery_", "speed_"],
            "failures": ["failure_", "fault_", "error_", "issue_", "problem_", "warning_"],
            "timestamps": ["time_", "date_", "timestamp", "created_", "logged_", "recorded_"],
            "identifiers": ["id", "_id", "identifier", "serial", "model", "type", "category"]
        }
        
        groups = {group: [] for group in patterns}
        
        # Assign columns to groups based on patterns
        for col in df.columns:
            assigned = False
            for group, keywords in patterns.items():
                if any(keyword.lower() in col.lower() for keyword in keywords):
                    groups[group].append(col)
                    assigned = True
                    break
            
            if not assigned:
                # Try to assign based on data type
                if np.issubdtype(df[col].dtype, np.number):
                    groups["robot_sensors"].append(col)  # Default numeric to sensors
                else:
                    groups["other"] = groups.get("other", []) + [col]
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with correlation analysis
        """
        logger.info("Analyzing correlations between numeric columns")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {"error": "Not enough numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Use find_correlation_patterns utility to find significant correlations
        correlations = find_correlation_patterns(df, min_corr=0.5, max_pvalue=0.05)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": correlations[:20]  # Top 20 correlations
        }
    
    def generate_column_insights(self, df: pd.DataFrame, column_groups: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Generate insights about column groups using the LLM.
        
        Args:
            df: Input DataFrame
            column_groups: Grouped columns
            
        Returns:
            Dictionary with LLM-generated insights for each group
        """
        logger.info("Generating insights for column groups using LLM")
        
        insights = {}
        
        for group_name, columns in column_groups.items():
            if not columns:
                continue
                
            # Create a sample of the data for these columns
            sample_size = min(5, len(df))
            sample_df = df[columns].head(sample_size)
            
            # Create a prompt for the LLM
            prompt = f"""Analyze these columns from a warehouse robotics dataset:
Columns: {', '.join(columns)}

Sample data:
{sample_df.to_string()}

Please provide insights about this group of columns ({group_name}):
1. What information might these columns capture about the warehouse robots?
2. How might this data be useful beyond maintenance (e.g., for optimization, performance analysis, etc.)?
3. What patterns or relationships might exist in this group of columns?
4. What additional derived metrics might be valuable to calculate from these columns?

Focus on providing specific, actionable insights beyond just predictive maintenance."""

            # Get insights from LLM using the call_llm utility
            insights[group_name] = call_llm(
                client=self.client,
                prompt=prompt,
                model_id=self.model_id,
                system_message=get_system_prompt('data_analysis'),
                temperature=0.2,
                max_tokens=750
            )
            
        return insights
    
    def identify_outliers(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier information for each numeric column
        """
        logger.info("Identifying outliers in numeric columns")
        
        outliers = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        for col in numeric_df.columns:
            # Calculate IQR for outlier detection
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            outlier_rows = df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            
            if len(outlier_rows) > 0:
                outliers[col] = []
                # Get sample of outliers (maximum 5)
                for idx, row in outlier_rows.head(5).iterrows():
                    outliers[col].append({
                        "row_index": int(idx),
                        "value": float(row[col]),
                        "is_high": bool(row[col] > upper_bound),
                        "threshold": float(upper_bound if row[col] > upper_bound else lower_bound)
                    })
                    
        return outliers
    
    def generate_comprehensive_report(self, df: pd.DataFrame, output_dir: str):
        """
        Generate a comprehensive analysis report for the dataset.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save the report and visualizations
        """
        logger.info("Generating comprehensive analysis report")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # Perform all analyses
        structure = self.analyze_dataset_structure(df)
        column_groups = self.identify_column_groups(df)
        correlations = self.analyze_correlations(df)
        insights = self.generate_column_insights(df, column_groups)
        outliers = self.identify_outliers(df)
        
        # Create full report JSON
        report = {
            "dataset_name": "Warehouse Robotics Dataset",
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "structure": structure,
            "column_groups": column_groups,
            "correlations": {
                "strong_correlations": correlations.get("strong_correlations", [])
            },
            "outliers": outliers,
            "llm_insights": insights
        }
        
        # Save report to JSON
        report_file = os.path.join(output_dir, "full_analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._create_visualizations(df, column_groups, correlations, output_dir)
        
        # Generate final insights using LLM
        final_insights = self._generate_final_insights(report)
        
        # Save final insights
        insights_file = os.path.join(output_dir, "final_insights.txt")
        with open(insights_file, 'w') as f:
            f.write(final_insights)
        
        logger.info(f"Comprehensive report saved to {report_file}")
        logger.info(f"Final insights saved to {insights_file}")
        
        return report, final_insights
    
    def _create_visualizations(self, df: pd.DataFrame, column_groups: Dict[str, List[str]], 
                              correlations: Dict[str, Any], output_dir: str):
        """Create visualizations for the dataset analysis."""
        vis_dir = os.path.join(output_dir, "visualizations")
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 15:
            # If too many columns, use only the most variable ones
            top_cols = numeric_df.var().sort_values(ascending=False).head(15).index.tolist()
            corr_matrix = numeric_df[top_cols].corr()
        else:
            corr_matrix = numeric_df.corr()
            
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        save_plot(plt.gcf(), "correlation_heatmap", vis_dir)
        plt.close()
        
        # 2. Column Type Distribution
        column_types = [str(df[col].dtype) for col in df.columns]
        type_counts = pd.Series(column_types).value_counts()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=type_counts.index, y=type_counts.values)
        plt.title('Column Data Type Distribution')
        plt.xlabel('Data Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot(plt.gcf(), "column_types", vis_dir)
        plt.close()
        
        # 3. Distribution of numeric columns
        for group, cols in column_groups.items():
            numeric_cols = [col for col in cols if col in numeric_df.columns]
            if not numeric_cols:
                continue
                
            # Limit to top 6 columns per group for readability
            for i in range(0, len(numeric_cols), 6):
                subset = numeric_cols[i:i+6]
                if not subset:
                    continue
                    
                fig, axes = plt.subplots(len(subset), 1, figsize=(12, 3*len(subset)))
                if len(subset) == 1:
                    axes = [axes]
                    
                for ax, col in zip(axes, subset):
                    sns.histplot(df[col].dropna(), ax=ax, kde=True)
                    ax.set_title(f'{col} Distribution')
                    
                plt.tight_layout()
                save_plot(fig, f"{group}_distribution_{i//6}", vis_dir)
                plt.close()
                
        # 4. Top correlations plot
        if "strong_correlations" in correlations and correlations["strong_correlations"]:
            top_5 = correlations["strong_correlations"][:5]
            
            plt.figure(figsize=(10, 6))
            for i, corr in enumerate(top_5):
                plt.subplot(2, 3, i+1)
                plt.scatter(df[corr["col1"]], df[corr["col2"]], alpha=0.5)
                plt.xlabel(corr["col1"])
                plt.ylabel(corr["col2"])
                plt.title(f"r = {corr['correlation']:.2f}")
                
            plt.tight_layout()
            save_plot(plt.gcf(), "top_correlations", vis_dir)
            plt.close()
    
    def _generate_final_insights(self, report: Dict[str, Any]) -> str:
        """Generate final insights from the complete analysis."""
        
        # Create a summary prompt based on the report
        column_count = report["structure"]["column_count"]
        numeric_count = len(report["structure"]["numeric_columns"])
        categorical_count = len(report["structure"]["categorical_columns"])
        
        # Extract top correlations
        top_corrs = report["correlations"].get("strong_correlations", [])
        corr_text = ""
        if top_corrs:
            corr_text = "Top correlations:\n"
            for i, corr in enumerate(top_corrs[:3]):
                corr_text += f"- {corr['col1']} and {corr['col2']}: {corr['correlation']:.2f}\n"
        
        # Group insights
        group_insights = ""
        for group, insight in report["llm_insights"].items():
            # Truncate to prevent overly long prompt
            short_insight = insight[:300] + "..." if len(insight) > 300 else insight
            group_insights += f"\n{group}:\n{short_insight}\n"
        
        prompt = f"""You've analyzed a warehouse robotics dataset with {column_count} columns 
({numeric_count} numeric, {categorical_count} categorical).

{corr_text}

The data has been grouped into these categories with the following initial insights:
{group_insights}

Based on this comprehensive analysis:

1. What are the 3-5 most valuable insights from this data that go BEYOND maintenance prediction?
2. What unexpected patterns or correlations might deserve further investigation?
3. What business value could be derived from this data that wasn't obvious initially?
4. How could this data be used to optimize warehouse operations beyond just robot maintenance?
5. What specific recommendations would you make for further data collection or analysis?

Provide a comprehensive summary that focuses on practical, actionable insights that would be valuable
for warehouse management and operations teams."""

        # Use the call_llm utility instead of the custom implementation
        return call_llm(
            client=self.client,
            prompt=prompt,
            model_id=self.model_id,
            system_message=get_system_prompt('data_analysis'),
            temperature=0.3,
            max_tokens=1000
        )

def print_colored(text, color=Fore.WHITE, bold=False):
    """Print colored text to the terminal."""
    if bold:
        print(color + Style.BRIGHT + text + Style.RESET_ALL)
    else:
        print(color + text + Style.RESET_ALL)

def print_header(text):
    """Print a section header."""
    width = min(80, len(text) + 10)
    print("\n" + "=" * width)
    print_colored(f"  {text}  ", Fore.CYAN, bold=True)
    print("=" * width)

def load_api_key(api_key_file: str) -> str:
    """Load the OpenAI API key from a file."""
    with open(api_key_file, 'r') as f:
        return f.read().strip()

def main():
    """Run the dataset analyzer from command line."""
    parser = argparse.ArgumentParser(description="Analyze all columns in the robotics dataset")
    parser.add_argument("--data_file", default="extended_robot_data.csv", help="Path to the dataset")
    parser.add_argument("--api_key_file", default="openai_api_key.txt", help="Path to the API key file")
    parser.add_argument("--model_id", default="gpt-3.5-turbo", help="Model ID to use")
    parser.add_argument("--output_dir", default="results/dataset_analysis", help="Output directory")
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    print_header("WAREHOUSE ROBOTICS DATASET ANALYZER")
    print_colored("Comprehensive analysis of all dataset columns", Fore.CYAN)
    
    try:
        # Load API key
        print_colored("\nLoading OpenAI API key...", Fore.BLUE)
        api_key = load_api_key(args.api_key_file)
        
        # Initialize analyzer
        print_colored("Initializing dataset analyzer...", Fore.BLUE)
        analyzer = DatasetAnalyzer(api_key, args.model_id)
        
        # Load data
        df = analyzer.load_data(args.data_file)
        print_colored(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns", Fore.GREEN)
        
        # Run analysis
        print_colored("\nAnalyzing dataset structure...", Fore.BLUE)
        structure = analyzer.analyze_dataset_structure(df)
        print_colored(f"Found {len(structure['numeric_columns'])} numeric columns, {len(structure['categorical_columns'])} categorical columns", Fore.GREEN)
        
        print_colored("\nIdentifying column groups...", Fore.BLUE)
        column_groups = analyzer.identify_column_groups(df)
        print_colored(f"Identified {len(column_groups)} logical column groups", Fore.GREEN)
        for group, columns in column_groups.items():
            print(f"  - {group}: {len(columns)} columns")
        
        print_colored("\nAnalyzing correlations...", Fore.BLUE)
        correlations = analyzer.analyze_correlations(df)
        strong_corr_count = len(correlations.get("strong_correlations", []))
        print_colored(f"Found {strong_corr_count} strong correlations between columns", Fore.GREEN)
        
        print_colored("\nIdentifying outliers...", Fore.BLUE)
        outliers = analyzer.identify_outliers(df)
        print_colored(f"Found outliers in {len(outliers)} columns", Fore.GREEN)
        
        print_colored("\nGenerating LLM insights for column groups...", Fore.BLUE)
        insights = analyzer.generate_column_insights(df, column_groups)
        print_colored(f"Generated insights for {len(insights)} column groups", Fore.GREEN)
        
        print_colored("\nGenerating comprehensive report...", Fore.BLUE)
        report, final_insights = analyzer.generate_comprehensive_report(df, args.output_dir)
        
        print_colored("\nAnalysis complete!", Fore.GREEN, bold=True)
        print_colored(f"Results saved to {args.output_dir}", Fore.GREEN)
        
        print_header("KEY INSIGHTS PREVIEW")
        # Print just the first part of the insights
        print(final_insights[:500] + "...\n")
        print_colored(f"Full insights available in {args.output_dir}/final_insights.txt", Fore.BLUE)
        
    except Exception as e:
        logger.error(f"Error in dataset analysis: {str(e)}", exc_info=True)
        print_colored(f"\nError: {str(e)}", Fore.RED)
        sys.exit(1)

if __name__ == "__main__":
    main() 