#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to run the thesis benchmark with proper path setup.
This resolves import issues by setting up the Python path correctly.
"""

import os
import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Set up paths and run the benchmark."""
    # Get the current directory (should be project root)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the Python path
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Add the src directory to the Python path
    src_dir = os.path.join(root_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    logger.info(f"Python path set to: {sys.path[:3]}")
    
    # Try to import and run the benchmark
    try:
        # Import the benchmark module
        from src.run_thesis_benchmark import main as run_benchmark
        
        # Print banner
        print("=" * 80)
        print(f"RUNNING PDF ANALYSIS BENCHMARK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Run the benchmark
        run_benchmark()
        
        # Get the current date
        current_date = datetime.now().strftime('%B %d, %Y')
        
        # Copy generated reports to docs directory
        generated_report = Path(root_dir) / 'src' / 'outputs' / 'thesis_benchmark' / 'results' / 'benchmark_report.md'
        detailed_report = Path(root_dir) / 'src' / 'outputs' / 'thesis_benchmark' / 'results' / 'detailed_benchmark_report.md'
        docs_report = Path(root_dir) / 'docs' / 'benchmark_report.md'
        docs_detailed_report = Path(root_dir) / 'docs' / 'detailed_benchmark_report.md'
        
        # Copy figures to docs directory
        figures_dir = Path(root_dir) / 'src' / 'outputs' / 'thesis_benchmark' / 'results' / 'figures'
        docs_figures_dir = Path(root_dir) / 'docs' / 'figures'
        
        # Create docs figures directory if it doesn't exist
        docs_figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy all figure files
        for figure_file in figures_dir.glob('*.png'):
            shutil.copy2(figure_file, docs_figures_dir / figure_file.name)
            
        # Update the report paths in the copied reports
        if generated_report.exists():
            with open(generated_report, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Update figure paths to use the docs/figures directory
            content = content.replace('`figures/', '`./figures/')
            
            # Add the date at the end of the report
            if '---' not in content:
                content += f"\n\n---\n\n*Report generated on: {current_date}*\n"
            else:
                # Update the date if it already exists
                content = content.replace('*Report generated on:*', f'*Report generated on: {current_date}*')
                
            # Copy to docs directory
            with open(docs_report, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Copied and updated benchmark report to {docs_report}")
            
        # Copy detailed report to docs directory
        if detailed_report.exists():
            shutil.copy2(detailed_report, docs_detailed_report)
            logger.info(f"Copied detailed report to {docs_detailed_report}")
        
        # Print completion message
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nResults saved to: src/outputs/thesis_benchmark/results/")
        print("Reports copied to: docs/benchmark_report.md and docs/detailed_benchmark_report.md")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required modules are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 