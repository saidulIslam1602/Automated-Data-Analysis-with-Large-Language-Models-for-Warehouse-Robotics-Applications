from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1.0",
    description="Advanced Anomaly Detection for Robotics Maintenance",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.7.3",
        "pandas>=1.3.5",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        "requests>=2.27.1",
        "imbalanced-learn>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.1",
            "black>=22.1.0",
            "flake8>=4.0.1",
        ],
        "web": [
            "streamlit>=1.8.1",
            "plotly>=5.6.0",
        ],
        "all": [
            "pytest>=7.0.1",
            "black>=22.1.0",
            "flake8>=4.0.1",
            "streamlit>=1.8.1",
            "plotly>=5.6.0",
            "shap>=0.40.0",
            "lime>=0.2.0.1",
            "tsfel>=0.1.4",
            "pywavelets>=1.2.0",
        ],
    },
    python_requires=">=3.8",
) 