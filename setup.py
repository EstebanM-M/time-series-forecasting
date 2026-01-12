from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forecasting-system",
    version="0.1.0",
    author="Esteban",
    author_email="tu_email@example.com",
    description="Multi-model time series forecasting system with interactive dashboard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/time-series-forecasting",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core ML
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        
        # Time Series Models
        "prophet>=1.1.5",
        "statsmodels>=0.14.0",
        "xgboost>=2.0.0",
        "tensorflow>=2.13.0",
        
        # Visualization
        "plotly>=5.18.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        
        # Dashboard
        "streamlit>=1.30.0",
        
        # Database
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        
        # Utils
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
)