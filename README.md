# ⚡ Energy Demand Forecasting System

Multi-model time series forecasting system with interactive dashboard comparing Prophet, ARIMA, XGBoost, and LSTM approaches.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Portfolio Project** by [Esteban](https://github.com/EstebanM-M) | [LinkedIn](https://www.linkedin.com/in/esteban-morales-mahecha/) | [Live Demo](#) (Coming Soon)

---

## 🎯 Overview

A production-ready time series forecasting platform that enables users to:
- Upload custom CSV data for analysis
- Train and compare multiple ML models in real-time
- Visualize predictions with confidence intervals
- Generate business-focused insights and reports
- Export predictions, models, and comprehensive reports

**Key Differentiators:**
- ✅ Real-time model training on user data
- ✅ Interactive dashboard with business metrics
- ✅ Multi-model comparison framework
- ✅ Professional software engineering practices

---

## 🚀 Features

### Core Functionality
- **4 ML Models**: Prophet, ARIMA/SARIMA, XGBoost, LSTM
- **CSV Upload**: Validate and process custom time series data
- **Real-time Training**: Train models on-demand (30s - 3min)
- **Interactive Dashboard**: 6-page Streamlit application
- **Model Comparison**: Side-by-side evaluation with multiple metrics
- **Business Metrics**: Cost analysis, ROI, forecast bias
- **Export Results**: Download predictions (CSV), reports (PDF), trained models (.pkl)

### Technical Features
- Automated data validation and cleaning
- Missing value handling and outlier detection
- Feature engineering (time-based features)
- Multiple evaluation metrics (MAE, RMSE, MAPE, SMAPE)
- Confidence intervals and prediction uncertainty
- Docker containerization

---

## 📊 Dataset

The system supports multiple data sources:

### Real Data (Current Setup) ⭐
Uses the **PJM Hourly Energy Consumption** dataset from Kaggle:
- 145K+ hourly records (2002-2018)
- Real power consumption data from PJM Interconnection LLC
- Demonstrates realistic patterns: daily, weekly, and seasonal cycles

**Kaggle API Setup** (if not configured):
```bash
# 1. Get API token from https://www.kaggle.com/settings
# 2. Place kaggle.json in ~/.kaggle/ (Mac/Linux) or %USERPROFILE%\.kaggle\ (Windows)
# 3. Dataset auto-downloads on first run
```

### Synthetic Data (Fallback)
Automatically generated realistic data if Kaggle unavailable:
- 2 years of hourly data with realistic patterns
- Daily peaks (5-7 PM), weekly patterns, seasonal variation

### Custom Data
Upload any CSV time series data through the dashboard:
- Minimum 30 data points required
- Automatic date/time detection
- Flexible frequency support (hourly, daily, weekly, monthly)

---

## 🛠️ Tech Stack

**Machine Learning:**
- Prophet (Facebook) - Robust seasonal forecasting
- Statsmodels - ARIMA/SARIMA statistical models
- XGBoost - Gradient boosting with feature engineering
- TensorFlow/Keras - LSTM neural networks

**Data & Visualization:**
- Pandas, NumPy - Data manipulation
- Plotly - Interactive visualizations
- Matplotlib, Seaborn - Statistical plots

**Application:**
- Streamlit - Interactive dashboard
- SQLAlchemy - Database ORM
- PostgreSQL/SQLite - Data storage

**DevOps:**
- Docker & Docker Compose - Containerization
- pytest - Testing framework
- GitHub Actions - CI/CD (planned)

---

## 💻 Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) Docker

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/EstebanM-M/time-series-forecasting.git
cd time-series-forecasting

# 2. Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# 3. Install package
pip install -e .

# 4. Download sample data (optional - auto-downloads if not present)
python -c "from forecasting.preprocessing.data_loader import download_sample_data; download_sample_data()"

# 5. Run dashboard (coming soon)
streamlit run dashboard/app.py
```

### Docker Installation (Alternative)
```bash
# Build and run
docker-compose up

# Access dashboard at http://localhost:8501
```

---

## 📖 Usage

### Command Line
```python
from forecasting.preprocessing.data_loader import DataLoader
from forecasting.preprocessing.cleaner import DataCleaner
from forecasting.models.prophet_model import ProphetForecaster

# Load data
loader = DataLoader()
df = loader.load_pjm_sample()

# Clean data
cleaner = DataCleaner()
df_clean = cleaner.clean(df, 'datetime', 'consumption_mw')

# Train Prophet model
forecaster = ProphetForecaster()
forecaster.fit(df_clean)
predictions = forecaster.predict(horizon=30)
```

### Dashboard (Coming Soon)
```bash
streamlit run dashboard/app.py
```

**Workflow:**
1. Upload CSV or use sample data
2. Configure models and forecast horizon
3. Train selected models
4. Compare results and metrics
5. Download predictions and reports

---

## 📈 Project Status

Current development phase: **Day 1-2 / 6 days**

- [x] Project setup and structure
- [x] Data download pipeline (Kaggle + synthetic)
- [x] Data validation and cleaning
- [x] Data exploration and analysis
- [x] Prophet model implementation
- [x] Evaluation metrics system
- [ ] ARIMA/SARIMA implementation
- [ ] XGBoost for time series
- [ ] LSTM neural network
- [ ] Interactive Streamlit dashboard
- [ ] Model comparison framework
- [ ] Business metrics calculation
- [ ] Export functionality (CSV, PDF, PKL)
- [ ] Docker deployment
- [ ] Documentation and testing
- [ ] Live deployment (Streamlit Cloud)

**Target Completion:** January 17-18, 2026

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=forecasting tests/

# Run specific test
pytest tests/test_preprocessing.py
```

---

## 📁 Project Structure
```
time-series-forecasting/
├── README.md
├── setup.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── src/forecasting/           # Core package
│   ├── config.py             # Configuration
│   ├── preprocessing/        # Data pipeline
│   │   ├── data_loader.py   # Download & load data
│   │   ├── validator.py     # Data validation
│   │   └── cleaner.py       # Cleaning & feature engineering
│   ├── models/              # ML models
│   │   ├── prophet_model.py
│   │   ├── arima_model.py
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   ├── evaluation/          # Metrics & comparison
│   │   ├── metrics.py
│   │   └── comparator.py
│   └── utils/              # Utilities
│       └── visualization.py
│
├── dashboard/              # Streamlit app
│   ├── app.py
│   ├── pages/
│   │   ├── 1_📊_Overview.py
│   │   ├── 2_⬆️_Upload_Data.py
│   │   ├── 3_⚙️_Configure.py
│   │   ├── 4_🔮_Train.py
│   │   ├── 5_📈_Results.py
│   │   └── 6_💼_Business.py
│   └── components/
│
├── data/                  # Data storage
│   ├── raw/              # Raw datasets
│   ├── processed/        # Cleaned data
│   └── sample/           # Sample for demos
│
├── models/               # Saved models
├── notebooks/            # Analysis notebooks
└── tests/               # Unit tests
```

---

## 🎯 Use Cases

This forecasting system is applicable to:

- **Energy & Utilities**: Demand forecasting, load balancing
- **Retail & E-commerce**: Sales prediction, inventory optimization
- **Finance**: Stock price trends, revenue forecasting
- **Manufacturing**: Production planning, maintenance scheduling
- **Healthcare**: Patient admissions, resource allocation
- **IoT & Smart Systems**: Sensor data prediction, anomaly detection

---

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Esteban**
- Electronic Engineer (Escuela Colombiana de Ingeniería, 2024)
- Transitioning to ML/AI Engineering
- LinkedIn: [profile](https://www.linkedin.com/in/esteban-morales-mahecha/)
- GitHub: [EstebanM-M](https://github.com/EstebanM-M)
- Email: tu_email@example.com

---

## 🙏 Acknowledgments

- PJM Interconnection LLC for the energy consumption dataset
- Kaggle community for data access
- Facebook Research for Prophet library
- Anthropic for Claude AI assistance

---

## 📚 Resources

**Documentation:**
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/tsa.html)
- [XGBoost](https://xgboost.readthedocs.io/)
- [TensorFlow](https://www.tensorflow.org/)

**Learning Resources:**
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)
- [Kaggle Time Series Course](https://www.kaggle.com/learn/time-series)

---

⭐ **If you find this project useful, please star the repository!**

---
