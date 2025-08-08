# 🚀 AI-Driven E-Commerce Order Recovery System

A comprehensive end-to-end system that detects sudden e-commerce order drop-offs, predicts demand-supply mismatches, and provides a real-time dashboard for monitoring and resolution.

## 📊 Features

### 🔍 Real-time Order Drop Detection
- Monitors daily/hourly order patterns
- Alerts when orders drop beyond configurable thresholds
- Automated anomaly detection using Z-score analysis

### 🤖 AI-Powered Risk Prediction
- Machine learning model to predict demand-supply mismatch risk
- XGBoost classifier with >80% F1 score
- Real-time SKU risk assessment (High/Medium/Low)

### 📈 Comprehensive Analytics
- Interactive Streamlit dashboard
- Real-time KPIs and performance metrics
- Category and regional performance analysis
- Delivery performance monitoring

### ⚠️ Smart Alerting System
- Email, Slack, and webhook notifications
- Configurable alert rules and thresholds
- Multi-channel alert delivery

## 🛠️ Tech Stack

- **Backend**: FastAPI with Python 3.9+
- **Frontend**: Streamlit dashboard
- **Database**: PostgreSQL with Redis caching
- **ML/AI**: XGBoost, Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Docker, Docker Compose, Nginx
- **CI/CD**: GitHub Actions ready

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (for local development)
- 8GB+ RAM recommended

### 1. Clone and Setup
```bash
git clone <repository-url>
cd MEESHO
cp env.example .env
# Edit .env with your configuration
```

### 2. Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### 3. Local Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Generate mock data
python data/mock_data_generator.py

# Run API server
uvicorn api.main:app --reload --port 8000

# Run dashboard (in another terminal)
streamlit run dashboard.py --server.port 8501
```

## 📱 Dashboard Features

### 📊 Overview Dashboard
- Real-time KPIs (orders, revenue, stock-outs)
- Critical alert notifications
- Order and revenue trends
- Category performance metrics

### 📈 Order Analytics
- Deep dive into order patterns
- Correlation analysis between metrics
- Statistical summaries and trends
- Interactive time series charts

### 🤖 AI Risk Prediction
- High-risk SKU identification
- Model performance metrics
- Risk distribution visualization
- Actionable AI recommendations

### 📍 Regional Analysis
- Performance by geographical region
- Interactive maps and heatmaps
- Regional comparison metrics

### 🚚 Delivery Performance
- Delivery delay monitoring
- Partner performance analysis
- Regional delivery insights

### ⚠️ Anomaly Detection
- Real-time anomaly identification
- Configurable sensitivity thresholds
- Historical anomaly timeline
- Severity-based categorization

## 🔧 API Endpoints

### Core Endpoints
- `GET /kpi-summary` - Current KPI metrics
- `GET /order-trend` - Time series order data
- `GET /high-risk-skus` - AI-predicted high-risk SKUs
- `GET /anomaly-detection` - Anomaly detection results
- `POST /train-model` - Train/retrain ML model
- `POST /ingest-data` - Reload data from sources

### Analytics Endpoints
- `GET /category-performance` - Category-wise metrics
- `GET /regional-performance` - Regional analysis
- `GET /delivery-performance` - Delivery metrics
- `GET /demand-forecast` - Demand forecasting

## 🔔 Alert Configuration

### Email Alerts
```bash
# Configure in .env
EMAIL_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
RECIPIENT_EMAILS=admin@company.com,ops@company.com
```

### Slack Alerts
```bash
# Configure webhook URL
SLACK_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### Custom Webhooks
```bash
# Configure custom endpoint
WEBHOOK_ENABLED=true
CUSTOM_WEBHOOK_URL=https://your-api.com/webhook/alerts
```

## 🧠 Machine Learning Model

### Features Used
- Order count and quantity patterns
- Stock levels and inventory ratios
- Price change percentages
- Demand-to-stock ratios
- Historical trends and seasonality

### Model Performance
- **Algorithm**: XGBoost Classifier
- **Target F1 Score**: >0.8
- **Risk Categories**: High, Medium, Low
- **Retraining**: Automated daily retraining
- **Feature Importance**: Available in dashboard

## 📊 Sample Data

The system includes a comprehensive mock data generator that creates:
- **20,000+ SKUs** across 8 categories
- **1 year** of historical orders (300K+ orders)
- **Inventory levels** with realistic stock patterns
- **Price history** with market fluctuations
- **Delivery performance** data

## 🔧 Configuration

### Alert Thresholds (configurable in .env)
```bash
ORDER_DROP_THRESHOLD=20.0        # Order drop % alert
STOCKOUT_THRESHOLD=15.0          # Stock-out % alert
DELIVERY_DELAY_THRESHOLD=25.0    # Delivery delay % alert
HIGH_RISK_SKU_THRESHOLD=100      # High-risk SKU count alert
```

### Model Configuration
```bash
MODEL_RETRAIN_INTERVAL_HOURS=24  # Auto-retrain interval
MODEL_PERFORMANCE_THRESHOLD=0.8  # Minimum F1 score
```

## 📈 Performance Metrics

### Expected System Performance
- **Anomaly Detection**: >90% accuracy
- **Risk Prediction**: F1 score >0.8
- **Dashboard Load Time**: <2 seconds for 20K records
- **API Response Time**: <100ms for most endpoints

### Scalability
- Handles 20,000+ SKUs efficiently
- Supports 1M+ orders with proper indexing
- Horizontal scaling via Docker containers
- Redis caching for improved performance

## 🚢 Deployment

### Production Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# With SSL (configure nginx.conf)
docker-compose -f docker-compose.ssl.yml up -d
```

### Cloud Deployment (AWS/GCP)
1. Configure environment variables
2. Set up managed database (RDS/Cloud SQL)
3. Deploy using container services (ECS/Cloud Run)
4. Configure load balancer and SSL certificates

## 🔍 Monitoring & Logging

### Application Logs
```bash
# View real-time logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Log files location
./logs/application.log
```

### Health Checks
- API Health: `http://localhost:8000/`
- Dashboard Health: `http://localhost:8501/`
- Database Health: Built into Docker Compose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request
5. Follow code style guidelines

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the dashboard system management page

## 🎯 Future Enhancements

- Integration with real marketplace APIs (Flipkart, Shopify)
- Advanced pricing optimization suggestions
- Natural language query interface for CXOs
- Mobile app for alerts and monitoring
- Advanced ML models with deep learning
- Multi-tenant support for enterprise use

---

Built with ❤️ for modern e-commerce operations
