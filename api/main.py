from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import DataProcessor
from src.ml_model import RiskPredictionModel, DemandForecastModel
import pandas as pd
from datetime import datetime

app = FastAPI(title="E-Commerce Order Recovery API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_processor = DataProcessor()
risk_model = RiskPredictionModel()
demand_model = DemandForecastModel()

# Load data on startup
@app.on_event("startup")
async def startup_event():
    """Initialize data and models on startup"""
    print("Loading data...")
    success = data_processor.load_data('data/data')
    if not success:
        print("Warning: Could not load data files")
    
    # Try to load existing model
    risk_model.load_model()

# Pydantic models
class KPIResponse(BaseModel):
    total_orders_today: int
    total_orders_yesterday: int
    order_drop_percent: float
    stockout_percent: float
    avg_price_change_percent: float
    total_revenue: float
    avg_order_value: float
    total_skus: int
    stockout_skus: int
    last_updated: str

class HighRiskSKU(BaseModel):
    sku_id: str
    predicted_risk: str
    confidence: float
    stock_level: int
    demand: float
    order_count: int
    recommendation: str

class AnomalyAlert(BaseModel):
    date: str
    metric: str
    value: float
    expected: float
    z_score: float
    severity: str

@app.get("/")
async def root():
    """API health check"""
    return {"message": "E-Commerce Order Recovery API", "status": "running"}

@app.get("/kpi-summary", response_model=KPIResponse)
async def get_kpi_summary():
    """Get current KPI summary"""
    try:
        kpis = data_processor.get_kpi_summary()
        if not kpis:
            raise HTTPException(status_code=500, detail="No data available")
        return KPIResponse(**kpis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order-trend")
async def get_order_trend(days: int = 30):
    """Get order trend data for the last N days"""
    try:
        trend_data = data_processor.get_order_trend(days=days)
        # Convert to dict for JSON serialization
        trend_dict = trend_data.to_dict(orient='records')
        
        # Convert dates to strings
        for record in trend_dict:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        return {"trend_data": trend_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/high-risk-skus")
async def get_high_risk_skus(top_n: int = 100):
    """Get high-risk SKUs predicted by ML model"""
    try:
        # Prepare ML features
        ml_data = data_processor.prepare_ml_features()
        
        # Check if model is trained
        if risk_model.model is None:
            # Train model if not available
            metrics = risk_model.train_model(ml_data)
            risk_model.save_model()
        
        # Get high-risk SKUs
        high_risk_skus = risk_model.get_high_risk_skus(ml_data, top_n=top_n)
        
        return {"high_risk_skus": high_risk_skus}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anomaly-detection")
async def detect_anomalies(metric: str = "order_count", threshold: float = 2.0):
    """Detect anomalies in order patterns"""
    try:
        valid_metrics = ["order_count", "total_qty", "avg_price", "revenue"]
        if metric not in valid_metrics:
            raise HTTPException(status_code=400, detail=f"Invalid metric. Choose from: {valid_metrics}")
        
        anomalies = data_processor.detect_anomalies(metric=metric, threshold=threshold)
        return {"anomalies": anomalies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/category-performance")
async def get_category_performance():
    """Get performance metrics by category"""
    try:
        category_data = data_processor.get_category_performance()
        return {"category_performance": category_data.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regional-performance")
async def get_regional_performance():
    """Get performance metrics by region"""
    try:
        regional_data = data_processor.get_regional_performance()
        return {"regional_performance": regional_data.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/delivery-performance")
async def get_delivery_performance():
    """Get delivery performance metrics"""
    try:
        delivery_metrics = data_processor.get_delivery_performance()
        # Convert DataFrame to dict
        if 'delay_by_region' in delivery_metrics:
            delivery_metrics['delay_by_region'] = delivery_metrics['delay_by_region'].to_dict(orient='records')
        return delivery_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/demand-forecast")
async def get_demand_forecast(sku_ids: str, days_ahead: int = 7):
    """Get demand forecast for specific SKUs"""
    try:
        sku_list = [sku.strip() for sku in sku_ids.split(',')]
        orders_df = data_processor.data['orders']
        
        forecasts = demand_model.forecast_demand(orders_df, sku_list, days_ahead)
        return {"demand_forecasts": forecasts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
async def train_model(background_tasks: BackgroundTasks):
    """Train/retrain the ML model"""
    try:
        # Prepare data
        ml_data = data_processor.prepare_ml_features()
        
        # Train model
        metrics = risk_model.train_model(ml_data)
        risk_model.save_model()
        
        return {
            "message": "Model training completed",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the current ML model"""
    try:
        model_info = risk_model.get_model_metrics()
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest-data")
async def ingest_data():
    """Reload data from CSV files"""
    try:
        success = data_processor.load_data('data/data')
        if success:
            return {"message": "Data ingested successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
