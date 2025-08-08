import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Tuple

class RiskPredictionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.model_path = 'models/risk_model.pkl'
        self.encoder_path = 'models/label_encoder.pkl'
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        # Select numeric features
        feature_cols = ['order_count', 'total_qty', 'avg_qty', 'avg_price', 
                       'stock_level', 'price_change_pct']
        
        # Add derived features
        data['demand_to_stock_ratio'] = data['total_qty'] / (data['stock_level'] + 1)
        data['order_frequency'] = data['order_count'] / data['total_qty'].max()
        data['price_volatility'] = abs(data['price_change_pct'])
        
        feature_cols.extend(['demand_to_stock_ratio', 'order_frequency', 'price_volatility'])
        
        self.feature_columns = feature_cols
        X = data[feature_cols].fillna(0)
        
        # Encode target labels
        y = self.label_encoder.fit_transform(data['risk_level'])
        
        return X.values, y
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """Train the XGBoost model"""
        print("Preparing features...")
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        feature_importance = {k: float(v) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}
        
        print(f"Model training completed! F1 Score: {f1:.3f}")
        
        return {
            'f1_score': f1,
            'classification_report': report,
            'feature_importance': feature_importance
        }
    
    def predict_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict risk levels for given data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        feature_cols = ['order_count', 'total_qty', 'avg_qty', 'avg_price', 
                       'stock_level', 'price_change_pct']
        
        # Add derived features
        data = data.copy()
        data['demand_to_stock_ratio'] = data['total_qty'] / (data['stock_level'] + 1)
        data['order_frequency'] = data['order_count'] / data['total_qty'].max()
        data['price_volatility'] = abs(data['price_change_pct'])
        
        feature_cols.extend(['demand_to_stock_ratio', 'order_frequency', 'price_volatility'])
        
        X = data[feature_cols].fillna(0)
        
        # Predict
        predictions = self.model.predict(X.values)
        probabilities = self.model.predict_proba(X.values)
        
        # Convert back to labels
        risk_labels = self.label_encoder.inverse_transform(predictions)
        
        # Create results dataframe
        results = data[['sku_id']].copy()
        results['predicted_risk'] = risk_labels
        results['confidence'] = np.max(probabilities, axis=1)
        
        # Add probability for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name.lower()}'] = probabilities[:, i]
        
        return results.sort_values('confidence', ascending=False)
    
    def get_high_risk_skus(self, data: pd.DataFrame, top_n: int = 100) -> List[Dict]:
        """Get top N high-risk SKUs with details"""
        predictions = self.predict_risk(data)
        
        # Filter high-risk SKUs
        high_risk = predictions[predictions['predicted_risk'] == 'High'].head(top_n)
        
        # Merge with original data for context
        detailed_results = high_risk.merge(data[['sku_id', 'stock_level', 'total_qty', 'order_count']], on='sku_id')
        
        risk_list = []
        for _, row in detailed_results.iterrows():
            risk_list.append({
                'sku_id': row['sku_id'],
                'predicted_risk': row['predicted_risk'],
                'confidence': round(row['confidence'], 3),
                'stock_level': row['stock_level'],
                'demand': row['total_qty'],
                'order_count': row['order_count'],
                'recommendation': self._generate_recommendation(row)
            })
        
        return risk_list
    
    def _generate_recommendation(self, row: pd.Series) -> str:
        """Generate actionable recommendation based on risk factors"""
        recommendations = []
        
        if row['stock_level'] < 10:
            recommendations.append("Urgent restocking required")
        elif row['stock_level'] < 50:
            recommendations.append("Monitor stock levels closely")
        
        if row['total_qty'] > row.get('avg_qty', 0) * 2:
            recommendations.append("High demand - consider increasing inventory")
        
        if row.get('price_volatility', 0) > 20:
            recommendations.append("Price instability detected - review pricing strategy")
        
        return "; ".join(recommendations) if recommendations else "Monitor closely"
    
    def save_model(self):
        """Save trained model and encoder"""
        os.makedirs('models', exist_ok=True)
        
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.label_encoder, self.encoder_path)
            print("Model saved successfully!")
    
    def load_model(self):
        """Load trained model and encoder"""
        try:
            self.model = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.encoder_path)
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved model found. Train the model first.")
            return False
    
    def get_model_metrics(self) -> Dict:
        """Get model performance metrics"""
        if self.model is None:
            return {"error": "Model not trained"}
        
        return {
            "model_type": "XGBoost Classifier",
            "features_count": len(self.feature_columns),
            "feature_names": self.feature_columns,
            "classes": list(self.label_encoder.classes_),
            "model_params": self.model.get_params()
        }

class DemandForecastModel:
    def __init__(self):
        self.models = {}  # Store models for different SKUs
        
    def prepare_time_series_data(self, orders_df: pd.DataFrame, sku_id: str) -> pd.DataFrame:
        """Prepare time series data for a specific SKU"""
        sku_orders = orders_df[orders_df['sku_id'] == sku_id].copy()
        sku_orders['order_date'] = pd.to_datetime(sku_orders['order_date'])
        
        # Aggregate by date
        daily_demand = sku_orders.groupby('order_date').agg({
            'qty': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        # Create features
        daily_demand['day_of_week'] = daily_demand['order_date'].dt.dayofweek
        daily_demand['month'] = daily_demand['order_date'].dt.month
        daily_demand['day_of_month'] = daily_demand['order_date'].dt.day
        
        # Lag features
        daily_demand['qty_lag1'] = daily_demand['qty'].shift(1)
        daily_demand['qty_lag7'] = daily_demand['qty'].shift(7)
        daily_demand['qty_rolling_7'] = daily_demand['qty'].rolling(7).mean()
        
        return daily_demand.dropna()
    
    def forecast_demand(self, orders_df: pd.DataFrame, sku_ids: List[str], days_ahead: int = 7) -> Dict:
        """Simple demand forecasting using moving averages"""
        forecasts = {}
        
        for sku_id in sku_ids:
            sku_data = self.prepare_time_series_data(orders_df, sku_id)
            
            if len(sku_data) > 14:  # Need minimum data
                # Simple forecast using exponential moving average
                recent_demand = sku_data['qty'].tail(14).mean()
                trend = sku_data['qty'].tail(7).mean() - sku_data['qty'].tail(14).head(7).mean()
                
                forecast = max(0, recent_demand + (trend * 0.5))  # Conservative trend adjustment
                
                forecasts[sku_id] = {
                    'forecasted_demand': round(forecast, 2),
                    'confidence': 'Medium' if len(sku_data) > 30 else 'Low',
                    'historical_avg': round(recent_demand, 2),
                    'trend': 'Increasing' if trend > 0 else 'Decreasing' if trend < 0 else 'Stable'
                }
            else:
                forecasts[sku_id] = {
                    'forecasted_demand': 0,
                    'confidence': 'Very Low',
                    'historical_avg': 0,
                    'trend': 'Insufficient Data'
                }
        
        return forecasts
