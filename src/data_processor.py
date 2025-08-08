import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.data = {}
        
    def load_data(self, data_dir='data'):
        """Load all CSV files into memory"""
        try:
            self.data['orders'] = pd.read_csv(f'{data_dir}/orders.csv')
            self.data['inventory'] = pd.read_csv(f'{data_dir}/inventory.csv')
            self.data['price_history'] = pd.read_csv(f'{data_dir}/price_history.csv')
            self.data['delivery'] = pd.read_csv(f'{data_dir}/delivery.csv')
            self.data['skus'] = pd.read_csv(f'{data_dir}/skus.csv')
            
            # Convert date columns
            self.data['orders']['order_date'] = pd.to_datetime(self.data['orders']['order_date'])
            self.data['price_history']['date'] = pd.to_datetime(self.data['price_history']['date'])
            self.data['inventory']['last_updated'] = pd.to_datetime(self.data['inventory']['last_updated'])
            
            print("Data loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_kpi_summary(self) -> Dict:
        """Calculate key performance indicators"""
        if 'orders' not in self.data:
            return {}
        
        orders_df = self.data['orders']
        today = orders_df['order_date'].max()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)
        
        # Filter for recent data
        today_orders = orders_df[orders_df['order_date'].dt.date == today.date()]
        yesterday_orders = orders_df[orders_df['order_date'].dt.date == yesterday.date()]
        week_orders = orders_df[orders_df['order_date'] >= last_week]
        
        # Calculate KPIs
        total_orders_today = len(today_orders)
        total_orders_yesterday = len(yesterday_orders)
        
        # Order drop percentage
        if total_orders_yesterday > 0:
            drop_percent = ((total_orders_yesterday - total_orders_today) / total_orders_yesterday) * 100
        else:
            drop_percent = 0
        
        # Stock-out percentage
        inventory_df = self.data['inventory']
        total_skus = len(inventory_df)
        stockout_skus = len(inventory_df[inventory_df['stock_level'] == 0])
        stockout_percent = (stockout_skus / total_skus) * 100
        
        # Average price change
        price_df = self.data['price_history']
        recent_prices = price_df.sort_values('date').groupby('sku_id').tail(2)
        price_changes = recent_prices.groupby('sku_id')['price'].pct_change().fillna(0)
        avg_price_change = price_changes.mean() * 100
        
        # Revenue metrics
        completed_orders = orders_df[orders_df['status'] == 'completed']
        total_revenue = (completed_orders['qty'] * completed_orders['price']).sum()
        avg_order_value = completed_orders['price'].mean()
        
        return {
            'total_orders_today': total_orders_today,
            'total_orders_yesterday': total_orders_yesterday,
            'order_drop_percent': round(drop_percent, 2),
            'stockout_percent': round(stockout_percent, 2),
            'avg_price_change_percent': round(avg_price_change, 2),
            'total_revenue': round(total_revenue, 2),
            'avg_order_value': round(avg_order_value, 2),
            'total_skus': total_skus,
            'stockout_skus': stockout_skus,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_order_trend(self, days=30) -> pd.DataFrame:
        """Get order trend data for the last N days"""
        orders_df = self.data['orders']
        cutoff_date = orders_df['order_date'].max() - timedelta(days=days)
        
        recent_orders = orders_df[orders_df['order_date'] >= cutoff_date]
        
        # Group by date and count orders
        trend_data = recent_orders.groupby(recent_orders['order_date'].dt.date).agg({
            'order_id': 'count',
            'qty': 'sum',
            'price': 'mean'
        }).reset_index()
        
        trend_data.columns = ['date', 'order_count', 'total_qty', 'avg_price']
        trend_data['revenue'] = recent_orders.groupby(recent_orders['order_date'].dt.date).apply(
            lambda x: (x['qty'] * x['price']).sum()
        ).values
        
        return trend_data
    
    def detect_anomalies(self, metric='order_count', window=7, threshold=2) -> List[Dict]:
        """Detect anomalies using Z-score method"""
        trend_data = self.get_order_trend(days=60)
        
        # Calculate rolling mean and std
        trend_data['rolling_mean'] = trend_data[metric].rolling(window=window).mean()
        trend_data['rolling_std'] = trend_data[metric].rolling(window=window).std()
        
        # Calculate Z-score
        trend_data['z_score'] = (trend_data[metric] - trend_data['rolling_mean']) / trend_data['rolling_std']
        
        # Identify anomalies
        anomalies = trend_data[abs(trend_data['z_score']) > threshold]
        
        anomaly_list = []
        for _, row in anomalies.iterrows():
            anomaly_list.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'metric': metric,
                'value': row[metric],
                'expected': row['rolling_mean'],
                'z_score': row['z_score'],
                'severity': 'High' if abs(row['z_score']) > 3 else 'Medium'
            })
        
        return anomaly_list
    
    def get_category_performance(self) -> pd.DataFrame:
        """Get performance metrics by category"""
        orders_df = self.data['orders']
        
        category_stats = orders_df.groupby('category').agg({
            'order_id': 'count',
            'qty': 'sum',
            'price': 'mean'
        }).reset_index()
        
        category_stats.columns = ['category', 'order_count', 'total_qty', 'avg_price']
        category_stats['revenue'] = orders_df.groupby('category').apply(
            lambda x: (x['qty'] * x['price']).sum()
        ).values
        
        return category_stats.sort_values('revenue', ascending=False)
    
    def get_regional_performance(self) -> pd.DataFrame:
        """Get performance metrics by region"""
        orders_df = self.data['orders']
        
        regional_stats = orders_df.groupby('region').agg({
            'order_id': 'count',
            'qty': 'sum',
            'price': 'mean'
        }).reset_index()
        
        regional_stats.columns = ['region', 'order_count', 'total_qty', 'avg_price']
        regional_stats['revenue'] = orders_df.groupby('region').apply(
            lambda x: (x['qty'] * x['price']).sum()
        ).values
        
        return regional_stats.sort_values('revenue', ascending=False)
    
    def get_delivery_performance(self) -> Dict:
        """Get delivery performance metrics"""
        delivery_df = self.data['delivery']
        
        total_deliveries = len(delivery_df)
        delayed_deliveries = len(delivery_df[delivery_df['delay_flag'] == True])
        avg_delivery_time = delivery_df['actual_delivery_days'].mean()
        
        delay_by_region = delivery_df.groupby('region')['delay_flag'].agg(['count', 'sum']).reset_index()
        delay_by_region['delay_rate'] = delay_by_region['sum'] / delay_by_region['count'] * 100
        
        return {
            'total_deliveries': total_deliveries,
            'delayed_deliveries': delayed_deliveries,
            'delay_rate': round((delayed_deliveries / total_deliveries) * 100, 2),
            'avg_delivery_time': round(avg_delivery_time, 2),
            'delay_by_region': delay_by_region
        }
    
    def prepare_ml_features(self) -> pd.DataFrame:
        """Prepare features for ML model"""
        # Merge relevant data
        orders_df = self.data['orders']
        inventory_df = self.data['inventory']
        price_df = self.data['price_history']
        
        # Get latest price for each SKU
        latest_prices = price_df.sort_values('date').groupby('sku_id').tail(1)
        
        # Calculate demand features
        demand_features = orders_df.groupby('sku_id').agg({
            'order_id': 'count',
            'qty': ['sum', 'mean'],
            'price': 'mean'
        }).reset_index()
        
        demand_features.columns = ['sku_id', 'order_count', 'total_qty', 'avg_qty', 'avg_price']
        
        # Merge with inventory and price data
        ml_data = demand_features.merge(inventory_df[['sku_id', 'stock_level']], on='sku_id', how='left')
        ml_data = ml_data.merge(latest_prices[['sku_id', 'price']], on='sku_id', how='left', suffixes=('', '_current'))
        
        # Calculate price change
        ml_data['price_change_pct'] = ((ml_data['price_current'] - ml_data['avg_price']) / ml_data['avg_price']) * 100
        
        # Create risk labels (simplified logic)
        ml_data['risk_level'] = 'Low'
        ml_data.loc[ml_data['stock_level'] < 10, 'risk_level'] = 'High'
        ml_data.loc[(ml_data['stock_level'] < 50) & (ml_data['order_count'] > ml_data['order_count'].median()), 'risk_level'] = 'Medium'
        
        return ml_data.fillna(0)
