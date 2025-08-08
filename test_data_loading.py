#!/usr/bin/env python3
"""
Quick test script to verify data loading functionality
"""

import sys
import os
sys.path.append('src')

from src.data_processor import DataProcessor

def test_data_loading():
    """Test if data loads correctly"""
    print("🔍 Testing data loading...")
    
    processor = DataProcessor()
    
    # Test loading from correct path
    success = processor.load_data('data/data')
    
    if success:
        print("✅ Data loaded successfully!")
        
        # Print summary statistics
        orders_df = processor.data['orders']
        print(f"📊 Orders loaded: {len(orders_df):,}")
        print(f"📅 Date range: {orders_df['order_date'].min()} to {orders_df['order_date'].max()}")
        
        inventory_df = processor.data['inventory']
        print(f"📦 Inventory records: {len(inventory_df):,}")
        
        skus_df = processor.data['skus']
        print(f"🏷️ SKUs: {len(skus_df):,}")
        
        # Test KPI calculation
        print("\n📈 Testing KPI calculation...")
        kpis = processor.get_kpi_summary()
        print(f"📊 Total orders today: {kpis['total_orders_today']}")
        print(f"📉 Order drop %: {kpis['order_drop_percent']}")
        print(f"📋 Stock-out %: {kpis['stockout_percent']}")
        
        # Test trend data
        print("\n📈 Testing trend analysis...")
        trend_data = processor.get_order_trend(days=7)
        print(f"📊 Trend data points: {len(trend_data)}")
        
        print("\n🎉 All tests passed! Data is ready for dashboard.")
        return True
        
    else:
        print("❌ Failed to load data!")
        print("🔍 Checking if data files exist...")
        
        data_files = ['orders.csv', 'inventory.csv', 'skus.csv', 'price_history.csv', 'delivery.csv']
        for file in data_files:
            file_path = f'data/data/{file}'
            if os.path.exists(file_path):
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
        
        return False

if __name__ == "__main__":
    test_data_loading()
