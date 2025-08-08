import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

fake = Faker()

class MockDataGenerator:
    def __init__(self, n_skus=20000, n_days=365):
        self.n_skus = n_skus
        self.n_days = n_days
        self.start_date = datetime.now() - timedelta(days=n_days)
        self.categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Books', 'Sports', 'Beauty', 'Toys', 'Automotive']
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        self.sources = ['Website', 'Mobile App', 'Social Media', 'Email Campaign', 'Direct']
        
    def generate_skus(self):
        """Generate SKU master data"""
        skus = []
        for i in range(self.n_skus):
            sku = {
                'sku_id': f'SKU_{i+1:06d}',
                'category': random.choice(self.categories),
                'base_price': round(random.uniform(10, 1000), 2),
                'brand': fake.company(),
                'product_name': fake.catch_phrase()
            }
            skus.append(sku)
        return pd.DataFrame(skus)
    
    def generate_orders(self, skus_df):
        """Generate historical orders data"""
        orders = []
        order_id = 1
        
        for day in range(self.n_days):
            current_date = self.start_date + timedelta(days=day)
            
            # Simulate seasonal patterns and weekly cycles
            base_orders = 1000
            if current_date.weekday() >= 5:  # Weekend boost
                base_orders *= 1.3
            if current_date.month in [11, 12]:  # Holiday season
                base_orders *= 1.5
            
            # Random fluctuation
            daily_orders = int(base_orders * random.uniform(0.7, 1.3))
            
            # Simulate sudden drops occasionally
            if random.random() < 0.05:  # 5% chance of drop
                daily_orders = int(daily_orders * random.uniform(0.3, 0.6))
            
            for _ in range(daily_orders):
                sku = skus_df.sample(1).iloc[0]
                
                order = {
                    'order_id': f'ORD_{order_id:08d}',
                    'sku_id': sku['sku_id'],
                    'order_date': current_date,
                    'qty': random.randint(1, 5),
                    'price': sku['base_price'] * random.uniform(0.8, 1.2),
                    'status': random.choices(['completed', 'cancelled', 'returned'], weights=[85, 10, 5])[0],
                    'region': random.choice(self.regions),
                    'source': random.choice(self.sources),
                    'category': sku['category']
                }
                orders.append(order)
                order_id += 1
        
        return pd.DataFrame(orders)
    
    def generate_inventory(self, skus_df):
        """Generate current inventory levels"""
        inventory = []
        for _, sku in skus_df.iterrows():
            stock_level = random.randint(0, 1000)
            # Some SKUs are out of stock
            if random.random() < 0.1:
                stock_level = 0
                
            inv = {
                'sku_id': sku['sku_id'],
                'stock_level': stock_level,
                'last_updated': datetime.now() - timedelta(hours=random.randint(0, 24)),
                'warehouse_location': random.choice(self.regions)
            }
            inventory.append(inv)
        
        return pd.DataFrame(inventory)
    
    def generate_price_history(self, skus_df):
        """Generate price change history"""
        price_history = []
        
        for _, sku in skus_df.iterrows():
            current_price = sku['base_price']
            
            for day in range(0, self.n_days, 7):  # Weekly price updates
                date = self.start_date + timedelta(days=day)
                
                # Random price changes
                if random.random() < 0.2:  # 20% chance of price change
                    price_change = random.uniform(-0.3, 0.5)
                    current_price = max(current_price * (1 + price_change), 5)
                
                price_hist = {
                    'sku_id': sku['sku_id'],
                    'price': round(current_price, 2),
                    'date': date,
                    'change_reason': random.choice(['market_adjustment', 'promotion', 'cost_increase', 'demand_based'])
                }
                price_history.append(price_hist)
        
        return pd.DataFrame(price_history)
    
    def generate_delivery_data(self, orders_df):
        """Generate delivery performance data"""
        delivery = []
        
        for _, order in orders_df.iterrows():
            if order['status'] == 'completed':
                expected_delivery = 3  # 3 days standard
                actual_delivery = random.randint(1, 10)
                
                delivery_record = {
                    'order_id': order['order_id'],
                    'expected_delivery_days': expected_delivery,
                    'actual_delivery_days': actual_delivery,
                    'delay_flag': actual_delivery > expected_delivery,
                    'delivery_partner': random.choice(['Partner_A', 'Partner_B', 'Partner_C']),
                    'region': order['region']
                }
                delivery.append(delivery_record)
        
        return pd.DataFrame(delivery)
    
    def generate_all_data(self):
        """Generate all mock datasets"""
        print("Generating SKU data...")
        skus_df = self.generate_skus()
        
        print("Generating orders data...")
        orders_df = self.generate_orders(skus_df)
        
        print("Generating inventory data...")
        inventory_df = self.generate_inventory(skus_df)
        
        print("Generating price history...")
        price_history_df = self.generate_price_history(skus_df)
        
        print("Generating delivery data...")
        delivery_df = self.generate_delivery_data(orders_df)
        
        return {
            'skus': skus_df,
            'orders': orders_df,
            'inventory': inventory_df,
            'price_history': price_history_df,
            'delivery': delivery_df
        }
    
    def save_data(self, data_dict, output_dir='data'):
        """Save all datasets to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            filepath = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"Saved {name} data to {filepath} ({len(df)} rows)")

if __name__ == "__main__":
    generator = MockDataGenerator(n_skus=20000, n_days=365)
    data = generator.generate_all_data()
    generator.save_data(data)
    print("Mock data generation completed!")
