-- Initialize PostgreSQL database for E-Commerce Order Recovery System

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS ecommerce_recovery;

-- Switch to the database
\c ecommerce_recovery;

-- Create tables
CREATE TABLE IF NOT EXISTS skus (
    sku_id VARCHAR(20) PRIMARY KEY,
    category VARCHAR(50),
    base_price DECIMAL(10,2),
    brand VARCHAR(100),
    product_name VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(20) PRIMARY KEY,
    sku_id VARCHAR(20) REFERENCES skus(sku_id),
    order_date TIMESTAMP,
    qty INTEGER,
    price DECIMAL(10,2),
    status VARCHAR(20),
    region VARCHAR(20),
    source VARCHAR(50),
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inventory (
    sku_id VARCHAR(20) PRIMARY KEY REFERENCES skus(sku_id),
    stock_level INTEGER,
    last_updated TIMESTAMP,
    warehouse_location VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY,
    sku_id VARCHAR(20) REFERENCES skus(sku_id),
    price DECIMAL(10,2),
    date TIMESTAMP,
    change_reason VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS delivery (
    order_id VARCHAR(20) PRIMARY KEY REFERENCES orders(order_id),
    expected_delivery_days INTEGER,
    actual_delivery_days INTEGER,
    delay_flag BOOLEAN,
    delivery_partner VARCHAR(50),
    region VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_sku ON orders(sku_id);
CREATE INDEX IF NOT EXISTS idx_orders_region ON orders(region);
CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date);
CREATE INDEX IF NOT EXISTS idx_price_history_sku ON price_history(sku_id);
CREATE INDEX IF NOT EXISTS idx_inventory_stock ON inventory(stock_level);

-- Create views for common queries
CREATE OR REPLACE VIEW daily_order_summary AS
SELECT 
    DATE(order_date) as order_date,
    COUNT(*) as order_count,
    SUM(qty) as total_qty,
    AVG(price) as avg_price,
    SUM(qty * price) as total_revenue
FROM orders 
WHERE status = 'completed'
GROUP BY DATE(order_date)
ORDER BY order_date DESC;

CREATE OR REPLACE VIEW category_performance AS
SELECT 
    category,
    COUNT(*) as order_count,
    SUM(qty) as total_qty,
    AVG(price) as avg_price,
    SUM(qty * price) as total_revenue
FROM orders 
WHERE status = 'completed'
GROUP BY category
ORDER BY total_revenue DESC;

CREATE OR REPLACE VIEW regional_performance AS
SELECT 
    region,
    COUNT(*) as order_count,
    SUM(qty) as total_qty,
    AVG(price) as avg_price,
    SUM(qty * price) as total_revenue
FROM orders 
WHERE status = 'completed'
GROUP BY region
ORDER BY total_revenue DESC;

-- Create alert configuration table
CREATE TABLE IF NOT EXISTS alert_config (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50),
    threshold_value DECIMAL(10,2),
    is_enabled BOOLEAN DEFAULT true,
    notification_channels TEXT[], -- JSON array of channels
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default alert configurations
INSERT INTO alert_config (alert_type, threshold_value, notification_channels) VALUES
('order_drop_percent', 20.0, '["email", "slack"]'),
('stockout_percent', 15.0, '["email", "webhook"]'),
('delivery_delay_percent', 25.0, '["email"]'),
('high_risk_sku_count', 100, '["slack", "webhook"]');

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for alert_config
CREATE TRIGGER update_alert_config_updated_at 
    BEFORE UPDATE ON alert_config 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
