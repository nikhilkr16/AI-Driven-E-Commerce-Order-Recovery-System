import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import sys
import os

# Add source directory to path
sys.path.append('src')
from data_processor import DataProcessor
from ml_model import RiskPredictionModel

# Page configuration
st.set_page_config(
    page_title="E-Commerce Order Recovery Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffcc02;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #e6ffe6;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
    st.session_state.risk_model = RiskPredictionModel()
    st.session_state.data_loaded = False

def load_data():
    """Load data and initialize models"""
    try:
        success = st.session_state.data_processor.load_data('data/data')
        if success:
            st.session_state.data_loaded = True
            return True
        return False
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return False

def generate_mock_data():
    """Generate mock data if not exists"""
    try:
        from data.mock_data_generator import MockDataGenerator
        generator = MockDataGenerator(n_skus=20000, n_days=365)
        data = generator.generate_all_data()
        generator.save_data(data)
        st.success("Mock data generated successfully!")
        return True
    except Exception as e:
        st.error(f"Error generating mock data: {e}")
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 E-Commerce Order Recovery Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Overview Dashboard",
        "📈 Order Analytics",
        "🤖 AI Risk Prediction",
        "📍 Regional Analysis",
        "🚚 Delivery Performance",
        "⚠️ Anomaly Detection",
        "🔧 System Management"
    ])
    
    # Data loading section
    if not st.session_state.data_loaded:
        st.sidebar.warning("Data not loaded")
        if st.sidebar.button("🔄 Load Data"):
            with st.spinner("Loading data..."):
                if not load_data():
                    st.sidebar.error("Failed to load data")
                    if st.sidebar.button("📊 Generate Mock Data"):
                        with st.spinner("Generating mock data..."):
                            if generate_mock_data():
                                load_data()
    else:
        st.sidebar.success("✅ Data loaded")
        if st.sidebar.button("🔄 Refresh Data"):
            with st.spinner("Refreshing data..."):
                load_data()
    
    # Main content based on selected page
    if not st.session_state.data_loaded:
        st.warning("Please load or generate data to view the dashboard")
        return
    
    if page == "📊 Overview Dashboard":
        show_overview_dashboard()
    elif page == "📈 Order Analytics":
        show_order_analytics()
    elif page == "🤖 AI Risk Prediction":
        show_ai_risk_prediction()
    elif page == "📍 Regional Analysis":
        show_regional_analysis()
    elif page == "🚚 Delivery Performance":
        show_delivery_performance()
    elif page == "⚠️ Anomaly Detection":
        show_anomaly_detection()
    elif page == "🔧 System Management":
        show_system_management()

def show_overview_dashboard():
    """Show main KPI dashboard"""
    st.title("📊 Real-time KPI Overview")
    
    # Get KPI data
    kpis = st.session_state.data_processor.get_kpi_summary()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Orders Today",
            value=kpis['total_orders_today'],
            delta=kpis['total_orders_today'] - kpis['total_orders_yesterday']
        )
    
    with col2:
        order_drop = kpis['order_drop_percent']
        st.metric(
            label="📉 Order Drop %",
            value=f"{order_drop:.1f}%",
            delta=f"{order_drop:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="📋 Stock-out %",
            value=f"{kpis['stockout_percent']:.1f}%",
            delta=f"{kpis['stockout_skus']} SKUs"
        )
    
    with col4:
        st.metric(
            label="💰 Avg Order Value",
            value=f"₹{kpis['avg_order_value']:.0f}",
            delta=f"{kpis['avg_price_change_percent']:.1f}%"
        )
    
    # Alert Section
    st.subheader("🚨 Critical Alerts")
    
    # Order drop alert
    if order_drop > 20:
        st.markdown(f"""
        <div class="alert-high">
            <strong>⚠️ CRITICAL: Order Drop Alert</strong><br>
            Orders have dropped by {order_drop:.1f}% compared to yesterday. Immediate investigation required.
        </div>
        """, unsafe_allow_html=True)
    elif order_drop > 10:
        st.markdown(f"""
        <div class="alert-medium">
            <strong>⚠️ WARNING: Order Drop Alert</strong><br>
            Orders have dropped by {order_drop:.1f}% compared to yesterday. Monitor closely.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-card">
            <strong>✅ Order Volume Normal</strong><br>
            No significant order drops detected.
        </div>
        """, unsafe_allow_html=True)
    
    # Stock-out alert
    if kpis['stockout_percent'] > 15:
        st.markdown(f"""
        <div class="alert-high">
            <strong>⚠️ CRITICAL: High Stock-out Rate</strong><br>
            {kpis['stockout_percent']:.1f}% of SKUs are out of stock ({kpis['stockout_skus']} SKUs).
        </div>
        """, unsafe_allow_html=True)
    elif kpis['stockout_percent'] > 10:
        st.markdown(f"""
        <div class="alert-medium">
            <strong>⚠️ WARNING: Elevated Stock-out Rate</strong><br>
            {kpis['stockout_percent']:.1f}% of SKUs are out of stock.
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Order Trend (Last 30 Days)")
        trend_data = st.session_state.data_processor.get_order_trend(days=30)
        
        fig = px.line(trend_data, x='date', y='order_count', 
                     title="Daily Order Count Trend")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue Trend (Last 30 Days)")
        
        fig = px.bar(trend_data, x='date', y='revenue', 
                    title="Daily Revenue Trend")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Category Performance
    st.subheader("🏷️ Category Performance")
    category_data = st.session_state.data_processor.get_category_performance()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(category_data, values='revenue', names='category', 
                    title="Revenue by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(category_data, x='category', y='order_count', 
                    title="Orders by Category")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_order_analytics():
    """Show detailed order analytics"""
    st.title("📈 Order Analytics Deep Dive")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2)
    with col2:
        metric = st.selectbox("Metric", ["order_count", "revenue", "avg_price", "total_qty"])
    
    # Get trend data
    trend_data = st.session_state.data_processor.get_order_trend(days=days_back)
    
    # Main trend chart
    st.subheader(f"📊 {metric.replace('_', ' ').title()} Trend")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data['date'], 
        y=trend_data[metric],
        mode='lines+markers',
        name=metric.replace('_', ' ').title(),
        line=dict(width=3)
    ))
    
    # Add moving average
    trend_data['ma_7'] = trend_data[metric].rolling(window=7).mean()
    fig.add_trace(go.Scatter(
        x=trend_data['date'], 
        y=trend_data['ma_7'],
        mode='lines',
        name='7-day Moving Average',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average", f"{trend_data[metric].mean():.1f}")
    with col2:
        st.metric("Maximum", f"{trend_data[metric].max():.1f}")
    with col3:
        st.metric("Minimum", f"{trend_data[metric].min():.1f}")
    with col4:
        st.metric("Std Dev", f"{trend_data[metric].std():.1f}")
    
    # Correlation analysis
    st.subheader("🔗 Metric Correlations")
    corr_data = trend_data[['order_count', 'revenue', 'avg_price', 'total_qty']].corr()
    
    fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                   title="Correlation Matrix of Key Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data table
    st.subheader("📋 Detailed Data")
    st.dataframe(trend_data.style.format({
        'revenue': '₹{:.0f}',
        'avg_price': '₹{:.2f}',
        'order_count': '{:.0f}',
        'total_qty': '{:.0f}'
    }))

def show_ai_risk_prediction():
    """Show AI-powered risk prediction dashboard"""
    st.title("🤖 AI Risk Prediction Dashboard")
    
    # Model training section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🔧 Model Status")
    with col2:
        if st.button("🚀 Train Model"):
            with st.spinner("Training AI model..."):
                try:
                    ml_data = st.session_state.data_processor.prepare_ml_features()
                    metrics = st.session_state.risk_model.train_model(ml_data)
                    st.session_state.risk_model.save_model()
                    st.success(f"Model trained! F1 Score: {metrics['f1_score']:.3f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    
    # Try to load existing model
    try:
        if st.session_state.risk_model.model is None:
            st.session_state.risk_model.load_model()
        
        # Model metrics
        if st.session_state.risk_model.model is not None:
            st.success("✅ AI Model Ready")
            
            # Get predictions
            ml_data = st.session_state.data_processor.prepare_ml_features()
            high_risk_skus = st.session_state.risk_model.get_high_risk_skus(ml_data, top_n=100)
            
            # Risk summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_risk_count = len([sku for sku in high_risk_skus if sku['predicted_risk'] == 'High'])
                st.metric("🔴 High Risk SKUs", high_risk_count)
            
            with col2:
                medium_risk_count = len(ml_data[ml_data['risk_level'] == 'Medium'])
                st.metric("🟡 Medium Risk SKUs", medium_risk_count)
            
            with col3:
                low_risk_count = len(ml_data[ml_data['risk_level'] == 'Low'])
                st.metric("🟢 Low Risk SKUs", low_risk_count)
            
            # Risk distribution
            st.subheader("📊 Risk Distribution")
            risk_counts = ml_data['risk_level'].value_counts()
            
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title="SKU Risk Level Distribution",
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
            st.plotly_chart(fig, use_container_width=True)
            
            # High-risk SKUs table
            st.subheader("🚨 High-Risk SKUs (Top 20)")
            
            if high_risk_skus:
                risk_df = pd.DataFrame(high_risk_skus[:20])
                
                # Color-coded display
                def highlight_risk(val):
                    if val == 'High':
                        return 'background-color: #ffebee'
                    return ''
                
                styled_df = risk_df.style.applymap(highlight_risk, subset=['predicted_risk'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Recommendations summary
                st.subheader("💡 AI Recommendations")
                recommendations = {}
                for sku in high_risk_skus[:10]:
                    rec = sku['recommendation']
                    recommendations[rec] = recommendations.get(rec, 0) + 1
                
                for rec, count in recommendations.items():
                    st.markdown(f"• **{rec}** - {count} SKUs")
            else:
                st.info("No high-risk SKUs identified")
                
        else:
            st.warning("⚠️ AI Model not available. Please train the model first.")
            
    except Exception as e:
        st.error(f"Error in AI prediction: {e}")

def show_regional_analysis():
    """Show regional performance analysis"""
    st.title("📍 Regional Performance Analysis")
    
    # Get regional data
    regional_data = st.session_state.data_processor.get_regional_performance()
    
    # Regional metrics
    st.subheader("🌏 Regional Overview")
    
    # Create columns for each region
    cols = st.columns(len(regional_data))
    
    for i, (_, region) in enumerate(regional_data.iterrows()):
        with cols[i]:
            st.metric(
                label=f"📍 {region['region']}",
                value=f"₹{region['revenue']:,.0f}",
                delta=f"{region['order_count']} orders"
            )
    
    # Regional charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(regional_data, x='region', y='revenue',
                    title="Revenue by Region",
                    color='revenue',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(regional_data, x='order_count', y='avg_price',
                        size='revenue', color='region',
                        title="Orders vs Avg Price by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional heatmap
    st.subheader("🗺️ Regional Performance Heatmap")
    
    # Simulate geographical coordinates for regions
    region_coords = {
        'North': [28.6139, 77.2090],
        'South': [13.0827, 80.2707],
        'East': [22.5726, 88.3639],
        'West': [19.0760, 72.8777],
        'Central': [23.2599, 77.4126]
    }
    
    # Create map data
    map_data = regional_data.copy()
    map_data['lat'] = map_data['region'].map(lambda x: region_coords.get(x, [0, 0])[0])
    map_data['lon'] = map_data['region'].map(lambda x: region_coords.get(x, [0, 0])[1])
    
    fig = px.scatter_mapbox(map_data, lat='lat', lon='lon', size='revenue',
                           color='order_count', hover_name='region',
                           hover_data={'revenue': ':,.0f', 'order_count': True},
                           mapbox_style='open-street-map',
                           title="Revenue and Orders by Region",
                           zoom=4, height=500)
    fig.update_layout(mapbox_center_lat=22, mapbox_center_lon=78)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed regional table
    st.subheader("📊 Detailed Regional Metrics")
    st.dataframe(regional_data.style.format({
        'revenue': '₹{:,.0f}',
        'avg_price': '₹{:.2f}',
        'order_count': '{:,.0f}',
        'total_qty': '{:,.0f}'
    }))

def show_delivery_performance():
    """Show delivery performance dashboard"""
    st.title("🚚 Delivery Performance Dashboard")
    
    # Get delivery data
    delivery_metrics = st.session_state.data_processor.get_delivery_performance()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Total Deliveries", f"{delivery_metrics['total_deliveries']:,}")
    
    with col2:
        st.metric("⏰ Delayed Deliveries", f"{delivery_metrics['delayed_deliveries']:,}")
    
    with col3:
        st.metric("📊 Delay Rate", f"{delivery_metrics['delay_rate']:.1f}%")
    
    with col4:
        st.metric("🕐 Avg Delivery Time", f"{delivery_metrics['avg_delivery_time']:.1f} days")
    
    # Delivery performance alerts
    if delivery_metrics['delay_rate'] > 25:
        st.markdown("""
        <div class="alert-high">
            <strong>⚠️ CRITICAL: High Delivery Delays</strong><br>
            Delivery delay rate is above 25%. Immediate action required.
        </div>
        """, unsafe_allow_html=True)
    elif delivery_metrics['delay_rate'] > 15:
        st.markdown("""
        <div class="alert-medium">
            <strong>⚠️ WARNING: Elevated Delivery Delays</strong><br>
            Delivery delay rate is above normal levels.
        </div>
        """, unsafe_allow_html=True)
    
    # Regional delivery performance
    st.subheader("📍 Delivery Performance by Region")
    
    delay_by_region = pd.DataFrame(delivery_metrics['delay_by_region'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(delay_by_region, x='region', y='delay_rate',
                    title="Delay Rate by Region (%)",
                    color='delay_rate',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(delay_by_region, values='count', names='region',
                    title="Total Deliveries by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    # Delivery performance table
    st.subheader("📊 Regional Delivery Details")
    
    styled_delay = delay_by_region.style.format({
        'delay_rate': '{:.1f}%',
        'count': '{:,}',
        'sum': '{:,}'
    })
    
    st.dataframe(styled_delay, use_container_width=True)

def show_anomaly_detection():
    """Show anomaly detection dashboard"""
    st.title("⚠️ Anomaly Detection Dashboard")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Metric to Analyze", 
                             ["order_count", "total_qty", "avg_price", "revenue"])
    with col2:
        threshold = st.slider("Anomaly Threshold (Z-score)", 1.5, 4.0, 2.0, 0.1)
    
    # Detect anomalies
    anomalies = st.session_state.data_processor.detect_anomalies(metric=metric, threshold=threshold)
    
    # Anomaly summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_anomalies = len(anomalies)
        st.metric("🚨 Total Anomalies", total_anomalies)
    
    with col2:
        high_severity = len([a for a in anomalies if a['severity'] == 'High'])
        st.metric("🔴 High Severity", high_severity)
    
    with col3:
        medium_severity = len([a for a in anomalies if a['severity'] == 'Medium'])
        st.metric("🟡 Medium Severity", medium_severity)
    
    if anomalies:
        # Anomaly timeline
        st.subheader("📅 Anomaly Timeline")
        
        anomaly_df = pd.DataFrame(anomalies)
        anomaly_df['date'] = pd.to_datetime(anomaly_df['date'])
        
        # Get trend data for context
        trend_data = st.session_state.data_processor.get_order_trend(days=60)
        
        fig = go.Figure()
        
        # Plot main trend
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data[metric],
            mode='lines+markers',
            name=f'{metric} Trend',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight anomalies
        high_anomalies = anomaly_df[anomaly_df['severity'] == 'High']
        medium_anomalies = anomaly_df[anomaly_df['severity'] == 'Medium']
        
        if not high_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=high_anomalies['date'],
                y=high_anomalies['value'],
                mode='markers',
                name='High Severity Anomalies',
                marker=dict(color='red', size=12, symbol='x')
            ))
        
        if not medium_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=medium_anomalies['date'],
                y=medium_anomalies['value'],
                mode='markers',
                name='Medium Severity Anomalies',
                marker=dict(color='orange', size=10, symbol='diamond')
            ))
        
        fig.update_layout(
            title=f'Anomaly Detection: {metric}',
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details
        st.subheader("📋 Anomaly Details")
        
        # Style anomalies based on severity
        def highlight_severity(val):
            if val == 'High':
                return 'background-color: #ffebee'
            elif val == 'Medium':
                return 'background-color: #fff3e0'
            return ''
        
        styled_anomalies = anomaly_df.style.applymap(highlight_severity, subset=['severity'])
        st.dataframe(styled_anomalies, use_container_width=True)
        
    else:
        st.success("✅ No anomalies detected in the selected time period!")
        st.info("This indicates normal business operations for the analyzed metric.")

def show_system_management():
    """Show system management and configuration"""
    st.title("🔧 System Management")
    
    # Data status
    st.subheader("📊 Data Status")
    
    if st.session_state.data_loaded:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            orders_count = len(st.session_state.data_processor.data['orders'])
            st.metric("📦 Total Orders", f"{orders_count:,}")
        
        with col2:
            skus_count = len(st.session_state.data_processor.data['skus'])
            st.metric("🏷️ Total SKUs", f"{skus_count:,}")
        
        with col3:
            last_order_date = st.session_state.data_processor.data['orders']['order_date'].max()
            st.metric("📅 Latest Order", last_order_date.strftime('%Y-%m-%d'))
    
    # Model management
    st.subheader("🤖 AI Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train New Model"):
            with st.spinner("Training model..."):
                try:
                    ml_data = st.session_state.data_processor.prepare_ml_features()
                    metrics = st.session_state.risk_model.train_model(ml_data)
                    st.session_state.risk_model.save_model()
                    st.success(f"Model trained successfully! F1 Score: {metrics['f1_score']:.3f}")
                    
                    # Show feature importance
                    st.subheader("📊 Feature Importance")
                    importance_df = pd.DataFrame(
                        list(metrics['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    )
                    fig = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h', title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Model training failed: {e}")
    
    with col2:
        if st.button("📈 Model Performance"):
            try:
                metrics = st.session_state.risk_model.get_model_metrics()
                if 'error' not in metrics:
                    st.json(metrics)
                else:
                    st.warning("No trained model available")
            except Exception as e:
                st.error(f"Error getting model metrics: {e}")
    
    # Data refresh
    st.subheader("🔄 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate New Mock Data"):
            with st.spinner("Generating fresh mock data..."):
                if generate_mock_data():
                    load_data()
    
    with col2:
        if st.button("🔄 Reload Existing Data"):
            with st.spinner("Reloading data..."):
                load_data()
    
    with col3:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    # System info
    st.subheader("ℹ️ System Information")
    
    system_info = {
        "Dashboard Version": "1.0.0",
        "Last Updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Data Source": "Local CSV Files",
        "ML Framework": "XGBoost + Scikit-learn",
        "Visualization": "Plotly + Streamlit"
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")

if __name__ == "__main__":
    main()
