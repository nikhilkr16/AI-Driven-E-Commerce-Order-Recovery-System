import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional
import os
from dataclasses import dataclass

@dataclass
class AlertConfig:
    email_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    recipient_emails: List[str] = None
    slack_webhook_url: str = ""
    custom_webhook_url: str = ""

class AlertManager:
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        if self.config.recipient_emails is None:
            self.config.recipient_emails = []
    
    def send_email_alert(self, subject: str, message: str, severity: str = "Medium"):
        """Send email alert"""
        if not self.config.email_enabled or not self.config.recipient_emails:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ", ".join(self.config.recipient_emails)
            msg['Subject'] = f"[{severity.upper()}] {subject}"
            
            # HTML email body
            html_body = f"""
            <html>
                <body>
                    <h2 style="color: {'red' if severity == 'High' else 'orange' if severity == 'Medium' else 'blue'};">
                        {severity} Alert: {subject}
                    </h2>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        {message.replace('\\n', '<br>')}
                    </div>
                    <hr>
                    <p style="font-size: 12px; color: #666;">
                        This is an automated alert from E-Commerce Order Recovery System
                    </p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            print(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
            return False
    
    def send_slack_alert(self, subject: str, message: str, severity: str = "Medium"):
        """Send Slack alert via webhook"""
        if not self.config.slack_enabled or not self.config.slack_webhook_url:
            return False
        
        try:
            # Color based on severity
            color_map = {
                "High": "#ff4444",
                "Medium": "#ffaa00", 
                "Low": "#00aa00"
            }
            
            slack_payload = {
                "text": f"{severity} Alert: {subject}",
                "attachments": [
                    {
                        "color": color_map.get(severity, "#cccccc"),
                        "fields": [
                            {
                                "title": "Alert Details",
                                "value": message,
                                "short": False
                            },
                            {
                                "title": "Severity",
                                "value": severity,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=slack_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Slack alert sent: {subject}")
                return True
            else:
                print(f"Slack alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
            return False
    
    def send_webhook_alert(self, subject: str, message: str, severity: str = "Medium", additional_data: Dict = None):
        """Send alert via custom webhook"""
        if not self.config.webhook_enabled or not self.config.custom_webhook_url:
            return False
        
        try:
            payload = {
                "alert_type": "order_recovery_system",
                "subject": subject,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "additional_data": additional_data or {}
            }
            
            response = requests.post(
                self.config.custom_webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                print(f"Webhook alert sent: {subject}")
                return True
            else:
                print(f"Webhook alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
            return False
    
    def send_alert(self, subject: str, message: str, severity: str = "Medium", additional_data: Dict = None):
        """Send alert via all configured channels"""
        results = {}
        
        if self.config.email_enabled:
            results['email'] = self.send_email_alert(subject, message, severity)
        
        if self.config.slack_enabled:
            results['slack'] = self.send_slack_alert(subject, message, severity)
        
        if self.config.webhook_enabled:
            results['webhook'] = self.send_webhook_alert(subject, message, severity, additional_data)
        
        return results

class AlertRules:
    """Define alert rules and conditions"""
    
    @staticmethod
    def check_order_drop_alert(current_orders: int, previous_orders: int, threshold_percent: float = 20.0) -> Optional[Dict]:
        """Check for sudden order drops"""
        if previous_orders == 0:
            return None
        
        drop_percent = ((previous_orders - current_orders) / previous_orders) * 100
        
        if drop_percent >= threshold_percent:
            severity = "High" if drop_percent >= 30 else "Medium"
            return {
                "subject": f"Order Drop Alert: {drop_percent:.1f}% decrease",
                "message": f"""
                Orders have dropped significantly:
                
                • Previous Period: {previous_orders:,} orders
                • Current Period: {current_orders:,} orders
                • Drop Percentage: {drop_percent:.1f}%
                
                Immediate investigation required to identify root cause.
                """,
                "severity": severity,
                "additional_data": {
                    "current_orders": current_orders,
                    "previous_orders": previous_orders,
                    "drop_percent": drop_percent
                }
            }
        return None
    
    @staticmethod
    def check_stockout_alert(stockout_percent: float, stockout_count: int, threshold_percent: float = 15.0) -> Optional[Dict]:
        """Check for high stock-out rates"""
        if stockout_percent >= threshold_percent:
            severity = "High" if stockout_percent >= 25 else "Medium"
            return {
                "subject": f"High Stock-out Alert: {stockout_percent:.1f}% of SKUs",
                "message": f"""
                High stock-out rate detected:
                
                • Stock-out Percentage: {stockout_percent:.1f}%
                • SKUs Out of Stock: {stockout_count:,}
                
                Urgent restocking may be required for critical SKUs.
                """,
                "severity": severity,
                "additional_data": {
                    "stockout_percent": stockout_percent,
                    "stockout_count": stockout_count
                }
            }
        return None
    
    @staticmethod
    def check_delivery_delay_alert(delay_rate: float, delayed_count: int, threshold_percent: float = 25.0) -> Optional[Dict]:
        """Check for high delivery delays"""
        if delay_rate >= threshold_percent:
            severity = "High" if delay_rate >= 35 else "Medium"
            return {
                "subject": f"Delivery Delay Alert: {delay_rate:.1f}% delayed",
                "message": f"""
                High delivery delay rate detected:
                
                • Delay Rate: {delay_rate:.1f}%
                • Delayed Deliveries: {delayed_count:,}
                
                Review delivery partner performance and logistics operations.
                """,
                "severity": severity,
                "additional_data": {
                    "delay_rate": delay_rate,
                    "delayed_count": delayed_count
                }
            }
        return None
    
    @staticmethod
    def check_high_risk_skus_alert(high_risk_count: int, total_skus: int, threshold_count: int = 100) -> Optional[Dict]:
        """Check for too many high-risk SKUs"""
        if high_risk_count >= threshold_count:
            risk_percent = (high_risk_count / total_skus) * 100
            severity = "High" if high_risk_count >= 200 else "Medium"
            return {
                "subject": f"High-Risk SKUs Alert: {high_risk_count} SKUs at risk",
                "message": f"""
                Large number of high-risk SKUs detected:
                
                • High-Risk SKUs: {high_risk_count:,}
                • Total SKUs: {total_skus:,}
                • Risk Percentage: {risk_percent:.1f}%
                
                Review inventory levels and demand patterns for these SKUs.
                """,
                "severity": severity,
                "additional_data": {
                    "high_risk_count": high_risk_count,
                    "total_skus": total_skus,
                    "risk_percent": risk_percent
                }
            }
        return None
    
    @staticmethod
    def check_anomaly_alert(anomalies: List[Dict], severity_threshold: str = "Medium") -> Optional[Dict]:
        """Check for significant anomalies"""
        high_severity_anomalies = [a for a in anomalies if a['severity'] == 'High']
        medium_severity_anomalies = [a for a in anomalies if a['severity'] == 'Medium']
        
        if high_severity_anomalies or (severity_threshold == "Medium" and medium_severity_anomalies):
            total_anomalies = len(anomalies)
            severity = "High" if high_severity_anomalies else "Medium"
            
            return {
                "subject": f"Anomaly Detection Alert: {total_anomalies} anomalies found",
                "message": f"""
                Significant anomalies detected in order patterns:
                
                • Total Anomalies: {total_anomalies}
                • High Severity: {len(high_severity_anomalies)}
                • Medium Severity: {len(medium_severity_anomalies)}
                
                Recent anomalies may indicate underlying business issues.
                """,
                "severity": severity,
                "additional_data": {
                    "total_anomalies": total_anomalies,
                    "high_severity_count": len(high_severity_anomalies),
                    "medium_severity_count": len(medium_severity_anomalies),
                    "anomalies": anomalies[:5]  # Include first 5 anomalies
                }
            }
        return None

class AlertScheduler:
    """Schedule and manage periodic alerts"""
    
    def __init__(self, alert_manager: AlertManager, data_processor):
        self.alert_manager = alert_manager
        self.data_processor = data_processor
        self.last_alert_times = {}
    
    def run_alert_checks(self):
        """Run all alert checks and send notifications"""
        alerts_sent = []
        
        try:
            # Get current KPIs
            kpis = self.data_processor.get_kpi_summary()
            
            # Check order drop
            order_alert = AlertRules.check_order_drop_alert(
                kpis['total_orders_today'],
                kpis['total_orders_yesterday']
            )
            if order_alert:
                results = self.alert_manager.send_alert(**order_alert)
                alerts_sent.append(("Order Drop", results))
            
            # Check stock-out
            stockout_alert = AlertRules.check_stockout_alert(
                kpis['stockout_percent'],
                kpis['stockout_skus']
            )
            if stockout_alert:
                results = self.alert_manager.send_alert(**stockout_alert)
                alerts_sent.append(("Stock-out", results))
            
            # Check delivery performance
            delivery_metrics = self.data_processor.get_delivery_performance()
            delivery_alert = AlertRules.check_delivery_delay_alert(
                delivery_metrics['delay_rate'],
                delivery_metrics['delayed_deliveries']
            )
            if delivery_alert:
                results = self.alert_manager.send_alert(**delivery_alert)
                alerts_sent.append(("Delivery Delay", results))
            
            # Check anomalies
            anomalies = self.data_processor.detect_anomalies()
            anomaly_alert = AlertRules.check_anomaly_alert(anomalies)
            if anomaly_alert:
                results = self.alert_manager.send_alert(**anomaly_alert)
                alerts_sent.append(("Anomaly Detection", results))
            
            print(f"Alert check completed. {len(alerts_sent)} alerts sent.")
            return alerts_sent
            
        except Exception as e:
            error_alert = {
                "subject": "Alert System Error",
                "message": f"Error during alert checks: {str(e)}",
                "severity": "High"
            }
            self.alert_manager.send_alert(**error_alert)
            print(f"Error in alert checks: {e}")
            return []

def create_sample_config():
    """Create a sample alert configuration"""
    return AlertConfig(
        email_enabled=False,  # Set to True and configure for email alerts
        slack_enabled=False,  # Set to True and configure for Slack alerts
        webhook_enabled=False,  # Set to True and configure for webhook alerts
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        email_username="your-email@gmail.com",
        email_password="your-app-password",
        recipient_emails=["admin@company.com", "operations@company.com"],
        slack_webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
        custom_webhook_url="https://your-api.com/webhook/alerts"
    )

# Example usage
if __name__ == "__main__":
    # Create sample configuration
    config = create_sample_config()
    
    # Initialize alert manager
    alert_manager = AlertManager(config)
    
    # Send test alert
    test_alert = {
        "subject": "Test Alert",
        "message": "This is a test alert from the E-Commerce Order Recovery System",
        "severity": "Low"
    }
    
    results = alert_manager.send_alert(**test_alert)
    print(f"Test alert results: {results}")
