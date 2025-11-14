"""
Alert Backend - Email & Slack Notifications

Sends real-time notifications for critical trading alerts via email and Slack.
Monitors drawdowns, Sharpe degradation, and strategy failures.

Usage:
    from dashboard.alert_backend import AlertManager

    manager = AlertManager()
    manager.send_alert(
        severity='CRITICAL',
        category='DRAWDOWN',
        message='Circuit breaker triggered: -20.5% drawdown',
        strategy='equity_momentum_90d'
    )
"""

import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import os
from pathlib import Path

from dashboard.database import DatabaseManager, Alert


class AlertConfig:
    """Alert notification configuration"""

    def __init__(self, config_path: str = 'dashboard/config/alert_config.json'):
        """Load alert configuration from JSON file"""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': '',
                    'sender_password': '',
                    'recipient_emails': []
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': '',
                    'channel': '#trading-alerts',
                    'username': 'Trading Alert Bot'
                },
                'alert_rules': {
                    'critical_only': False,
                    'throttle_minutes': 15,  # Don't spam same alert
                    'business_hours_only': False
                },
                'severity_colors': {
                    'CRITICAL': '#ff4444',
                    'WARNING': '#ffaa00',
                    'INFO': '#00d4ff'
                }
            }

    def save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def is_email_enabled(self) -> bool:
        """Check if email notifications are enabled"""
        return self.config['email']['enabled']

    def is_slack_enabled(self) -> bool:
        """Check if Slack notifications are enabled"""
        return self.config['slack']['enabled']


class EmailNotifier:
    """Email notification handler"""

    def __init__(self, config: AlertConfig):
        """Initialize email notifier with configuration"""
        self.config = config
        self.email_config = config.config['email']

    def send_email(self,
                   severity: str,
                   category: str,
                   message: str,
                   strategy: str,
                   threshold: Optional[float] = None,
                   actual: Optional[float] = None) -> bool:
        """
        Send email notification

        Args:
            severity: Alert severity (CRITICAL, WARNING, INFO)
            category: Alert category (DRAWDOWN, SHARPE, etc)
            message: Alert message
            strategy: Strategy name
            threshold: Threshold value that was breached
            actual: Actual value

        Returns:
            True if sent successfully
        """
        if not self.config.is_email_enabled():
            print("⚠️  Email notifications disabled in config")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 {severity}: {category} - {strategy}"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])

            # Create HTML email body
            html_body = self._create_html_email(
                severity, category, message, strategy,
                threshold, actual
            )

            # Attach HTML
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

            # Connect to SMTP server
            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['sender_email'],
                    self.email_config['sender_password']
                )
                server.send_message(msg)

            print(f"✅ Email sent: {severity} - {category}")
            return True

        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False

    def _create_html_email(self,
                          severity: str,
                          category: str,
                          message: str,
                          strategy: str,
                          threshold: Optional[float],
                          actual: Optional[float]) -> str:
        """Create HTML email body"""

        severity_colors = self.config.config['severity_colors']
        color = severity_colors.get(severity, '#00d4ff')

        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Courier New', monospace;
                    background-color: #0a0e27;
                    color: #e0e0e0;
                    padding: 20px;
                }}
                .alert-container {{
                    background: linear-gradient(135deg, #1a1f3a 0%, #2a3f5f 100%);
                    border-left: 6px solid {color};
                    padding: 30px;
                    border-radius: 8px;
                    max-width: 600px;
                    margin: 0 auto;
                }}
                .alert-header {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {color};
                    margin-bottom: 20px;
                }}
                .alert-body {{
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .metric {{
                    background-color: #151a30;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .metric-label {{
                    color: #8a9ba8;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                .metric-value {{
                    color: #00d4ff;
                    font-size: 20px;
                    font-weight: bold;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #2a3f5f;
                    font-size: 12px;
                    color: #8a9ba8;
                }}
            </style>
        </head>
        <body>
            <div class="alert-container">
                <div class="alert-header">
                    🚨 {severity}: {category}
                </div>
                <div class="alert-body">
                    <p><strong>Strategy:</strong> {strategy}</p>
                    <p><strong>Message:</strong> {message}</p>

                    {f'''
                    <div class="metric">
                        <div class="metric-label">Threshold</div>
                        <div class="metric-value">{threshold}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Actual Value</div>
                        <div class="metric-value">{actual}</div>
                    </div>
                    ''' if threshold is not None else ''}

                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="footer">
                    <p>⚡ Quantitative Trading Platform - Alert System</p>
                    <p>This is an automated notification. Please review the dashboard for details.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html


class SlackNotifier:
    """Slack notification handler"""

    def __init__(self, config: AlertConfig):
        """Initialize Slack notifier with configuration"""
        self.config = config
        self.slack_config = config.config['slack']

    def send_slack_message(self,
                          severity: str,
                          category: str,
                          message: str,
                          strategy: str,
                          threshold: Optional[float] = None,
                          actual: Optional[float] = None) -> bool:
        """
        Send Slack notification

        Args:
            severity: Alert severity (CRITICAL, WARNING, INFO)
            category: Alert category (DRAWDOWN, SHARPE, etc)
            message: Alert message
            strategy: Strategy name
            threshold: Threshold value
            actual: Actual value

        Returns:
            True if sent successfully
        """
        if not self.config.is_slack_enabled():
            print("⚠️  Slack notifications disabled in config")
            return False

        try:
            # Create Slack message payload
            payload = self._create_slack_payload(
                severity, category, message, strategy,
                threshold, actual
            )

            # Send to Slack webhook
            response = requests.post(
                self.slack_config['webhook_url'],
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                print(f"✅ Slack notification sent: {severity} - {category}")
                return True
            else:
                print(f"❌ Slack notification failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Failed to send Slack notification: {e}")
            return False

    def _create_slack_payload(self,
                             severity: str,
                             category: str,
                             message: str,
                             strategy: str,
                             threshold: Optional[float],
                             actual: Optional[float]) -> Dict:
        """Create Slack message payload with blocks"""

        severity_colors = self.config.config['severity_colors']
        color = severity_colors.get(severity, '#00d4ff')

        # Emoji based on severity
        emoji_map = {
            'CRITICAL': ':rotating_light:',
            'WARNING': ':warning:',
            'INFO': ':information_source:'
        }
        emoji = emoji_map.get(severity, ':bell:')

        # Build blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {severity}: {category}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Strategy:*\n{strategy}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Message:*\n{message}"
                }
            }
        ]

        # Add threshold/actual values if present
        if threshold is not None and actual is not None:
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Threshold:*\n`{threshold}`"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Actual Value:*\n`{actual}`"
                    }
                ]
            })

        # Add divider
        blocks.append({"type": "divider"})

        # Create payload
        payload = {
            "username": self.slack_config['username'],
            "channel": self.slack_config['channel'],
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks
                }
            ]
        }

        return payload


class AlertManager:
    """
    Alert Management System

    Coordinates email and Slack notifications for trading alerts.
    Handles throttling, alert history, and notification logic.
    """

    def __init__(self,
                 db_path: str = 'dashboard/data/dashboard.db',
                 config_path: str = 'dashboard/config/alert_config.json'):
        """Initialize alert manager"""
        self.db_manager = DatabaseManager(db_path)
        self.config = AlertConfig(config_path)
        self.email_notifier = EmailNotifier(self.config)
        self.slack_notifier = SlackNotifier(self.config)
        self._alert_cache = {}  # Track recent alerts for throttling

    def send_alert(self,
                   severity: str,
                   category: str,
                   message: str,
                   strategy: str,
                   threshold: Optional[float] = None,
                   actual: Optional[float] = None,
                   force: bool = False) -> bool:
        """
        Send alert via configured channels (email and/or Slack)

        Args:
            severity: CRITICAL, WARNING, or INFO
            category: Alert category (DRAWDOWN, SHARPE, etc)
            message: Alert message
            strategy: Strategy name
            threshold: Threshold value that was breached
            actual: Actual value
            force: Force send even if throttled

        Returns:
            True if at least one notification was sent
        """
        # Check if we should throttle this alert
        if not force and self._should_throttle(severity, category, strategy):
            print(f"⚠️  Alert throttled: {severity} - {category} (sent recently)")
            return False

        # Check business hours restriction
        if self.config.config['alert_rules']['business_hours_only']:
            if not self._is_business_hours():
                print("⚠️  Alert suppressed: outside business hours")
                return False

        # Check severity filter
        if self.config.config['alert_rules']['critical_only']:
            if severity != 'CRITICAL':
                print(f"⚠️  Alert suppressed: non-critical (config set to critical_only)")
                return False

        print(f"\n{'='*80}")
        print(f"SENDING ALERT: {severity} - {category}")
        print(f"Strategy: {strategy}")
        print(f"Message: {message}")
        print(f"{'='*80}\n")

        sent = False

        # Send email notification
        if self.config.is_email_enabled():
            email_sent = self.email_notifier.send_email(
                severity, category, message, strategy,
                threshold, actual
            )
            sent = sent or email_sent

        # Send Slack notification
        if self.config.is_slack_enabled():
            slack_sent = self.slack_notifier.send_slack_message(
                severity, category, message, strategy,
                threshold, actual
            )
            sent = sent or slack_sent

        if sent:
            # Update throttle cache
            alert_key = f"{strategy}:{category}:{severity}"
            self._alert_cache[alert_key] = datetime.now()

            # Log to database
            self._log_notification(severity, category, message, strategy)

        return sent

    def _should_throttle(self, severity: str, category: str, strategy: str) -> bool:
        """Check if alert should be throttled (sent too recently)"""
        alert_key = f"{strategy}:{category}:{severity}"

        if alert_key in self._alert_cache:
            last_sent = self._alert_cache[alert_key]
            throttle_minutes = self.config.config['alert_rules']['throttle_minutes']

            time_since = (datetime.now() - last_sent).total_seconds() / 60
            if time_since < throttle_minutes:
                return True

        return False

    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours (9 AM - 5 PM ET, weekdays)"""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Business hours check (9 AM - 5 PM)
        if now.hour < 9 or now.hour >= 17:
            return False

        return True

    def _log_notification(self,
                         severity: str,
                         category: str,
                         message: str,
                         strategy: str):
        """Log notification to database"""
        session = self.db_manager.get_session()
        try:
            self.db_manager.add_alert(
                session,
                strategy_name=strategy,
                severity=severity,
                category=category,
                message=message,
                threshold_value=None,
                actual_value=None
            )
        finally:
            session.close()

    def monitor_alerts(self, strategy: str) -> List[Dict]:
        """
        Monitor recent alerts and send notifications for unacknowledged ones

        Args:
            strategy: Strategy name to monitor

        Returns:
            List of alerts that triggered notifications
        """
        session = self.db_manager.get_session()
        notifications_sent = []

        try:
            # Get unacknowledged alerts from last hour
            recent_time = datetime.now() - timedelta(hours=1)

            alerts = session.query(Alert).filter(
                Alert.strategy_name == strategy,
                Alert.acknowledged == False,
                Alert.resolved == False,
                Alert.timestamp >= recent_time
            ).all()

            for alert in alerts:
                # Send notification
                sent = self.send_alert(
                    severity=alert.severity,
                    category=alert.category,
                    message=alert.message,
                    strategy=alert.strategy_name,
                    threshold=alert.threshold_value,
                    actual=alert.actual_value
                )

                if sent:
                    notifications_sent.append({
                        'id': alert.id,
                        'severity': alert.severity,
                        'category': alert.category,
                        'message': alert.message
                    })

        finally:
            session.close()

        return notifications_sent


def create_default_config():
    """Create default configuration file with instructions"""
    config_path = 'dashboard/config/alert_config.json'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config = {
        'email': {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your-email@gmail.com',
            'sender_password': 'your-app-password',
            'recipient_emails': ['recipient1@example.com', 'recipient2@example.com']
        },
        'slack': {
            'enabled': False,
            'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            'channel': '#trading-alerts',
            'username': 'Trading Alert Bot'
        },
        'alert_rules': {
            'critical_only': False,
            'throttle_minutes': 15,
            'business_hours_only': False
        },
        'severity_colors': {
            'CRITICAL': '#ff4444',
            'WARNING': '#ffaa00',
            'INFO': '#00d4ff'
        },
        '_instructions': {
            'email_setup': 'For Gmail: Enable 2FA and create an App Password at https://myaccount.google.com/apppasswords',
            'slack_setup': 'Create a webhook at https://api.slack.com/messaging/webhooks',
            'throttle_minutes': 'Minimum time between duplicate alerts (prevents spam)',
            'business_hours_only': 'Set to true to only send alerts during market hours (9 AM - 5 PM ET, weekdays)'
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created default config: {config_path}")
    print("\nTo enable notifications:")
    print("1. Edit dashboard/config/alert_config.json")
    print("2. Set 'enabled': true for email and/or Slack")
    print("3. Add your credentials (email) or webhook URL (Slack)")
    print("\nFor Gmail: https://myaccount.google.com/apppasswords")
    print("For Slack: https://api.slack.com/messaging/webhooks")


if __name__ == '__main__':
    """Test alert backend and create config"""

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'create-config':
        create_default_config()
    else:
        print("\nTesting Alert Backend\n")

        # Create config if doesn't exist
        if not os.path.exists('dashboard/config/alert_config.json'):
            create_default_config()

        # Test alert manager
        manager = AlertManager()

        print("\nSending test alert...")
        manager.send_alert(
            severity='WARNING',
            category='DRAWDOWN',
            message='Test alert: Drawdown approaching -15%',
            strategy='equity_momentum_90d',
            threshold=-15.0,
            actual=-14.5,
            force=True
        )

        print("\n✅ Alert backend test complete")
        print("\nNote: Notifications will only be sent if enabled in config")
        print("Run: python dashboard/alert_backend.py create-config")
