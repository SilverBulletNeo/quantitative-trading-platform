# Alert Notification Setup Guide

Quick guide to configure email and Slack notifications for trading alerts.

## Prerequisites

- Gmail account (for email) OR Slack workspace (for Slack)
- Dashboard installed and running

## 1. Email Notifications (Gmail)

### Step 1: Enable 2-Factor Authentication

1. Go to https://myaccount.google.com/security
2. Enable "2-Step Verification"

### Step 2: Create App Password

1. Go to https://myaccount.google.com/apppasswords
2. Select "Mail" and "Other (Custom name)"
3. Enter "Trading Dashboard"
4. Click "Generate"
5. **Copy the 16-character password** (you'll need this)

### Step 3: Configure Dashboard

Edit `dashboard/config/alert_config.json`:

```json
{
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-16-char-app-password",
    "recipient_emails": [
      "your-email@gmail.com",
      "additional-recipient@example.com"
    ]
  }
}
```

### Step 4: Test

```bash
python -c "
from dashboard.alert_backend import AlertManager
manager = AlertManager()
manager.send_alert(
    severity='INFO',
    category='TEST',
    message='Email notification test',
    strategy='test',
    force=True
)
"
```

Check your email inbox!

## 2. Slack Notifications

### Step 1: Create Incoming Webhook

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. App Name: "Trading Alert Bot"
4. Select your workspace
5. Navigate to "Incoming Webhooks"
6. Activate "Incoming Webhooks"
7. Click "Add New Webhook to Workspace"
8. Select channel (e.g., #trading-alerts)
9. **Copy the webhook URL** (starts with `https://hooks.slack.com/services/...`)

### Step 2: Configure Dashboard

Edit `dashboard/config/alert_config.json`:

```json
{
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    "channel": "#trading-alerts",
    "username": "Trading Alert Bot"
  }
}
```

### Step 3: Test

```bash
python -c "
from dashboard.alert_backend import AlertManager
manager = AlertManager()
manager.send_alert(
    severity='INFO',
    category='TEST',
    message='Slack notification test',
    strategy='test',
    force=True
)
"
```

Check your Slack channel!

## 3. Alert Rules Configuration

### Throttling (Prevent Spam)

```json
{
  "alert_rules": {
    "throttle_minutes": 15
  }
}
```

Same alert won't be sent more than once every 15 minutes.

### Severity Filtering

```json
{
  "alert_rules": {
    "critical_only": true
  }
}
```

Only send CRITICAL alerts (ignore WARNING and INFO).

### Business Hours Only

```json
{
  "alert_rules": {
    "business_hours_only": true
  }
}
```

Only send alerts during market hours (9 AM - 5 PM ET, weekdays).

## 4. Alert Types

The dashboard automatically generates alerts for:

| Alert Category | Severity | Trigger |
|----------------|----------|---------|
| DRAWDOWN | CRITICAL | Current drawdown < -20% |
| DRAWDOWN | WARNING | Current drawdown < -15% |
| SHARPE | WARNING | Sharpe ratio < 1.0 |
| SHARPE | CRITICAL | Sharpe ratio < 0.5 |
| VOLATILITY | WARNING | 30-day vol > 25% |
| VAR | CRITICAL | VaR (95%) < -4% |
| OVERFITTING | CRITICAL | OOS Sharpe << in-sample Sharpe |

## 5. Custom Alerts

### From Dashboard Code

```python
from dashboard.alert_backend import AlertManager

manager = AlertManager()

# Send custom alert
manager.send_alert(
    severity='WARNING',  # CRITICAL, WARNING, INFO
    category='CUSTOM',
    message='Your custom alert message',
    strategy='equity_momentum_90d',
    threshold=-15.0,  # Optional
    actual=-14.5,      # Optional
    force=False        # Respect throttling
)
```

### Automated Monitoring

```python
# Monitor all unacknowledged alerts
alerts = manager.monitor_alerts('equity_momentum_90d')

# Returns list of alerts that triggered notifications
for alert in alerts:
    print(f"Sent: {alert['message']}")
```

## 6. Production Setup

### Scheduled Monitoring (Cron)

Add to crontab:

```bash
# Monitor every 15 minutes during market hours
*/15 9-17 * * 1-5 cd /path/to/dashboard && python -c "from alert_backend import AlertManager; AlertManager().monitor_alerts('equity_momentum_90d')"
```

### Docker Setup

If running in Docker, alerts will work automatically. Ensure config file is mounted:

```yaml
volumes:
  - ./config:/app/dashboard/config
```

## 7. Troubleshooting

### Email Not Sending

**Error: "Authentication failed"**
- Verify 2FA is enabled
- Regenerate app password
- Check sender_email matches Gmail account

**Error: "Connection refused"**
- Check smtp_port (587 for TLS, 465 for SSL)
- Verify firewall allows outbound connections

### Slack Not Sending

**Error: "Invalid webhook"**
- Verify webhook URL is correct
- Ensure webhook hasn't been deleted in Slack
- Check channel exists

**Messages not appearing**
- Verify channel name matches (#trading-alerts)
- Check bot isn't muted in channel

### Alerts Not Triggering

**No alerts appearing**
- Check alert thresholds in config
- Verify `critical_only` isn't filtering out alerts
- Check business_hours_only setting
- Look for throttling (same alert sent recently)

**Too many alerts**
- Increase throttle_minutes
- Enable critical_only
- Adjust thresholds (see database.py)

## 8. Security Best Practices

### Protect Credentials

```bash
# Never commit alert_config.json with real credentials
echo "dashboard/config/alert_config.json" >> .gitignore

# Use environment variables (optional)
export SMTP_PASSWORD="your-app-password"
export SLACK_WEBHOOK="your-webhook-url"
```

### Use Separate Email

Consider using a dedicated email account for trading notifications instead of your personal Gmail.

### Slack Permissions

- Create private channel for alerts
- Restrict webhook to specific channel
- Rotate webhook URL periodically

## 9. Testing

### Full Test Suite

```bash
cd dashboard
python alert_backend.py
```

### Individual Tests

```python
from dashboard.alert_backend import AlertManager

manager = AlertManager()

# Test email
manager.email_notifier.send_email(
    severity='INFO',
    category='TEST',
    message='Email test',
    strategy='test'
)

# Test Slack
manager.slack_notifier.send_slack_message(
    severity='INFO',
    category='TEST',
    message='Slack test',
    strategy='test'
)
```

## 10. Support

If you encounter issues:

1. Check logs: `dashboard/logs/alerts.log`
2. Verify configuration: `cat dashboard/config/alert_config.json`
3. Test connectivity: `ping smtp.gmail.com`
4. Review documentation: `dashboard/alert_backend.py`

## Complete Example Configuration

```json
{
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "trading.bot@gmail.com",
    "sender_password": "abcd efgh ijkl mnop",
    "recipient_emails": [
      "trader1@example.com",
      "trader2@example.com"
    ]
  },
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/T00/B00/XXXX",
    "channel": "#trading-alerts",
    "username": "Trading Alert Bot"
  },
  "alert_rules": {
    "critical_only": false,
    "throttle_minutes": 15,
    "business_hours_only": false
  },
  "severity_colors": {
    "CRITICAL": "#ff4444",
    "WARNING": "#ffaa00",
    "INFO": "#00d4ff"
  }
}
```

Now you're ready to receive real-time trading alerts! 🚨
