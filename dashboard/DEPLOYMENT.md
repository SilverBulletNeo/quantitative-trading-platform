# Dashboard Deployment Guide

Complete guide for deploying the Quantitative Trading Dashboard in production using Docker.

## Quick Start

### Development (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
python app.py
```

Access at: http://localhost:8050

### Production (Docker)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Stop
docker-compose down
```

Access at: http://localhost:8050

## Docker Deployment

### 1. Basic Deployment

**Build the image:**
```bash
cd dashboard
docker build -t trading-dashboard:latest .
```

**Run the container:**
```bash
docker run -d \
  --name trading-dashboard \
  -p 8050:8050 \
  -v $(pwd)/data:/app/dashboard/data \
  -v $(pwd)/reports:/app/dashboard/reports \
  -v $(pwd)/exports:/app/dashboard/exports \
  -v $(pwd)/config:/app/dashboard/config \
  trading-dashboard:latest
```

**Check logs:**
```bash
docker logs -f trading-dashboard
```

### 2. Docker Compose (Recommended)

**Start all services:**
```bash
docker-compose up -d
```

**View running containers:**
```bash
docker-compose ps
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dashboard
```

**Stop services:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose up -d --build
```

### 3. Production Deployment with Nginx

**Enable Nginx reverse proxy:**
```bash
docker-compose --profile production up -d
```

This starts:
- Dashboard (port 8050, internal)
- Nginx (port 80, public)

**Configure SSL (recommended):**

1. Obtain SSL certificates (Let's Encrypt):
```bash
# Install certbot
sudo apt-get install certbot

# Get certificates
sudo certbot certonly --standalone -d your-domain.com
```

2. Copy certificates:
```bash
mkdir -p ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
```

3. Update nginx.conf:
   - Uncomment HTTPS server block
   - Update `server_name` to your domain
   - Uncomment SSL certificate paths

4. Restart Nginx:
```bash
docker-compose --profile production restart nginx
```

### 4. Monitoring (Optional)

**Start with monitoring stack:**
```bash
docker-compose --profile monitoring up -d
```

This adds:
- Prometheus (port 9090) - Metrics collection
- Grafana (port 3000) - Visualization
  - Username: admin
  - Password: admin (change on first login)

**Configure Prometheus:**

Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dashboard'
    static_configs:
      - targets: ['dashboard:8050']
```

**Access Grafana:**
1. Navigate to http://localhost:3000
2. Login (admin/admin)
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboard or create custom

## Environment Variables

Configure via docker-compose.yml or .env file:

```bash
# Dashboard settings
DASH_DEBUG=false
DASH_HOST=0.0.0.0
DASH_PORT=8050

# Database
DB_PATH=/app/dashboard/data/dashboard.db

# Timezone
TZ=America/New_York

# Alert notifications (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## Data Persistence

All important data is persisted via Docker volumes:

```
dashboard/
├── data/           # SQLite database
├── reports/        # Generated PDF reports
├── exports/        # CSV exports
├── config/         # Alert configuration
└── logs/           # Application logs
```

**Backup data:**
```bash
# Backup database
docker cp trading-dashboard:/app/dashboard/data/dashboard.db ./backup/

# Or use volumes
tar -czf dashboard-backup-$(date +%Y%m%d).tar.gz data/ reports/ config/
```

**Restore data:**
```bash
# Stop container
docker-compose down

# Restore files
tar -xzf dashboard-backup-20251114.tar.gz

# Start container
docker-compose up -d
```

## Performance Tuning

### 1. Resource Limits

Add to docker-compose.yml:
```yaml
services:
  dashboard:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 2. Connection Pooling

For high traffic, increase workers:
```bash
# Use gunicorn for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8050", "app:server"]
```

Install gunicorn:
```bash
pip install gunicorn
```

### 3. Caching

Enable Redis for caching (optional):
```yaml
services:
  redis:
    image: redis:alpine
    container_name: trading-redis
    ports:
      - "6379:6379"
    networks:
      - trading-network
```

## Security Considerations

### 1. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8050/tcp  # Block direct dashboard access
sudo ufw enable
```

### 2. Environment Secrets

Use Docker secrets for sensitive data:
```yaml
secrets:
  smtp_password:
    file: ./secrets/smtp_password.txt
  slack_webhook:
    file: ./secrets/slack_webhook.txt

services:
  dashboard:
    secrets:
      - smtp_password
      - slack_webhook
```

### 3. Network Isolation

```yaml
networks:
  trading-network:
    driver: bridge
    internal: true  # Prevent external access
  web:
    driver: bridge
```

### 4. Read-Only Filesystem

Add to dashboard service:
```yaml
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

## Troubleshooting

### Container won't start

**Check logs:**
```bash
docker-compose logs dashboard
```

**Common issues:**
- Port 8050 already in use: Change port in docker-compose.yml
- Permission denied: Check volume permissions
- Missing dependencies: Rebuild image

### Database locked

```bash
# Stop all containers
docker-compose down

# Remove lock file
rm data/dashboard.db-journal

# Restart
docker-compose up -d
```

### High memory usage

```bash
# Check resource usage
docker stats

# Restart container
docker-compose restart dashboard
```

### Connection refused

```bash
# Check if container is running
docker ps

# Check network
docker network inspect trading-network

# Test internal connection
docker exec -it trading-dashboard curl http://localhost:8050
```

## Monitoring & Logs

### Real-time logs

```bash
# All services
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100 dashboard

# Specific time range
docker-compose logs --since 2h dashboard
```

### Log rotation

Add to docker-compose.yml:
```yaml
services:
  dashboard:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Health checks

```bash
# Manual health check
curl http://localhost:8050/

# Docker health status
docker inspect --format='{{.State.Health.Status}}' trading-dashboard
```

## Updating

### Update dashboard code

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose up -d --build
```

### Update dependencies

```bash
# Update requirements.txt
pip freeze > requirements.txt

# Rebuild image
docker-compose build --no-cache
docker-compose up -d
```

### Rolling updates (zero downtime)

```bash
# Scale up
docker-compose up -d --scale dashboard=2

# Wait for health check
sleep 30

# Scale down old
docker-compose stop dashboard
docker-compose rm -f dashboard

# Scale back to 1
docker-compose up -d --scale dashboard=1
```

## Production Checklist

Before deploying to production:

- [ ] Update Nginx configuration with domain name
- [ ] Install SSL certificates
- [ ] Configure firewall rules
- [ ] Set strong passwords in config files
- [ ] Enable log rotation
- [ ] Set resource limits
- [ ] Configure backup schedule
- [ ] Test health checks
- [ ] Configure monitoring alerts
- [ ] Document recovery procedures
- [ ] Test rollback process
- [ ] Enable HTTPS redirect
- [ ] Configure CORS if needed
- [ ] Set up reverse proxy
- [ ] Test email/Slack notifications

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-repo/issues
- Documentation: README.md
- Email: support@your-domain.com
