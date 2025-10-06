# Voice AI Agent - Hetzner VPS Deployment Guide

Complete deployment guide for deploying Voice AI Agent on Hetzner VPS with production-ready configuration.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Server Setup](#server-setup)
- [Security Hardening](#security-hardening)
- [Application Deployment](#application-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Backup Configuration](#backup-configuration)
- [SSL/TLS Setup](#ssltls-setup)
- [CI/CD Pipeline](#cicd-pipeline)
- [Maintenance](#maintenance)
- [Troubleshooting](#troubleshooting)

## Overview

This guide covers deploying Voice AI Agent on a Hetzner VPS with:
- Production-optimized Docker configuration
- Comprehensive security hardening
- Automated backup and recovery
- Monitoring with Prometheus and Grafana
- CI/CD pipeline with GitHub Actions
- SSL/TLS encryption with Let's Encrypt

### Architecture Overview

```
Internet → Cloudflare → NGINX → API Gateway → Voice AI Services
                    ↓
              [Hetzner VPS]
              ├── Docker Containers
              │   ├── API (FastAPI)
              │   ├── PostgreSQL
              │   ├── Redis
              │   ├── Ollama LLM
              │   ├── Celery Workers
              │   └── Monitoring Stack
              ├── Backup System
              └── Security Hardening
```

## Prerequisites

### Hetzner VPS Requirements

**Recommended Specifications:**
- **CPU**: 8+ vCPUs (Intel or AMD)
- **RAM**: 32GB+ (64GB recommended for production)
- **Storage**: 160GB+ SSD
- **Network**: 1 Gbit/s connection
- **OS**: Ubuntu 22.04 LTS

**Cost Estimate:** ~€50-80/month for suitable configuration

### Domain and DNS
- Domain name pointing to your Hetzner VPS IP
- Cloudflare account (recommended for DDoS protection)

### Local Development Tools
- Docker and Docker Compose
- Git
- SSH client
- Text editor (VS Code recommended)

## Server Setup

### 1. Initial Server Configuration

```bash
# Connect to your Hetzner VPS
ssh root@your-server-ip

# Update system packages
apt update && apt upgrade -y

# Create deployment user
adduser voiceai
usermod -aG sudo voiceai
usermod -aG docker voiceai

# Set up SSH key authentication
mkdir -p /home/voiceai/.ssh
cp ~/.ssh/authorized_keys /home/voiceai/.ssh/
chown -R voiceai:voiceai /home/voiceai/.ssh
chmod 700 /home/voiceai/.ssh
chmod 600 /home/voiceai/.ssh/authorized_keys
```

### 2. Install Required Software

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install additional tools
apt install -y git curl wget unzip htop neofetch

# Verify installations
docker --version
docker-compose --version
```

### 3. Prepare Application Directory

```bash
# Switch to deployment user
su - voiceai

# Create application directory
sudo mkdir -p /opt/voiceai
sudo chown voiceai:voiceai /opt/voiceai
cd /opt/voiceai

# Clone repository (or upload files)
git clone https://github.com/yourusername/voice-ai-agent.git .
```

## Security Hardening

### 1. Run Security Hardening Script

```bash
# Make script executable
chmod +x scripts/security/harden-server.sh

# Run as root
sudo ./scripts/security/harden-server.sh
```

The script configures:
- UFW firewall with restrictive rules
- Fail2Ban intrusion detection
- SSH hardening
- Kernel security parameters
- File integrity monitoring
- Docker security

### 2. Configure SSH Key Authentication

```bash
# Generate SSH key pair on your local machine
ssh-keygen -t ed25519 -C "your-email@example.com"

# Copy public key to server
ssh-copy-id voiceai@your-server-ip

# Test SSH connection
ssh voiceai@your-server-ip
```

### 3. Update Environment Configuration

```bash
# Copy production environment template
cp .env.production .env

# Edit configuration
nano .env
```

**Key Configuration Items:**
```bash
# Database
POSTGRES_PASSWORD=your-secure-password
DATABASE_URL=postgresql+asyncpg://voiceai:your-secure-password@postgres:5432/voiceai_db

# Security
SECRET_KEY=your-super-secret-key-change-this
CORS_ORIGINS=https://yourdomain.com

# External Services
ZOHO_CLIENT_ID=your-zoho-client-id
ZOHO_CLIENT_SECRET=your-zoho-client-secret
ELEVENLABS_API_KEY=your-elevenlabs-key

# Monitoring
GRAFANA_ADMIN_PASSWORD=your-grafana-password
```

## Application Deployment

### 1. Initial Deployment

```bash
# Navigate to project directory
cd /opt/voiceai

# Build and start services
docker-compose -f docker/docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker/docker-compose.prod.yml ps

# View logs
docker-compose -f docker/docker-compose.prod.yml logs -f api
```

### 2. Database Initialization

```bash
# Run database migrations
docker-compose -f docker/docker-compose.prod.yml exec api alembic upgrade head

# Initialize database schemas
docker-compose -f docker/docker-compose.prod.yml exec postgres psql -U voiceai -d voiceai_db -f /docker-entrypoint-initdb.d/init.sql

# Optimize database
./scripts/db/optimize-db.sh
```

### 3. Download AI Models

```bash
# Download Whisper models
docker-compose -f docker/docker-compose.prod.yml exec api python -c "import whisper; whisper.load_model('base')"

# Download Ollama models
docker-compose -f docker/docker-compose.prod.yml exec ollama ollama pull llama2:7b-chat

# Verify models
docker-compose -f docker/docker-compose.prod.yml exec api ls -la /app/models/
```

## Monitoring Setup

### 1. Access Monitoring Services

**Grafana Dashboard:**
- URL: `http://your-domain:3000`
- Username: `admin`
- Password: Set in `.env` file

**Prometheus:**
- URL: `http://your-domain:9090`

### 2. Configure Grafana Dashboards

Import pre-configured dashboards:
1. Voice AI System Overview
2. Application Performance
3. Infrastructure Metrics
4. Security Monitoring

### 3. Set Up Alerting

Configure alerts for:
- High CPU/Memory usage
- API response time degradation
- Database connectivity issues
- Failed voice processing

## Backup Configuration

### 1. Set Up Automated Backups

```bash
# Run backup setup script
sudo ./scripts/backup/setup-cron.sh

# Test manual backup
./scripts/backup/backup.sh

# Verify backup
ls -la /app/data/backups/
```

### 2. Configure Remote Backup (Optional)

For additional security, configure S3 backup:

```bash
# Install AWS CLI
sudo apt install awscli

# Configure AWS credentials
aws configure

# Update environment variables
echo "BACKUP_S3_ENABLED=true" >> .env
echo "BACKUP_S3_BUCKET=your-backup-bucket" >> .env
```

### 3. Test Backup Recovery

```bash
# Test restore process
./scripts/backup/restore.sh backup_name

# Verify system after restore
./scripts/deploy/smoke-tests.sh
```

## SSL/TLS Setup

### 1. Install Certbot

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Stop nginx temporarily
sudo docker-compose -f docker/docker-compose.prod.yml stop nginx
```

### 2. Obtain SSL Certificate

```bash
# Get certificate
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Verify certificate
sudo certbot certificates
```

### 3. Update NGINX Configuration

```bash
# Update nginx configuration to use Let's Encrypt certificates
sudo nano docker/nginx/sites-enabled/voiceai.conf

# Update certificate paths:
ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
```

### 4. Set Up Auto-Renewal

```bash
# Add renewal cron job
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -

# Test renewal
sudo certbot renew --dry-run
```

## CI/CD Pipeline

### 1. GitHub Repository Setup

1. Fork or create repository
2. Add repository secrets in GitHub Settings > Secrets:

```
STAGING_HOST=staging.yourdomain.com
STAGING_USER=voiceai
STAGING_SSH_KEY=<private-key-content>

PRODUCTION_HOST=yourdomain.com
PRODUCTION_USER=voiceai
PRODUCTION_SSH_KEY=<private-key-content>

SLACK_WEBHOOK_URL=<slack-webhook-url>
```

### 2. Workflow Configuration

The provided GitHub Actions workflow (`.github/workflows/deploy-hetzner.yml`) includes:
- Code quality checks
- Security scanning
- Automated testing
- Docker image building
- Staging deployment
- Production deployment
- Smoke tests

### 3. Deployment Process

**Automatic Deployment:**
- Push to `main` branch → deploys to staging
- Push to `production` branch → deploys to production
- Create tag `v*` → creates release and deploys to production

**Manual Deployment:**
```bash
# Using deployment script
./scripts/deploy/deploy.sh production latest

# Run smoke tests
./scripts/deploy/smoke-tests.sh
```

## Maintenance

### 1. Regular Maintenance Tasks

**Daily (Automated):**
- Database backups
- Log rotation
- Security updates
- Health checks

**Weekly:**
- Backup verification
- Performance monitoring review
- Security scan
- Disk cleanup

**Monthly:**
- Full system backup
- Security audit
- Performance optimization
- Update dependencies

### 2. Monitoring and Alerts

**Key Metrics to Monitor:**
- Response time < 2 seconds
- CPU usage < 80%
- Memory usage < 85%
- Disk usage < 80%
- Error rate < 1%

**Alert Channels:**
- Slack notifications
- Email alerts
- SMS for critical issues

### 3. Update Process

```bash
# Pull latest changes
git pull origin main

# Run deployment
./scripts/deploy/deploy.sh production $(git rev-parse --short HEAD)

# Run smoke tests
./scripts/deploy/smoke-tests.sh
```

## Troubleshooting

### Common Issues

**1. Container Won't Start**
```bash
# Check logs
docker-compose -f docker/docker-compose.prod.yml logs service-name

# Check disk space
df -h

# Check memory
free -h
```

**2. Database Connection Issues**
```bash
# Check PostgreSQL status
docker-compose -f docker/docker-compose.prod.yml exec postgres pg_isready

# Check database logs
docker-compose -f docker/docker-compose.prod.yml logs postgres

# Reset database connection
docker-compose -f docker/docker-compose.prod.yml restart postgres
```

**3. High Memory Usage**
```bash
# Check container memory usage
docker stats

# Restart memory-intensive services
docker-compose -f docker/docker-compose.prod.yml restart api ollama

# Clear cache
docker system prune -f
```

**4. SSL Certificate Issues**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate
sudo certbot renew

# Update nginx configuration
docker-compose -f docker/docker-compose.prod.yml restart nginx
```

### Performance Optimization

**1. Database Optimization**
```bash
# Run optimization script
./scripts/db/optimize-db.sh

# Monitor query performance
docker-compose -f docker/docker-compose.prod.yml exec postgres psql -U voiceai -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**2. Application Tuning**
```bash
# Adjust worker processes
docker-compose -f docker/docker-compose.prod.yml up -d --scale api=4

# Monitor resource usage
htop
```

### Emergency Procedures

**1. Service Outage**
```bash
# Quick health check
./scripts/deploy/smoke-tests.sh

# Emergency restart
docker-compose -f docker/docker-compose.prod.yml restart

# Rollback to previous version
./scripts/deploy/deploy.sh production previous-tag
```

**2. Data Corruption**
```bash
# Restore from latest backup
./scripts/backup/restore.sh latest-backup-name

# Verify data integrity
./scripts/deploy/smoke-tests.sh
```

## Support and Resources

### Documentation
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
- [API Documentation](../api/README.md) - API reference
- [Architecture Overview](../architecture/README.md) - System design

### Monitoring URLs
- Grafana: `https://yourdomain.com:3000`
- Prometheus: `https://yourdomain.com:9090`
- API Health: `https://yourdomain.com/api/v1/health`

### Emergency Contacts
- System Administrator: admin@yourdomain.com
- On-call Engineer: oncall@yourdomain.com
- Hetzner Support: https://console.hetzner.cloud

---

## Deployment Checklist

- [ ] Hetzner VPS provisioned with sufficient resources
- [ ] Domain name configured and DNS pointing to VPS
- [ ] SSH key authentication set up
- [ ] Security hardening script executed
- [ ] Environment variables configured
- [ ] Docker containers deployed and running
- [ ] Database initialized and optimized
- [ ] AI models downloaded
- [ ] SSL certificates installed and configured
- [ ] Monitoring and alerting configured
- [ ] Backup system set up and tested
- [ ] CI/CD pipeline configured
- [ ] Smoke tests passing
- [ ] Documentation updated
- [ ] Team trained on deployment procedures

**Deployment Complete! 🚀**

Your Voice AI Agent is now running securely on Hetzner VPS with production-grade configuration.