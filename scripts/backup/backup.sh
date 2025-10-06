#!/bin/bash
# Backup Script for Voice AI Agent - Hetzner VPS Production
# Creates comprehensive backups of database, application data, and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="/app/data/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PREFIX="voiceai_backup_${TIMESTAMP}"

# Docker containers
DB_CONTAINER="voiceai_postgres_prod"
REDIS_CONTAINER="voiceai_redis_prod"
API_CONTAINER="voiceai_api_prod"

# Database configuration
DB_NAME="voiceai_db"
DB_USER="voiceai"

# Retention settings
KEEP_DAILY=7    # Keep daily backups for 7 days
KEEP_WEEKLY=4   # Keep weekly backups for 4 weeks
KEEP_MONTHLY=3  # Keep monthly backups for 3 months

echo -e "${GREEN}Voice AI Agent - Backup Script${NC}"
echo "==============================="
echo "Timestamp: $(date)"
echo "Backup directory: $BACKUP_DIR"
echo ""

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to handle errors
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Function to check if container is running
check_container() {
    local container=$1
    if ! docker ps | grep -q "$container"; then
        error_exit "Container $container is not running"
    fi
}

# Check if all required containers are running
log "Checking container status..."
check_container "$DB_CONTAINER"
check_container "$REDIS_CONTAINER"
check_container "$API_CONTAINER"

# Create backup subdirectory
BACKUP_PATH="$BACKUP_DIR/$BACKUP_PREFIX"
mkdir -p "$BACKUP_PATH"

log "Created backup directory: $BACKUP_PATH"

# 1. Database Backup
log "Starting PostgreSQL database backup..."
DB_BACKUP_FILE="$BACKUP_PATH/database_backup.sql"

# Create compressed database dump
if docker exec "$DB_CONTAINER" pg_dump -U "$DB_USER" -d "$DB_NAME" --verbose --no-owner --no-privileges > "$DB_BACKUP_FILE"; then
    log "Database backup completed: $DB_BACKUP_FILE"
    # Compress database backup
    gzip "$DB_BACKUP_FILE"
    log "Database backup compressed: ${DB_BACKUP_FILE}.gz"
else
    error_exit "Database backup failed"
fi

# 2. Redis Backup
log "Starting Redis backup..."
REDIS_BACKUP_FILE="$BACKUP_PATH/redis_backup.rdb"

if docker exec "$REDIS_CONTAINER" redis-cli BGSAVE; then
    # Wait for background save to complete
    sleep 5
    while docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE | grep -q "$(docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE)"; do
        sleep 1
    done
    
    # Copy the RDB file
    docker cp "$REDIS_CONTAINER:/data/dump.rdb" "$REDIS_BACKUP_FILE"
    log "Redis backup completed: $REDIS_BACKUP_FILE"
    
    # Compress Redis backup
    gzip "$REDIS_BACKUP_FILE"
    log "Redis backup compressed: ${REDIS_BACKUP_FILE}.gz"
else
    error_exit "Redis backup failed"
fi

# 3. Application Data Backup
log "Starting application data backup..."

# Backup models and cache
MODELS_BACKUP="$BACKUP_PATH/models_backup.tar.gz"
if docker exec "$API_CONTAINER" tar -czf - -C /app models cache 2>/dev/null > "$MODELS_BACKUP"; then
    log "Models and cache backup completed: $MODELS_BACKUP"
else
    log "Warning: Models and cache backup failed (may be empty)"
fi

# Backup uploaded files and user data
UPLOADS_BACKUP="$BACKUP_PATH/uploads_backup.tar.gz"
if docker exec "$API_CONTAINER" tar -czf - -C /app data/uploads 2>/dev/null > "$UPLOADS_BACKUP"; then
    log "Uploads backup completed: $UPLOADS_BACKUP"
else
    log "Warning: Uploads backup failed (may be empty)"
fi

# 4. Configuration Backup
log "Starting configuration backup..."
CONFIG_BACKUP="$BACKUP_PATH/config_backup.tar.gz"

# Create temporary directory for configuration files
TEMP_CONFIG_DIR=$(mktemp -d)
trap "rm -rf $TEMP_CONFIG_DIR" EXIT

# Copy configuration files
cp -r "$PROJECT_ROOT/docker" "$TEMP_CONFIG_DIR/"
cp "$PROJECT_ROOT/.env.production" "$TEMP_CONFIG_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/docker-compose.prod.yml" "$TEMP_CONFIG_DIR/" 2>/dev/null || true

# Create configuration backup
tar -czf "$CONFIG_BACKUP" -C "$TEMP_CONFIG_DIR" .
log "Configuration backup completed: $CONFIG_BACKUP"

# 5. Application Logs Backup
log "Starting logs backup..."
LOGS_BACKUP="$BACKUP_PATH/logs_backup.tar.gz"

# Copy logs from all containers
TEMP_LOGS_DIR=$(mktemp -d)
mkdir -p "$TEMP_LOGS_DIR/container_logs"

# Get logs from each container
for container in "$DB_CONTAINER" "$REDIS_CONTAINER" "$API_CONTAINER"; do
    if docker ps | grep -q "$container"; then
        docker logs "$container" > "$TEMP_LOGS_DIR/container_logs/${container}.log" 2>&1 || true
    fi
done

# Copy application logs from API container
docker cp "$API_CONTAINER:/app/logs" "$TEMP_LOGS_DIR/app_logs" 2>/dev/null || true

# Create logs backup
tar -czf "$LOGS_BACKUP" -C "$TEMP_LOGS_DIR" .
log "Logs backup completed: $LOGS_BACKUP"

# Clean up temporary directories
rm -rf "$TEMP_LOGS_DIR"

# 6. Create backup manifest
log "Creating backup manifest..."
MANIFEST_FILE="$BACKUP_PATH/backup_manifest.txt"

cat > "$MANIFEST_FILE" << EOF
Voice AI Agent Backup Manifest
==============================
Backup Date: $(date)
Backup Directory: $BACKUP_PATH
Hostname: $(hostname)
Docker Version: $(docker --version)

Backup Contents:
- database_backup.sql.gz: PostgreSQL database dump
- redis_backup.rdb.gz: Redis data dump
- models_backup.tar.gz: AI models and cache data
- uploads_backup.tar.gz: User uploaded files
- config_backup.tar.gz: Application configuration files
- logs_backup.tar.gz: Application and container logs
- backup_manifest.txt: This manifest file

Database Information:
- Database: $DB_NAME
- User: $DB_USER
- Tables: $(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "\dt" | wc -l) tables

File Sizes:
$(ls -lh "$BACKUP_PATH")

Container Status at Backup Time:
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

System Information:
- Memory Usage: $(free -h | grep Mem)
- Disk Usage: $(df -h | grep -E "/$|/app")
- Load Average: $(uptime)
EOF

log "Backup manifest created: $MANIFEST_FILE"

# 7. Calculate backup size and create summary
BACKUP_SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)
log "Total backup size: $BACKUP_SIZE"

# 8. Verify backup integrity
log "Verifying backup integrity..."

# Test database backup
if zcat "${DB_BACKUP_FILE}.gz" | head -n 10 | grep -q "PostgreSQL database dump"; then
    log "✓ Database backup integrity verified"
else
    error_exit "Database backup integrity check failed"
fi

# Test Redis backup
if file "${REDIS_BACKUP_FILE}.gz" | grep -q "gzip compressed"; then
    log "✓ Redis backup integrity verified"
else
    error_exit "Redis backup integrity check failed"
fi

# Test configuration backup
if tar -tzf "$CONFIG_BACKUP" | grep -q "docker"; then
    log "✓ Configuration backup integrity verified"
else
    error_exit "Configuration backup integrity check failed"
fi

# 9. Cleanup old backups
log "Cleaning up old backups..."

# Function to clean up old backups
cleanup_old_backups() {
    local backup_dir="$1"
    local keep_days="$2"
    local pattern="$3"
    
    find "$backup_dir" -name "$pattern" -type d -mtime +$keep_days -exec rm -rf {} + 2>/dev/null || true
}

# Clean up based on retention policy
cleanup_old_backups "$BACKUP_DIR" "$KEEP_DAILY" "voiceai_backup_*"

# Keep weekly backups (every Sunday)
if [ "$(date +%u)" = "7" ]; then
    log "Creating weekly backup marker..."
    touch "$BACKUP_PATH/.weekly_backup"
fi

# Keep monthly backups (first day of month)
if [ "$(date +%d)" = "01" ]; then
    log "Creating monthly backup marker..."
    touch "$BACKUP_PATH/.monthly_backup"
fi

# Clean up weekly backups older than KEEP_WEEKLY weeks
find "$BACKUP_DIR" -name ".weekly_backup" -type f -mtime +$((KEEP_WEEKLY * 7)) -delete 2>/dev/null || true

# Clean up monthly backups older than KEEP_MONTHLY months
find "$BACKUP_DIR" -name ".monthly_backup" -type f -mtime +$((KEEP_MONTHLY * 30)) -delete 2>/dev/null || true

log "Old backup cleanup completed"

# 10. Optional: Upload to remote storage (S3, etc.)
if [ "$BACKUP_S3_ENABLED" = "true" ] && [ -n "$BACKUP_S3_BUCKET" ]; then
    log "Uploading backup to S3..."
    
    # Create archive of entire backup
    BACKUP_ARCHIVE="$BACKUP_DIR/${BACKUP_PREFIX}.tar.gz"
    tar -czf "$BACKUP_ARCHIVE" -C "$BACKUP_DIR" "$BACKUP_PREFIX"
    
    # Upload to S3 (requires AWS CLI configured)
    if command -v aws &> /dev/null; then
        aws s3 cp "$BACKUP_ARCHIVE" "s3://$BACKUP_S3_BUCKET/voiceai-backups/"
        log "Backup uploaded to S3: s3://$BACKUP_S3_BUCKET/voiceai-backups/$(basename "$BACKUP_ARCHIVE")"
        
        # Remove local archive after successful upload
        rm "$BACKUP_ARCHIVE"
    else
        log "Warning: AWS CLI not found, skipping S3 upload"
    fi
fi

# 11. Final summary
echo ""
echo -e "${GREEN}Backup completed successfully!${NC}"
echo "==============================="
echo "Backup location: $BACKUP_PATH"
echo "Backup size: $BACKUP_SIZE"
echo "Files created:"
echo "  - database_backup.sql.gz"
echo "  - redis_backup.rdb.gz"
echo "  - models_backup.tar.gz"
echo "  - uploads_backup.tar.gz"
echo "  - config_backup.tar.gz"
echo "  - logs_backup.tar.gz"
echo "  - backup_manifest.txt"
echo ""
echo -e "${YELLOW}To restore from this backup, run:${NC}"
echo "./scripts/backup/restore.sh $BACKUP_PREFIX"
echo ""
echo -e "${BLUE}Backup completed at: $(date)${NC}"

# Exit successfully
exit 0