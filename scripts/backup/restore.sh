#!/bin/bash
# Restore Script for Voice AI Agent - Hetzner VPS Production
# Restores from backup created by backup.sh

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

# Docker containers
DB_CONTAINER="voiceai_postgres_prod"
REDIS_CONTAINER="voiceai_redis_prod"
API_CONTAINER="voiceai_api_prod"

# Database configuration
DB_NAME="voiceai_db"
DB_USER="voiceai"

echo -e "${GREEN}Voice AI Agent - Restore Script${NC}"
echo "==============================="
echo "Timestamp: $(date)"
echo ""

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to handle errors
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Function to prompt for confirmation
confirm() {
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    read -p "Do you want to continue? (yes/no): " response
    case "$response" in
        yes|YES|y|Y) return 0 ;;
        *) error_exit "Operation cancelled by user" ;;
    esac
}

# Check if backup name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_name>"
    echo ""
    echo "Available backups:"
    ls -la "$BACKUP_DIR" | grep "voiceai_backup_" || echo "No backups found"
    exit 1
fi

BACKUP_NAME="$1"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Check if backup exists
if [ ! -d "$BACKUP_PATH" ]; then
    error_exit "Backup directory not found: $BACKUP_PATH"
fi

log "Backup found: $BACKUP_PATH"

# Display backup information
if [ -f "$BACKUP_PATH/backup_manifest.txt" ]; then
    echo -e "${YELLOW}Backup Information:${NC}"
    echo "==================="
    head -n 20 "$BACKUP_PATH/backup_manifest.txt"
    echo ""
fi

# Confirm restoration
confirm "This will restore the Voice AI Agent from backup: $BACKUP_NAME"
confirm "WARNING: This will overwrite current data. Are you sure?"

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

# 1. Stop application services (keep databases running)
log "Stopping application services..."
docker stop "$API_CONTAINER" || true

# Give services time to shut down gracefully
sleep 5

# 2. Restore Database
log "Restoring PostgreSQL database..."
DB_BACKUP_FILE="$BACKUP_PATH/database_backup.sql.gz"

if [ ! -f "$DB_BACKUP_FILE" ]; then
    error_exit "Database backup file not found: $DB_BACKUP_FILE"
fi

# Drop existing connections
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d postgres -c "
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '$DB_NAME'
  AND pid <> pg_backend_pid();
"

# Drop and recreate database
log "Dropping existing database..."
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"

# Restore database from backup
log "Restoring database from backup..."
if zcat "$DB_BACKUP_FILE" | docker exec -i "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME"; then
    log "Database restoration completed successfully"
else
    error_exit "Database restoration failed"
fi

# 3. Restore Redis
log "Restoring Redis data..."
REDIS_BACKUP_FILE="$BACKUP_PATH/redis_backup.rdb.gz"

if [ -f "$REDIS_BACKUP_FILE" ]; then
    # Stop Redis temporarily
    docker stop "$REDIS_CONTAINER"
    
    # Remove existing data
    docker volume rm voiceai_redis_data 2>/dev/null || true
    
    # Start Redis container
    docker start "$REDIS_CONTAINER"
    
    # Wait for Redis to start
    sleep 5
    
    # Flush existing data
    docker exec "$REDIS_CONTAINER" redis-cli FLUSHALL
    
    # Stop Redis for file restoration
    docker stop "$REDIS_CONTAINER"
    
    # Copy backup file to Redis data directory
    zcat "$REDIS_BACKUP_FILE" > /tmp/dump.rdb
    docker cp /tmp/dump.rdb "$REDIS_CONTAINER:/data/dump.rdb"
    rm /tmp/dump.rdb
    
    # Start Redis
    docker start "$REDIS_CONTAINER"
    
    # Wait for Redis to load data
    sleep 10
    
    log "Redis restoration completed successfully"
else
    log "Warning: Redis backup file not found, skipping Redis restoration"
fi

# 4. Restore Application Data
log "Restoring application data..."

# Restore models and cache
MODELS_BACKUP="$BACKUP_PATH/models_backup.tar.gz"
if [ -f "$MODELS_BACKUP" ]; then
    log "Restoring models and cache..."
    docker exec "$API_CONTAINER" rm -rf /app/models /app/cache
    docker exec -i "$API_CONTAINER" tar -xzf - -C /app < "$MODELS_BACKUP"
    log "Models and cache restored successfully"
else
    log "Warning: Models backup not found, skipping models restoration"
fi

# Restore uploaded files
UPLOADS_BACKUP="$BACKUP_PATH/uploads_backup.tar.gz"
if [ -f "$UPLOADS_BACKUP" ]; then
    log "Restoring uploaded files..."
    docker exec "$API_CONTAINER" rm -rf /app/data/uploads
    docker exec -i "$API_CONTAINER" tar -xzf - -C /app < "$UPLOADS_BACKUP"
    log "Uploaded files restored successfully"
else
    log "Warning: Uploads backup not found, skipping uploads restoration"
fi

# 5. Restore Configuration (Optional)
CONFIG_BACKUP="$BACKUP_PATH/config_backup.tar.gz"
if [ -f "$CONFIG_BACKUP" ]; then
    confirm "Do you want to restore configuration files? This will overwrite current configuration."
    
    if [ $? -eq 0 ]; then
        log "Restoring configuration files..."
        
        # Create temporary directory
        TEMP_CONFIG_DIR=$(mktemp -d)
        trap "rm -rf $TEMP_CONFIG_DIR" EXIT
        
        # Extract configuration backup
        tar -xzf "$CONFIG_BACKUP" -C "$TEMP_CONFIG_DIR"
        
        # Copy configuration files back
        cp -r "$TEMP_CONFIG_DIR/docker"/* "$PROJECT_ROOT/docker/" 2>/dev/null || true
        cp "$TEMP_CONFIG_DIR/.env.production" "$PROJECT_ROOT/" 2>/dev/null || true
        
        log "Configuration files restored successfully"
        log "Note: You may need to restart services for configuration changes to take effect"
    fi
else
    log "Warning: Configuration backup not found, skipping configuration restoration"
fi

# 6. Start application services
log "Starting application services..."
docker start "$API_CONTAINER"

# Wait for services to start
sleep 10

# Check if API service is responding
log "Checking service health..."
for i in {1..30}; do
    if docker exec "$API_CONTAINER" curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        log "API service is responding"
        break
    fi
    if [ $i -eq 30 ]; then
        error_exit "API service failed to start properly"
    fi
    sleep 2
done

# 7. Run database maintenance
log "Running post-restoration database maintenance..."
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "ANALYZE;"
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;"

# 8. Verify restoration
log "Verifying restoration..."

# Check database tables
TABLE_COUNT=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog');")
log "Database tables restored: $TABLE_COUNT"

# Check Redis keys
REDIS_KEYS=$(docker exec "$REDIS_CONTAINER" redis-cli DBSIZE)
log "Redis keys restored: $REDIS_KEYS"

# Check API health
if docker exec "$API_CONTAINER" curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    log "✓ API service health check passed"
else
    log "⚠ API service health check failed"
fi

# 9. Create restoration log
RESTORE_LOG="$BACKUP_PATH/restoration_log_$(date +%Y%m%d_%H%M%S).txt"
cat > "$RESTORE_LOG" << EOF
Voice AI Agent Restoration Log
==============================
Restoration Date: $(date)
Backup Used: $BACKUP_NAME
Restored By: $(whoami)
Hostname: $(hostname)

Restoration Summary:
- Database: Restored from database_backup.sql.gz
- Redis: Restored from redis_backup.rdb.gz
- Models: Restored from models_backup.tar.gz
- Uploads: Restored from uploads_backup.tar.gz
- Configuration: $([ -f "$CONFIG_BACKUP" ] && echo "Available (may have been restored)" || echo "Not available")

Post-Restoration Status:
- Database Tables: $TABLE_COUNT
- Redis Keys: $REDIS_KEYS
- API Health: $(docker exec "$API_CONTAINER" curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1 && echo "OK" || echo "FAILED")

Container Status After Restoration:
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

System Status:
- Memory Usage: $(free -h | grep Mem)
- Disk Usage: $(df -h | grep -E "/$|/app")
- Load Average: $(uptime)
EOF

log "Restoration log created: $RESTORE_LOG"

# 10. Final summary
echo ""
echo -e "${GREEN}Restoration completed successfully!${NC}"
echo "================================="
echo "Backup restored: $BACKUP_NAME"
echo "Database tables: $TABLE_COUNT"
echo "Redis keys: $REDIS_KEYS"
echo "API health: $(docker exec "$API_CONTAINER" curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1 && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAILED${NC}")"
echo ""
echo -e "${YELLOW}Post-restoration checklist:${NC}"
echo "1. Verify application functionality"
echo "2. Check all integrations (CRM, telephony)"
echo "3. Test voice processing features"
echo "4. Verify monitoring dashboards"
echo "5. Check backup and restore procedures"
echo ""
echo -e "${BLUE}Restoration completed at: $(date)${NC}"

# Exit successfully
exit 0