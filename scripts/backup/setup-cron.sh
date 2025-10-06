#!/bin/bash
# Setup Automated Backups for Voice AI Agent - Hetzner VPS
# Configures cron jobs for regular backups

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${GREEN}Voice AI Agent - Backup Cron Setup${NC}"
echo "=================================="

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to handle errors
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    error_exit "This script must be run as root or with sudo"
fi

# Check if backup script exists
BACKUP_SCRIPT="$SCRIPT_DIR/backup.sh"
if [ ! -f "$BACKUP_SCRIPT" ]; then
    error_exit "Backup script not found: $BACKUP_SCRIPT"
fi

# Make backup script executable
chmod +x "$BACKUP_SCRIPT"

log "Setting up automated backup cron jobs..."

# Create cron job for daily backups
CRON_FILE="/etc/cron.d/voiceai-backup"

cat > "$CRON_FILE" << EOF
# Voice AI Agent Automated Backup Schedule
# Daily backup at 2:00 AM
0 2 * * * root $BACKUP_SCRIPT >> /var/log/voiceai-backup.log 2>&1

# Weekly backup verification at 3:00 AM on Sundays
0 3 * * 0 root $SCRIPT_DIR/verify-backups.sh >> /var/log/voiceai-backup-verify.log 2>&1

# Monthly cleanup at 4:00 AM on the 1st of each month
0 4 1 * * root $SCRIPT_DIR/cleanup-old-backups.sh >> /var/log/voiceai-backup-cleanup.log 2>&1
EOF

# Set proper permissions for cron file
chmod 644 "$CRON_FILE"

log "Cron job created: $CRON_FILE"

# Create backup verification script
VERIFY_SCRIPT="$SCRIPT_DIR/verify-backups.sh"
cat > "$VERIFY_SCRIPT" << 'EOF'
#!/bin/bash
# Backup Verification Script for Voice AI Agent

set -e

BACKUP_DIR="/app/data/backups"
LOG_FILE="/var/log/voiceai-backup-verify.log"

echo "$(date): Starting backup verification" >> "$LOG_FILE"

# Find latest backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | grep "voiceai_backup_" | head -n 1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "$(date): ERROR: No backups found" >> "$LOG_FILE"
    exit 1
fi

BACKUP_PATH="$BACKUP_DIR/$LATEST_BACKUP"

echo "$(date): Verifying backup: $LATEST_BACKUP" >> "$LOG_FILE"

# Check if all expected files exist
EXPECTED_FILES=(
    "database_backup.sql.gz"
    "redis_backup.rdb.gz"
    "models_backup.tar.gz"
    "config_backup.tar.gz"
    "backup_manifest.txt"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$BACKUP_PATH/$file" ]; then
        echo "$(date): ERROR: Missing backup file: $file" >> "$LOG_FILE"
        exit 1
    fi
done

# Test file integrity
if ! zcat "$BACKUP_PATH/database_backup.sql.gz" | head -n 1 | grep -q "PostgreSQL"; then
    echo "$(date): ERROR: Database backup appears corrupted" >> "$LOG_FILE"
    exit 1
fi

if ! file "$BACKUP_PATH/redis_backup.rdb.gz" | grep -q "gzip"; then
    echo "$(date): ERROR: Redis backup appears corrupted" >> "$LOG_FILE"
    exit 1
fi

# Check backup age (should not be older than 2 days)
BACKUP_AGE=$(find "$BACKUP_PATH" -mtime +2 | wc -l)
if [ "$BACKUP_AGE" -gt 0 ]; then
    echo "$(date): WARNING: Backup is older than 2 days" >> "$LOG_FILE"
fi

echo "$(date): Backup verification completed successfully" >> "$LOG_FILE"
EOF

chmod +x "$VERIFY_SCRIPT"

# Create cleanup script
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup-old-backups.sh"
cat > "$CLEANUP_SCRIPT" << 'EOF'
#!/bin/bash
# Cleanup Old Backups Script for Voice AI Agent

set -e

BACKUP_DIR="/app/data/backups"
LOG_FILE="/var/log/voiceai-backup-cleanup.log"

echo "$(date): Starting backup cleanup" >> "$LOG_FILE"

# Remove backups older than 30 days
REMOVED_COUNT=0
while IFS= read -r -d '' backup; do
    echo "$(date): Removing old backup: $(basename "$backup")" >> "$LOG_FILE"
    rm -rf "$backup"
    ((REMOVED_COUNT++))
done < <(find "$BACKUP_DIR" -name "voiceai_backup_*" -type d -mtime +30 -print0 2>/dev/null)

echo "$(date): Cleanup completed. Removed $REMOVED_COUNT old backups" >> "$LOG_FILE"

# Check disk space and warn if low
DISK_USAGE=$(df "$BACKUP_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "$(date): WARNING: Backup disk usage is high: ${DISK_USAGE}%" >> "$LOG_FILE"
fi
EOF

chmod +x "$CLEANUP_SCRIPT"

# Create log rotation configuration
LOGROTATE_CONFIG="/etc/logrotate.d/voiceai-backup"
cat > "$LOGROTATE_CONFIG" << EOF
/var/log/voiceai-backup*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

log "Log rotation configured: $LOGROTATE_CONFIG"

# Restart cron service
systemctl restart cron
log "Cron service restarted"

# Create monitoring script
MONITOR_SCRIPT="$SCRIPT_DIR/monitor-backups.sh"
cat > "$MONITOR_SCRIPT" << 'EOF'
#!/bin/bash
# Backup Monitoring Script for Voice AI Agent

BACKUP_DIR="/app/data/backups"
ALERT_EMAIL="${BACKUP_ALERT_EMAIL:-admin@yourdomain.com}"

# Check if latest backup exists and is recent
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | grep "voiceai_backup_" | head -n 1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "CRITICAL: No backups found in $BACKUP_DIR"
    exit 2
fi

# Check backup age (should be less than 25 hours old)
BACKUP_PATH="$BACKUP_DIR/$LATEST_BACKUP"
BACKUP_AGE_HOURS=$(( ($(date +%s) - $(stat -c %Y "$BACKUP_PATH")) / 3600 ))

if [ "$BACKUP_AGE_HOURS" -gt 25 ]; then
    echo "WARNING: Latest backup is $BACKUP_AGE_HOURS hours old"
    exit 1
fi

# Check backup size (should be reasonable)
BACKUP_SIZE_MB=$(du -sm "$BACKUP_PATH" | cut -f1)

if [ "$BACKUP_SIZE_MB" -lt 10 ]; then
    echo "WARNING: Backup size is very small: ${BACKUP_SIZE_MB}MB"
    exit 1
fi

echo "OK: Latest backup is $BACKUP_AGE_HOURS hours old, ${BACKUP_SIZE_MB}MB"
exit 0
EOF

chmod +x "$MONITOR_SCRIPT"

# Add monitoring to cron (every 4 hours)
echo "0 */4 * * * root $MONITOR_SCRIPT >> /var/log/voiceai-backup-monitor.log 2>&1" >> "$CRON_FILE"

# Test backup script
log "Testing backup script..."
if "$BACKUP_SCRIPT" --test > /dev/null 2>&1; then
    log "✓ Backup script test successful"
else
    log "⚠ Backup script test failed (may require running services)"
fi

# Create status script
STATUS_SCRIPT="$SCRIPT_DIR/backup-status.sh"
cat > "$STATUS_SCRIPT" << 'EOF'
#!/bin/bash
# Backup Status Script for Voice AI Agent

BACKUP_DIR="/app/data/backups"

echo "Voice AI Agent - Backup Status"
echo "=============================="
echo "Date: $(date)"
echo ""

# Show backup directory info
echo "Backup Directory: $BACKUP_DIR"
echo "Total Size: $(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "Unknown")"
echo ""

# List recent backups
echo "Recent Backups:"
echo "---------------"
ls -la "$BACKUP_DIR" | grep "voiceai_backup_" | head -n 5 || echo "No backups found"
echo ""

# Show cron jobs
echo "Scheduled Backup Jobs:"
echo "----------------------"
cat /etc/cron.d/voiceai-backup 2>/dev/null || echo "No cron jobs configured"
echo ""

# Show recent backup log entries
echo "Recent Backup Log Entries:"
echo "--------------------------"
tail -n 10 /var/log/voiceai-backup.log 2>/dev/null || echo "No backup logs found"
echo ""

# Disk space
echo "Disk Space:"
echo "-----------"
df -h "$BACKUP_DIR" 2>/dev/null || df -h /
EOF

chmod +x "$STATUS_SCRIPT"

# Final summary
echo ""
echo -e "${GREEN}Automated backup setup completed!${NC}"
echo "=================================="
echo "Backup schedule:"
echo "  - Daily backups at 2:00 AM"
echo "  - Weekly verification at 3:00 AM on Sundays"
echo "  - Monthly cleanup at 4:00 AM on 1st of month"
echo "  - Monitoring every 4 hours"
echo ""
echo "Configuration files created:"
echo "  - Cron job: $CRON_FILE"
echo "  - Log rotation: $LOGROTATE_CONFIG"
echo "  - Verification script: $VERIFY_SCRIPT"
echo "  - Cleanup script: $CLEANUP_SCRIPT"
echo "  - Monitor script: $MONITOR_SCRIPT"
echo "  - Status script: $STATUS_SCRIPT"
echo ""
echo "Log files:"
echo "  - Backup log: /var/log/voiceai-backup.log"
echo "  - Verification log: /var/log/voiceai-backup-verify.log"
echo "  - Cleanup log: /var/log/voiceai-backup-cleanup.log"
echo "  - Monitor log: /var/log/voiceai-backup-monitor.log"
echo ""
echo -e "${YELLOW}To check backup status:${NC}"
echo "$STATUS_SCRIPT"
echo ""
echo -e "${YELLOW}To run manual backup:${NC}"
echo "$BACKUP_SCRIPT"
echo ""
echo -e "${BLUE}Setup completed at: $(date)${NC}"
EOF