#!/bin/bash
# Database Optimization Script for Voice AI Agent - Hetzner VPS
# Optimizes PostgreSQL for voice processing workload

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DB_CONTAINER="voiceai_postgres_prod"
DB_NAME="voiceai_db"
DB_USER="voiceai"

echo -e "${GREEN}Voice AI Agent - Database Optimization Script${NC}"
echo "=============================================="

# Check if PostgreSQL container is running
if ! docker ps | grep -q $DB_CONTAINER; then
    echo -e "${RED}Error: PostgreSQL container $DB_CONTAINER is not running${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting database optimization...${NC}"

# Function to execute SQL commands
execute_sql() {
    local sql="$1"
    echo "Executing: $sql"
    docker exec -it $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c "$sql"
}

# Function to execute SQL from file
execute_sql_file() {
    local file="$1"
    echo "Executing SQL file: $file"
    docker exec -i $DB_CONTAINER psql -U $DB_USER -d $DB_NAME < "$file"
}

# 1. Analyze all tables to update statistics
echo -e "${YELLOW}1. Updating table statistics...${NC}"
execute_sql "ANALYZE;"

# 2. Create indexes for voice processing optimization
echo -e "${YELLOW}2. Creating optimized indexes...${NC}"

# Conversation indexes
execute_sql "
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_session_id 
ON conversations.conversation_sessions(session_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_created_at 
ON conversations.conversation_sessions(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_status 
ON conversations.conversation_sessions(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversation_turns_session_id 
ON conversations.conversation_turns(session_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversation_turns_timestamp 
ON conversations.conversation_turns(timestamp);
"

# Voice data indexes
execute_sql "
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_voice_transcriptions_session_id 
ON voice_data.transcriptions(session_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_voice_transcriptions_created_at 
ON voice_data.transcriptions(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_voice_syntheses_created_at 
ON voice_data.syntheses(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_voice_processing_metrics_timestamp 
ON voice_data.processing_metrics(timestamp);
"

# CRM indexes
execute_sql "
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leads_email 
ON crm.leads(email);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leads_phone 
ON crm.leads(phone);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leads_created_at 
ON crm.leads(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leads_status 
ON crm.leads(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leads_practice_areas 
ON crm.leads USING GIN(practice_areas);
"

# Telephony indexes
execute_sql "
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_records_call_id 
ON telephony.call_records(call_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_records_initiated_at 
ON telephony.call_records(initiated_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_records_status 
ON telephony.call_records(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_records_direction 
ON telephony.call_records(direction);
"

# Campaign indexes
execute_sql "
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_status 
ON campaigns.campaigns(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_created_at 
ON campaigns.campaigns(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_contacts_phone 
ON campaigns.campaign_contacts(phone_number);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_contacts_status 
ON campaigns.campaign_contacts(call_status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dnc_phone_number 
ON campaigns.do_not_call_list(phone_number);
"

# 3. Create partial indexes for frequently queried data
echo -e "${YELLOW}3. Creating partial indexes...${NC}"

execute_sql "
-- Active sessions only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_sessions 
ON conversations.conversation_sessions(session_id) 
WHERE status = 'active';

-- Recent transcriptions (last 30 days)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_transcriptions 
ON voice_data.transcriptions(created_at) 
WHERE created_at > NOW() - INTERVAL '30 days';

-- Failed calls for retry processing
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failed_calls 
ON telephony.call_records(initiated_at) 
WHERE status = 'failed';

-- Pending campaign contacts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pending_contacts 
ON campaigns.campaign_contacts(priority, campaign_list_id) 
WHERE call_status = 'pending';
"

# 4. Update table statistics after index creation
echo -e "${YELLOW}4. Updating statistics after index creation...${NC}"
execute_sql "ANALYZE;"

# 5. Optimize vacuum settings
echo -e "${YELLOW}5. Optimizing vacuum settings...${NC}"

execute_sql "
-- Adjust autovacuum settings for high-traffic tables
ALTER TABLE conversations.conversation_turns SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE voice_data.transcriptions SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE voice_data.processing_metrics SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);
"

# 6. Create materialized views for analytics
echo -e "${YELLOW}6. Creating materialized views for analytics...${NC}"

execute_sql "
-- Daily conversation summary
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_conversation_summary AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_sessions,
    AVG(EXTRACT(EPOCH FROM (ended_at - created_at))) as avg_duration_seconds,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions
FROM conversations.conversation_sessions
WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Daily voice processing metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_voice_metrics AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_transcriptions,
    AVG(processing_time_ms) as avg_processing_time_ms,
    AVG(confidence_score) as avg_confidence_score,
    COUNT(CASE WHEN confidence_score > 0.9 THEN 1 END) as high_confidence_count
FROM voice_data.transcriptions
WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Create indexes on materialized views
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_conv_summary_date 
ON analytics.daily_conversation_summary(date);

CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_voice_metrics_date 
ON analytics.daily_voice_metrics(date);
"

# 7. Set up automated refresh for materialized views
echo -e "${YELLOW}7. Setting up materialized view refresh...${NC}"

# Create function to refresh materialized views
execute_sql "
CREATE OR REPLACE FUNCTION analytics.refresh_daily_views()
RETURNS void AS \$\$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.daily_conversation_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.daily_voice_metrics;
END;
\$\$ LANGUAGE plpgsql;
"

# 8. Performance monitoring views
echo -e "${YELLOW}8. Creating performance monitoring views...${NC}"

execute_sql "
-- Slow query monitoring view
CREATE OR REPLACE VIEW analytics.slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE mean_time > 1000  -- Queries taking more than 1 second
ORDER BY mean_time DESC;

-- Table size monitoring
CREATE OR REPLACE VIEW analytics.table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables
WHERE schemaname IN ('conversations', 'voice_data', 'crm', 'telephony', 'campaigns', 'analytics')
ORDER BY size_bytes DESC;
"

# 9. Database maintenance function
echo -e "${YELLOW}9. Creating maintenance function...${NC}"

execute_sql "
CREATE OR REPLACE FUNCTION admin.daily_maintenance()
RETURNS void AS \$\$
BEGIN
    -- Update statistics
    ANALYZE;
    
    -- Refresh materialized views
    PERFORM analytics.refresh_daily_views();
    
    -- Clean up old data (older than 90 days)
    DELETE FROM voice_data.processing_metrics 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    DELETE FROM analytics.session_summary 
    WHERE date < CURRENT_DATE - INTERVAL '90 days';
    
    -- Log maintenance completion
    INSERT INTO analytics.maintenance_log (operation, completed_at) 
    VALUES ('daily_maintenance', NOW());
    
    RAISE NOTICE 'Daily maintenance completed successfully';
END;
\$\$ LANGUAGE plpgsql;
"

# 10. Final optimization
echo -e "${YELLOW}10. Running final optimization...${NC}"

execute_sql "
-- Vacuum analyze all tables
VACUUM ANALYZE;

-- Reindex system catalogs
REINDEX SYSTEM $DB_NAME;
"

# 11. Display optimization results
echo -e "${YELLOW}11. Optimization summary...${NC}"

echo "Database indexes created:"
execute_sql "
SELECT 
    schemaname, 
    tablename, 
    indexname, 
    indexdef 
FROM pg_indexes 
WHERE schemaname IN ('conversations', 'voice_data', 'crm', 'telephony', 'campaigns')
ORDER BY schemaname, tablename;
"

echo "Table sizes after optimization:"
execute_sql "SELECT * FROM analytics.table_sizes;"

echo -e "${GREEN}Database optimization completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Recommendations:${NC}"
echo "1. Schedule daily maintenance: SELECT admin.daily_maintenance();"
echo "2. Monitor slow queries: SELECT * FROM analytics.slow_queries;"
echo "3. Check table sizes regularly: SELECT * FROM analytics.table_sizes;"
echo "4. Monitor database performance with Grafana dashboards"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "- Set up automated backups"
echo "- Configure monitoring alerts"
echo "- Test application performance"