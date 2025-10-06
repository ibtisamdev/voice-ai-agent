#!/bin/bash
# Deployment Script for Voice AI Agent - Hetzner VPS
# Blue-green deployment with health checks and rollback capability

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/voiceai"
BACKUP_DIR="$PROJECT_ROOT/backups"
DEPLOYMENT_LOG="/var/log/voiceai-deployment.log"

# Default values
ENVIRONMENT="${1:-staging}"
IMAGE_TAG="${2:-latest}"
COMPOSE_FILE="docker-compose.prod.yml"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_ENABLED=true

echo -e "${GREEN}Voice AI Agent - Deployment Script${NC}"
echo "=================================="
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"
echo "Timestamp: $(date)"
echo ""

# Function to log messages
log() {
    local message="$1"
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $message${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $message" >> "$DEPLOYMENT_LOG"
}

# Function to handle errors
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$DEPLOYMENT_LOG"
    exit 1
}

# Function to check if running as root
check_permissions() {
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        error_exit "This script must be run as root or user must be in docker group"
    fi
}

# Function to validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        staging|production)
            log "Valid environment: $ENVIRONMENT"
            ;;
        *)
            error_exit "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error_exit "Docker is not running"
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi
    
    # Check if project directory exists
    if [ ! -d "$PROJECT_ROOT" ]; then
        error_exit "Project directory not found: $PROJECT_ROOT"
    fi
    
    # Check if compose file exists
    if [ ! -f "$PROJECT_ROOT/docker/$COMPOSE_FILE" ]; then
        error_exit "Compose file not found: $PROJECT_ROOT/docker/$COMPOSE_FILE"
    fi
    
    log "Prerequisites check passed"
}

# Function to create backup before deployment
create_backup() {
    log "Creating pre-deployment backup..."
    
    if [ -f "$PROJECT_ROOT/scripts/backup/backup.sh" ]; then
        if "$PROJECT_ROOT/scripts/backup/backup.sh"; then
            log "Backup created successfully"
        else
            if [ "$ENVIRONMENT" = "production" ]; then
                error_exit "Backup failed - aborting production deployment"
            else
                log "Warning: Backup failed - continuing with staging deployment"
            fi
        fi
    else
        log "Warning: Backup script not found"
    fi
}

# Function to pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export IMAGE_TAG="$IMAGE_TAG"
    export ENVIRONMENT="$ENVIRONMENT"
    
    # Pull images
    if docker-compose -f "docker/$COMPOSE_FILE" pull; then
        log "Images pulled successfully"
    else
        error_exit "Failed to pull Docker images"
    fi
}

# Function to stop current services
stop_services() {
    log "Stopping current services..."
    
    cd "$PROJECT_ROOT"
    
    # Stop services gracefully
    if docker-compose -f "docker/$COMPOSE_FILE" down --timeout 30; then
        log "Services stopped successfully"
    else
        log "Warning: Some services may not have stopped gracefully"
    fi
}

# Function to start new services
start_services() {
    log "Starting new services..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export IMAGE_TAG="$IMAGE_TAG"
    export ENVIRONMENT="$ENVIRONMENT"
    
    # Start services
    if docker-compose -f "docker/$COMPOSE_FILE" up -d; then
        log "Services started successfully"
    else
        error_exit "Failed to start services"
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local timeout=$HEALTH_CHECK_TIMEOUT
    local interval=10
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        log "Health check attempt $((elapsed / interval + 1))..."
        
        # Check API health
        if curl -f -s http://localhost/api/v1/health > /dev/null 2>&1; then
            log "API service is healthy"
            break
        fi
        
        if [ $elapsed -ge $timeout ]; then
            error_exit "Services failed to become healthy within $timeout seconds"
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
    done
}

# Function to run post-deployment tests
run_tests() {
    log "Running post-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run basic health checks
    local tests_passed=true
    
    # Test API endpoints
    if ! curl -f -s http://localhost/api/v1/health | grep -q "healthy"; then
        log "Error: API health check failed"
        tests_passed=false
    fi
    
    # Test database connectivity
    if ! docker-compose -f "docker/$COMPOSE_FILE" exec -T postgres pg_isready -U voiceai > /dev/null 2>&1; then
        log "Error: Database connectivity test failed"
        tests_passed=false
    fi
    
    # Test Redis connectivity
    if ! docker-compose -f "docker/$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log "Error: Redis connectivity test failed"
        tests_passed=false
    fi
    
    # Test voice processing endpoint (if available)
    if curl -f -s http://localhost/api/v1/voice/voices > /dev/null 2>&1; then
        log "Voice processing endpoint is responding"
    else
        log "Warning: Voice processing endpoint test failed"
    fi
    
    if [ "$tests_passed" = true ]; then
        log "All tests passed"
        return 0
    else
        log "Some tests failed"
        return 1
    fi
}

# Function to rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Get previous image tag from deployment history
    local previous_tag
    if [ -f "$PROJECT_ROOT/.deployment_history" ]; then
        previous_tag=$(tail -n 2 "$PROJECT_ROOT/.deployment_history" | head -n 1 | cut -d' ' -f2)
        
        if [ -n "$previous_tag" ]; then
            log "Rolling back to previous version: $previous_tag"
            
            # Stop current services
            docker-compose -f "docker/$COMPOSE_FILE" down --timeout 30
            
            # Start with previous image
            export IMAGE_TAG="$previous_tag"
            docker-compose -f "docker/$COMPOSE_FILE" up -d
            
            # Wait for rollback to complete
            sleep 30
            
            # Test rollback
            if curl -f -s http://localhost/api/v1/health > /dev/null 2>&1; then
                log "Rollback successful"
                return 0
            else
                error_exit "Rollback failed - manual intervention required"
            fi
        else
            error_exit "No previous version found for rollback"
        fi
    else
        error_exit "Deployment history not found - cannot rollback"
    fi
}

# Function to update deployment history
update_deployment_history() {
    log "Updating deployment history..."
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') $IMAGE_TAG $ENVIRONMENT $(whoami)" >> "$PROJECT_ROOT/.deployment_history"
    
    # Keep only last 10 deployments
    tail -n 10 "$PROJECT_ROOT/.deployment_history" > "$PROJECT_ROOT/.deployment_history.tmp"
    mv "$PROJECT_ROOT/.deployment_history.tmp" "$PROJECT_ROOT/.deployment_history"
}

# Function to send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Send Slack notification if webhook is configured
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        local color="good"
        if [ "$status" = "error" ]; then
            color="danger"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\", \"color\":\"$color\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    # Log notification
    log "Notification sent: $message"
}

# Function to cleanup old images
cleanup_images() {
    log "Cleaning up old Docker images..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove old Voice AI images (keep last 3)
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | \
        grep "voiceai" | \
        tail -n +4 | \
        awk '{print $1}' | \
        xargs -r docker rmi || true
    
    log "Image cleanup completed"
}

# Function to generate deployment report
generate_report() {
    local status="$1"
    local report_file="/var/log/deployment-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Voice AI Agent Deployment Report
===============================
Date: $(date)
Environment: $ENVIRONMENT
Image Tag: $IMAGE_TAG
Status: $status
Deployed By: $(whoami)
Host: $(hostname)

Pre-deployment State:
- Running containers: $(docker ps --format "{{.Names}}" | wc -l)
- Available disk space: $(df -h / | awk 'NR==2 {print $4}')
- Available memory: $(free -h | awk 'NR==2 {print $7}')

Post-deployment State:
- Running containers: $(docker ps --format "{{.Names}}" | wc -l)
- API health: $(curl -f -s http://localhost/api/v1/health | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
- Database status: $(docker-compose -f docker/$COMPOSE_FILE exec -T postgres pg_isready -U voiceai 2>/dev/null | grep "accepting" > /dev/null && echo "healthy" || echo "unhealthy")
- Redis status: $(docker-compose -f docker/$COMPOSE_FILE exec -T redis redis-cli ping 2>/dev/null | grep "PONG" > /dev/null && echo "healthy" || echo "unhealthy")

Container Status:
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

Recent Logs:
$(tail -n 20 "$DEPLOYMENT_LOG")
EOF

    log "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log "Starting deployment process..."
    
    # Validate inputs and environment
    check_permissions
    validate_environment
    check_prerequisites
    
    # Pre-deployment steps
    if [ "$ENVIRONMENT" = "production" ] || [ "$ROLLBACK_ENABLED" = true ]; then
        create_backup
    fi
    
    # Store current state for potential rollback
    local current_containers
    current_containers=$(docker ps --format "{{.Names}}" | grep voiceai || echo "")
    
    # Deployment steps
    pull_images
    stop_services
    start_services
    wait_for_services
    
    # Post-deployment validation
    if run_tests; then
        log "Deployment completed successfully"
        update_deployment_history
        send_notification "success" "Voice AI Agent deployed successfully to $ENVIRONMENT (tag: $IMAGE_TAG)"
        cleanup_images
        generate_report "SUCCESS"
        
        echo ""
        echo -e "${GREEN}Deployment completed successfully!${NC}"
        echo "Environment: $ENVIRONMENT"
        echo "Image Tag: $IMAGE_TAG"
        echo "Health Status: $(curl -f -s http://localhost/api/v1/health | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")"
        echo ""
        
    else
        log "Post-deployment tests failed"
        
        if [ "$ROLLBACK_ENABLED" = true ] && [ "$ENVIRONMENT" = "production" ]; then
            send_notification "error" "Voice AI Agent deployment failed on $ENVIRONMENT - attempting rollback"
            rollback_deployment
            generate_report "ROLLED_BACK"
        else
            send_notification "error" "Voice AI Agent deployment failed on $ENVIRONMENT"
            generate_report "FAILED"
            error_exit "Deployment failed and rollback is disabled"
        fi
    fi
}

# Handle script interruption
trap 'log "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"

# Exit successfully
exit 0