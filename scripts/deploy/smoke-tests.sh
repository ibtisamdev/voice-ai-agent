#!/bin/bash
# Smoke Tests for Voice AI Agent - Production Deployment Validation
# Comprehensive post-deployment testing to ensure system health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="${BASE_URL:-http://localhost}"
TIMEOUT=30
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

echo -e "${GREEN}Voice AI Agent - Smoke Tests${NC}"
echo "============================"
echo "Base URL: $BASE_URL"
echo "Timestamp: $(date)"
echo ""

# Function to log test results
log_test() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓ $test_name${NC}: $message"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗ $test_name${NC}: $message"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    else
        echo -e "${YELLOW}⚠ $test_name${NC}: $message"
    fi
}

# Function to make HTTP requests with timeout
http_request() {
    local url="$1"
    local method="${2:-GET}"
    local data="${3:-}"
    local expected_status="${4:-200}"
    
    local response
    local status_code
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            --max-time $TIMEOUT \
            "$url" 2>/dev/null || echo -e "\n000")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            --max-time $TIMEOUT \
            "$url" 2>/dev/null || echo -e "\n000")
    fi
    
    status_code=$(echo "$response" | tail -n1)
    response=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        echo "$response"
        return 0
    else
        return 1
    fi
}

# Test 1: API Health Check
test_api_health() {
    local response
    if response=$(http_request "$BASE_URL/api/v1/health"); then
        if echo "$response" | grep -q '"status".*"healthy"'; then
            log_test "API Health Check" "PASS" "API is healthy"
        else
            log_test "API Health Check" "FAIL" "API status is not healthy"
        fi
    else
        log_test "API Health Check" "FAIL" "API health endpoint not responding"
    fi
}

# Test 2: Database Connectivity
test_database_connectivity() {
    # Test through API endpoint that requires database
    local response
    if response=$(http_request "$BASE_URL/api/v1/health/db"); then
        if echo "$response" | grep -q '"database".*"connected"'; then
            log_test "Database Connectivity" "PASS" "Database connection is healthy"
        else
            log_test "Database Connectivity" "FAIL" "Database connection issue detected"
        fi
    else
        log_test "Database Connectivity" "FAIL" "Database health endpoint not responding"
    fi
}

# Test 3: Redis Connectivity
test_redis_connectivity() {
    local response
    if response=$(http_request "$BASE_URL/api/v1/health/redis"); then
        if echo "$response" | grep -q '"redis".*"connected"'; then
            log_test "Redis Connectivity" "PASS" "Redis connection is healthy"
        else
            log_test "Redis Connectivity" "FAIL" "Redis connection issue detected"
        fi
    else
        log_test "Redis Connectivity" "FAIL" "Redis health endpoint not responding"
    fi
}

# Test 4: Authentication Endpoints
test_authentication() {
    # Test invalid authentication
    local response
    if http_request "$BASE_URL/api/v1/protected" "GET" "" "401" > /dev/null 2>&1; then
        log_test "Authentication" "PASS" "Authentication properly rejects unauthorized requests"
    else
        log_test "Authentication" "FAIL" "Authentication not working properly"
    fi
}

# Test 5: Voice Processing Endpoints
test_voice_endpoints() {
    # Test voice engines list
    local response
    if response=$(http_request "$BASE_URL/api/v1/voice/voices"); then
        if echo "$response" | grep -q "voices"; then
            log_test "Voice Endpoints" "PASS" "Voice processing endpoints are responding"
        else
            log_test "Voice Endpoints" "FAIL" "Voice endpoints responding but invalid data"
        fi
    else
        log_test "Voice Endpoints" "FAIL" "Voice processing endpoints not responding"
    fi
}

# Test 6: Conversation Endpoints
test_conversation_endpoints() {
    # Test conversation session creation
    local session_data='{"session_type": "voice_call", "direction": "inbound"}'
    local response
    if response=$(http_request "$BASE_URL/api/v1/conversation/sessions" "POST" "$session_data" "201"); then
        local session_id
        session_id=$(echo "$response" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$session_id" ]; then
            log_test "Conversation Endpoints" "PASS" "Conversation session created successfully"
            
            # Test session retrieval
            if http_request "$BASE_URL/api/v1/conversation/sessions/$session_id" > /dev/null 2>&1; then
                log_test "Conversation Session Retrieval" "PASS" "Session retrieved successfully"
            else
                log_test "Conversation Session Retrieval" "FAIL" "Failed to retrieve session"
            fi
        else
            log_test "Conversation Endpoints" "FAIL" "Session created but no session ID returned"
        fi
    else
        log_test "Conversation Endpoints" "FAIL" "Failed to create conversation session"
    fi
}

# Test 7: CRM Integration
test_crm_integration() {
    # Test CRM status endpoint
    local response
    if response=$(http_request "$BASE_URL/api/v1/crm/status"); then
        if echo "$response" | grep -q "status"; then
            log_test "CRM Integration" "PASS" "CRM integration endpoints are responding"
        else
            log_test "CRM Integration" "FAIL" "CRM endpoints responding but invalid data"
        fi
    else
        log_test "CRM Integration" "WARN" "CRM integration endpoints not responding (may not be configured)"
    fi
}

# Test 8: Telephony System
test_telephony_system() {
    # Test telephony status
    local response
    if response=$(http_request "$BASE_URL/api/v1/telephony/status"); then
        if echo "$response" | grep -q "status"; then
            log_test "Telephony System" "PASS" "Telephony system is responding"
        else
            log_test "Telephony System" "FAIL" "Telephony system responding but invalid data"
        fi
    else
        log_test "Telephony System" "WARN" "Telephony system not responding (may not be configured)"
    fi
}

# Test 9: Campaign Management
test_campaign_management() {
    # Test campaign list endpoint
    local response
    if response=$(http_request "$BASE_URL/api/v1/campaigns"); then
        log_test "Campaign Management" "PASS" "Campaign management endpoints are responding"
    else
        log_test "Campaign Management" "FAIL" "Campaign management endpoints not responding"
    fi
}

# Test 10: Monitoring Endpoints
test_monitoring_endpoints() {
    # Test Prometheus metrics
    if http_request "$BASE_URL/metrics" > /dev/null 2>&1; then
        log_test "Monitoring Endpoints" "PASS" "Prometheus metrics endpoint is accessible"
    else
        log_test "Monitoring Endpoints" "FAIL" "Prometheus metrics endpoint not accessible"
    fi
}

# Test 11: WebSocket Connectivity
test_websocket_connectivity() {
    # Test WebSocket endpoint (basic connectivity)
    if command -v wscat &> /dev/null; then
        # Use wscat if available
        if timeout 5 wscat -c "${BASE_URL/http/ws}/ws/voice/stream/test" -x "ping" 2>/dev/null; then
            log_test "WebSocket Connectivity" "PASS" "WebSocket endpoints are accessible"
        else
            log_test "WebSocket Connectivity" "FAIL" "WebSocket connection failed"
        fi
    else
        log_test "WebSocket Connectivity" "WARN" "WebSocket test skipped (wscat not available)"
    fi
}

# Test 12: Performance Check
test_performance() {
    # Test response time for health endpoint
    local start_time
    local end_time
    local response_time
    
    start_time=$(date +%s%N)
    if http_request "$BASE_URL/api/v1/health" > /dev/null 2>&1; then
        end_time=$(date +%s%N)
        response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        if [ "$response_time" -lt 2000 ]; then
            log_test "Performance Check" "PASS" "Response time: ${response_time}ms"
        elif [ "$response_time" -lt 5000 ]; then
            log_test "Performance Check" "WARN" "Slow response time: ${response_time}ms"
        else
            log_test "Performance Check" "FAIL" "Very slow response time: ${response_time}ms"
        fi
    else
        log_test "Performance Check" "FAIL" "Performance test failed"
    fi
}

# Test 13: Memory and CPU Usage
test_resource_usage() {
    # Check system resources
    local memory_usage
    local cpu_usage
    
    memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    if (( $(echo "$memory_usage < 85" | bc -l) )); then
        log_test "Memory Usage" "PASS" "Memory usage: ${memory_usage}%"
    elif (( $(echo "$memory_usage < 95" | bc -l) )); then
        log_test "Memory Usage" "WARN" "High memory usage: ${memory_usage}%"
    else
        log_test "Memory Usage" "FAIL" "Critical memory usage: ${memory_usage}%"
    fi
    
    if (( $(echo "$cpu_usage < 80" | bc -l) )); then
        log_test "CPU Usage" "PASS" "CPU usage: ${cpu_usage}%"
    elif (( $(echo "$cpu_usage < 95" | bc -l) )); then
        log_test "CPU Usage" "WARN" "High CPU usage: ${cpu_usage}%"
    else
        log_test "CPU Usage" "FAIL" "Critical CPU usage: ${cpu_usage}%"
    fi
}

# Test 14: Disk Space
test_disk_space() {
    local disk_usage
    disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -lt 80 ]; then
        log_test "Disk Space" "PASS" "Disk usage: ${disk_usage}%"
    elif [ "$disk_usage" -lt 90 ]; then
        log_test "Disk Space" "WARN" "High disk usage: ${disk_usage}%"
    else
        log_test "Disk Space" "FAIL" "Critical disk usage: ${disk_usage}%"
    fi
}

# Test 15: Container Health
test_container_health() {
    local unhealthy_containers
    unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | wc -l)
    
    if [ "$unhealthy_containers" -eq 0 ]; then
        log_test "Container Health" "PASS" "All containers are healthy"
    else
        log_test "Container Health" "FAIL" "$unhealthy_containers unhealthy containers detected"
    fi
}

# Run all tests
run_all_tests() {
    echo -e "${BLUE}Running smoke tests...${NC}"
    echo ""
    
    # Core system tests
    test_api_health
    test_database_connectivity
    test_redis_connectivity
    test_authentication
    
    # Feature tests
    test_voice_endpoints
    test_conversation_endpoints
    test_crm_integration
    test_telephony_system
    test_campaign_management
    
    # Infrastructure tests
    test_monitoring_endpoints
    test_websocket_connectivity
    test_performance
    
    # System resource tests
    test_resource_usage
    test_disk_space
    test_container_health
}

# Generate test report
generate_report() {
    echo ""
    echo -e "${BLUE}Test Summary${NC}"
    echo "============"
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo ""
    
    if [ "$FAILED_TESTS" -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! System is ready for production.${NC}"
        return 0
    elif [ "$FAILED_TESTS" -lt 3 ]; then
        echo -e "${YELLOW}⚠️  Some tests failed. Review failures before proceeding.${NC}"
        return 1
    else
        echo -e "${RED}❌ Multiple test failures detected. System may not be ready.${NC}"
        return 2
    fi
}

# Main execution
main() {
    # Check if required tools are available
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl is required for smoke tests${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}Warning: jq not found, JSON parsing may be limited${NC}"
    fi
    
    # Run tests
    run_all_tests
    
    # Generate report and exit with appropriate code
    generate_report
    exit $?
}

# Run main function
main "$@"