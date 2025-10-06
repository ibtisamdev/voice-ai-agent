#!/bin/bash

# Voice AI Agent - Voice Services Test Runner
set -e

echo "🧪 Voice AI Agent - Voice Services Test Suite"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default test options
VERBOSE=false
COVERAGE=false
SPECIFIC_TEST=""
INTEGRATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -i|--integration)
            INTEGRATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose      Run tests with verbose output"
            echo "  -c, --coverage     Run tests with coverage report"
            echo "  -t, --test FILE    Run specific test file"
            echo "  -i, --integration  Run integration tests"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all voice tests"
            echo "  $0 -v                        # Run with verbose output"
            echo "  $0 -c                        # Run with coverage"
            echo "  $0 -t test_voice_stt.py      # Run specific test"
            echo "  $0 -i                        # Run integration tests"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${BLUE}🐳 Checking if services are running...${NC}"

# Check if services are running
if ! docker-compose -f docker/docker-compose.yml ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Services are not running. Starting them now...${NC}"
    docker-compose -f docker/docker-compose.yml up -d
    echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
    sleep 10
fi

# Build test command
TEST_CMD="pytest"

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD -v"
fi

# Add coverage if requested
if [ "$COVERAGE" = true ]; then
    TEST_CMD="$TEST_CMD --cov=ai --cov=backend/app --cov-report=html --cov-report=term"
fi

# Add specific test file if provided
if [ -n "$SPECIFIC_TEST" ]; then
    TEST_CMD="$TEST_CMD tests/$SPECIFIC_TEST"
else
    # Run voice-specific tests
    TEST_CMD="$TEST_CMD tests/test_voice_stt.py tests/test_voice_tts.py tests/test_conversation.py tests/test_voice_websocket.py"
fi

# Add integration tests if requested
if [ "$INTEGRATION" = true ]; then
    TEST_CMD="$TEST_CMD -m integration"
else
    TEST_CMD="$TEST_CMD -m 'not integration'"
fi

echo -e "${BLUE}🧪 Running voice service tests...${NC}"
echo -e "${YELLOW}Command: $TEST_CMD${NC}"
echo ""

# Run tests in Docker container
docker-compose -f docker/docker-compose.yml exec -T api $TEST_CMD

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All voice service tests passed!${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo -e "${BLUE}📊 Coverage report generated in htmlcov/index.html${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo -e "  • Run integration tests: $0 -i"
    echo -e "  • Test specific component: $0 -t test_voice_stt.py"
    echo -e "  • Generate coverage report: $0 -c"
    echo -e "  • Test voice endpoints manually: make voice-test"
    
else
    echo ""
    echo -e "${RED}❌ Some tests failed. Exit code: $TEST_EXIT_CODE${NC}"
    echo ""
    echo -e "${YELLOW}🔍 Debugging tips:${NC}"
    echo -e "  • Check test logs above for specific failures"
    echo -e "  • Run with verbose output: $0 -v"
    echo -e "  • Check service logs: make logs"
    echo -e "  • Verify services are healthy: curl http://localhost:8000/api/v1/health"
    
    exit $TEST_EXIT_CODE
fi

echo ""
echo -e "${BLUE}🎯 Voice Testing Summary:${NC}"
echo -e "  • STT (Speech-to-Text) tests"
echo -e "  • TTS (Text-to-Speech) tests" 
echo -e "  • Conversation management tests"
echo -e "  • WebSocket streaming tests"
echo -e "  • Intent classification tests"
echo -e "  • Dialog flow engine tests"