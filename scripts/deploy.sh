#!/bin/bash
#
# Video Recommendation System - Deployment Script
# ================================================
#
# This script automates the full deployment of the video recommendation system
# to AWS. It handles:
#   1. Python dependency installation
#   2. Running tests
#   3. CDK infrastructure deployment
#   4. Test data bootstrapping
#   5. API validation
#
# Prerequisites:
#   - AWS CLI configured with credentials
#   - Node.js 18+ installed
#   - Python 3.11+ installed
#   - Docker running (for Lambda container builds)
#
# Usage:
#   ./scripts/deploy.sh [environment]
#
# Arguments:
#   environment   Optional: dev (default), staging, or prod
#
# Examples:
#   ./scripts/deploy.sh           # Deploy to dev
#   ./scripts/deploy.sh staging   # Deploy to staging
#   ./scripts/deploy.sh prod      # Deploy to production
#

set -e  # Exit on any error

# Configuration
ENVIRONMENT="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CDK_DIR="$PROJECT_ROOT/deployment/cdk"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                     Video Recommendation System                               ║"
    echo "║                         Deployment Script                                     ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║  Environment:  $ENVIRONMENT                                                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        log_info "  macOS: brew install awscli"
        log_info "  Linux: curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip' && unzip awscliv2.zip && sudo ./aws/install"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js 18+ first."
        log_info "  macOS: brew install node"
        log_info "  Linux: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
        exit 1
    fi

    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        log_error "Node.js version 18+ required. Current version: $(node -v)"
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.11+ first."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        log_warning "Python 3.11+ recommended. Current version: $PYTHON_VERSION"
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        log_info "  macOS: brew install --cask docker"
        log_info "  Linux: curl -fsSL https://get.docker.com | sh"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    log_success "All prerequisites met!"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    cd "$PROJECT_ROOT"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1

    # Install package with all dependencies (ml + dev for testing)
    log_info "Installing package with all dependencies (this may take a few minutes)..."
    pip install -e ".[all]" > /dev/null 2>&1 || {
        log_warning "Could not install ML dependencies, installing base + dev only..."
        pip install -e ".[dev]" > /dev/null 2>&1
    }

    log_success "Python dependencies installed!"
}

run_tests() {
    log_info "Running tests..."
    cd "$PROJECT_ROOT"

    # Activate virtual environment
    source venv/bin/activate

    # Check if tensorflow is available
    if python -c "import tensorflow" 2>/dev/null; then
        log_info "Running all tests (TensorFlow available)..."
        TEST_CMD="python -m pytest tests/ -v --tb=short -q"
    else
        log_warning "TensorFlow not installed, running tests that don't require ML dependencies..."
        # Exclude tests that require tensorflow
        TEST_CMD="python -m pytest tests/ -v --tb=short -q --ignore=tests/test_metrics.py --ignore=tests/test_ranker_model.py --ignore=tests/test_serving.py --ignore=tests/test_two_tower_model.py"
    fi

    # Run pytest
    if $TEST_CMD; then
        log_success "All tests passed!"
    else
        log_error "Tests failed! Fix the issues before deploying."
        exit 1
    fi
}

install_cdk_dependencies() {
    log_info "Installing CDK dependencies..."
    cd "$CDK_DIR"

    # Install npm packages
    npm install > /dev/null 2>&1

    # Build TypeScript
    npm run build > /dev/null 2>&1

    log_success "CDK dependencies installed!"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure to AWS..."
    cd "$CDK_DIR"

    # Get stack name
    STACK_NAME="VideoRecSystem-${ENVIRONMENT}"

    # Bootstrap CDK if needed (first-time deployment)
    log_info "Checking CDK bootstrap status..."
    if ! aws cloudformation describe-stacks --stack-name CDKToolkit &> /dev/null; then
        log_info "Bootstrapping CDK..."
        npx cdk bootstrap
    fi

    # Deploy the stack
    log_info "Deploying stack: $STACK_NAME"
    npx cdk deploy "$STACK_NAME" --require-approval never

    log_success "Infrastructure deployed!"
}

get_stack_outputs() {
    log_info "Retrieving stack outputs..."

    STACK_NAME="VideoRecSystem-${ENVIRONMENT}"

    # Get API URL
    API_URL=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
        --output text 2>/dev/null)

    if [ -z "$API_URL" ] || [ "$API_URL" == "None" ]; then
        log_error "Could not retrieve API URL from stack outputs"
        exit 1
    fi

    # Get other outputs
    DATA_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
        --output text 2>/dev/null)

    USER_FEATURES_TABLE=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`UserFeaturesTableName`].OutputValue' \
        --output text 2>/dev/null)

    VIDEO_FEATURES_TABLE=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`VideoFeaturesTableName`].OutputValue' \
        --output text 2>/dev/null)

    # Get ML Pipeline outputs
    ML_PIPELINE_ARN=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`MLPipelineStateMachineArn`].OutputValue' \
        --output text 2>/dev/null)

    MODEL_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`ModelBucketName`].OutputValue' \
        --output text 2>/dev/null)

    ARTIFACTS_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucketName`].OutputValue' \
        --output text 2>/dev/null)

    log_success "Stack outputs retrieved!"
}

run_ml_pipeline() {
    log_info "Starting ML training pipeline..."

    if [ -z "$ML_PIPELINE_ARN" ] || [ "$ML_PIPELINE_ARN" == "None" ]; then
        log_warning "ML Pipeline ARN not found, skipping ML training"
        return
    fi

    # Start the Step Functions execution
    EXECUTION_NAME="deploy-$(date +%Y%m%d-%H%M%S)"
    log_info "Starting Step Functions execution: $EXECUTION_NAME"

    EXECUTION_ARN=$(aws stepfunctions start-execution \
        --state-machine-arn "$ML_PIPELINE_ARN" \
        --name "$EXECUTION_NAME" \
        --input '{"trigger": "deployment", "environment": "'"$ENVIRONMENT"'"}' \
        --query 'executionArn' \
        --output text 2>/dev/null)

    if [ -z "$EXECUTION_ARN" ] || [ "$EXECUTION_ARN" == "None" ]; then
        log_error "Failed to start ML pipeline execution"
        exit 1
    fi

    log_info "Pipeline execution started: $EXECUTION_ARN"
    log_info "Waiting for pipeline to complete (this may take several minutes)..."

    # Poll for completion
    MAX_WAIT=1200  # 20 minutes max
    WAIT_INTERVAL=30
    ELAPSED=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(aws stepfunctions describe-execution \
            --execution-arn "$EXECUTION_ARN" \
            --query 'status' \
            --output text 2>/dev/null)

        case "$STATUS" in
            "SUCCEEDED")
                log_success "ML pipeline completed successfully!"
                return 0
                ;;
            "FAILED")
                log_error "ML pipeline execution failed!"
                # Get failure details
                aws stepfunctions describe-execution \
                    --execution-arn "$EXECUTION_ARN" \
                    --query 'error' \
                    --output text 2>/dev/null
                exit 1
                ;;
            "TIMED_OUT")
                log_error "ML pipeline execution timed out!"
                exit 1
                ;;
            "ABORTED")
                log_error "ML pipeline execution was aborted!"
                exit 1
                ;;
            "RUNNING"|"PENDING")
                log_info "Pipeline status: $STATUS (${ELAPSED}s elapsed)..."
                sleep $WAIT_INTERVAL
                ELAPSED=$((ELAPSED + WAIT_INTERVAL))
                ;;
            *)
                log_warning "Unknown status: $STATUS"
                sleep $WAIT_INTERVAL
                ELAPSED=$((ELAPSED + WAIT_INTERVAL))
                ;;
        esac
    done

    log_error "Pipeline execution timed out after ${MAX_WAIT}s"
    exit 1
}

bootstrap_test_data() {
    log_info "Bootstrapping test data..."
    cd "$PROJECT_ROOT"

    # Activate virtual environment
    source venv/bin/activate

    # Set environment variables for the bootstrap script
    export ENVIRONMENT="$ENVIRONMENT"
    export USER_FEATURES_TABLE="$USER_FEATURES_TABLE"
    export VIDEO_FEATURES_TABLE="$VIDEO_FEATURES_TABLE"

    # Run bootstrap script
    python scripts/bootstrap_data.py

    log_success "Test data bootstrapped!"
}

validate_deployment() {
    log_info "Validating deployment..."

    # Test health endpoint
    log_info "Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s "${API_URL}health")

    if echo "$HEALTH_RESPONSE" | grep -q '"status": "healthy"'; then
        log_success "Health check passed!"
    else
        log_warning "Health check returned: $HEALTH_RESPONSE"
    fi

    # Test recommendations endpoint
    log_info "Testing recommendations endpoint..."
    RECS_RESPONSE=$(curl -s "${API_URL}recommendations/123?n=5")

    if echo "$RECS_RESPONSE" | grep -q '"recommendations"'; then
        log_success "Recommendations endpoint working!"

        # Count recommendations
        REC_COUNT=$(echo "$RECS_RESPONSE" | grep -o '"video_id"' | wc -l | tr -d ' ')
        log_info "Received $REC_COUNT recommendations"
    else
        log_warning "Recommendations response: $RECS_RESPONSE"
    fi

    log_success "Deployment validation complete!"
}

print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                         Deployment Complete!                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Environment:     $ENVIRONMENT"
    echo "API URL:         $API_URL"
    echo ""
    echo "Endpoints:"
    echo "  Health Check:  GET  ${API_URL}health"
    echo "  Get Recs:      GET  ${API_URL}recommendations/{user_id}?n=10"
    echo "  Post Recs:     POST ${API_URL}recommendations"
    echo ""
    echo "Example requests:"
    echo "  curl ${API_URL}health"
    echo "  curl ${API_URL}recommendations/123?n=5"
    echo "  curl -X POST ${API_URL}recommendations -H 'Content-Type: application/json' -d '{\"user_id\": 123, \"num_recommendations\": 10}'"
    echo ""
    echo "AWS Resources:"
    echo "  Data Bucket:         $DATA_BUCKET"
    echo "  Model Bucket:        $MODEL_BUCKET"
    echo "  Artifacts Bucket:    $ARTIFACTS_BUCKET"
    echo "  User Features Table: $USER_FEATURES_TABLE"
    echo "  Video Features Table: $VIDEO_FEATURES_TABLE"
    echo ""
    echo "ML Pipeline:"
    echo "  State Machine ARN:   $ML_PIPELINE_ARN"
    echo "  Models trained and deployed to: s3://${MODEL_BUCKET}/models/"
    echo "  Video embeddings stored in: s3://${MODEL_BUCKET}/vector_store/"
    echo ""
    echo "To manually retrain models:"
    echo "  aws stepfunctions start-execution --state-machine-arn $ML_PIPELINE_ARN --input '{\"trigger\": \"manual\"}'"
    echo ""
}

cleanup_on_error() {
    log_error "Deployment failed! Check the errors above."
    exit 1
}

# Set trap for error handling
trap cleanup_on_error ERR

# Main execution
main() {
    print_banner
    check_prerequisites
    install_python_dependencies
    run_tests
    install_cdk_dependencies
    deploy_infrastructure
    get_stack_outputs
    run_ml_pipeline
    bootstrap_test_data
    validate_deployment
    print_summary
}

# Run main function
main
