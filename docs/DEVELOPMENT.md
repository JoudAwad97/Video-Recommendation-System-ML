# Local Development Guide

This guide covers setting up and running the Video Recommendation System locally for development.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Setup](#quick-setup)
- [Local Services](#local-services)
- [Running the Application](#running-the-application)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Debugging](#debugging)
- [Common Issues](#common-issues)

## Prerequisites

### Required Software

| Software | Version | Installation |
|----------|---------|--------------|
| Python | 3.11+ | `brew install python@3.11` |
| Docker | 24.0+ | [Docker Desktop](https://docs.docker.com/desktop/) |
| Docker Compose | 2.0+ | Included with Docker Desktop |
| Node.js | 18+ | `brew install node@18` (for CDK) |
| Make | Any | Pre-installed on macOS/Linux |

### Verify Installation

```bash
python --version    # Python 3.11.x
docker --version    # Docker 24.x.x
docker compose version  # Docker Compose v2.x.x
node --version      # v18.x.x
make --version      # GNU Make x.x
```

## Quick Setup

The fastest way to get started:

```bash
# Clone repository
git clone https://github.com/your-org/video-recommendation-system.git
cd video-recommendation-system

# Complete setup (venv, dependencies, pre-commit hooks)
make setup

# Activate virtual environment
source .venv/bin/activate

# Start local services
docker compose up -d

# Generate synthetic data
make bootstrap-data-local

# Run tests to verify
make test

# Start local server
make local-serve
```

## Local Services

### Docker Compose Services

The `docker-compose.yml` provides local versions of AWS services:

| Service | Port | Purpose |
|---------|------|---------|
| DynamoDB Local | 8000 | Feature store, user data |
| Redis | 6379 | Caching layer |
| LocalStack | 4566 | S3 buckets |
| Lambda | 8080 | Lambda runtime (optional) |

### Starting Services

```bash
# Start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f dynamodb-local

# Stop services
docker compose down

# Stop and remove volumes (reset data)
docker compose down -v
```

### Service Endpoints

Once running, services are available at:

```bash
# DynamoDB Local
aws dynamodb list-tables \
  --endpoint-url http://localhost:8000

# Redis
redis-cli -h localhost -p 6379 ping

# LocalStack S3
aws s3 ls --endpoint-url http://localhost:4566
```

## Running the Application

### Local Recommendation Server

```bash
# Start the server
make local-serve

# Or manually
python -m src.serving.local_server
```

The server runs on `http://localhost:8080`.

### API Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Get recommendations
curl http://localhost:8080/recommendations/12345

# Get recommendations with parameters
curl "http://localhost:8080/recommendations/12345?n=10&exclude_watched=true"
```

### Environment Variables

For local development, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key variables for local development:

```bash
# Environment
ENVIRONMENT=dev
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# Local services
DYNAMODB_ENDPOINT=http://localhost:8000
S3_ENDPOINT=http://localhost:4566
REDIS_HOST=localhost
REDIS_PORT=6379

# Feature store
USER_FEATURES_TABLE=user-features
VIDEO_FEATURES_TABLE=video-features

# Model paths (local)
MODEL_BUCKET=models
VECTOR_STORE_PATH=./data/vector_store
RANKER_MODEL_PATH=./data/models/ranker.cbm
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature
```

### 2. Make Changes

Edit code following the [Contributing Guidelines](../CONTRIBUTING.md).

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linter
make lint

# Run type checker
make type-check

# Run all checks
make check
```

### 4. Run Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_serving/test_recommendation_service.py -v

# Run tests matching pattern
pytest tests/ -k "ranker" -v

# Run with coverage
make test-cov
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit (pre-commit hooks will run)
git commit -m "feat: add new feature"
```

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_config/             # Configuration tests
├── test_data/               # Data schema tests
├── test_preprocessing/      # Preprocessing tests
├── test_feature_engineering/# Feature engineering tests
├── test_models/             # Model tests
├── test_serving/            # Serving tests
├── test_monitoring/         # Monitoring tests
└── test_integration/        # Integration tests
```

### Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Parallel execution (faster)
make test-fast

# Specific directory
pytest tests/test_serving/ -v

# Specific test class
pytest tests/test_serving/test_recommendation_service.py::TestRecommendationService -v

# Specific test
pytest tests/test_serving/test_recommendation_service.py::TestRecommendationService::test_get_recommendations -v
```

### Test Fixtures

Common fixtures in `conftest.py`:

```python
@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(user_id=12345, country="US", ...)

@pytest.fixture
def mock_feature_store():
    """Create a mock feature store client."""
    return Mock(spec=FeatureStoreClient)

@pytest.fixture
def recommendation_service(mock_feature_store):
    """Create a recommendation service with mocks."""
    return RecommendationService(feature_client=mock_feature_store)
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestRecommendationService:
    """Tests for RecommendationService."""

    def test_get_recommendations_returns_list(self, recommendation_service):
        """Test that recommendations are returned as a list."""
        result = recommendation_service.get_recommendations(user_id=123)
        assert isinstance(result, list)

    @pytest.mark.parametrize("n,expected", [
        (5, 5),
        (10, 10),
        (100, 100),
    ])
    def test_get_recommendations_respects_n(self, recommendation_service, n, expected):
        """Test that n parameter limits results."""
        result = recommendation_service.get_recommendations(user_id=123, n=n)
        assert len(result) <= expected
```

## Debugging

### Logging

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=text
```

### Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

### VS Code Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Python: Local Server",
      "type": "python",
      "request": "launch",
      "module": "src.serving.local_server",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Python: Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

### Inspecting DynamoDB

```bash
# List tables
aws dynamodb list-tables --endpoint-url http://localhost:8000

# Scan table
aws dynamodb scan \
  --table-name user-features \
  --endpoint-url http://localhost:8000

# Get specific item
aws dynamodb get-item \
  --table-name user-features \
  --key '{"user_id": {"N": "12345"}}' \
  --endpoint-url http://localhost:8000
```

### Inspecting Redis

```bash
# Connect to Redis CLI
redis-cli -h localhost -p 6379

# List keys
KEYS *

# Get value
GET recommendations:12345

# Check TTL
TTL recommendations:12345
```

## Common Issues

### Docker Services Not Starting

```bash
# Check if ports are in use
lsof -i :8000  # DynamoDB
lsof -i :6379  # Redis

# Reset Docker
docker compose down -v
docker compose up -d
```

### Import Errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### DynamoDB Connection Refused

```bash
# Check if DynamoDB Local is running
docker compose ps dynamodb-local

# Restart DynamoDB
docker compose restart dynamodb-local

# Check logs
docker compose logs dynamodb-local
```

### Tests Failing

```bash
# Ensure services are running
docker compose up -d

# Wait for services to be ready
sleep 5

# Run tests
make test
```

### Pre-commit Hooks Failing

```bash
# Run hooks manually to see details
pre-commit run --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "message"

# Reinstall hooks
pre-commit install
```

### Model Files Not Found

```bash
# Generate synthetic data and train models locally
make bootstrap-data-local

# Or download pre-trained models
aws s3 cp s3://your-bucket/models/ ./data/models/ --recursive
```

## IDE Setup

### VS Code Extensions

Recommended extensions:

- **Python** - Microsoft Python extension
- **Pylance** - Fast Python language server
- **Python Test Explorer** - Test discovery and running
- **Docker** - Docker support
- **YAML** - YAML support
- **GitLens** - Git integration

### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".ruff_cache": true,
    ".mypy_cache": true
  }
}
```

### PyCharm Setup

1. Open project in PyCharm
2. Set Python interpreter to `.venv/bin/python`
3. Mark `src/` as Sources Root
4. Mark `tests/` as Test Sources Root
5. Configure pytest as test runner

## Performance Profiling

### cProfile

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = service.get_recommendations(user_id=123)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Run with profiler
python -m memory_profiler script.py
```

### Line Profiling

```bash
# Install line profiler
pip install line_profiler

# Add @profile decorator to functions
# Run with kernprof
kernprof -l -v script.py
```

## Next Steps

- Read the [Architecture Overview](ARCHITECTURE.md) for system design details
- Review [Contributing Guidelines](../CONTRIBUTING.md) before making changes
- Check [Deployment Guide](../deployment/README.md) for AWS deployment
