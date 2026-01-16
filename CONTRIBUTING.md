# Contributing to Video Recommendation System

Thank you for considering contributing to this project. This document provides guidelines and best practices for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for CDK)
- AWS CLI (for deployment)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/video-recommendation-system.git
cd video-recommendation-system

# Run the complete setup
make setup

# Activate virtual environment
source .venv/bin/activate

# Start local services
docker-compose up -d

# Verify setup
make test
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Sync with main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the [Code Standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Run all checks
make check

# Or run individually:
make lint      # Linting with ruff
make format    # Auto-format code
make test      # Run tests
make test-cov  # Tests with coverage
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with conventional message
git commit -m "feat: add user preference filtering"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications enforced by `ruff`:

- **Line length**: 100 characters max
- **Imports**: Sorted with `isort` rules
- **Quotes**: Double quotes for strings
- **Type hints**: Required for public functions

### Code Organization

```python
# Standard library imports
import json
import os
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

# Local imports
from src.config import get_settings
from src.utils.logging_utils import get_logger
```

### Type Hints

```python
# Good
def get_recommendations(
    user_id: int,
    n: int = 20,
    exclude_watched: bool = True,
) -> List[Dict[str, Any]]:
    ...

# Avoid
def get_recommendations(user_id, n=20, exclude_watched=True):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Calculate ranking metrics for model evaluation.

    Args:
        predictions: Predicted scores array of shape (n_samples,).
        labels: Ground truth labels array of shape (n_samples,).
        k: Number of top items to consider for metrics.

    Returns:
        Dictionary containing:
            - recall_at_k: Recall@K score
            - ndcg: Normalized Discounted Cumulative Gain
            - map: Mean Average Precision

    Raises:
        ValueError: If predictions and labels have different lengths.
    """
```

### Error Handling

```python
# Good - specific exception with context
try:
    model = load_model(model_path)
except FileNotFoundError:
    logger.error(f"Model not found at {model_path}")
    raise ModelNotFoundError(f"Failed to load model from {model_path}")

# Avoid - bare except
try:
    model = load_model(model_path)
except:
    pass
```

### Logging

```python
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug info")
logger.info("General operational info")
logger.warning("Something unexpected but not critical")
logger.error("Error that needs attention")
```

## Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_models/             # Model tests
│   ├── test_two_tower.py
│   └── test_ranker.py
├── test_serving/            # Serving tests
│   ├── test_recommendation_service.py
│   └── test_vector_store.py
├── test_preprocessing/      # Preprocessing tests
└── test_integration/        # Integration tests
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from src.serving.recommendation_service import RecommendationService


class TestRecommendationService:
    """Tests for RecommendationService."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return RecommendationService(
            vector_store=Mock(),
            feature_client=Mock(),
        )

    def test_get_recommendations_returns_expected_count(self, service):
        """Test that get_recommendations returns the requested number of items."""
        # Arrange
        user_id = 12345
        n = 10

        # Act
        results = service.get_recommendations(user_id, n=n)

        # Assert
        assert len(results) == n

    def test_get_recommendations_excludes_watched(self, service):
        """Test that watched videos are excluded from recommendations."""
        # ...

    @pytest.mark.parametrize("user_id,expected", [
        (1, True),
        (0, False),
        (-1, False),
    ])
    def test_validate_user_id(self, service, user_id, expected):
        """Test user ID validation with various inputs."""
        assert service._validate_user_id(user_id) == expected
```

### Test Coverage

- Minimum coverage: 80%
- New code should have >90% coverage
- Critical paths (serving, models) should have >95% coverage

```bash
# Run with coverage report
make test-cov

# Generate HTML report
pytest --cov=src --cov-report=html tests/
```

## Pull Request Process

### Before Submitting

1. **Run all checks**: `make check`
2. **Update tests**: Add/update tests for changes
3. **Update docs**: Update relevant documentation
4. **Self-review**: Review your own code first

### PR Description Template

```markdown
## Summary
Brief description of changes (1-2 sentences).

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks**: CI must pass
2. **Code review**: At least one approval required
3. **Address feedback**: Respond to all comments
4. **Squash and merge**: Use squash merge for clean history

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |
| `ci` | CI/CD changes |

### Examples

```bash
# Feature
git commit -m "feat(serving): add caching layer for recommendations"

# Bug fix
git commit -m "fix(ranker): handle missing features gracefully"

# Documentation
git commit -m "docs: update API reference with new endpoints"

# Breaking change
git commit -m "feat(api)!: change response format for recommendations

BREAKING CHANGE: Response now includes metadata field"
```

### Commit Hooks

Pre-commit hooks are installed automatically and run:

- `ruff` - Linting and formatting
- `mypy` - Type checking
- `bandit` - Security scanning
- `commitizen` - Commit message validation

To skip hooks (not recommended):

```bash
git commit --no-verify -m "message"
```

## Project-Specific Guidelines

### Adding New Features

1. **Discuss first**: Open an issue to discuss major features
2. **Design doc**: For large changes, write a brief design document
3. **Incremental PRs**: Break large features into smaller PRs

### Adding New Dependencies

1. Add to `requirements.txt` (production) or `requirements-dev.txt` (development)
2. Pin version ranges appropriately
3. Document why the dependency is needed
4. Check for security vulnerabilities

### CDK Infrastructure Changes

1. Preview changes: `make deploy-diff`
2. Test in dev environment first
3. Document new resources in `deployment/README.md`
4. Consider cost implications

### Model Changes

1. Run full evaluation pipeline
2. Compare metrics against baseline
3. Document model architecture changes
4. Update preprocessing if input features change

## Getting Help

- **Issues**: Open a GitHub issue for bugs or features
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` for detailed guides

Thank you for contributing!
