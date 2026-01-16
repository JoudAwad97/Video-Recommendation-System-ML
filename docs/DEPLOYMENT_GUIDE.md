# Video Recommendation System - Deployment Guide

This guide walks you through deploying the Video Recommendation System to your AWS account.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Manual Deployment](#manual-deployment)
5. [Configuration](#configuration)
6. [Verifying the Deployment](#verifying-the-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Cleanup](#cleanup)
9. [CI/CD Setup](#cicd-setup)

---

## Overview

The Video Recommendation System is a production-ready ML-powered recommendation engine that uses:

- **Two-Tower Neural Network** for candidate generation
- **CatBoost Ranker** for personalized ranking
- **AWS Lambda** for serverless inference (~2ms latency)
- **API Gateway** for REST API
- **DynamoDB** for feature storage
- **S3** for model artifacts

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Your Application                                │
│                         (Web, Mobile, Backend)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API Gateway                                       │
│                    /recommendations/{user_id}                               │
│                         /health                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Lambda Function                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Feature   │→ │  Candidate  │→ │  Filtering  │→ │   Ranking   │        │
│  │   Fetch     │  │ Generation  │  │   Service   │  │   Service   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │     DynamoDB      │           │        S3         │
        │  Feature Tables   │           │  Model Artifacts  │
        └───────────────────┘           └───────────────────┘
```

---

## Prerequisites

### Required Software

| Software | Minimum Version | Installation |
|----------|-----------------|--------------|
| AWS CLI | 2.x | [Install Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Node.js | 18+ | [Download](https://nodejs.org/) |
| Python | 3.11+ | [Download](https://www.python.org/downloads/) |
| Docker | 20+ | [Download](https://www.docker.com/products/docker-desktop/) |

### Installation Commands

**macOS (using Homebrew):**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install awscli node python@3.11
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Python 3.11
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

**Windows (using Chocolatey):**
```powershell
# Install Chocolatey if not installed
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install prerequisites
choco install awscli nodejs python311 docker-desktop -y
```

### AWS Account Setup

1. **Create an AWS Account** (if you don't have one):
   - Go to [aws.amazon.com](https://aws.amazon.com)
   - Click "Create an AWS Account"
   - Follow the signup process

2. **Create IAM User with Programmatic Access**:
   ```bash
   # Create IAM user (run in AWS Console CloudShell or with admin credentials)
   aws iam create-user --user-name video-rec-deployer

   # Attach AdministratorAccess policy (for deployment)
   aws iam attach-user-policy \
     --user-name video-rec-deployer \
     --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

   # Create access keys
   aws iam create-access-key --user-name video-rec-deployer
   ```

   > **Note**: For production, use more restrictive policies. See [Required IAM Permissions](#required-iam-permissions).

3. **Configure AWS CLI**:
   ```bash
   aws configure
   # Enter your Access Key ID
   # Enter your Secret Access Key
   # Enter your preferred region (e.g., us-east-2)
   # Enter output format: json
   ```

4. **Verify Configuration**:
   ```bash
   aws sts get-caller-identity
   ```

   Expected output:
   ```json
   {
       "UserId": "AIDAXXXXXXXXXXXXXXXXX",
       "Account": "123456789012",
       "Arn": "arn:aws:iam::123456789012:user/video-rec-deployer"
   }
   ```

---

## Quick Start

The fastest way to deploy is using the automated script:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/video-recommendation-system.git
cd video-recommendation-system

# Run deployment script
./scripts/deploy.sh
```

The script will:
1. ✅ Check prerequisites (AWS CLI, Node.js, Python, Docker)
2. ✅ Install Python dependencies
3. ✅ Run tests (251 tests)
4. ✅ Install CDK dependencies
5. ✅ Deploy AWS infrastructure
6. ✅ **Automatically run ML training pipeline**:
   - Generate synthetic training data (1000 users, 500 videos, 10000 interactions)
   - Train Two-Tower model and generate video embeddings
   - Train CatBoost Ranker model
   - Evaluate models against quality thresholds
   - Deploy models if evaluation passes
   - Update DynamoDB with video embeddings
7. ✅ Bootstrap test data
8. ✅ Validate the API

**After deployment, you'll have a fully operational recommendation system with trained models!**

At the end, you'll see your API URL and ML pipeline status:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Deployment Complete!                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Environment:     dev
API URL:         https://xxxxxxxxxx.execute-api.us-east-2.amazonaws.com/dev/

Endpoints:
  Health Check:  GET  https://xxxxxxxxxx.execute-api.us-east-2.amazonaws.com/dev/health
  Get Recs:      GET  https://xxxxxxxxxx.execute-api.us-east-2.amazonaws.com/dev/recommendations/{user_id}?n=10
  Post Recs:     POST https://xxxxxxxxxx.execute-api.us-east-2.amazonaws.com/dev/recommendations

AWS Resources:
  Data Bucket:         videorecsystem-dev-databucket-xxxxx
  Model Bucket:        videorecsystem-dev-modelbucket-xxxxx
  Artifacts Bucket:    videorecsystem-dev-artifactsbucket-xxxxx

ML Pipeline:
  State Machine ARN:   arn:aws:states:us-east-2:123456789:stateMachine:MLPipeline
  Models trained and deployed to: s3://videorecsystem-dev-modelbucket-xxxxx/models/
  Video embeddings stored in: s3://videorecsystem-dev-modelbucket-xxxxx/vector_store/

To manually retrain models:
  aws stepfunctions start-execution --state-machine-arn <ARN> --input '{"trigger": "manual"}'
```

---

## Manual Deployment

If you prefer step-by-step control:

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/video-recommendation-system.git
cd video-recommendation-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python package
pip install -e .
```

### Step 2: Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Expected: 251 passed
```

### Step 3: Deploy Infrastructure

```bash
# Navigate to CDK directory
cd deployment/cdk

# Install Node dependencies
npm install

# Build TypeScript
npm run build

# Bootstrap CDK (first time only)
npx cdk bootstrap

# Deploy stack
npx cdk deploy VideoRecSystem-dev --require-approval never
```

### Step 4: Bootstrap Test Data

```bash
# Return to project root
cd ../..

# Run bootstrap script
python scripts/bootstrap_data.py
```

### Step 5: Test the API

```bash
# Get your API URL from the deployment output, then:
API_URL="https://YOUR-API-ID.execute-api.YOUR-REGION.amazonaws.com/dev/"

# Health check
curl $API_URL/health

# Get recommendations
curl "$API_URL/recommendations/123?n=5"
```

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# =============================================================================
# Video Recommendation System Configuration
# =============================================================================

# Environment: dev, staging, or prod
ENVIRONMENT=dev

# AWS Configuration
AWS_REGION=us-east-2

# -----------------------------------------------------------------------------
# Feature Store (auto-populated by CDK deployment)
# -----------------------------------------------------------------------------
USER_FEATURES_TABLE=VideoRecSystem-dev-user-features
VIDEO_FEATURES_TABLE=VideoRecSystem-dev-video-features

# -----------------------------------------------------------------------------
# Serving Configuration
# -----------------------------------------------------------------------------
SERVING_DEFAULT_NUM_RECOMMENDATIONS=10
SERVING_MAX_NUM_RECOMMENDATIONS=100
SERVING_CANDIDATE_MULTIPLIER=5

# -----------------------------------------------------------------------------
# Vector Store Configuration
# -----------------------------------------------------------------------------
VECTOR_STORE_EMBEDDING_DIM=32
# VECTOR_STORE_S3_BUCKET=your-bucket  # Optional: Load embeddings from S3
# VECTOR_STORE_S3_KEY=embeddings/video_embeddings.npz

# -----------------------------------------------------------------------------
# Redis Configuration (optional - for caching)
# -----------------------------------------------------------------------------
REDIS_ENABLED=false
# REDIS_HOST=your-redis-host.cache.amazonaws.com
# REDIS_PORT=6379

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
```

### CDK Configuration

Environment-specific settings are in `deployment/cdk/lib/config/environments.ts`:

```typescript
export const environments: Record<string, EnvironmentConfig> = {
  dev: {
    environment: 'dev',
    region: 'us-east-2',
    // ... Lambda settings, DynamoDB settings, etc.
  },
  staging: {
    environment: 'staging',
    region: 'us-east-2',
    // Higher capacity for staging
  },
  prod: {
    environment: 'prod',
    region: 'us-east-1',
    // Production settings with high availability
  },
};
```

### Deploying to Different Environments

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh prod
```

---

## Verifying the Deployment

### Health Check

```bash
curl https://YOUR-API-URL/dev/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-16T15:36:09.212539Z",
  "checks": [
    {
      "name": "dynamodb:VideoRecSystem-dev-user-features",
      "status": "healthy",
      "message": "Table is active",
      "latency_ms": 103.48
    }
  ]
}
```

### Get Recommendations

```bash
curl "https://YOUR-API-URL/dev/recommendations/123?n=5"
```

Expected response:
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "video_id": 202,
      "score": 0.59,
      "rank": 1,
      "source": "ranking",
      "metadata": {
        "category": "Cooking",
        "video_language": "English",
        "video_duration": 841
      }
    }
  ],
  "total_latency_ms": 1.71
}
```

### POST Recommendations

```bash
curl -X POST "https://YOUR-API-URL/dev/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 456, "num_recommendations": 10}'
```

### Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| Cold Start Latency | ~3-5 seconds |
| Warm Request Latency | ~1-2 ms |
| Throughput | 1000+ requests/second |

---

## Troubleshooting

### Common Issues

#### 1. "CDKToolkit stack does not exist"

```bash
# Bootstrap CDK for your account/region
cd deployment/cdk
npx cdk bootstrap aws://YOUR_ACCOUNT_ID/YOUR_REGION
```

#### 2. Docker Build Fails

```bash
# Ensure Docker is running
docker info

# If on M1/M2 Mac, ensure Rosetta is enabled in Docker settings
# Docker Desktop > Settings > General > Use Rosetta for x86/amd64 emulation
```

#### 3. Lambda Returns 502 Error

Check CloudWatch Logs:
```bash
aws logs tail /aws/lambda/VideoRecSystem-dev-RecommendationFunction --follow
```

Common causes:
- Missing environment variables
- DynamoDB table not populated
- Import errors in Lambda code

#### 4. "Access Denied" Errors

Ensure your IAM user has sufficient permissions:
```bash
aws iam attach-user-policy \
  --user-name YOUR_USER \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
```

#### 5. Tests Fail Locally

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -e .

# Run tests with verbose output
python -m pytest tests/ -v --tb=long
```

### Getting Help

1. Check CloudWatch Logs for Lambda errors
2. Check CloudFormation events for deployment issues
3. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, etc.)

---

## Cleanup

To remove all deployed resources:

```bash
cd deployment/cdk

# Destroy the stack
npx cdk destroy VideoRecSystem-dev

# Confirm when prompted
```

This will delete:
- Lambda functions
- API Gateway
- DynamoDB tables
- S3 buckets (if empty)
- CloudWatch log groups
- IAM roles

> **Warning**: This is irreversible. All data will be lost.

---

## CI/CD Setup

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Video Recommendation System

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-2

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e .

      - name: Run tests
        run: python -m pytest tests/ -v

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Install dependencies
        run: |
          pip install -e .
          cd deployment/cdk && npm ci

      - name: Deploy
        working-directory: deployment/cdk
        run: npx cdk deploy VideoRecSystem-dev --require-approval never

      - name: Bootstrap data
        run: python scripts/bootstrap_data.py

      - name: Smoke test
        run: |
          API_URL=$(aws cloudformation describe-stacks \
            --stack-name VideoRecSystem-dev \
            --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
            --output text)
          curl -f "${API_URL}health"
```

### Required GitHub Secrets

Add these secrets in your GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |

---

## Required IAM Permissions

For production, use this minimal IAM policy instead of AdministratorAccess:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:*",
        "lambda:*",
        "apigateway:*",
        "dynamodb:*",
        "s3:*",
        "iam:*",
        "logs:*",
        "ecr:*",
        "states:*",
        "kinesis:*",
        "events:*",
        "glue:*",
        "ssm:*",
        "secretsmanager:*"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## Cost Estimates

| Service | Dev Environment | Production |
|---------|-----------------|------------|
| Lambda | ~$0/month (free tier) | ~$5-50/month |
| API Gateway | ~$0/month (free tier) | ~$3.50/million requests |
| DynamoDB | ~$0/month (free tier) | ~$10-100/month |
| S3 | ~$0.02/GB/month | ~$0.02/GB/month |
| CloudWatch | ~$0/month (free tier) | ~$5-20/month |

**Total estimated cost for dev**: ~$0-5/month (within free tier)
**Total estimated cost for production**: ~$50-200/month (depends on traffic)

---

## Next Steps

After deployment:

1. **Integrate with your application** - Use the API URL to fetch recommendations
2. **Send user events** - Use Kinesis to send interaction events for model improvement
3. **Monitor performance** - Check the CloudWatch dashboard
4. **Retrain models** - Run the ML pipeline manually when you have new data:
   ```bash
   aws stepfunctions start-execution \
     --state-machine-arn <ML_PIPELINE_ARN> \
     --input '{"trigger": "manual"}'
   ```
5. **View ML pipeline status** - Check Step Functions in AWS Console for pipeline execution details

## ML Pipeline

The ML pipeline runs automatically during deployment. Here's what it does:

### Pipeline Steps

1. **Preprocessing** (Lambda)
   - Generates synthetic training data (1000 users, 500 videos, 10000 interactions)
   - Builds feature vocabularies
   - Creates training datasets for both models
   - Uploads artifacts to S3

2. **Training** (Parallel Lambdas)
   - **Two-Tower**: Generates 64-dimensional video embeddings
   - **Ranker**: Trains CatBoost classification model

3. **Evaluation** (Lambda)
   - Validates model quality against thresholds:
     - Two-Tower: Recall@100 ≥ 0.1, NDCG ≥ 0.3
     - Ranker: AUC ≥ 0.6, NDCG ≥ 0.3

4. **Deployment** (Conditional Lambda)
   - Copies embeddings to production S3 location
   - Updates DynamoDB video features table
   - Updates model registry

### Model Artifacts

After successful pipeline execution:

```
s3://{model-bucket}/
├── models/
│   ├── two_tower/{version}/
│   │   ├── video_embeddings.npz
│   │   ├── model_config.json
│   │   └── vocabularies.json
│   └── ranker/{version}/
│       └── ranker_model.cbm
├── vector_store/
│   └── video_embeddings.npz    # Used by Lambda for serving
└── preprocessing/{job_id}/
    ├── vocabularies.json
    └── ranker_train.parquet
```

For questions or issues, please open a GitHub issue.
