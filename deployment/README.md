# Video Recommendation System - AWS CDK Deployment

Infrastructure as Code deployment for the Video Recommendation System using AWS CDK (TypeScript).

## Prerequisites

1. **AWS CLI** - Configured with valid credentials
   ```bash
   aws configure
   ```

2. **Node.js 18+** - Required for CDK
   ```bash
   node --version  # Should be 18.x or higher
   ```

3. **Python 3.11+** - Required for Lambda functions
   ```bash
   python3 --version  # Should be 3.11.x or higher
   ```

4. **Docker** - Required for Lambda container builds
   ```bash
   docker --version
   ```

5. **AWS CDK CLI** - Installed globally (optional, can use npx)
   ```bash
   npm install -g aws-cdk
   ```

## Quick Start (Recommended)

The easiest way to deploy is using the automated deployment script:

```bash
# From project root
./scripts/deploy.sh
```

This single command will:
1. Check all prerequisites
2. Install dependencies (Python + CDK)
3. Run tests
4. Deploy infrastructure
5. **Automatically run ML training pipeline**
6. Bootstrap test data
7. Validate deployment

### Manual CDK Deployment

```bash
# Navigate to CDK directory
cd deployment/cdk

# Install dependencies
npm install

# Bootstrap CDK (first time only)
npm run bootstrap

# Deploy to dev environment
npm run deploy:dev
```

## Deployment Commands

```bash
# Deploy to specific environment
npm run deploy:dev       # Development
npm run deploy:staging   # Staging
npm run deploy:prod      # Production

# Deploy with custom options
npx cdk deploy -c environment=staging -c region=eu-west-1

# Preview changes
npm run diff

# Destroy all resources
npm run destroy
```

## Project Structure

```
deployment/cdk/
├── bin/
│   └── app.ts                    # CDK app entry point
├── lib/
│   ├── config/
│   │   └── environment.ts        # Environment configurations
│   ├── constructs/
│   │   ├── storage.ts            # S3 & DynamoDB (core storage)
│   │   ├── networking.ts         # VPC & ElastiCache
│   │   ├── compute.ts            # Lambda functions (serving)
│   │   ├── api.ts                # API Gateway
│   │   ├── monitoring.ts         # CloudWatch & Alarms
│   │   ├── ml-pipeline.ts        # ML Pipeline (preprocessing, training, evaluation)
│   │   ├── data-collection.ts    # Kinesis, Firehose, inference tracking
│   │   ├── feature-store.ts      # Online (DynamoDB) & Offline (Glue) feature store
│   │   ├── secrets.ts            # AWS Secrets Manager (credentials, API keys)
│   │   └── index.ts
│   └── stacks/
│       ├── video-recommendation-stack.ts
│       └── index.ts
├── package.json
├── tsconfig.json
└── cdk.json
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                      AWS Cloud                                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────── SERVING LAYER ──────────────────────────────────┐│
│  │                                                                                     ││
│  │  ┌─────────────┐         ┌─────────────────────────────────────────┐               ││
│  │  │ API Gateway │────────▶│              Lambda Function            │               ││
│  │  │  (REST API) │         │       (Recommendation Service)          │               ││
│  │  └─────────────┘         └─────────────────────────────────────────┘               ││
│  │                                       │                                             ││
│  │            ┌──────────────────────────┼──────────────────────────┐                 ││
│  │            │                          │                          │                 ││
│  │            ▼                          ▼                          ▼                 ││
│  │    ┌─────────────┐         ┌─────────────────┐         ┌─────────────┐            ││
│  │    │     S3      │         │    DynamoDB     │         │ ElastiCache │            ││
│  │    │  (Models)   │         │  (Features)     │         │  (Redis)*   │            ││
│  │    └─────────────┘         └─────────────────┘         └─────────────┘            ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────── ML PIPELINE LAYER ──────────────────────────────────┐│
│  │                                                                                     ││
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐ ││
│  │  │                        Step Functions State Machine                            │ ││
│  │  │  ┌──────────────┐   ┌────────────────────────┐   ┌────────────┐   ┌─────────┐ │ ││
│  │  │  │ Preprocessing│──▶│ Training (Parallel)    │──▶│ Evaluation │──▶│ Deploy  │ │ ││
│  │  │  │   Lambda     │   │ • Two-Tower Lambda     │   │   Lambda   │   │ Lambda  │ │ ││
│  │  │  │              │   │ • Ranker Lambda        │   │            │   │         │ │ ││
│  │  │  └──────────────┘   └────────────────────────┘   └────────────┘   └─────────┘ │ ││
│  │  └───────────────────────────────────────────────────────────────────────────────┘ ││
│  │                            │                                                       ││
│  │                            ▼                                                       ││
│  │           ┌─────────────────────────────────────────────────────┐                 ││
│  │           │                    SageMaker                         │                 ││
│  │           │    (Training Jobs for Two-Tower & Ranker Models)    │                 ││
│  │           └─────────────────────────────────────────────────────┘                 ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌──────────────────────────── DATA COLLECTION LAYER ─────────────────────────────────┐│
│  │                                                                                     ││
│  │  ┌───────────┐      ┌───────────────┐      ┌─────────────────┐                     ││
│  │  │  Kinesis  │─────▶│   Firehose    │─────▶│       S3        │                     ││
│  │  │  Stream   │      │   Delivery    │      │  (Raw Events)   │                     ││
│  │  └───────────┘      └───────────────┘      └─────────────────┘                     ││
│  │        │                                                                            ││
│  │        ▼                                                                            ││
│  │  ┌───────────────────┐      ┌─────────────────────────┐                            ││
│  │  │ Processing Lambda │─────▶│  DynamoDB (Inference    │                            ││
│  │  │                   │      │   Tracking Table)       │                            ││
│  │  └───────────────────┘      └─────────────────────────┘                            ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────── FEATURE STORE LAYER ────────────────────────────────┐│
│  │                                                                                     ││
│  │  ┌─────────────────────────────┐      ┌────────────────────────────────────────┐   ││
│  │  │     Online Store (DynamoDB) │      │       Offline Store (Glue + S3)        │   ││
│  │  │  • User Features Table      │      │  • Glue Database                       │   ││
│  │  │  • Video Features Table     │      │  • User Features Table (Parquet)       │   ││
│  │  │  • User Activity Table      │      │  • Video Features Table (Parquet)      │   ││
│  │  └─────────────────────────────┘      └────────────────────────────────────────┘   ││
│  │                     │                                                               ││
│  │                     ▼                                                               ││
│  │           ┌─────────────────────────┐                                              ││
│  │           │   Ingestion Lambda      │                                              ││
│  │           │  (Feature Ingestion)    │                                              ││
│  │           └─────────────────────────┘                                              ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           CloudWatch (Monitoring)                                  │ │
│  │                   Dashboard │ Alarms │ Logs │ Metrics │ X-Ray                     │ │
│  └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                              * Redis enabled in staging/prod only
```

## Environment Configurations

| Feature | Dev | Staging | Prod |
|---------|-----|---------|------|
| Lambda Memory | 1024 MB | 1024 MB | 2048 MB |
| Lambda Timeout | 30s | 30s | 30s |
| API Rate Limit | 50 req/s | 200 req/s | 1000 req/s |
| DynamoDB Mode | On-Demand | On-Demand | Provisioned |
| VPC | No | Yes | Yes |
| ElastiCache Redis | No | Yes (t3.small) | Yes (r6g.large) |
| CloudWatch Alarms | No | No | Yes |
| Removal Policy | Destroy | Destroy | Retain |

## AWS Resources Created

### Storage
- **S3 Data Bucket** - Training data, raw events
- **S3 Model Bucket** - Trained models (Two-Tower, Ranker)
- **S3 Artifacts Bucket** - Pipeline artifacts (vocabularies, normalizers, embeddings)
- **DynamoDB User Features** - User feature store (serving)
- **DynamoDB Video Features** - Video feature store (serving)
- **DynamoDB Recommendations Cache** - Pre-computed recommendations

### Compute (Serving)
- **Lambda Function** - Recommendation service API

### ML Pipeline
- **Step Functions State Machine** - ML pipeline orchestration
- **Preprocessing Lambda** - Data preprocessing and feature engineering
- **Two-Tower Training Lambda** - Two-Tower model training orchestration
- **Ranker Training Lambda** - CatBoost Ranker model training orchestration
- **Evaluation Lambda** - Model evaluation and metrics computation
- **Deployment Lambda** - Model deployment to production
- **SageMaker Role** - IAM role for SageMaker training jobs
- **EventBridge Rule** - Weekly scheduled training (prod only)

### Data Collection
- **Kinesis Data Stream** - Real-time event ingestion
- **Firehose Delivery Stream** - S3 event persistence with partitioning
- **DynamoDB Inference Table** - Inference request tracking
- **Processing Lambda** - Event processing from Kinesis
- **Merge Lambda** - Feedback data merging

### Feature Store
- **DynamoDB User Features (Online)** - Real-time user features
- **DynamoDB Video Features (Online)** - Real-time video features
- **DynamoDB User Activity (Online)** - Real-time user activity tracking
- **Glue Database (Offline)** - Offline feature store catalog
- **Glue Tables (Offline)** - Parquet tables for batch processing
- **Ingestion Lambda** - Feature ingestion pipeline

### Networking (Staging/Prod)
- **VPC** - Isolated network
- **Security Groups** - Network access control
- **ElastiCache Redis** - Session caching

### API
- **API Gateway** - REST API with Lambda integration
- **Endpoints**: `/health`, `/recommendations`, `/interactions`, `/cached`

### Monitoring
- **CloudWatch Dashboard** - Metrics visualization
- **CloudWatch Alarms** - Error/latency alerts (prod)
- **X-Ray Tracing** - Request tracing
- **Step Functions Logging** - ML pipeline execution logs

### Secrets Management
- **Redis Credentials Secret** - Redis authentication (staging/prod)
- **API Keys Secret** - External API keys (model serving, etc.)
- **Config Secret** - Feature flags and configuration overrides

## Secrets Management

The system uses AWS Secrets Manager for secure credential storage. Secrets are created automatically during deployment.

### Available Secrets

| Secret | Description | Environments |
|--------|-------------|--------------|
| `{stack}/redis-credentials` | Redis host, port, auth token | staging, prod |
| `{stack}/api-keys` | External API keys | all |
| `{stack}/config` | Feature flags, overrides | all |

### Accessing Secrets in Lambda

Secrets are automatically loaded via the centralized settings system:

```python
from src.config import get_settings

settings = get_settings()

# Redis credentials (auto-loaded from Secrets Manager in prod)
redis_host = settings.redis.host
redis_port = settings.redis.port

# API keys
api_key = settings.get_secret("external_api_key")
```

### Manual Secret Updates

```bash
# Update a secret value
aws secretsmanager update-secret \
  --secret-id VideoRecSystem-dev/api-keys \
  --secret-string '{"model_api_key": "new-key-value"}'

# Rotate Redis credentials (staging/prod only)
aws secretsmanager rotate-secret \
  --secret-id VideoRecSystem-prod/redis-credentials
```

### Secret Rotation

- **Redis credentials**: Automatic rotation every 30 days (production)
- **API keys**: Manual rotation as needed
- **Config secrets**: No rotation (configuration only)

### Security Best Practices

1. **Never commit secrets** - Use `.env` for local development only
2. **Least privilege** - Lambda functions only have read access to required secrets
3. **Audit access** - CloudTrail logs all secret access
4. **Rotation** - Enable automatic rotation for production credentials

## Outputs

After deployment, the following outputs are displayed:

| Output | Description |
|--------|-------------|
| ApiUrl | API Gateway endpoint URL |
| LambdaFunctionName | Lambda function name |
| LambdaFunctionArn | Lambda function ARN |
| DataBucketName | S3 data bucket name |
| ModelBucketName | S3 model bucket name |
| ArtifactsBucketName | S3 artifacts bucket name |
| UserFeaturesTableName | DynamoDB user features table |
| VideoFeaturesTableName | DynamoDB video features table |
| DashboardUrl | CloudWatch dashboard URL |
| RedisEndpoint | Redis endpoint (staging/prod) |
| MLPipelineStateMachineArn | Step Functions state machine ARN |
| EventStreamName | Kinesis event stream name |
| InferenceTableName | DynamoDB inference tracking table |
| FeatureStoreUserTableName | Feature store user features table |
| FeatureStoreVideoTableName | Feature store video features table |
| GlueDatabaseName | Glue database for offline features |
| RedisCredentialsSecretArn | Redis credentials secret ARN |
| ApiKeysSecretArn | API keys secret ARN |
| ConfigSecretArn | Configuration secret ARN |

## Testing the Deployment

```bash
# Get the API URL from stack outputs
API_URL=$(aws cloudformation describe-stacks \
  --stack-name VideoRecSystem-dev \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
  --output text)

# Health check
curl ${API_URL}health

# Get recommendations
curl ${API_URL}recommendations/12345

# Post interaction
curl -X POST ${API_URL}interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "video_id": 101, "category": "gaming"}'
```

## Triggering the ML Pipeline

The ML pipeline can be triggered manually or runs automatically (weekly in production).

```bash
# Get the state machine ARN
STATE_MACHINE_ARN=$(aws cloudformation describe-stacks \
  --stack-name VideoRecSystem-dev \
  --query "Stacks[0].Outputs[?OutputKey=='MLPipelineStateMachineArn'].OutputValue" \
  --output text)

# Trigger the ML pipeline manually
aws stepfunctions start-execution \
  --state-machine-arn ${STATE_MACHINE_ARN} \
  --input '{"trigger": "manual", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'

# Check pipeline execution status
aws stepfunctions list-executions \
  --state-machine-arn ${STATE_MACHINE_ARN} \
  --max-results 5
```

### Pipeline Stages

1. **Preprocessing** - Generates synthetic data (1000 users, 500 videos, 10000 interactions), builds vocabularies, creates training datasets
2. **Training (Parallel)**
   - Two-Tower Model - Generates 64-dimensional video embeddings, uploads to S3
   - Ranker Model - Trains CatBoost classification model for ranking
3. **Evaluation** - Validates model quality against configurable thresholds:
   - Two-Tower: Recall@100 ≥ 0.1, NDCG ≥ 0.3
   - Ranker: AUC ≥ 0.6, NDCG ≥ 0.3
4. **Deployment** (Conditional - only if evaluation passes)
   - Copies embeddings to production S3 location (`vector_store/video_embeddings.npz`)
   - Updates DynamoDB video features table with embeddings
   - Updates model registry with new versions

### Pipeline Artifacts

After successful execution, artifacts are stored:

```
s3://{model-bucket}/
├── models/
│   ├── two_tower/{version}/
│   │   ├── video_embeddings.npz    # Video embeddings
│   │   ├── model_config.json       # Model configuration
│   │   └── vocabularies.json       # Feature vocabularies
│   └── ranker/{version}/
│       └── ranker_model.cbm        # CatBoost model
├── vector_store/
│   └── video_embeddings.npz        # Production embeddings (used by Lambda)
└── preprocessing/{job_id}/
    ├── vocabularies.json           # Shared vocabularies
    ├── video_catalog.parquet       # Video metadata
    └── ranker_train.parquet        # Training data
```

## Cost Estimates

### Development (~$15-30/month)
- Lambda: Pay per request (serving + pipeline functions)
- DynamoDB: On-demand pricing (all tables)
- S3: Minimal storage costs
- Kinesis: 1 shard (~$15/month)
- Step Functions: Pay per state transition

### Staging (~$100-200/month)
- Above + VPC NAT Gateway
- ElastiCache t3.small
- Kinesis: 2 shards
- More frequent pipeline runs

### Production (~$400-800/month)
- Above + larger instances
- ElastiCache r6g.large
- Kinesis: 4 shards
- Provisioned DynamoDB capacity
- SageMaker training jobs (on-demand)
- Weekly automated training

## Troubleshooting

### CDK Bootstrap Failed
```bash
# Ensure AWS credentials are configured
aws sts get-caller-identity

# Bootstrap with explicit account/region
npx cdk bootstrap aws://ACCOUNT_ID/REGION
```

### Lambda Deployment Failed
```bash
# Check Lambda logs
aws logs tail /aws/lambda/VideoRecSystem-dev-... --follow
```

### Permission Denied
```bash
# Ensure IAM user has required permissions
# Required: CloudFormation, Lambda, S3, DynamoDB, API Gateway, IAM
```

## Clean Up

```bash
# Destroy all resources
npm run destroy

# Or destroy specific stack
npx cdk destroy VideoRecSystem-dev
```

**Note:** Production stacks have `RETAIN` removal policy. You may need to manually delete S3 buckets and DynamoDB tables.
