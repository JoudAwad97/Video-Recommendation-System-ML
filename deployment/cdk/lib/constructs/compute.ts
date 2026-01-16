import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ecr_assets from 'aws-cdk-lib/aws-ecr-assets';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';
import * as path from 'path';

/**
 * Properties for the Compute construct.
 */
export interface ComputeProps {
  readonly config: EnvironmentConfig;

  /** S3 bucket for data */
  readonly dataBucket: s3.Bucket;

  /** S3 bucket for models */
  readonly modelBucket: s3.Bucket;

  /** DynamoDB table for user features */
  readonly userFeaturesTable: dynamodb.Table;

  /** DynamoDB table for video features */
  readonly videoFeaturesTable: dynamodb.Table;

  /** DynamoDB table for recommendations cache */
  readonly recommendationsCacheTable: dynamodb.Table;

  /** VPC for Lambda (optional) */
  readonly vpc?: ec2.Vpc;

  /** Security group for Lambda (optional) */
  readonly lambdaSecurityGroup?: ec2.SecurityGroup;

  /** Redis endpoint (optional) */
  readonly redisEndpoint?: string;
}

/**
 * Compute construct for Lambda functions.
 *
 * Creates:
 * - Lambda execution role with required permissions
 * - Recommendation service Lambda function (Docker-based with ML dependencies)
 * - CloudWatch log group with retention
 */
export class Compute extends Construct {
  /** Lambda function for recommendations */
  public readonly recommendationFunction: lambda.DockerImageFunction;

  /** Lambda execution role */
  public readonly executionRole: iam.Role;

  constructor(scope: Construct, id: string, props: ComputeProps) {
    super(scope, id);

    const { config } = props;

    // =========================================================================
    // IAM Role
    // =========================================================================

    this.executionRole = new iam.Role(this, 'LambdaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Execution role for Video Recommendation Lambda',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaBasicExecutionRole'
        ),
      ],
    });

    // Add VPC access policy if VPC is enabled
    if (props.vpc) {
      this.executionRole.addManagedPolicy(
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaVPCAccessExecutionRole'
        )
      );
    }

    // Grant S3 read access
    props.dataBucket.grantRead(this.executionRole);
    props.modelBucket.grantRead(this.executionRole);

    // Grant DynamoDB access
    props.userFeaturesTable.grantReadData(this.executionRole);
    props.videoFeaturesTable.grantReadData(this.executionRole);
    props.recommendationsCacheTable.grantReadWriteData(this.executionRole);

    // =========================================================================
    // Lambda Function (Docker-based for ML dependencies)
    // =========================================================================

    // Build environment variables
    const environment: Record<string, string> = {
      ENVIRONMENT: config.environment,
      LOG_LEVEL: config.logging.level,
      USE_REDIS: config.redis.enabled ? 'true' : 'false',
      REDIS_HOST: props.redisEndpoint || 'localhost',
      REDIS_PORT: '6379',
      USE_SAGEMAKER_FEATURE_STORE: 'false',
      MODEL_BUCKET: props.modelBucket.bucketName,
      DATA_BUCKET: props.dataBucket.bucketName,
      USER_FEATURES_TABLE: props.userFeaturesTable.tableName,
      VIDEO_FEATURES_TABLE: props.videoFeaturesTable.tableName,
      RECOMMENDATIONS_CACHE_TABLE: props.recommendationsCacheTable.tableName,
      DEFAULT_NUM_RECOMMENDATIONS: '20',
      MAX_NUM_RECOMMENDATIONS: '100',
    };

    // Docker image code from local Dockerfile
    // Use ARM64 architecture for better price/performance with Graviton2
    const dockerImageCode = lambda.DockerImageCode.fromImageAsset(
      path.join(__dirname, '../../../../'),
      {
        file: 'deployment/docker/Dockerfile.lambda',
        platform: ecr_assets.Platform.LINUX_ARM64,
        exclude: [
          '.git',
          '.venv',
          'venv',
          '__pycache__',
          '*.pyc',
          '.pytest_cache',
          '.mypy_cache',
          '*.egg-info',
          'dist',
          'build',
          '.coverage',
          'htmlcov',
          '*.log',
          'node_modules',
          'cdk.out',
          'deployment/cdk/cdk.out',
          'deployment/cdk/node_modules',
          'tests',
          'notebooks',
        ],
      }
    );

    // Create Docker-based Lambda function
    // Use ARM64 (Graviton2) for better price/performance
    this.recommendationFunction = new lambda.DockerImageFunction(
      this,
      'RecommendationFunction',
      {
        code: dockerImageCode,
        architecture: lambda.Architecture.ARM_64,
        timeout: cdk.Duration.seconds(config.lambda.timeout),
        memorySize: config.lambda.memorySize,
        role: this.executionRole,
        environment,
        tracing: lambda.Tracing.ACTIVE,
        logRetention: this.getLogRetention(config.logging.retentionDays),
        ...(props.vpc && props.lambdaSecurityGroup
          ? {
              vpc: props.vpc,
              vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
              securityGroups: [props.lambdaSecurityGroup],
            }
          : {}),
        ...(config.lambda.reservedConcurrency
          ? { reservedConcurrentExecutions: config.lambda.reservedConcurrency }
          : {}),
      }
    );
  }

  /**
   * Convert retention days to CDK RetentionDays enum.
   */
  private getLogRetention(days: number): logs.RetentionDays {
    const retentionMap: Record<number, logs.RetentionDays> = {
      1: logs.RetentionDays.ONE_DAY,
      3: logs.RetentionDays.THREE_DAYS,
      5: logs.RetentionDays.FIVE_DAYS,
      7: logs.RetentionDays.ONE_WEEK,
      14: logs.RetentionDays.TWO_WEEKS,
      30: logs.RetentionDays.ONE_MONTH,
      60: logs.RetentionDays.TWO_MONTHS,
      90: logs.RetentionDays.THREE_MONTHS,
    };

    return retentionMap[days] || logs.RetentionDays.ONE_WEEK;
  }
}
