import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';

/**
 * Properties for the Storage construct.
 */
export interface StorageProps {
  readonly config: EnvironmentConfig;
}

/**
 * Storage construct for S3 buckets and DynamoDB tables.
 *
 * Creates:
 * - S3 bucket for training data
 * - S3 bucket for model artifacts
 * - S3 bucket for pipeline artifacts (vocabularies, normalizers, embeddings)
 * - DynamoDB table for user features
 * - DynamoDB table for video features
 * - DynamoDB table for recommendations cache
 */
export class Storage extends Construct {
  /** S3 bucket for data storage */
  public readonly dataBucket: s3.Bucket;

  /** S3 bucket for model artifacts */
  public readonly modelBucket: s3.Bucket;

  /** S3 bucket for pipeline artifacts (vocabularies, normalizers, etc.) */
  public readonly artifactsBucket: s3.Bucket;

  /** DynamoDB table for user features */
  public readonly userFeaturesTable: dynamodb.Table;

  /** DynamoDB table for video features */
  public readonly videoFeaturesTable: dynamodb.Table;

  /** DynamoDB table for recommendations cache */
  public readonly recommendationsCacheTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props: StorageProps) {
    super(scope, id);

    const { config } = props;
    const removalPolicy = this.getRemovalPolicy(config.removalPolicy);

    // =========================================================================
    // S3 Buckets
    // =========================================================================

    this.dataBucket = new s3.Bucket(this, 'DataBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      removalPolicy,
      autoDeleteObjects: !config.isProduction,
      lifecycleRules: [
        {
          id: 'DeleteOldVersions',
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
        {
          id: 'TransitionToIA',
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(90),
            },
          ],
        },
      ],
    });

    this.modelBucket = new s3.Bucket(this, 'ModelBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      removalPolicy,
      autoDeleteObjects: !config.isProduction,
    });

    this.artifactsBucket = new s3.Bucket(this, 'ArtifactsBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      removalPolicy,
      autoDeleteObjects: !config.isProduction,
      lifecycleRules: [
        {
          id: 'DeleteOldVersions',
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
      ],
    });

    // =========================================================================
    // DynamoDB Tables
    // =========================================================================

    const billingMode = config.dynamodb.billingMode === 'PROVISIONED'
      ? dynamodb.BillingMode.PROVISIONED
      : dynamodb.BillingMode.PAY_PER_REQUEST;

    this.userFeaturesTable = new dynamodb.Table(this, 'UserFeaturesTable', {
      partitionKey: {
        name: 'user_id',
        type: dynamodb.AttributeType.NUMBER,
      },
      billingMode,
      readCapacity: config.dynamodb.readCapacity,
      writeCapacity: config.dynamodb.writeCapacity,
      removalPolicy,
      pointInTimeRecovery: config.dynamodb.pointInTimeRecovery,
    });

    this.videoFeaturesTable = new dynamodb.Table(this, 'VideoFeaturesTable', {
      partitionKey: {
        name: 'video_id',
        type: dynamodb.AttributeType.NUMBER,
      },
      billingMode,
      readCapacity: config.dynamodb.readCapacity,
      writeCapacity: config.dynamodb.writeCapacity,
      removalPolicy,
      pointInTimeRecovery: config.dynamodb.pointInTimeRecovery,
    });

    this.recommendationsCacheTable = new dynamodb.Table(this, 'RecommendationsCacheTable', {
      partitionKey: {
        name: 'user_id',
        type: dynamodb.AttributeType.NUMBER,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy,
    });
  }

  /**
   * Convert string removal policy to CDK RemovalPolicy.
   */
  private getRemovalPolicy(policy: string): cdk.RemovalPolicy {
    switch (policy) {
      case 'RETAIN':
        return cdk.RemovalPolicy.RETAIN;
      case 'SNAPSHOT':
        return cdk.RemovalPolicy.SNAPSHOT;
      default:
        return cdk.RemovalPolicy.DESTROY;
    }
  }
}
