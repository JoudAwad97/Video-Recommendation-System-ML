import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as glue from 'aws-cdk-lib/aws-glue';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';
import * as path from 'path';

/**
 * Properties for the FeatureStore construct.
 */
export interface FeatureStoreProps {
  readonly config: EnvironmentConfig;

  /** S3 bucket for offline feature storage */
  readonly dataBucket: s3.Bucket;
}

/**
 * Feature Store construct for online and offline features.
 *
 * Creates:
 * - DynamoDB tables for online feature serving
 * - Glue database and tables for offline feature storage
 * - Lambda function for feature ingestion
 * - IAM roles with appropriate permissions
 */
export class FeatureStore extends Construct {
  /** DynamoDB table for user features (online) */
  public readonly userFeaturesTable: dynamodb.Table;

  /** DynamoDB table for video features (online) */
  public readonly videoFeaturesTable: dynamodb.Table;

  /** DynamoDB table for real-time user activity */
  public readonly userActivityTable: dynamodb.Table;

  /** Glue database for offline features */
  public readonly glueDatabase: glue.CfnDatabase;

  /** Lambda function for feature ingestion */
  public readonly ingestionFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: FeatureStoreProps) {
    super(scope, id);

    const { config, dataBucket } = props;
    const removalPolicy = config.isProduction
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;

    const billingMode = config.dynamodb.billingMode === 'PROVISIONED'
      ? dynamodb.BillingMode.PROVISIONED
      : dynamodb.BillingMode.PAY_PER_REQUEST;

    // =========================================================================
    // Online Feature Store (DynamoDB)
    // =========================================================================

    this.userFeaturesTable = new dynamodb.Table(this, 'UserFeaturesTable', {
      tableName: `${cdk.Stack.of(this).stackName}-user-features`,
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
      tableName: `${cdk.Stack.of(this).stackName}-video-features`,
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

    this.userActivityTable = new dynamodb.Table(this, 'UserActivityTable', {
      tableName: `${cdk.Stack.of(this).stackName}-user-activity`,
      partitionKey: {
        name: 'user_id',
        type: dynamodb.AttributeType.NUMBER,
      },
      sortKey: {
        name: 'timestamp',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy,
    });

    // Add GSI for category lookups
    this.userActivityTable.addGlobalSecondaryIndex({
      indexName: 'category-index',
      partitionKey: {
        name: 'user_id',
        type: dynamodb.AttributeType.NUMBER,
      },
      sortKey: {
        name: 'category',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // =========================================================================
    // Offline Feature Store (Glue)
    // =========================================================================

    this.glueDatabase = new glue.CfnDatabase(this, 'FeatureStoreDatabase', {
      catalogId: cdk.Stack.of(this).account,
      databaseInput: {
        name: `${cdk.Stack.of(this).stackName.toLowerCase().replace(/-/g, '_')}_features`,
        description: 'Offline feature store for video recommendation system',
      },
    });

    // User features table
    new glue.CfnTable(this, 'UserFeaturesGlueTable', {
      catalogId: cdk.Stack.of(this).account,
      databaseName: this.glueDatabase.ref,
      tableInput: {
        name: 'user_features',
        description: 'Offline user features',
        tableType: 'EXTERNAL_TABLE',
        parameters: {
          'classification': 'parquet',
        },
        storageDescriptor: {
          location: `s3://${dataBucket.bucketName}/features/user_features/`,
          inputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
          outputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
          serdeInfo: {
            serializationLibrary: 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
          },
          columns: [
            { name: 'user_id', type: 'bigint' },
            { name: 'country', type: 'string' },
            { name: 'language', type: 'string' },
            { name: 'age', type: 'int' },
            { name: 'age_bucket', type: 'int' },
            { name: 'updated_at', type: 'timestamp' },
          ],
        },
        partitionKeys: [
          { name: 'dt', type: 'string' },
        ],
      },
    });

    // Video features table
    new glue.CfnTable(this, 'VideoFeaturesGlueTable', {
      catalogId: cdk.Stack.of(this).account,
      databaseName: this.glueDatabase.ref,
      tableInput: {
        name: 'video_features',
        description: 'Offline video features',
        tableType: 'EXTERNAL_TABLE',
        parameters: {
          'classification': 'parquet',
        },
        storageDescriptor: {
          location: `s3://${dataBucket.bucketName}/features/video_features/`,
          inputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
          outputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
          serdeInfo: {
            serializationLibrary: 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
          },
          columns: [
            { name: 'video_id', type: 'bigint' },
            { name: 'title', type: 'string' },
            { name: 'category', type: 'string' },
            { name: 'duration', type: 'int' },
            { name: 'language', type: 'string' },
            { name: 'view_count', type: 'bigint' },
            { name: 'like_count', type: 'bigint' },
            { name: 'updated_at', type: 'timestamp' },
          ],
        },
        partitionKeys: [
          { name: 'dt', type: 'string' },
        ],
      },
    });

    // =========================================================================
    // Feature Ingestion Lambda
    // =========================================================================

    const ingestionRole = new iam.Role(this, 'IngestionLambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    dataBucket.grantReadWrite(ingestionRole);
    this.userFeaturesTable.grantReadWriteData(ingestionRole);
    this.videoFeaturesTable.grantReadWriteData(ingestionRole);
    this.userActivityTable.grantReadWriteData(ingestionRole);

    this.ingestionFunction = new lambda.Function(this, 'IngestionFunction', {
      functionName: `${cdk.Stack.of(this).stackName}-feature-ingestion`,
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'src.lambdas.feature_ingestion.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../../'), {
        exclude: [
          'deployment',
          'tests',
          'notebooks',
          '.git',
          '.venv',
          'venv',
          '__pycache__',
          '*.pyc',
        ],
      }),
      role: ingestionRole,
      timeout: cdk.Duration.minutes(5),
      memorySize: 1024,
      environment: {
        ENVIRONMENT: config.environment,
        DATA_BUCKET: dataBucket.bucketName,
        USER_FEATURES_TABLE: this.userFeaturesTable.tableName,
        VIDEO_FEATURES_TABLE: this.videoFeaturesTable.tableName,
        USER_ACTIVITY_TABLE: this.userActivityTable.tableName,
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });
  }
}
