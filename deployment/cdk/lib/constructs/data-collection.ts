import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as kinesis from 'aws-cdk-lib/aws-kinesis';
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';
import * as path from 'path';

/**
 * Properties for the DataCollection construct.
 */
export interface DataCollectionProps {
  readonly config: EnvironmentConfig;

  /** S3 bucket for data storage */
  readonly dataBucket: s3.Bucket;
}

/**
 * Data Collection construct for inference tracking and feedback.
 *
 * Creates:
 * - Kinesis stream for real-time event ingestion
 * - Firehose delivery stream for S3 persistence
 * - DynamoDB table for inference tracking
 * - Lambda functions for data processing
 */
export class DataCollection extends Construct {
  /** Kinesis stream for event ingestion */
  public readonly eventStream: kinesis.Stream;

  /** Firehose delivery stream */
  public readonly deliveryStream: firehose.CfnDeliveryStream;

  /** DynamoDB table for inference tracking */
  public readonly inferenceTable: dynamodb.Table;

  /** Lambda function for processing events */
  public readonly processingFunction: lambda.Function;

  /** Lambda function for merging feedback */
  public readonly mergeFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: DataCollectionProps) {
    super(scope, id);

    const { config, dataBucket } = props;
    const removalPolicy = config.isProduction
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;

    // =========================================================================
    // Kinesis Data Stream
    // =========================================================================

    this.eventStream = new kinesis.Stream(this, 'EventStream', {
      streamName: `${cdk.Stack.of(this).stackName}-events`,
      shardCount: config.isProduction ? 4 : 1,
      retentionPeriod: cdk.Duration.hours(24),
    });

    // =========================================================================
    // Firehose Delivery Stream Role
    // =========================================================================

    const firehoseRole = new iam.Role(this, 'FirehoseRole', {
      assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
      inlinePolicies: {
        KinesisAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                'kinesis:DescribeStream',
                'kinesis:DescribeStreamSummary',
                'kinesis:GetRecords',
                'kinesis:GetShardIterator',
                'kinesis:ListShards',
                'kinesis:SubscribeToShard',
              ],
              resources: [this.eventStream.streamArn],
            }),
          ],
        }),
        S3Access: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                's3:AbortMultipartUpload',
                's3:GetBucketLocation',
                's3:GetObject',
                's3:ListBucket',
                's3:ListBucketMultipartUploads',
                's3:PutObject',
              ],
              resources: [
                dataBucket.bucketArn,
                `${dataBucket.bucketArn}/*`,
              ],
            }),
          ],
        }),
        CloudWatchLogsAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                'logs:CreateLogGroup',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
              ],
              resources: ['*'],
            }),
          ],
        }),
      },
    });

    // =========================================================================
    // Firehose Delivery Stream
    // =========================================================================

    this.deliveryStream = new firehose.CfnDeliveryStream(this, 'DeliveryStream', {
      deliveryStreamName: `${cdk.Stack.of(this).stackName}-events-delivery`,
      deliveryStreamType: 'KinesisStreamAsSource',
      kinesisStreamSourceConfiguration: {
        kinesisStreamArn: this.eventStream.streamArn,
        roleArn: firehoseRole.roleArn,
      },
      extendedS3DestinationConfiguration: {
        bucketArn: dataBucket.bucketArn,
        roleArn: firehoseRole.roleArn,
        prefix: 'events/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/',
        errorOutputPrefix: 'errors/!{firehose:error-output-type}/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/',
        bufferingHints: {
          intervalInSeconds: config.isProduction ? 60 : 300,
          sizeInMBs: config.isProduction ? 64 : 5,
        },
        compressionFormat: 'GZIP',
        cloudWatchLoggingOptions: {
          enabled: true,
          logGroupName: `/aws/firehose/${cdk.Stack.of(this).stackName}-events`,
          logStreamName: 'delivery',
        },
      },
    });

    // Ensure role is fully created before Firehose
    this.deliveryStream.node.addDependency(firehoseRole);

    // =========================================================================
    // Inference Tracking Table
    // =========================================================================

    this.inferenceTable = new dynamodb.Table(this, 'InferenceTable', {
      tableName: `${cdk.Stack.of(this).stackName}-inference-tracking`,
      partitionKey: {
        name: 'request_id',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'timestamp',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy,
      stream: dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
    });

    // Add GSI for user queries
    this.inferenceTable.addGlobalSecondaryIndex({
      indexName: 'user-index',
      partitionKey: {
        name: 'user_id',
        type: dynamodb.AttributeType.NUMBER,
      },
      sortKey: {
        name: 'timestamp',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // =========================================================================
    // Lambda Execution Role
    // =========================================================================

    const lambdaRole = new iam.Role(this, 'DataCollectionLambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    dataBucket.grantReadWrite(lambdaRole);
    this.eventStream.grantRead(lambdaRole);
    this.inferenceTable.grantReadWriteData(lambdaRole);

    // =========================================================================
    // Event Processing Lambda
    // =========================================================================

    this.processingFunction = new lambda.Function(this, 'ProcessingFunction', {
      functionName: `${cdk.Stack.of(this).stackName}-event-processing`,
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'src.lambdas.event_processing.handler',
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
      role: lambdaRole,
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      environment: {
        ENVIRONMENT: config.environment,
        DATA_BUCKET: dataBucket.bucketName,
        INFERENCE_TABLE: this.inferenceTable.tableName,
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // Add Kinesis trigger
    this.processingFunction.addEventSource(new lambdaEventSources.KinesisEventSource(this.eventStream, {
      batchSize: 100,
      startingPosition: lambda.StartingPosition.LATEST,
      maxBatchingWindow: cdk.Duration.seconds(10),
      retryAttempts: 3,
    }));

    // =========================================================================
    // Merge Job Lambda
    // =========================================================================

    this.mergeFunction = new lambda.Function(this, 'MergeFunction', {
      functionName: `${cdk.Stack.of(this).stackName}-feedback-merge`,
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'src.lambdas.merge_job.handler',
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
      role: lambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        ENVIRONMENT: config.environment,
        DATA_BUCKET: dataBucket.bucketName,
        INFERENCE_TABLE: this.inferenceTable.tableName,
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });
  }
}
