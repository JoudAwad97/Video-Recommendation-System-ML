import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';
import {
  Storage,
  Networking,
  Compute,
  Api,
  Monitoring,
  MLPipeline,
  DataCollection,
  FeatureStore,
} from '../constructs';

/**
 * Properties for VideoRecommendationStack.
 */
export interface VideoRecommendationStackProps extends cdk.StackProps {
  readonly config: EnvironmentConfig;
}

/**
 * Main stack for the Video Recommendation System.
 *
 * This stack orchestrates all the infrastructure components:
 * - Storage (S3, DynamoDB)
 * - Networking (VPC, Security Groups, ElastiCache)
 * - Compute (Lambda)
 * - API (API Gateway)
 * - Monitoring (CloudWatch Dashboard, Alarms)
 * - ML Pipeline (Preprocessing, Training, Evaluation, Deployment)
 * - Data Collection (Kinesis, Firehose, Inference Tracking)
 * - Feature Store (Online DynamoDB, Offline Glue)
 */
export class VideoRecommendationStack extends cdk.Stack {
  /** Storage construct */
  public readonly storage: Storage;

  /** Networking construct */
  public readonly networking: Networking;

  /** Compute construct */
  public readonly compute: Compute;

  /** API construct */
  public readonly api: Api;

  /** Monitoring construct */
  public readonly monitoring: Monitoring;

  /** ML Pipeline construct */
  public readonly mlPipeline: MLPipeline;

  /** Data Collection construct */
  public readonly dataCollection: DataCollection;

  /** Feature Store construct */
  public readonly featureStore: FeatureStore;

  constructor(scope: Construct, id: string, props: VideoRecommendationStackProps) {
    super(scope, id, props);

    const { config } = props;

    // =========================================================================
    // Storage Layer
    // =========================================================================

    this.storage = new Storage(this, 'Storage', {
      config,
    });

    // =========================================================================
    // Networking Layer
    // =========================================================================

    this.networking = new Networking(this, 'Networking', {
      config,
    });

    // =========================================================================
    // Compute Layer
    // =========================================================================

    this.compute = new Compute(this, 'Compute', {
      config,
      dataBucket: this.storage.dataBucket,
      modelBucket: this.storage.modelBucket,
      userFeaturesTable: this.storage.userFeaturesTable,
      videoFeaturesTable: this.storage.videoFeaturesTable,
      recommendationsCacheTable: this.storage.recommendationsCacheTable,
      vpc: this.networking.vpc,
      lambdaSecurityGroup: this.networking.lambdaSecurityGroup,
      redisEndpoint: this.networking.redisEndpoint,
    });

    // =========================================================================
    // API Layer
    // =========================================================================

    this.api = new Api(this, 'Api', {
      config,
      lambdaFunction: this.compute.recommendationFunction,
    });

    // =========================================================================
    // Monitoring Layer
    // =========================================================================

    this.monitoring = new Monitoring(this, 'Monitoring', {
      config,
      lambdaFunction: this.compute.recommendationFunction,
      restApi: this.api.restApi,
    });

    // =========================================================================
    // ML Pipeline Layer
    // =========================================================================

    this.mlPipeline = new MLPipeline(this, 'MLPipeline', {
      config,
      dataBucket: this.storage.dataBucket,
      modelBucket: this.storage.modelBucket,
      artifactsBucket: this.storage.artifactsBucket,
      videoFeaturesTableName: this.storage.videoFeaturesTable.tableName,
    });

    // =========================================================================
    // Data Collection Layer
    // =========================================================================

    this.dataCollection = new DataCollection(this, 'DataCollection', {
      config,
      dataBucket: this.storage.dataBucket,
    });

    // =========================================================================
    // Feature Store Layer
    // =========================================================================

    this.featureStore = new FeatureStore(this, 'FeatureStore', {
      config,
      dataBucket: this.storage.dataBucket,
    });

    // =========================================================================
    // Stack Outputs
    // =========================================================================

    new cdk.CfnOutput(this, 'ApiUrl', {
      value: this.api.apiUrl,
      description: 'API Gateway URL',
      exportName: `${id}-ApiUrl`,
    });

    new cdk.CfnOutput(this, 'LambdaFunctionName', {
      value: this.compute.recommendationFunction.functionName,
      description: 'Lambda Function Name',
      exportName: `${id}-LambdaFunctionName`,
    });

    new cdk.CfnOutput(this, 'LambdaFunctionArn', {
      value: this.compute.recommendationFunction.functionArn,
      description: 'Lambda Function ARN',
      exportName: `${id}-LambdaFunctionArn`,
    });

    new cdk.CfnOutput(this, 'DataBucketName', {
      value: this.storage.dataBucket.bucketName,
      description: 'S3 Data Bucket Name',
      exportName: `${id}-DataBucketName`,
    });

    new cdk.CfnOutput(this, 'ModelBucketName', {
      value: this.storage.modelBucket.bucketName,
      description: 'S3 Model Bucket Name',
      exportName: `${id}-ModelBucketName`,
    });

    new cdk.CfnOutput(this, 'UserFeaturesTableName', {
      value: this.storage.userFeaturesTable.tableName,
      description: 'DynamoDB User Features Table',
      exportName: `${id}-UserFeaturesTableName`,
    });

    new cdk.CfnOutput(this, 'VideoFeaturesTableName', {
      value: this.storage.videoFeaturesTable.tableName,
      description: 'DynamoDB Video Features Table',
      exportName: `${id}-VideoFeaturesTableName`,
    });

    new cdk.CfnOutput(this, 'DashboardUrl', {
      value: `https://${this.region}.console.aws.amazon.com/cloudwatch/home?region=${this.region}#dashboards:name=VideoRecSystem-${config.environment}`,
      description: 'CloudWatch Dashboard URL',
    });

    if (this.networking.redisEndpoint) {
      new cdk.CfnOutput(this, 'RedisEndpoint', {
        value: this.networking.redisEndpoint,
        description: 'ElastiCache Redis Endpoint',
        exportName: `${id}-RedisEndpoint`,
      });
    }

    // ML Pipeline Outputs
    new cdk.CfnOutput(this, 'ArtifactsBucketName', {
      value: this.storage.artifactsBucket.bucketName,
      description: 'S3 Artifacts Bucket Name',
      exportName: `${id}-ArtifactsBucketName`,
    });

    new cdk.CfnOutput(this, 'MLPipelineStateMachineArn', {
      value: this.mlPipeline.stateMachine.stateMachineArn,
      description: 'ML Pipeline Step Functions State Machine ARN',
      exportName: `${id}-MLPipelineArn`,
    });

    // Data Collection Outputs
    new cdk.CfnOutput(this, 'EventStreamName', {
      value: this.dataCollection.eventStream.streamName,
      description: 'Kinesis Event Stream Name',
      exportName: `${id}-EventStreamName`,
    });

    new cdk.CfnOutput(this, 'InferenceTableName', {
      value: this.dataCollection.inferenceTable.tableName,
      description: 'DynamoDB Inference Tracking Table',
      exportName: `${id}-InferenceTableName`,
    });

    // Feature Store Outputs
    new cdk.CfnOutput(this, 'FeatureStoreUserTableName', {
      value: this.featureStore.userFeaturesTable.tableName,
      description: 'Feature Store User Features Table',
      exportName: `${id}-FeatureStoreUserTableName`,
    });

    new cdk.CfnOutput(this, 'FeatureStoreVideoTableName', {
      value: this.featureStore.videoFeaturesTable.tableName,
      description: 'Feature Store Video Features Table',
      exportName: `${id}-FeatureStoreVideoTableName`,
    });

    new cdk.CfnOutput(this, 'GlueDatabaseName', {
      value: this.featureStore.glueDatabase.ref,
      description: 'Glue Database for Offline Feature Store',
      exportName: `${id}-GlueDatabaseName`,
    });
  }
}
