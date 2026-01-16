import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as stepfunctions from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as ecr_assets from 'aws-cdk-lib/aws-ecr-assets';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';
import * as path from 'path';

/**
 * Properties for the MLPipeline construct.
 */
export interface MLPipelineProps {
  readonly config: EnvironmentConfig;

  /** S3 bucket for data */
  readonly dataBucket: s3.Bucket;

  /** S3 bucket for models */
  readonly modelBucket: s3.Bucket;

  /** S3 bucket for artifacts */
  readonly artifactsBucket: s3.Bucket;

  /** DynamoDB table name for video features (optional) */
  readonly videoFeaturesTableName?: string;
}

/**
 * ML Pipeline construct for training and preprocessing.
 *
 * Creates:
 * - Docker-based Lambda functions for preprocessing, training, and evaluation
 * - Step Functions state machine for pipeline orchestration
 * - EventBridge rule for scheduled training (optional)
 * - IAM roles with appropriate permissions
 */
export class MLPipeline extends Construct {
  /** Step Functions state machine for the ML pipeline */
  public readonly stateMachine: stepfunctions.StateMachine;

  /** Lambda function for data preprocessing */
  public readonly preprocessingFunction: lambda.DockerImageFunction;

  /** Lambda function for Two-Tower training */
  public readonly twoTowerTrainingFunction: lambda.DockerImageFunction;

  /** Lambda function for Ranker training */
  public readonly rankerTrainingFunction: lambda.DockerImageFunction;

  /** Lambda function for model evaluation */
  public readonly evaluationFunction: lambda.DockerImageFunction;

  /** Lambda function for model deployment */
  public readonly deploymentFunction: lambda.DockerImageFunction;

  /** IAM role for SageMaker */
  public readonly sagemakerRole: iam.Role;

  constructor(scope: Construct, id: string, props: MLPipelineProps) {
    super(scope, id);

    const { config, dataBucket, modelBucket, artifactsBucket } = props;

    // =========================================================================
    // SageMaker Execution Role
    // =========================================================================

    this.sagemakerRole = new iam.Role(this, 'SageMakerRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: 'Execution role for SageMaker training jobs',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });

    // Grant S3 access to SageMaker role
    dataBucket.grantReadWrite(this.sagemakerRole);
    modelBucket.grantReadWrite(this.sagemakerRole);
    artifactsBucket.grantReadWrite(this.sagemakerRole);

    // =========================================================================
    // Lambda Execution Role for Pipeline
    // =========================================================================

    const pipelineLambdaRole = new iam.Role(this, 'PipelineLambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Execution role for ML pipeline Lambda functions',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    // Grant S3 access
    dataBucket.grantReadWrite(pipelineLambdaRole);
    modelBucket.grantReadWrite(pipelineLambdaRole);
    artifactsBucket.grantReadWrite(pipelineLambdaRole);

    // Grant SageMaker permissions
    pipelineLambdaRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'sagemaker:CreateTrainingJob',
        'sagemaker:DescribeTrainingJob',
        'sagemaker:StopTrainingJob',
        'sagemaker:CreateModel',
        'sagemaker:CreateEndpointConfig',
        'sagemaker:CreateEndpoint',
        'sagemaker:UpdateEndpoint',
        'sagemaker:DescribeEndpoint',
        'sagemaker:DeleteEndpoint',
        'sagemaker:DeleteEndpointConfig',
        'sagemaker:DeleteModel',
      ],
      resources: ['*'],
    }));

    // Allow passing SageMaker role
    pipelineLambdaRole.addToPolicy(new iam.PolicyStatement({
      actions: ['iam:PassRole'],
      resources: [this.sagemakerRole.roleArn],
    }));

    // Grant DynamoDB write access for video features table (deployment)
    if (props.videoFeaturesTableName) {
      pipelineLambdaRole.addToPolicy(new iam.PolicyStatement({
        actions: [
          'dynamodb:PutItem',
          'dynamodb:BatchWriteItem',
          'dynamodb:UpdateItem',
        ],
        resources: [
          `arn:aws:dynamodb:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:table/${props.videoFeaturesTableName}`,
        ],
      }));
    }

    // =========================================================================
    // Docker Image for ML Pipeline Lambdas
    // =========================================================================

    // Common environment variables
    const commonEnvironment = {
      ENVIRONMENT: config.environment,
      DATA_BUCKET: dataBucket.bucketName,
      MODEL_BUCKET: modelBucket.bucketName,
      ARTIFACTS_BUCKET: artifactsBucket.bucketName,
      SAGEMAKER_ROLE_ARN: this.sagemakerRole.roleArn,
    };

    // Docker image code from local Dockerfile
    // Using ARM64 architecture for better price/performance with Graviton2
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

    // =========================================================================
    // Preprocessing Lambda (Docker-based)
    // =========================================================================

    this.preprocessingFunction = new lambda.DockerImageFunction(this, 'PreprocessingFunction', {
      code: dockerImageCode,
      architecture: lambda.Architecture.ARM_64,
      // Note: Using CDK auto-generated names to allow resource replacement
      description: 'Data preprocessing and feature engineering',
      role: pipelineLambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        ...commonEnvironment,
        LAMBDA_HANDLER: 'preprocessing',
      },
      logRetention: logs.RetentionDays.TWO_WEEKS,
    });

    // Override the handler using escape hatch (Docker image uses CMD)
    const preprocessingCfn = this.preprocessingFunction.node.defaultChild as lambda.CfnFunction;
    preprocessingCfn.addPropertyOverride('ImageConfig', {
      Command: ['src.lambdas.preprocessing.handler'],
    });

    // =========================================================================
    // Two-Tower Training Lambda (Docker-based)
    // =========================================================================

    this.twoTowerTrainingFunction = new lambda.DockerImageFunction(this, 'TwoTowerTrainingFunction', {
      code: dockerImageCode,
      architecture: lambda.Architecture.ARM_64,
      // Note: Using CDK auto-generated names to allow resource replacement
      description: 'Two-Tower model training orchestration',
      role: pipelineLambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        ...commonEnvironment,
        LAMBDA_HANDLER: 'two_tower_training',
      },
      logRetention: logs.RetentionDays.TWO_WEEKS,
    });

    const twoTowerCfn = this.twoTowerTrainingFunction.node.defaultChild as lambda.CfnFunction;
    twoTowerCfn.addPropertyOverride('ImageConfig', {
      Command: ['src.lambdas.training.two_tower_handler'],
    });

    // =========================================================================
    // Ranker Training Lambda (Docker-based)
    // =========================================================================

    this.rankerTrainingFunction = new lambda.DockerImageFunction(this, 'RankerTrainingFunction', {
      code: dockerImageCode,
      architecture: lambda.Architecture.ARM_64,
      // Note: Using CDK auto-generated names to allow resource replacement
      description: 'Ranker model training orchestration',
      role: pipelineLambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        ...commonEnvironment,
        LAMBDA_HANDLER: 'ranker_training',
      },
      logRetention: logs.RetentionDays.TWO_WEEKS,
    });

    const rankerCfn = this.rankerTrainingFunction.node.defaultChild as lambda.CfnFunction;
    rankerCfn.addPropertyOverride('ImageConfig', {
      Command: ['src.lambdas.training.ranker_handler'],
    });

    // =========================================================================
    // Evaluation Lambda (Docker-based)
    // =========================================================================

    this.evaluationFunction = new lambda.DockerImageFunction(this, 'EvaluationFunction', {
      code: dockerImageCode,
      architecture: lambda.Architecture.ARM_64,
      // Note: Using CDK auto-generated names to allow resource replacement
      description: 'Model evaluation and metrics computation',
      role: pipelineLambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        ...commonEnvironment,
        LAMBDA_HANDLER: 'evaluation',
      },
      logRetention: logs.RetentionDays.TWO_WEEKS,
    });

    const evaluationCfn = this.evaluationFunction.node.defaultChild as lambda.CfnFunction;
    evaluationCfn.addPropertyOverride('ImageConfig', {
      Command: ['src.lambdas.evaluation.handler'],
    });

    // =========================================================================
    // Deployment Lambda (Docker-based)
    // =========================================================================

    this.deploymentFunction = new lambda.DockerImageFunction(this, 'DeploymentFunction', {
      code: dockerImageCode,
      architecture: lambda.Architecture.ARM_64,
      // Note: Using CDK auto-generated names to allow resource replacement
      description: 'Model deployment to SageMaker endpoints',
      role: pipelineLambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        ...commonEnvironment,
        LAMBDA_HANDLER: 'deployment',
        VIDEO_FEATURES_TABLE: props.videoFeaturesTableName || '',
      },
      logRetention: logs.RetentionDays.TWO_WEEKS,
    });

    const deploymentCfn = this.deploymentFunction.node.defaultChild as lambda.CfnFunction;
    deploymentCfn.addPropertyOverride('ImageConfig', {
      Command: ['src.lambdas.deployment.handler'],
    });

    // =========================================================================
    // Step Functions State Machine
    // =========================================================================

    // Define tasks - payloadResponseOnly extracts just the Payload from Lambda response
    const preprocessingTask = new tasks.LambdaInvoke(this, 'PreprocessingTask', {
      lambdaFunction: this.preprocessingFunction,
      payloadResponseOnly: true,
      resultPath: '$.preprocessing',
    });

    const twoTowerTrainingTask = new tasks.LambdaInvoke(this, 'TwoTowerTrainingTask', {
      lambdaFunction: this.twoTowerTrainingFunction,
      payloadResponseOnly: true,
      resultPath: '$.two_tower_training',
    });

    const rankerTrainingTask = new tasks.LambdaInvoke(this, 'RankerTrainingTask', {
      lambdaFunction: this.rankerTrainingFunction,
      payloadResponseOnly: true,
      resultPath: '$.ranker_training',
    });

    const evaluationTask = new tasks.LambdaInvoke(this, 'EvaluationTask', {
      lambdaFunction: this.evaluationFunction,
      payloadResponseOnly: true,
      resultPath: '$.evaluation',
    });

    const deploymentTask = new tasks.LambdaInvoke(this, 'DeploymentTask', {
      lambdaFunction: this.deploymentFunction,
      payloadResponseOnly: true,
      resultPath: '$.deployment',
    });

    // Parallel training for Two-Tower and Ranker
    const parallelTraining = new stepfunctions.Parallel(this, 'ParallelTraining', {
      resultPath: '$.training_results',
    });
    parallelTraining.branch(twoTowerTrainingTask);
    parallelTraining.branch(rankerTrainingTask);

    // Check if deployment should proceed
    const shouldDeploy = new stepfunctions.Choice(this, 'ShouldDeploy')
      .when(
        stepfunctions.Condition.booleanEquals('$.evaluation.should_deploy', true),
        deploymentTask
      )
      .otherwise(new stepfunctions.Succeed(this, 'SkipDeployment', {
        comment: 'Model did not meet deployment criteria',
      }));

    // Define the workflow
    const definition = preprocessingTask
      .next(parallelTraining)
      .next(evaluationTask)
      .next(shouldDeploy);

    // Create state machine
    this.stateMachine = new stepfunctions.StateMachine(this, 'MLPipelineStateMachine', {
      stateMachineName: `${cdk.Stack.of(this).stackName}-ml-pipeline`,
      definition,
      timeout: cdk.Duration.hours(6),
      tracingEnabled: true,
      logs: {
        destination: new logs.LogGroup(this, 'StateMachineLogGroup', {
          logGroupName: `/aws/stepfunctions/${cdk.Stack.of(this).stackName}-ml-pipeline`,
          retention: logs.RetentionDays.TWO_WEEKS,
        }),
        level: stepfunctions.LogLevel.ALL,
      },
    });

    // =========================================================================
    // Scheduled Training (Production Only)
    // =========================================================================

    if (config.isProduction) {
      // Schedule weekly training
      new events.Rule(this, 'WeeklyTrainingRule', {
        ruleName: `${cdk.Stack.of(this).stackName}-weekly-training`,
        schedule: events.Schedule.cron({
          weekDay: 'SUN',
          hour: '2',
          minute: '0',
        }),
        targets: [new targets.SfnStateMachine(this.stateMachine, {
          input: events.RuleTargetInput.fromObject({
            trigger: 'scheduled',
            timestamp: events.EventField.time,
          }),
        })],
        description: 'Weekly model retraining schedule',
      });
    }
  }
}
