import * as cdk from 'aws-cdk-lib';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';

/**
 * Properties for the Api construct.
 */
export interface ApiProps {
  readonly config: EnvironmentConfig;

  /** Lambda function to integrate with */
  readonly lambdaFunction: lambda.Function;
}

/**
 * API construct for API Gateway.
 *
 * Creates:
 * - REST API with Lambda proxy integration
 * - Routes for recommendations, interactions, health check
 * - CORS configuration
 * - Throttling and rate limiting
 */
export class Api extends Construct {
  /** API Gateway REST API */
  public readonly restApi: apigateway.RestApi;

  /** API URL */
  public readonly apiUrl: string;

  constructor(scope: Construct, id: string, props: ApiProps) {
    super(scope, id);

    const { config, lambdaFunction } = props;

    // =========================================================================
    // REST API
    // =========================================================================

    this.restApi = new apigateway.RestApi(this, 'RecommendationApi', {
      description: 'Video Recommendation System API',
      deployOptions: {
        stageName: config.environment,
        throttlingBurstLimit: config.apiGateway.throttlingBurstLimit,
        throttlingRateLimit: config.apiGateway.throttlingRateLimit,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: !config.isProduction,
        metricsEnabled: true,
        tracingEnabled: true,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: [
          'Content-Type',
          'Authorization',
          'X-Api-Key',
          'X-Amz-Date',
          'X-Amz-Security-Token',
        ],
        allowCredentials: true,
      },
      cloudWatchRole: true,
    });

    // Lambda integration
    const lambdaIntegration = new apigateway.LambdaIntegration(lambdaFunction, {
      proxy: true,
    });

    // =========================================================================
    // Routes
    // =========================================================================

    // GET /health
    const healthResource = this.restApi.root.addResource('health');
    healthResource.addMethod('GET', lambdaIntegration);

    // /recommendations
    const recommendationsResource = this.restApi.root.addResource('recommendations');

    // POST /recommendations
    recommendationsResource.addMethod('POST', lambdaIntegration);

    // GET /recommendations/{user_id}
    const userRecommendationsResource = recommendationsResource.addResource('{user_id}');
    userRecommendationsResource.addMethod('GET', lambdaIntegration);

    // POST /interactions
    const interactionsResource = this.restApi.root.addResource('interactions');
    interactionsResource.addMethod('POST', lambdaIntegration);

    // /cached/{user_id}
    const cachedResource = this.restApi.root.addResource('cached');
    const cachedUserResource = cachedResource.addResource('{user_id}');
    cachedUserResource.addMethod('GET', lambdaIntegration);

    // Store API URL
    this.apiUrl = this.restApi.url;
  }
}
