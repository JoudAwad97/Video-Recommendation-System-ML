import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as cloudwatch_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';

/**
 * Properties for the Monitoring construct.
 */
export interface MonitoringProps {
  readonly config: EnvironmentConfig;

  /** Lambda function to monitor */
  readonly lambdaFunction: lambda.Function;

  /** API Gateway to monitor */
  readonly restApi: apigateway.RestApi;
}

/**
 * Monitoring construct for CloudWatch dashboards and alarms.
 *
 * Creates:
 * - CloudWatch dashboard with key metrics
 * - Alarms for errors and latency
 * - SNS topic for alarm notifications (production only)
 */
export class Monitoring extends Construct {
  /** CloudWatch dashboard */
  public readonly dashboard: cloudwatch.Dashboard;

  /** SNS topic for alarms (production only) */
  public readonly alarmTopic?: sns.Topic;

  constructor(scope: Construct, id: string, props: MonitoringProps) {
    super(scope, id);

    const { config, lambdaFunction, restApi } = props;

    // =========================================================================
    // CloudWatch Dashboard
    // =========================================================================

    this.dashboard = new cloudwatch.Dashboard(this, 'Dashboard', {
      dashboardName: `VideoRecSystem-${config.environment}`,
    });

    // Lambda metrics
    const invocationsWidget = new cloudwatch.GraphWidget({
      title: 'Lambda Invocations',
      left: [lambdaFunction.metricInvocations({ period: cdk.Duration.minutes(1) })],
      width: 8,
      height: 6,
    });

    const durationWidget = new cloudwatch.GraphWidget({
      title: 'Lambda Duration (ms)',
      left: [
        lambdaFunction.metricDuration({
          period: cdk.Duration.minutes(1),
          statistic: 'Average',
        }),
        lambdaFunction.metricDuration({
          period: cdk.Duration.minutes(1),
          statistic: 'p99',
        }),
      ],
      width: 8,
      height: 6,
    });

    const errorsWidget = new cloudwatch.GraphWidget({
      title: 'Lambda Errors',
      left: [lambdaFunction.metricErrors({ period: cdk.Duration.minutes(1) })],
      width: 8,
      height: 6,
    });

    // API Gateway metrics
    const apiRequestsWidget = new cloudwatch.GraphWidget({
      title: 'API Requests',
      left: [
        restApi.metricCount({ period: cdk.Duration.minutes(1) }),
      ],
      width: 8,
      height: 6,
    });

    const apiLatencyWidget = new cloudwatch.GraphWidget({
      title: 'API Latency (ms)',
      left: [
        restApi.metricLatency({
          period: cdk.Duration.minutes(1),
          statistic: 'Average',
        }),
        restApi.metricLatency({
          period: cdk.Duration.minutes(1),
          statistic: 'p99',
        }),
      ],
      width: 8,
      height: 6,
    });

    const api4xxWidget = new cloudwatch.GraphWidget({
      title: 'API 4XX Errors',
      left: [restApi.metricClientError({ period: cdk.Duration.minutes(1) })],
      width: 8,
      height: 6,
    });

    const api5xxWidget = new cloudwatch.GraphWidget({
      title: 'API 5XX Errors',
      left: [restApi.metricServerError({ period: cdk.Duration.minutes(1) })],
      width: 8,
      height: 6,
    });

    // Concurrent executions
    const concurrencyWidget = new cloudwatch.GraphWidget({
      title: 'Concurrent Executions',
      left: [
        lambdaFunction.metric('ConcurrentExecutions', {
          period: cdk.Duration.minutes(1),
          statistic: 'Maximum',
        }),
      ],
      width: 8,
      height: 6,
    });

    // Add widgets to dashboard
    this.dashboard.addWidgets(invocationsWidget, durationWidget, errorsWidget);
    this.dashboard.addWidgets(apiRequestsWidget, apiLatencyWidget, concurrencyWidget);
    this.dashboard.addWidgets(api4xxWidget, api5xxWidget);

    // =========================================================================
    // Alarms (Production Only)
    // =========================================================================

    if (config.isProduction) {
      // Create SNS topic for alarms
      this.alarmTopic = new sns.Topic(this, 'AlarmTopic', {
        displayName: `VideoRecSystem-Alarms-${config.environment}`,
      });

      // Lambda error rate alarm
      const errorAlarm = new cloudwatch.Alarm(this, 'LambdaErrorAlarm', {
        metric: lambdaFunction.metricErrors({
          period: cdk.Duration.minutes(5),
        }),
        threshold: 10,
        evaluationPeriods: 2,
        alarmDescription: 'Lambda function error rate is too high',
        treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      });
      errorAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(this.alarmTopic));

      // Lambda duration alarm (p99 > 5 seconds)
      const durationAlarm = new cloudwatch.Alarm(this, 'LambdaDurationAlarm', {
        metric: lambdaFunction.metricDuration({
          period: cdk.Duration.minutes(5),
          statistic: 'p99',
        }),
        threshold: 5000,
        evaluationPeriods: 3,
        alarmDescription: 'Lambda p99 latency is too high',
        treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      });
      durationAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(this.alarmTopic));

      // API 5XX error alarm
      const api5xxAlarm = new cloudwatch.Alarm(this, 'Api5xxAlarm', {
        metric: restApi.metricServerError({
          period: cdk.Duration.minutes(5),
        }),
        threshold: 5,
        evaluationPeriods: 2,
        alarmDescription: 'API 5XX error rate is too high',
        treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      });
      api5xxAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(this.alarmTopic));
    }
  }
}
