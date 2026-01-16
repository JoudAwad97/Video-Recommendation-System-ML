import * as cdk from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';

/**
 * Properties for the Secrets construct.
 */
export interface SecretsProps {
  readonly config: EnvironmentConfig;
}

/**
 * Secrets Management construct.
 *
 * Creates and manages secrets for:
 * - Redis/ElastiCache credentials
 * - API keys for external services (Pinecone, etc.)
 * - Database credentials
 */
export class Secrets extends Construct {
  /** Secret for Redis credentials */
  public readonly redisSecret?: secretsmanager.Secret;

  /** Secret for external API keys */
  public readonly apiKeysSecret: secretsmanager.Secret;

  /** Secret for service configuration */
  public readonly configSecret: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props: SecretsProps) {
    super(scope, id);

    const { config } = props;

    // =========================================================================
    // Redis Credentials Secret (Staging/Production only)
    // =========================================================================

    if (config.redis.enabled) {
      this.redisSecret = new secretsmanager.Secret(this, 'RedisSecret', {
        secretName: `${cdk.Stack.of(this).stackName}/redis`,
        description: 'Redis/ElastiCache credentials',
        generateSecretString: {
          secretStringTemplate: JSON.stringify({
            host: '',  // Will be populated after ElastiCache creation
            port: 6379,
          }),
          generateStringKey: 'password',
          excludePunctuation: true,
          passwordLength: 32,
        },
      });

      // Add rotation schedule for production
      if (config.isProduction) {
        this.redisSecret.addRotationSchedule('RedisSecretRotation', {
          automaticallyAfter: cdk.Duration.days(30),
          rotateImmediatelyOnUpdate: false,
        });
      }
    }

    // =========================================================================
    // API Keys Secret
    // =========================================================================

    this.apiKeysSecret = new secretsmanager.Secret(this, 'ApiKeysSecret', {
      secretName: `${cdk.Stack.of(this).stackName}/api-keys`,
      description: 'API keys for external services',
      secretObjectValue: {
        // Pinecone API key (for vector store)
        pinecone_api_key: cdk.SecretValue.unsafePlainText(''),
        pinecone_environment: cdk.SecretValue.unsafePlainText(''),
        pinecone_index_name: cdk.SecretValue.unsafePlainText(''),

        // OpenSearch credentials (if using managed OpenSearch)
        opensearch_endpoint: cdk.SecretValue.unsafePlainText(''),
        opensearch_username: cdk.SecretValue.unsafePlainText(''),
        opensearch_password: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // =========================================================================
    // Service Configuration Secret
    // =========================================================================

    this.configSecret = new secretsmanager.Secret(this, 'ConfigSecret', {
      secretName: `${cdk.Stack.of(this).stackName}/config`,
      description: 'Service configuration values',
      secretObjectValue: {
        // Feature flags
        enable_ab_testing: cdk.SecretValue.unsafePlainText('false'),
        ab_test_traffic_split: cdk.SecretValue.unsafePlainText('0.5'),

        // Rate limiting
        rate_limit_requests_per_second: cdk.SecretValue.unsafePlainText(
          config.apiGateway.throttlingRateLimit.toString()
        ),

        // Model configuration
        default_model_version: cdk.SecretValue.unsafePlainText('v1'),
        enable_model_fallback: cdk.SecretValue.unsafePlainText('true'),
      },
    });

    // =========================================================================
    // Tags
    // =========================================================================

    cdk.Tags.of(this).add('Component', 'Secrets');
    cdk.Tags.of(this).add('Environment', config.environment);
  }

  /**
   * Grant read access to secrets for a given principal.
   */
  public grantRead(grantee: iam.IGrantable): void {
    if (this.redisSecret) {
      this.redisSecret.grantRead(grantee);
    }
    this.apiKeysSecret.grantRead(grantee);
    this.configSecret.grantRead(grantee);
  }

  /**
   * Get environment variables for Lambda function.
   */
  public getEnvironmentVariables(): Record<string, string> {
    const env: Record<string, string> = {
      API_KEYS_SECRET_ARN: this.apiKeysSecret.secretArn,
      CONFIG_SECRET_ARN: this.configSecret.secretArn,
    };

    if (this.redisSecret) {
      env.REDIS_SECRET_ARN = this.redisSecret.secretArn;
    }

    return env;
  }
}
