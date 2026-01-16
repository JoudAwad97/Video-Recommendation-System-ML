/**
 * Environment configuration for the Video Recommendation System.
 *
 * Defines settings for different deployment environments (dev, staging, prod).
 */

export type Environment = 'dev' | 'staging' | 'prod';

export interface EnvironmentConfig {
  /** Environment name */
  readonly environment: Environment;

  /** Whether this is a production environment */
  readonly isProduction: boolean;

  /** Lambda configuration */
  readonly lambda: {
    readonly memorySize: number;
    readonly timeout: number;
    readonly reservedConcurrency?: number;
  };

  /** API Gateway configuration */
  readonly apiGateway: {
    readonly throttlingBurstLimit: number;
    readonly throttlingRateLimit: number;
  };

  /** DynamoDB configuration */
  readonly dynamodb: {
    readonly billingMode: 'PAY_PER_REQUEST' | 'PROVISIONED';
    readonly readCapacity?: number;
    readonly writeCapacity?: number;
    readonly pointInTimeRecovery: boolean;
  };

  /** Redis/ElastiCache configuration */
  readonly redis: {
    readonly enabled: boolean;
    readonly nodeType: string;
    readonly numNodes: number;
  };

  /** VPC configuration */
  readonly vpc: {
    readonly enabled: boolean;
    readonly maxAzs: number;
    readonly natGateways: number;
  };

  /** Logging configuration */
  readonly logging: {
    readonly level: string;
    readonly retentionDays: number;
  };

  /** Removal policy for resources */
  readonly removalPolicy: 'DESTROY' | 'RETAIN' | 'SNAPSHOT';
}

/**
 * Configuration for development environment.
 */
const devConfig: EnvironmentConfig = {
  environment: 'dev',
  isProduction: false,
  lambda: {
    memorySize: 1024,
    timeout: 30,
  },
  apiGateway: {
    throttlingBurstLimit: 100,
    throttlingRateLimit: 50,
  },
  dynamodb: {
    billingMode: 'PAY_PER_REQUEST',
    pointInTimeRecovery: false,
  },
  redis: {
    enabled: false,
    nodeType: 'cache.t3.micro',
    numNodes: 1,
  },
  vpc: {
    enabled: false,
    maxAzs: 2,
    natGateways: 0,
  },
  logging: {
    level: 'DEBUG',
    retentionDays: 7,
  },
  removalPolicy: 'DESTROY',
};

/**
 * Configuration for staging environment.
 */
const stagingConfig: EnvironmentConfig = {
  environment: 'staging',
  isProduction: false,
  lambda: {
    memorySize: 1024,
    timeout: 30,
  },
  apiGateway: {
    throttlingBurstLimit: 500,
    throttlingRateLimit: 200,
  },
  dynamodb: {
    billingMode: 'PAY_PER_REQUEST',
    pointInTimeRecovery: true,
  },
  redis: {
    enabled: true,
    nodeType: 'cache.t3.small',
    numNodes: 1,
  },
  vpc: {
    enabled: true,
    maxAzs: 2,
    natGateways: 1,
  },
  logging: {
    level: 'INFO',
    retentionDays: 14,
  },
  removalPolicy: 'DESTROY',
};

/**
 * Configuration for production environment.
 */
const prodConfig: EnvironmentConfig = {
  environment: 'prod',
  isProduction: true,
  lambda: {
    memorySize: 2048,
    timeout: 30,
    reservedConcurrency: 100,
  },
  apiGateway: {
    throttlingBurstLimit: 2000,
    throttlingRateLimit: 1000,
  },
  dynamodb: {
    billingMode: 'PROVISIONED',
    readCapacity: 100,
    writeCapacity: 50,
    pointInTimeRecovery: true,
  },
  redis: {
    enabled: true,
    nodeType: 'cache.r6g.large',
    numNodes: 2,
  },
  vpc: {
    enabled: true,
    maxAzs: 3,
    natGateways: 2,
  },
  logging: {
    level: 'INFO',
    retentionDays: 30,
  },
  removalPolicy: 'RETAIN',
};

/**
 * Get configuration for the specified environment.
 */
export function getEnvironmentConfig(environment: Environment): EnvironmentConfig {
  switch (environment) {
    case 'dev':
      return devConfig;
    case 'staging':
      return stagingConfig;
    case 'prod':
      return prodConfig;
    default:
      throw new Error(`Unknown environment: ${environment}`);
  }
}
