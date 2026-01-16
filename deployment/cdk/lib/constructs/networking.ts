import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import { Construct } from 'constructs';
import { EnvironmentConfig } from '../config/environment';

/**
 * Properties for the Networking construct.
 */
export interface NetworkingProps {
  readonly config: EnvironmentConfig;
}

/**
 * Networking construct for VPC and ElastiCache.
 *
 * Creates:
 * - VPC with public, private, and isolated subnets
 * - Security groups for Lambda and Redis
 * - ElastiCache Redis cluster (when enabled)
 */
export class Networking extends Construct {
  /** VPC for the application (undefined if VPC is disabled) */
  public readonly vpc?: ec2.Vpc;

  /** Security group for Lambda functions */
  public readonly lambdaSecurityGroup?: ec2.SecurityGroup;

  /** Security group for Redis */
  public readonly redisSecurityGroup?: ec2.SecurityGroup;

  /** Redis endpoint (undefined if Redis is disabled) */
  public readonly redisEndpoint?: string;

  /** Redis port */
  public readonly redisPort: number = 6379;

  constructor(scope: Construct, id: string, props: NetworkingProps) {
    super(scope, id);

    const { config } = props;

    // Skip networking if VPC is disabled
    if (!config.vpc.enabled) {
      return;
    }

    // =========================================================================
    // VPC
    // =========================================================================

    this.vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: config.vpc.maxAzs,
      natGateways: config.vpc.natGateways,
      subnetConfiguration: [
        {
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: 'Isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
    });

    // =========================================================================
    // Security Groups
    // =========================================================================

    this.lambdaSecurityGroup = new ec2.SecurityGroup(this, 'LambdaSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for Lambda functions',
      allowAllOutbound: true,
    });

    this.redisSecurityGroup = new ec2.SecurityGroup(this, 'RedisSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for ElastiCache Redis',
      allowAllOutbound: false,
    });

    // Allow Lambda to access Redis
    this.redisSecurityGroup.addIngressRule(
      this.lambdaSecurityGroup,
      ec2.Port.tcp(this.redisPort),
      'Allow Lambda to access Redis'
    );

    // =========================================================================
    // ElastiCache Redis
    // =========================================================================

    if (config.redis.enabled) {
      const subnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
        description: 'Subnet group for Redis cluster',
        subnetIds: this.vpc.selectSubnets({
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
        }).subnetIds,
      });

      const redisCluster = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
        engine: 'redis',
        cacheNodeType: config.redis.nodeType,
        numCacheNodes: config.redis.numNodes,
        vpcSecurityGroupIds: [this.redisSecurityGroup.securityGroupId],
        cacheSubnetGroupName: subnetGroup.ref,
        engineVersion: '7.0',
        autoMinorVersionUpgrade: true,
      });

      redisCluster.addDependency(subnetGroup);

      this.redisEndpoint = redisCluster.attrRedisEndpointAddress;
    }
  }
}
