#!/usr/bin/env node
/**
 * CDK Application Entry Point for Video Recommendation System.
 *
 * Usage:
 *   npm run deploy              # Deploy to default environment (dev)
 *   npm run deploy:dev          # Deploy to dev
 *   npm run deploy:staging      # Deploy to staging
 *   npm run deploy:prod         # Deploy to prod
 *   npm run destroy             # Destroy all stacks
 *
 * Custom deployment:
 *   npx cdk deploy -c environment=staging -c region=eu-west-1
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { VideoRecommendationStack } from '../lib/stacks';
import { getEnvironmentConfig, Environment } from '../lib/config/environment';

// Create CDK app
const app = new cdk.App();

// Get environment from context or default to 'dev'
const environmentName = (app.node.tryGetContext('environment') || 'dev') as Environment;

// Validate environment
if (!['dev', 'staging', 'prod'].includes(environmentName)) {
  throw new Error(
    `Invalid environment: ${environmentName}. Must be one of: dev, staging, prod`
  );
}

// Get environment-specific configuration
const config = getEnvironmentConfig(environmentName);

// Get AWS account and region
const account = process.env.CDK_DEFAULT_ACCOUNT || process.env.AWS_ACCOUNT_ID;
const region = app.node.tryGetContext('region') ||
  process.env.CDK_DEFAULT_REGION ||
  process.env.AWS_REGION ||
  'us-east-1';

// Validate AWS account
if (!account) {
  console.warn(
    'Warning: AWS account not detected. Make sure you have configured AWS credentials.'
  );
}

// Stack naming
const stackName = `VideoRecSystem-${environmentName}`;

// Create the main stack
new VideoRecommendationStack(app, stackName, {
  config,
  env: {
    account,
    region,
  },
  description: `Video Recommendation System - ${environmentName} environment`,
  tags: {
    Environment: environmentName,
    Project: 'video-recommendation-system',
    ManagedBy: 'CDK',
  },
});

// Synthesize the app
app.synth();

// Print deployment information
console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Video Recommendation System - CDK                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Environment:  ${environmentName.padEnd(60)}║
║  Region:       ${region.padEnd(60)}║
║  Stack:        ${stackName.padEnd(60)}║
╚══════════════════════════════════════════════════════════════════════════════╝
`);
