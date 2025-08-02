/**
 * Comprehensive Test Configuration
 * Centralizes test settings, fixtures, and utilities
 */

export interface TestConfig {
  timeouts: {
    unit: number;
    integration: number;
    e2e: number;
    performance: number;
  };
  database: {
    url: string;
    testDatabase: string;
    maxConnections: number;
  };
  api: {
    baseUrl: string;
    timeout: number;
    retries: number;
  };
  fixtures: {
    dataPath: string;
    sampleSize: number;
  };
  performance: {
    thresholds: {
      responseTime: number;
      throughput: number;
      errorRate: number;
    };
  };
  security: {
    scanDepth: 'basic' | 'thorough';
    vulnerabilityThreshold: 'low' | 'medium' | 'high';
  };
}

export const testConfig: TestConfig = {
  timeouts: {
    unit: 5000,        // 5 seconds
    integration: 30000, // 30 seconds
    e2e: 60000,        // 1 minute
    performance: 300000 // 5 minutes
  },
  database: {
    url: process.env.TEST_DATABASE_URL || 'postgresql://test:test@localhost:5433/synthetic_guardian_test',
    testDatabase: 'synthetic_guardian_test',
    maxConnections: 10
  },
  api: {
    baseUrl: process.env.TEST_API_URL || 'http://localhost:8080',
    timeout: 10000,
    retries: 3
  },
  fixtures: {
    dataPath: './tests/fixtures/data',
    sampleSize: 1000
  },
  performance: {
    thresholds: {
      responseTime: 500,   // 500ms
      throughput: 100,     // 100 req/s
      errorRate: 0.01      // 1%
    }
  },
  security: {
    scanDepth: 'thorough',
    vulnerabilityThreshold: 'medium'
  }
};

export const getTestConfig = (): TestConfig => testConfig;

export const isCI = (): boolean => process.env.CI === 'true';

export const isDebug = (): boolean => process.env.DEBUG === 'true';

export const getTestEnvironment = (): 'local' | 'ci' | 'staging' => {
  if (isCI()) return 'ci';
  if (process.env.TEST_ENV === 'staging') return 'staging';
  return 'local';
};

export const shouldRunPerformanceTests = (): boolean => {
  return process.env.RUN_PERFORMANCE_TESTS === 'true' || getTestEnvironment() === 'ci';
};

export const shouldRunSecurityTests = (): boolean => {
  return process.env.RUN_SECURITY_TESTS === 'true' || getTestEnvironment() === 'ci';
};