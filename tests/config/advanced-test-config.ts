/**
 * Advanced Testing Configuration for Synthetic Data Guardian
 * Provides enhanced testing utilities and configurations for comprehensive test coverage
 */

import { config } from 'dotenv';
import { Config } from 'jest';

// Load test environment variables
config({ path: '.env.test' });

export interface TestEnvironmentConfig {
  database: {
    url: string;
    testDb: string;
    migrations: boolean;
    seedData: boolean;
  };
  redis: {
    url: string;
    testDb: number;
    flushOnStart: boolean;
  };
  neo4j: {
    uri: string;
    user: string;
    password: string;
    testDb: string;
  };
  api: {
    baseUrl: string;
    timeout: number;
    retries: number;
  };
  performance: {
    cpuThreshold: number;
    memoryThreshold: number;
    responseTimeThreshold: number;
  };
  security: {
    enableScanningTests: boolean;
    testApiKeys: boolean;
    validateInputs: boolean;
  };
}

export const testConfig: TestEnvironmentConfig = {
  database: {
    url: process.env.TEST_DATABASE_URL || 'postgresql://test:test@localhost:5432/synthetic_guardian_test',
    testDb: 'synthetic_guardian_test',
    migrations: true,
    seedData: false
  },
  redis: {
    url: process.env.TEST_REDIS_URL || 'redis://localhost:6379',
    testDb: 15, // Use database 15 for tests
    flushOnStart: true
  },
  neo4j: {
    uri: process.env.TEST_NEO4J_URI || 'bolt://localhost:7687',
    user: process.env.TEST_NEO4J_USER || 'neo4j',
    password: process.env.TEST_NEO4J_PASSWORD || 'test',
    testDb: 'test'
  },
  api: {
    baseUrl: process.env.TEST_API_BASE_URL || 'http://localhost:8080',
    timeout: 30000,
    retries: 3
  },
  performance: {
    cpuThreshold: 80, // CPU usage percentage
    memoryThreshold: 512, // Memory usage in MB
    responseTimeThreshold: 500 // Response time in ms
  },
  security: {
    enableScanningTests: true,
    testApiKeys: false, // Don't test real API keys in CI
    validateInputs: true
  }
};

/**
 * Jest configuration for comprehensive testing
 */
export const advancedJestConfig: Config = {
  // Core configuration
  preset: 'ts-jest',
  testEnvironment: 'node',
  
  // Test discovery
  testMatch: [
    '**/__tests__/**/*.+(ts|tsx|js)',
    '**/*.(test|spec).+(ts|tsx|js)',
  ],
  
  // Coverage configuration
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
    '!src/**/*.interface.ts',
    '!src/**/*.type.ts',
    '!src/**/*.config.ts',
    '!src/**/migrations/**',
    '!src/**/seeds/**'
  ],
  
  coverageThreshold: {
    global: {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85,
    },
    // Specific thresholds for critical modules
    'src/generators/': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
    'src/validators/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95,
    },
    'src/security/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95,
    }
  },
  
  // Performance and reliability
  testTimeout: 30000,
  maxWorkers: '50%',
  maxConcurrency: 5,
  
  // Enhanced reporting
  reporters: [
    'default',
    ['jest-html-reporters', {
      publicPath: './coverage/html-report',
      filename: 'test-report.html',
      expand: true,
      hideIcon: false,
      pageTitle: 'Synthetic Data Guardian Test Report'
    }],
    ['jest-junit', {
      outputDirectory: './test-results',
      outputName: 'junit.xml',
      ancestorSeparator: ' › ',
      uniqueOutputName: 'false',
      suiteNameTemplate: '{filepath}',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}'
    }],
    ['jest-sonar', {
      outputDirectory: './coverage',
      outputName: 'sonar-report.xml',
      reportedFilePath: 'relative'
    }]
  ],
  
  // Test environment setup
  setupFilesAfterEnv: [
    '<rootDir>/tests/config/test-setup.ts',
    '<rootDir>/tests/config/performance-setup.ts',
    '<rootDir>/tests/config/security-setup.ts'
  ],
  
  // Module resolution
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '^@fixtures/(.*)$': '<rootDir>/tests/fixtures/$1',
    '^@helpers/(.*)$': '<rootDir>/tests/helpers/$1'
  },
  
  // Global variables for tests
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json',
    },
    TEST_CONFIG: testConfig
  },
  
  // Test isolation
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,
  
  // File extensions
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  
  // Transform configuration
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
    '^.+\\.(js|jsx)$': 'babel-jest'
  },
  
  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/coverage/',
    '/test-results/'
  ],
  
  // Watch mode configuration
  watchPathIgnorePatterns: [
    '/node_modules/',
    '/coverage/',
    '/dist/',
    '/test-results/'
  ],
  
  // Error handling
  errorOnDeprecated: true,
  bail: 0, // Continue running tests after failures
  
  // Snapshot configuration
  updateSnapshot: process.env.CI ? false : true,
  
  // Test result caching
  cache: true,
  cacheDirectory: '<rootDir>/.jest-cache'
};

/**
 * Test utilities and helpers
 */
export class TestUtils {
  /**
   * Create a test database transaction that automatically rolls back
   */
  static async withTestTransaction<T>(callback: () => Promise<T>): Promise<T> {
    // Implementation would depend on your database setup
    // This is a placeholder for the actual implementation
    return callback();
  }
  
  /**
   * Clear all test data from Redis
   */
  static async clearRedisTestData(): Promise<void> {
    // Implementation for Redis cleanup
  }
  
  /**
   * Clear all test data from Neo4j
   */
  static async clearNeo4jTestData(): Promise<void> {
    // Implementation for Neo4j cleanup
  }
  
  /**
   * Generate test data fixtures
   */
  static generateTestData(type: string, count: number = 10): any[] {
    // Implementation for test data generation
    return [];
  }
  
  /**
   * Mock external API responses
   */
  static mockExternalAPIs(): void {
    // Implementation for API mocking
  }
  
  /**
   * Performance testing utilities
   */
  static async measurePerformance<T>(
    fn: () => Promise<T>,
    expectedMaxTime: number
  ): Promise<{ result: T; duration: number }> {
    const start = Date.now();
    const result = await fn();
    const duration = Date.now() - start;
    
    if (duration > expectedMaxTime) {
      throw new Error(`Performance test failed: ${duration}ms > ${expectedMaxTime}ms`);
    }
    
    return { result, duration };
  }
  
  /**
   * Security testing utilities
   */
  static validateSecurityHeaders(headers: Record<string, string>): boolean {
    const requiredHeaders = [
      'x-content-type-options',
      'x-frame-options',
      'x-xss-protection',
      'strict-transport-security'
    ];
    
    return requiredHeaders.every(header => 
      headers[header] !== undefined
    );
  }
  
  /**
   * Load testing data generator
   */
  static async generateLoadTestData(
    endpoint: string,
    concurrentUsers: number,
    duration: number
  ): Promise<any> {
    // Implementation for load test data generation
    return {};
  }
}

/**
 * Custom Jest matchers for Synthetic Data Guardian
 */
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeValidSyntheticData(): R;
      toHavePrivacyScore(score: number): R;
      toHaveQualityScore(score: number): R;
      toBeWatermarked(): R;
      toHaveValidLineage(): R;
      toMeetPerformanceThreshold(threshold: number): R;
      toPassSecurityValidation(): R;
    }
  }
}

export default advancedJestConfig;