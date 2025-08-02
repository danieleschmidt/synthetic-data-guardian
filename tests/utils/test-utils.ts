/**
 * Comprehensive Test Utilities
 * Provides common testing helpers, mocks, and fixtures
 */

import { faker } from '@faker-js/faker';
import { testConfig } from '../config/test-config';

// Test Data Generators
export class TestDataGenerator {
  static generateUserProfile(overrides: Partial<any> = {}) {
    return {
      id: faker.string.uuid(),
      name: faker.person.fullName(),
      email: faker.internet.email(),
      age: faker.number.int({ min: 18, max: 80 }),
      income: faker.number.int({ min: 20000, max: 200000 }),
      location: {
        city: faker.location.city(),
        country: faker.location.country(),
        coordinates: {
          lat: faker.location.latitude(),
          lng: faker.location.longitude()
        }
      },
      preferences: {
        newsletter: faker.datatype.boolean(),
        marketing: faker.datatype.boolean(),
        analytics: faker.datatype.boolean()
      },
      created_at: faker.date.past(),
      updated_at: faker.date.recent(),
      ...overrides
    };
  }

  static generateTransactionData(count: number = 100) {
    return Array.from({ length: count }, () => ({
      id: faker.string.uuid(),
      user_id: faker.string.uuid(),
      amount: faker.number.float({ min: 0.01, max: 10000, precision: 0.01 }),
      currency: faker.finance.currencyCode(),
      merchant: faker.company.name(),
      category: faker.helpers.arrayElement([
        'retail', 'food', 'transport', 'utilities', 'entertainment', 'healthcare'
      ]),
      timestamp: faker.date.past(),
      status: faker.helpers.arrayElement(['pending', 'completed', 'failed']),
      metadata: {
        channel: faker.helpers.arrayElement(['online', 'mobile', 'in-store']),
        device: faker.helpers.arrayElement(['desktop', 'mobile', 'tablet']),
        ip_address: faker.internet.ip()
      }
    }));
  }

  static generateMedicalData(count: number = 50) {
    return Array.from({ length: count }, () => ({
      patient_id: faker.string.uuid(),
      age: faker.number.int({ min: 0, max: 100 }),
      gender: faker.helpers.arrayElement(['M', 'F', 'O']),
      diagnosis: faker.helpers.arrayElement([
        'Hypertension', 'Diabetes', 'Asthma', 'Depression', 'Arthritis'
      ]),
      treatment: faker.lorem.words(3),
      visit_date: faker.date.past(),
      duration_minutes: faker.number.int({ min: 15, max: 120 }),
      cost: faker.number.float({ min: 50, max: 5000, precision: 0.01 }),
      insurance_claim: faker.datatype.boolean(),
      symptoms: faker.lorem.words(5).split(' '),
      medications: Array.from(
        { length: faker.number.int({ min: 0, max: 5 }) },
        () => faker.lorem.word()
      )
    }));
  }

  static generateTimeSeriesData(points: number = 100, startDate?: Date) {
    const start = startDate || faker.date.past();
    const interval = 24 * 60 * 60 * 1000; // 1 day in milliseconds
    
    return Array.from({ length: points }, (_, index) => ({
      timestamp: new Date(start.getTime() + (index * interval)),
      value: faker.number.float({ min: 0, max: 100, precision: 0.01 }),
      metric: faker.helpers.arrayElement(['cpu_usage', 'memory_usage', 'disk_io', 'network_traffic']),
      tags: {
        host: faker.internet.domainName(),
        region: faker.location.countryCode(),
        environment: faker.helpers.arrayElement(['dev', 'staging', 'prod'])
      }
    }));
  }
}

// Test Environment Setup
export class TestEnvironment {
  static async setupDatabase() {
    // Database setup logic would go here
    console.log('Setting up test database...');
  }

  static async teardownDatabase() {
    // Database cleanup logic would go here
    console.log('Tearing down test database...');
  }

  static async setupRedis() {
    console.log('Setting up test Redis...');
  }

  static async teardownRedis() {
    console.log('Tearing down test Redis...');
  }

  static async setupAll() {
    await this.setupDatabase();
    await this.setupRedis();
  }

  static async teardownAll() {
    await this.teardownDatabase();
    await this.teardownRedis();
  }
}

// API Test Helpers
export class APITestHelper {
  private baseURL: string;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || testConfig.api.baseUrl;
  }

  async makeRequest(method: string, endpoint: string, data?: any, headers?: any) {
    const fetch = (await import('node-fetch')).default;
    
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers
      },
      body: data ? JSON.stringify(data) : undefined
    });

    return {
      status: response.status,
      data: await response.json(),
      headers: response.headers
    };
  }

  async post(endpoint: string, data: any, headers?: any) {
    return this.makeRequest('POST', endpoint, data, headers);
  }

  async get(endpoint: string, headers?: any) {
    return this.makeRequest('GET', endpoint, undefined, headers);
  }

  async put(endpoint: string, data: any, headers?: any) {
    return this.makeRequest('PUT', endpoint, data, headers);
  }

  async delete(endpoint: string, headers?: any) {
    return this.makeRequest('DELETE', endpoint, undefined, headers);
  }
}

// Performance Test Helpers
export class PerformanceTestHelper {
  static measureExecutionTime<T>(fn: () => Promise<T>): Promise<{ result: T; duration: number }> {
    return new Promise(async (resolve) => {
      const start = performance.now();
      const result = await fn();
      const end = performance.now();
      resolve({ result, duration: end - start });
    });
  }

  static async loadTest(
    testFunction: () => Promise<any>,
    options: {
      concurrency: number;
      duration: number; // in milliseconds
      rampUp?: number;  // in milliseconds
    }
  ) {
    const { concurrency, duration, rampUp = 0 } = options;
    const results: Array<{ success: boolean; duration: number; error?: any }> = [];
    const startTime = Date.now();
    const endTime = startTime + duration;
    
    const workers = Array.from({ length: concurrency }, async (_, index) => {
      // Stagger start times for ramp-up
      if (rampUp > 0) {
        const delay = (rampUp / concurrency) * index;
        await new Promise(resolve => setTimeout(resolve, delay));
      }

      while (Date.now() < endTime) {
        try {
          const { duration } = await this.measureExecutionTime(testFunction);
          results.push({ success: true, duration });
        } catch (error) {
          results.push({ success: false, duration: 0, error });
        }
      }
    });

    await Promise.all(workers);

    // Calculate statistics
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    const durations = successful.map(r => r.duration);
    
    return {
      totalRequests: results.length,
      successfulRequests: successful.length,
      failedRequests: failed.length,
      successRate: (successful.length / results.length) * 100,
      averageResponseTime: durations.reduce((a, b) => a + b, 0) / durations.length,
      minResponseTime: Math.min(...durations),
      maxResponseTime: Math.max(...durations),
      p95ResponseTime: this.percentile(durations, 0.95),
      p99ResponseTime: this.percentile(durations, 0.99),
      throughput: (successful.length / (duration / 1000)) // requests per second
    };
  }

  private static percentile(arr: number[], percentile: number): number {
    const sorted = arr.sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[index];
  }
}

// Security Test Helpers
export class SecurityTestHelper {
  static generateSQLInjectionPayloads(): string[] {
    return [
      "'; DROP TABLE users; --",
      "' OR '1'='1",
      "' UNION SELECT * FROM users --",
      "'; EXEC xp_cmdshell('dir'); --",
      "' OR 1=1#"
    ];
  }

  static generateXSSPayloads(): string[] {
    return [
      "<script>alert('XSS')</script>",
      "javascript:alert('XSS')",
      "<img src=x onerror=alert('XSS')>",
      "';alert('XSS');//",
      "<svg onload=alert('XSS')>"
    ];
  }

  static generateCSRFToken(): string {
    return faker.string.alphanumeric(32);
  }

  static async testEndpointSecurity(
    apiHelper: APITestHelper,
    endpoint: string,
    payload: any
  ) {
    const results = [];

    // Test SQL injection
    for (const injection of this.generateSQLInjectionPayloads()) {
      const testPayload = { ...payload };
      // Inject into string fields
      Object.keys(testPayload).forEach(key => {
        if (typeof testPayload[key] === 'string') {
          testPayload[key] = injection;
        }
      });

      try {
        const response = await apiHelper.post(endpoint, testPayload);
        results.push({
          type: 'sql_injection',
          payload: injection,
          status: response.status,
          vulnerable: response.status === 200
        });
      } catch (error) {
        results.push({
          type: 'sql_injection',
          payload: injection,
          error: error.message,
          vulnerable: false
        });
      }
    }

    // Test XSS
    for (const xss of this.generateXSSPayloads()) {
      const testPayload = { ...payload };
      Object.keys(testPayload).forEach(key => {
        if (typeof testPayload[key] === 'string') {
          testPayload[key] = xss;
        }
      });

      try {
        const response = await apiHelper.post(endpoint, testPayload);
        results.push({
          type: 'xss',
          payload: xss,
          status: response.status,
          vulnerable: response.status === 200
        });
      } catch (error) {
        results.push({
          type: 'xss',
          payload: xss,
          error: error.message,
          vulnerable: false
        });
      }
    }

    return results;
  }
}

// Mock Factories
export class MockFactory {
  static createMockDatabase() {
    return {
      query: jest.fn(),
      transaction: jest.fn(),
      close: jest.fn()
    };
  }

  static createMockRedisClient() {
    return {
      get: jest.fn(),
      set: jest.fn(),
      del: jest.fn(),
      expire: jest.fn(),
      quit: jest.fn()
    };
  }

  static createMockLogger() {
    return {
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
      debug: jest.fn()
    };
  }

  static createMockGenerationRequest() {
    return {
      pipeline: 'test_pipeline',
      num_records: 1000,
      format: 'json',
      parameters: {
        model: 'gaussian_copula',
        epochs: 100
      },
      validation: {
        statistical: true,
        privacy: true
      }
    };
  }
}

// Test Assertions
export class TestAssertions {
  static assertValidUUID(value: string) {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    expect(value).toMatch(uuidRegex);
  }

  static assertValidEmail(value: string) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    expect(value).toMatch(emailRegex);
  }

  static assertValidTimestamp(value: string) {
    const timestamp = new Date(value);
    expect(timestamp).toBeInstanceOf(Date);
    expect(timestamp.getTime()).not.toBeNaN();
  }

  static assertResponseTime(duration: number, threshold: number = testConfig.performance.thresholds.responseTime) {
    expect(duration).toBeLessThan(threshold);
  }

  static assertSuccessRate(rate: number, threshold: number = 95) {
    expect(rate).toBeGreaterThanOrEqual(threshold);
  }
}

export {
  TestDataGenerator,
  TestEnvironment,
  APITestHelper,
  PerformanceTestHelper,
  SecurityTestHelper,
  MockFactory,
  TestAssertions
};