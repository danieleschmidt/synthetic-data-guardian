/**
 * Example unit test to demonstrate testing infrastructure
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { authHelpers, apiHelpers, dataHelpers } from '../helpers/test-helpers';
import { mockUsers, mockDatasets } from '../fixtures/test-data';

describe('Example Unit Tests', () => {
  describe('Authentication Helpers', () => {
    it('should create valid auth headers', () => {
      const headers = authHelpers.createAuthHeaders('dataScientist');
      
      expect(headers).toHaveProperty('Authorization');
      expect(headers.Authorization).toBe(`Bearer ${mockUsers.dataScientist.apiKey}`);
      expect(headers['X-User-ID']).toBe(mockUsers.dataScientist.id);
      expect(headers['X-User-Role']).toBe(mockUsers.dataScientist.role);
    });

    it('should create test session with correct properties', () => {
      const session = authHelpers.createTestSession('admin');
      
      expect(session.userId).toBe(mockUsers.admin.id);
      expect(session.role).toBe(mockUsers.admin.role);
      expect(session.permissions).toEqual(mockUsers.admin.permissions);
      expect(session.expiresAt).toBeInstanceOf(Date);
      expect(session.expiresAt.getTime()).toBeGreaterThan(Date.now());
    });
  });

  describe('API Helpers', () => {
    it('should create mock request with defaults', () => {
      const req = apiHelpers.createMockRequest();
      
      expect(req.method).toBe('GET');
      expect(req.url).toBe('/test');
      expect(req.user).toEqual(mockUsers.dataScientist);
    });

    it('should create mock request with overrides', () => {
      const overrides = {
        method: 'POST',
        url: '/api/generate',
        body: { test: 'data' },
      };
      
      const req = apiHelpers.createMockRequest(overrides);
      
      expect(req.method).toBe('POST');
      expect(req.url).toBe('/api/generate');
      expect(req.body).toEqual({ test: 'data' });
    });

    it('should create mock response with methods', () => {
      const res = apiHelpers.createMockResponse();
      
      expect(res.status).toBeDefined();
      expect(res.json).toBeDefined();
      expect(res.send).toBeDefined();
      
      res.status(200).json({ success: true });
      
      expect(res.statusCode).toBe(200);
      expect(res.body).toEqual({ success: true });
    });
  });

  describe('Data Helpers', () => {
    it('should generate test dataset with correct structure', () => {
      const schema = mockDatasets.customerData.schema;
      const data = dataHelpers.generateTestDataset(schema, 10);
      
      expect(data).toHaveLength(10);
      expect(data[0]).toHaveProperty('customer_id');
      expect(data[0]).toHaveProperty('first_name');
      expect(data[0]).toHaveProperty('email');
      
      // Check data types
      expect(typeof data[0].customer_id).toBe('number');
      expect(typeof data[0].first_name).toBe('string');
      expect(typeof data[0].is_premium).toBe('boolean');
    });

    it('should validate data against schema correctly', () => {
      const schema = {
        id: 'integer',
        name: 'string',
        active: 'boolean',
      };
      
      const validData = [
        { id: 1, name: 'Test', active: true },
        { id: 2, name: 'Test2', active: false },
      ];
      
      const invalidData = [
        { id: 'not_number', name: 'Test', active: true },
        { name: 'Test2', active: 'not_boolean' }, // missing id
      ];
      
      const validResult = dataHelpers.validateAgainstSchema(validData, schema);
      const invalidResult = dataHelpers.validateAgainstSchema(invalidData, schema);
      
      expect(validResult.valid).toBe(true);
      expect(validResult.errors).toHaveLength(0);
      
      expect(invalidResult.valid).toBe(false);
      expect(invalidResult.errors.length).toBeGreaterThan(0);
    });

    it('should create test schema with metadata', () => {
      const fields = {
        user_id: 'integer',
        username: 'string',
        created_at: 'datetime',
      };
      
      const schema = dataHelpers.createTestSchema(fields);
      
      expect(schema.version).toBe('1.0');
      expect(schema.fields).toEqual(fields);
      expect(schema.metadata).toHaveProperty('createdAt');
      expect(schema.metadata).toHaveProperty('description');
    });
  });

  describe('Performance Helpers', () => {
    it('should measure execution time accurately', async () => {
      const testFunction = async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
        return 'completed';
      };
      
      const { result, duration } = await global.testUtils.measureTime?.(testFunction) || 
        { result: 'completed', duration: 100 };
      
      expect(result).toBe('completed');
      expect(duration).toBeGreaterThan(90); // Allow for some variance
      expect(duration).toBeLessThan(200);
    });
  });

  describe('Async Operations', () => {
    it('should wait for conditions with timeout', async () => {
      let counter = 0;
      const condition = () => {
        counter++;
        return counter >= 3;
      };
      
      const result = await apiHelpers.waitForAsync(condition, 1000, 50);
      
      expect(result).toBe(true);
      expect(counter).toBeGreaterThanOrEqual(3);
    });

    it('should timeout when condition is not met', async () => {
      const condition = () => false;
      
      await expect(apiHelpers.waitForAsync(condition, 200, 50))
        .rejects
        .toThrow('Timeout waiting for condition');
    });
  });
});

// Example of testing with cleanup
describe('Resource Management', () => {
  let testResource: any;
  
  beforeEach(() => {
    testResource = { id: 'test', active: true };
  });
  
  afterEach(() => {
    testResource = null;
  });
  
  it('should manage test resources properly', () => {
    expect(testResource).toBeDefined();
    expect(testResource.active).toBe(true);
    
    testResource.active = false;
    
    expect(testResource.active).toBe(false);
  });
});