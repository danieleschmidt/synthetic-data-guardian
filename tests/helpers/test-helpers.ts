/**
 * Test helper utilities for Synthetic Data Guardian
 */

import { jest } from '@jest/globals';
import fs from 'fs/promises';
import path from 'path';
import { mockUsers, mockDatasets, generateMockCsvData } from '../fixtures/test-data';

/**
 * Authentication helpers
 */
export const authHelpers = {
  /**
   * Create mock authentication headers
   */
  createAuthHeaders: (userType: keyof typeof mockUsers = 'dataScientist') => {
    const user = mockUsers[userType];
    return {
      'Authorization': `Bearer ${user.apiKey}`,
      'Content-Type': 'application/json',
      'X-User-ID': user.id,
      'X-User-Role': user.role,
    };
  },

  /**
   * Mock JWT token validation
   */
  mockJwtValidation: (userType: keyof typeof mockUsers = 'dataScientist') => {
    const user = mockUsers[userType];
    return jest.fn().mockResolvedValue({
      valid: true,
      user: {
        id: user.id,
        email: user.email,
        role: user.role,
        permissions: user.permissions,
      },
    });
  },

  /**
   * Create test session
   */
  createTestSession: (userType: keyof typeof mockUsers = 'dataScientist') => {
    const user = mockUsers[userType];
    return {
      sessionId: `test-session-${Date.now()}`,
      userId: user.id,
      email: user.email,
      role: user.role,
      permissions: user.permissions,
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
    };
  },
};

/**
 * API testing helpers
 */
export const apiHelpers = {
  /**
   * Create mock request object
   */
  createMockRequest: (overrides: any = {}) => {
    const defaults = {
      method: 'GET',
      url: '/test',
      headers: {},
      body: {},
      query: {},
      params: {},
      user: mockUsers.dataScientist,
    };
    return { ...defaults, ...overrides };
  },

  /**
   * Create mock response object
   */
  createMockResponse: () => {
    const res: any = {
      statusCode: 200,
      headers: {},
      body: null,
    };
    
    res.status = jest.fn().mockImplementation((code: number) => {
      res.statusCode = code;
      return res;
    });
    
    res.json = jest.fn().mockImplementation((data: any) => {
      res.body = data;
      return res;
    });
    
    res.send = jest.fn().mockImplementation((data: any) => {
      res.body = data;
      return res;
    });
    
    res.set = jest.fn().mockImplementation((key: string, value: string) => {
      res.headers[key] = value;
      return res;
    });
    
    res.cookie = jest.fn().mockReturnValue(res);
    res.clearCookie = jest.fn().mockReturnValue(res);
    res.redirect = jest.fn().mockReturnValue(res);
    
    return res;
  },

  /**
   * Create mock next function
   */
  createMockNext: () => jest.fn(),

  /**
   * Wait for async operation with timeout
   */
  waitForAsync: async (fn: () => boolean, timeout: number = 5000, interval: number = 100) => {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      if (fn()) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    throw new Error(`Timeout waiting for condition after ${timeout}ms`);
  },
};

/**
 * File system helpers
 */
export const fileHelpers = {
  /**
   * Create temporary test file
   */
  createTempFile: async (content: string, extension: string = '.txt') => {
    const tempDir = path.join(process.cwd(), 'temp', 'tests');
    await fs.mkdir(tempDir, { recursive: true });
    
    const fileName = `test-${Date.now()}${extension}`;
    const filePath = path.join(tempDir, fileName);
    
    await fs.writeFile(filePath, content, 'utf8');
    return filePath;
  },

  /**
   * Create temporary CSV file with mock data
   */
  createTempCsvFile: async (datasetType: keyof typeof mockDatasets, numRows: number = 100) => {
    const dataset = mockDatasets[datasetType];
    const csvContent = generateMockCsvData(dataset.schema, numRows);
    return await fileHelpers.createTempFile(csvContent, '.csv');
  },

  /**
   * Clean up temporary files
   */
  cleanupTempFiles: async () => {
    const tempDir = path.join(process.cwd(), 'temp', 'tests');
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (error) {
      // Directory might not exist
    }
  },

  /**
   * Read file content
   */
  readFile: async (filePath: string) => {
    return await fs.readFile(filePath, 'utf8');
  },
};

/**
 * Database testing helpers
 */
export const dbHelpers = {
  /**
   * Create mock database client
   */
  createMockDbClient: () => {
    const client = {
      connect: jest.fn().mockResolvedValue(undefined),
      disconnect: jest.fn().mockResolvedValue(undefined),
      query: jest.fn().mockResolvedValue({ rows: [], rowCount: 0 }),
      transaction: jest.fn().mockImplementation(async (callback) => {
        return await callback(client);
      }),
    };
    return client;
  },

  /**
   * Create mock Redis client
   */
  createMockRedisClient: () => {
    const cache = new Map<string, string>();
    return {
      connect: jest.fn().mockResolvedValue(undefined),
      disconnect: jest.fn().mockResolvedValue(undefined),
      get: jest.fn().mockImplementation((key: string) => {
        return Promise.resolve(cache.get(key) || null);
      }),
      set: jest.fn().mockImplementation((key: string, value: string, ttl?: number) => {
        cache.set(key, value);
        if (ttl) {
          setTimeout(() => cache.delete(key), ttl * 1000);
        }
        return Promise.resolve('OK');
      }),
      del: jest.fn().mockImplementation((key: string) => {
        const existed = cache.has(key);
        cache.delete(key);
        return Promise.resolve(existed ? 1 : 0);
      }),
      flushall: jest.fn().mockImplementation(() => {
        cache.clear();
        return Promise.resolve('OK');
      }),
    };
  },

  /**
   * Create mock Neo4j client
   */
  createMockNeo4jClient: () => {
    const records: any[] = [];
    return {
      connect: jest.fn().mockResolvedValue(undefined),
      disconnect: jest.fn().mockResolvedValue(undefined),
      run: jest.fn().mockImplementation((query: string, params: any) => {
        return Promise.resolve({
          records: records.filter(r => r.query === query),
          summary: { counters: { nodesCreated: 0, relationshipsCreated: 0 } },
        });
      }),
      addRecord: (record: any) => {
        records.push(record);
      },
      clearRecords: () => {
        records.length = 0;
      },
    };
  },
};

/**
 * Data generation helpers
 */
export const dataHelpers = {
  /**
   * Generate test dataset
   */
  generateTestDataset: (schema: Record<string, string>, size: number = 100) => {
    const data = [];
    for (let i = 0; i < size; i++) {
      const row: Record<string, any> = {};
      for (const [field, type] of Object.entries(schema)) {
        row[field] = generateValueByType(type, i);
      }
      data.push(row);
    }
    return data;
  },

  /**
   * Create test schema
   */
  createTestSchema: (fields: Record<string, string>) => {
    return {
      version: '1.0',
      fields,
      constraints: [],
      metadata: {
        createdAt: new Date().toISOString(),
        description: 'Test schema',
      },
    };
  },

  /**
   * Validate data against schema
   */
  validateAgainstSchema: (data: any[], schema: Record<string, string>) => {
    const errors: string[] = [];
    
    for (let i = 0; i < data.length; i++) {
      const row = data[i];
      for (const [field, type] of Object.entries(schema)) {
        if (!(field in row)) {
          errors.push(`Row ${i}: Missing field '${field}'`);
          continue;
        }
        
        const value = row[field];
        if (!isValidType(value, type)) {
          errors.push(`Row ${i}: Field '${field}' has invalid type. Expected ${type}, got ${typeof value}`);
        }
      }
    }
    
    return {
      valid: errors.length === 0,
      errors,
    };
  },
};

/**
 * Performance testing helpers
 */
export const performanceHelpers = {
  /**
   * Measure execution time
   */
  measureTime: async <T>(fn: () => Promise<T>): Promise<{ result: T; duration: number }> => {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;
    return { result, duration };
  },

  /**
   * Create load test scenario
   */
  createLoadTest: (concurrency: number, requests: number, fn: () => Promise<any>) => {
    return async () => {
      const results: any[] = [];
      const errors: any[] = [];
      
      const batches = Math.ceil(requests / concurrency);
      
      for (let batch = 0; batch < batches; batch++) {
        const batchPromises = [];
        const batchSize = Math.min(concurrency, requests - batch * concurrency);
        
        for (let i = 0; i < batchSize; i++) {
          batchPromises.push(
            fn().then(result => results.push(result)).catch(error => errors.push(error))
          );
        }
        
        await Promise.all(batchPromises);
      }
      
      return {
        totalRequests: requests,
        successfulRequests: results.length,
        failedRequests: errors.length,
        results,
        errors,
      };
    };
  },
};

/**
 * Cleanup helper
 */
export const cleanup = {
  /**
   * Clean up all test resources
   */
  all: async () => {
    await fileHelpers.cleanupTempFiles();
    jest.clearAllMocks();
    jest.restoreAllMocks();
  },
};

// Private helper functions
function generateValueByType(type: string, index: number): any {
  if (type === 'integer') {
    return index + 1;
  }
  if (type.startsWith('integer[')) {
    const [min, max] = type.match(/\[(\d+):(\d+)\]/)!.slice(1).map(Number);
    return min + (index % (max - min));
  }
  if (type === 'string') {
    return `String Value ${index + 1}`;
  }
  if (type === 'email') {
    return `user${index + 1}@example.com`;
  }
  if (type === 'boolean') {
    return index % 2 === 0;
  }
  if (type === 'datetime') {
    return new Date(2024, 0, 1 + (index % 365)).toISOString();
  }
  if (type.startsWith('categorical[')) {
    const options = type.match(/\[(.+)\]/)![1].split(',');
    return options[index % options.length];
  }
  if (type.startsWith('float[')) {
    const [min, max] = type.match(/\[([\d.]+):([\d.]+)\]/)!.slice(1).map(Number);
    return min + (index / 100) * (max - min);
  }
  
  return `Unknown Type: ${type}`;
}

function isValidType(value: any, type: string): boolean {
  if (type === 'integer') {
    return Number.isInteger(value);
  }
  if (type === 'string' || type === 'email') {
    return typeof value === 'string';
  }
  if (type === 'boolean') {
    return typeof value === 'boolean';
  }
  if (type === 'datetime') {
    return value instanceof Date || !isNaN(Date.parse(value));
  }
  if (type.startsWith('float[')) {
    return typeof value === 'number' && !Number.isInteger(value);
  }
  
  return true; // Default to valid for unknown types
}