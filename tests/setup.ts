import { jest } from '@jest/globals';

// Set up test environment
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';

// Mock external dependencies
jest.mock('../src/lib/external-apis', () => ({
  OpenAIClient: jest.fn().mockImplementation(() => ({
    generate: jest.fn(),
  })),
  AnthropicClient: jest.fn().mockImplementation(() => ({
    generate: jest.fn(),
  })),
  StabilityAIClient: jest.fn().mockImplementation(() => ({
    generate: jest.fn(),
  })),
}));

// Mock database connections
jest.mock('../src/lib/database', () => ({
  PostgreSQLClient: jest.fn().mockImplementation(() => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    query: jest.fn(),
  })),
  RedisClient: jest.fn().mockImplementation(() => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    get: jest.fn(),
    set: jest.fn(),
    del: jest.fn(),
  })),
  Neo4jClient: jest.fn().mockImplementation(() => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    run: jest.fn(),
  })),
}));

// Mock file system operations
jest.mock('fs/promises', () => ({
  readFile: jest.fn(),
  writeFile: jest.fn(),
  mkdir: jest.fn(),
  stat: jest.fn(),
  access: jest.fn(),
}));

// Mock cloud storage
jest.mock('../src/lib/storage', () => ({
  S3Client: jest.fn().mockImplementation(() => ({
    upload: jest.fn(),
    download: jest.fn(),
    delete: jest.fn(),
  })),
  GCSClient: jest.fn().mockImplementation(() => ({
    upload: jest.fn(),
    download: jest.fn(),
    delete: jest.fn(),
  })),
}));

// Global test utilities
global.testUtils = {
  generateMockData: (count: number = 10) => {
    return Array.from({ length: count }, (_, i) => ({
      id: i + 1,
      name: `Test Item ${i + 1}`,
      value: Math.random() * 100,
      timestamp: new Date().toISOString(),
    }));
  },
  
  waitFor: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)),
  
  createMockRequest: (overrides = {}) => ({
    method: 'GET',
    url: '/test',
    headers: {},
    body: {},
    query: {},
    params: {},
    ...overrides,
  }),
  
  createMockResponse: () => {
    const res: any = {};
    res.status = jest.fn().mockReturnValue(res);
    res.json = jest.fn().mockReturnValue(res);
    res.send = jest.fn().mockReturnValue(res);
    res.set = jest.fn().mockReturnValue(res);
    return res;
  },
};

// Setup and teardown hooks
beforeEach(() => {
  jest.clearAllMocks();
});

afterEach(() => {
  jest.restoreAllMocks();
});

// Increase timeout for async operations
jest.setTimeout(30000);