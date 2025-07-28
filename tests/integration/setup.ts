import { jest } from '@jest/globals';
import { PostgreSQLContainer } from '@testcontainers/postgresql';
import { RedisContainer } from '@testcontainers/redis';
import { Neo4jContainer } from '@testcontainers/neo4j';

// Test containers for integration tests
let postgresContainer: PostgreSQLContainer;
let redisContainer: RedisContainer;
let neo4jContainer: Neo4jContainer;

// Global setup for integration tests
beforeAll(async () => {
  console.log('Starting integration test containers...');
  
  // Start PostgreSQL container
  postgresContainer = await new PostgreSQLContainer('postgres:15')
    .withDatabase('synthetic_guardian_test')
    .withUsername('test_user')
    .withPassword('test_password')
    .withExposedPorts(5432)
    .start();
  
  // Start Redis container
  redisContainer = await new RedisContainer('redis:7-alpine')
    .withExposedPorts(6379)
    .start();
  
  // Start Neo4j container
  neo4jContainer = await new Neo4jContainer('neo4j:5')
    .withAdminPassword('test_password')
    .withExposedPorts(7687, 7474)
    .start();
  
  // Set environment variables for containers
  process.env.DATABASE_URL = postgresContainer.getConnectionUri();
  process.env.REDIS_URL = `redis://${redisContainer.getHost()}:${redisContainer.getMappedPort(6379)}`;
  process.env.NEO4J_URI = `bolt://${neo4jContainer.getHost()}:${neo4jContainer.getMappedPort(7687)}`;
  process.env.NEO4J_USER = 'neo4j';
  process.env.NEO4J_PASSWORD = 'test_password';
  
  // Set test environment
  process.env.NODE_ENV = 'test';
  process.env.LOG_LEVEL = 'error';
  process.env.TESTING_MODE = 'true';
  process.env.MOCK_EXTERNAL_APIS = 'true';
  
  console.log('Integration test containers started successfully');
}, 120000); // 2 minute timeout for container startup

// Global teardown for integration tests
afterAll(async () => {
  console.log('Stopping integration test containers...');
  
  try {
    await postgresContainer?.stop();
    await redisContainer?.stop();
    await neo4jContainer?.stop();
    console.log('Integration test containers stopped successfully');
  } catch (error) {
    console.error('Error stopping containers:', error);
  }
}, 60000); // 1 minute timeout for container cleanup

// Test utilities for integration tests
global.integrationTestUtils = {
  getPostgresConnectionString: () => postgresContainer.getConnectionUri(),
  getRedisConnectionString: () => `redis://${redisContainer.getHost()}:${redisContainer.getMappedPort(6379)}`,
  getNeo4jConnectionString: () => `bolt://${neo4jContainer.getHost()}:${neo4jContainer.getMappedPort(7687)}`,
  
  waitForService: async (url: string, timeout: number = 30000) => {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      try {
        const response = await fetch(url);
        if (response.ok) return true;
      } catch (error) {
        // Service not ready yet
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error(`Service at ${url} did not become ready within ${timeout}ms`);
  },
  
  cleanupDatabase: async () => {
    // Clean up test data between tests
    const { Client } = await import('pg');
    const client = new Client({ connectionString: postgresContainer.getConnectionUri() });
    await client.connect();
    
    // Truncate all tables
    await client.query(`
      TRUNCATE TABLE IF EXISTS 
        synthetic_datasets,
        generation_jobs,
        validation_results,
        audit_logs,
        user_sessions
      CASCADE;
    `);
    
    await client.end();
  },
  
  seedTestData: async () => {
    // Seed common test data
    const { Client } = await import('pg');
    const client = new Client({ connectionString: postgresContainer.getConnectionUri() });
    await client.connect();
    
    // Create test user
    await client.query(`
      INSERT INTO users (id, email, role, created_at) 
      VALUES ('test-user-1', 'test@example.com', 'data_scientist', NOW())
      ON CONFLICT (id) DO NOTHING;
    `);
    
    // Create test pipeline
    await client.query(`
      INSERT INTO pipelines (id, name, configuration, user_id, created_at)
      VALUES ('test-pipeline-1', 'Test Pipeline', '{}', 'test-user-1', NOW())
      ON CONFLICT (id) DO NOTHING;
    `);
    
    await client.end();
  },
};

// Setup hooks for each test
beforeEach(async () => {
  await global.integrationTestUtils.cleanupDatabase();
  await global.integrationTestUtils.seedTestData();
});

// Extended timeout for integration tests
jest.setTimeout(60000);