/**
 * Example integration test to demonstrate testing infrastructure
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import { Client } from 'pg';
import Redis from 'ioredis';
import neo4j from 'neo4j-driver';
import { authHelpers, fileHelpers, dbHelpers } from '../helpers/test-helpers';
import { mockUsers, mockDatasets } from '../fixtures/test-data';

describe('Integration Tests', () => {
  let postgresClient: Client;
  let redisClient: Redis;
  let neo4jDriver: any;
  
  beforeAll(async () => {
    // Initialize database connections using test containers
    postgresClient = new Client({
      connectionString: global.integrationTestUtils.getPostgresConnectionString(),
    });
    await postgresClient.connect();
    
    redisClient = new Redis(global.integrationTestUtils.getRedisConnectionString());
    
    neo4jDriver = neo4j.driver(
      global.integrationTestUtils.getNeo4jConnectionString(),
      neo4j.auth.basic('neo4j', 'test_password')
    );
    
    // Create test tables
    await setupTestTables();
  });
  
  afterAll(async () => {
    await postgresClient.end();
    await redisClient.quit();
    await neo4jDriver.close();
  });
  
  beforeEach(async () => {
    // Clean up data before each test
    await global.integrationTestUtils.cleanupDatabase();
    await global.integrationTestUtils.seedTestData();
  });
  
  describe('Database Operations', () => {
    it('should connect to PostgreSQL and perform queries', async () => {
      const result = await postgresClient.query('SELECT NOW() as current_time');
      
      expect(result.rows).toHaveLength(1);
      expect(result.rows[0]).toHaveProperty('current_time');
      expect(result.rows[0].current_time).toBeInstanceOf(Date);
    });
    
    it('should store and retrieve user data', async () => {
      const user = mockUsers.dataScientist;
      
      // Insert user
      await postgresClient.query(
        'INSERT INTO users (id, email, role) VALUES ($1, $2, $3)',
        [user.id, user.email, user.role]
      );
      
      // Retrieve user
      const result = await postgresClient.query(
        'SELECT * FROM users WHERE id = $1',
        [user.id]
      );
      
      expect(result.rows).toHaveLength(1);
      expect(result.rows[0].email).toBe(user.email);
      expect(result.rows[0].role).toBe(user.role);
    });
    
    it('should handle database transactions', async () => {
      const user1 = { id: 'tx-user-1', email: 'tx1@test.com', role: 'user' };
      const user2 = { id: 'tx-user-2', email: 'tx2@test.com', role: 'user' };
      
      try {
        await postgresClient.query('BEGIN');
        
        await postgresClient.query(
          'INSERT INTO users (id, email, role) VALUES ($1, $2, $3)',
          [user1.id, user1.email, user1.role]
        );
        
        await postgresClient.query(
          'INSERT INTO users (id, email, role) VALUES ($1, $2, $3)',
          [user2.id, user2.email, user2.role]
        );
        
        await postgresClient.query('COMMIT');
        
        // Verify both users were inserted
        const result = await postgresClient.query(
          'SELECT COUNT(*) as count FROM users WHERE id IN ($1, $2)',
          [user1.id, user2.id]
        );
        
        expect(parseInt(result.rows[0].count)).toBe(2);
      } catch (error) {
        await postgresClient.query('ROLLBACK');
        throw error;
      }
    });
  });
  
  describe('Redis Operations', () => {
    it('should connect to Redis and perform operations', async () => {
      const key = 'test:key';
      const value = 'test value';
      
      // Set value
      await redisClient.set(key, value);
      
      // Get value
      const retrieved = await redisClient.get(key);
      
      expect(retrieved).toBe(value);
    });
    
    it('should handle Redis expiration', async () => {
      const key = 'test:expiring';
      const value = 'will expire';
      
      // Set with 1 second expiration
      await redisClient.setex(key, 1, value);
      
      // Verify it exists
      const immediate = await redisClient.get(key);
      expect(immediate).toBe(value);
      
      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 1100));
      
      // Verify it's gone
      const expired = await redisClient.get(key);
      expect(expired).toBeNull();
    });
    
    it('should handle Redis hash operations', async () => {
      const hashKey = 'test:hash';
      const data = {
        name: 'Test User',
        email: 'test@example.com',
        role: 'data_scientist',
      };
      
      // Set hash fields
      await redisClient.hmset(hashKey, data);
      
      // Get all hash fields
      const retrieved = await redisClient.hgetall(hashKey);
      
      expect(retrieved).toEqual(data);
    });
  });
  
  describe('Neo4j Operations', () => {
    let session: any;
    
    beforeEach(() => {
      session = neo4jDriver.session();
    });
    
    afterEach(async () => {
      await session.close();
    });
    
    it('should connect to Neo4j and create nodes', async () => {
      const result = await session.run(
        'CREATE (u:User {id: $id, name: $name}) RETURN u',
        { id: 'neo4j-user-1', name: 'Neo4j Test User' }
      );
      
      expect(result.records).toHaveLength(1);
      
      const user = result.records[0].get('u');
      expect(user.properties.id).toBe('neo4j-user-1');
      expect(user.properties.name).toBe('Neo4j Test User');
    });
    
    it('should create relationships between nodes', async () => {
      // Create dataset node
      await session.run(
        'CREATE (d:Dataset {id: $id, name: $name})',
        { id: 'dataset-1', name: 'Test Dataset' }
      );
      
      // Create synthetic dataset node
      await session.run(
        'CREATE (s:SyntheticDataset {id: $id, name: $name})',
        { id: 'synthetic-1', name: 'Synthetic Test Dataset' }
      );
      
      // Create relationship
      await session.run(`
        MATCH (d:Dataset {id: $datasetId})
        MATCH (s:SyntheticDataset {id: $syntheticId})
        CREATE (d)-[:GENERATED]->(s)
      `, { datasetId: 'dataset-1', syntheticId: 'synthetic-1' });
      
      // Query relationship
      const result = await session.run(`
        MATCH (d:Dataset)-[r:GENERATED]->(s:SyntheticDataset)
        RETURN d, r, s
      `);
      
      expect(result.records).toHaveLength(1);
      
      const record = result.records[0];
      expect(record.get('d').properties.id).toBe('dataset-1');
      expect(record.get('s').properties.id).toBe('synthetic-1');
    });
  });
  
  describe('File System Integration', () => {
    it('should create and read temporary files', async () => {
      const content = 'Test file content\nSecond line';
      const filePath = await fileHelpers.createTempFile(content, '.txt');
      
      expect(filePath).toBeDefined();
      
      const readContent = await fileHelpers.readFile(filePath);
      
      expect(readContent).toBe(content);
    });
    
    it('should create CSV files with mock data', async () => {
      const filePath = await fileHelpers.createTempCsvFile('customerData', 50);
      
      expect(filePath).toBeDefined();
      expect(filePath.endsWith('.csv')).toBe(true);
      
      const content = await fileHelpers.readFile(filePath);
      const lines = content.split('\n');
      
      // Header + 50 data rows
      expect(lines.length).toBe(51);
      
      // Check header
      const headers = lines[0].split(',');
      expect(headers).toContain('customer_id');
      expect(headers).toContain('first_name');
      expect(headers).toContain('email');
    });
  });
  
  describe('End-to-End Data Flow', () => {
    it('should simulate complete data generation workflow', async () => {
      // 1. Create user session in Redis
      const sessionId = `session-${Date.now()}`;
      const session = authHelpers.createTestSession('dataScientist');
      
      await redisClient.setex(
        `session:${sessionId}`,
        3600,
        JSON.stringify(session)
      );
      
      // 2. Store generation job in PostgreSQL
      const jobId = `job-${Date.now()}`;
      await postgresClient.query(`
        INSERT INTO generation_jobs (id, user_id, status, pipeline_config, created_at)
        VALUES ($1, $2, $3, $4, NOW())
      `, [jobId, session.userId, 'pending', JSON.stringify({ generator: 'mock' })]);
      
      // 3. Create lineage in Neo4j
      await session.run(`
        CREATE (j:GenerationJob {id: $jobId, userId: $userId, status: $status})
      `, { jobId, userId: session.userId, status: 'pending' });
      
      // 4. Verify all components are connected
      // Check Redis session
      const storedSession = await redisClient.get(`session:${sessionId}`);
      expect(JSON.parse(storedSession!).userId).toBe(session.userId);
      
      // Check PostgreSQL job
      const jobResult = await postgresClient.query(
        'SELECT * FROM generation_jobs WHERE id = $1',
        [jobId]
      );
      expect(jobResult.rows[0].user_id).toBe(session.userId);
      
      // Check Neo4j lineage
      const lineageResult = await session.run(
        'MATCH (j:GenerationJob {id: $jobId}) RETURN j',
        { jobId }
      );
      expect(lineageResult.records[0].get('j').properties.userId).toBe(session.userId);
    });
  });
  
  // Helper function to set up test tables
  async function setupTestTables() {
    await postgresClient.query(`
      CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(255) PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        role VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);
    
    await postgresClient.query(`
      CREATE TABLE IF NOT EXISTS generation_jobs (
        id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) REFERENCES users(id),
        status VARCHAR(50) NOT NULL,
        pipeline_config JSONB,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);
    
    await postgresClient.query(`
      CREATE TABLE IF NOT EXISTS synthetic_datasets (
        id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) REFERENCES generation_jobs(id),
        name VARCHAR(255) NOT NULL,
        schema_definition JSONB,
        size INTEGER,
        quality_score DECIMAL(3,2),
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);
  }
});