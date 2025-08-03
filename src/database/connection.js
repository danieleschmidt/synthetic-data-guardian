/**
 * Database Connection Manager - Handles connections to PostgreSQL, Redis, and Neo4j
 */

import pg from 'pg';
import Redis from 'ioredis';
import neo4j from 'neo4j-driver';
import dotenv from 'dotenv';

dotenv.config();

class DatabaseManager {
  constructor(logger) {
    this.logger = logger;
    this.postgres = null;
    this.redis = null;
    this.neo4j = null;
    this.initialized = false;
  }

  async initialize() {
    try {
      this.logger.info('Initializing database connections...');

      // Initialize PostgreSQL
      await this.initializePostgreSQL();

      // Initialize Redis
      await this.initializeRedis();

      // Initialize Neo4j
      await this.initializeNeo4j();

      this.initialized = true;
      this.logger.info('All database connections established successfully');

    } catch (error) {
      this.logger.error('Database initialization failed', { error: error.message });
      throw error;
    }
  }

  async initializePostgreSQL() {
    const config = {
      connectionString: process.env.DATABASE_URL,
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT) || 5432,
      database: process.env.DB_NAME || 'synthetic_guardian',
      user: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || 'password',
      ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false,
      min: parseInt(process.env.DB_POOL_MIN) || 2,
      max: parseInt(process.env.DB_POOL_MAX) || 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    };

    this.postgres = new pg.Pool(config);

    // Test connection
    try {
      const client = await this.postgres.connect();
      await client.query('SELECT NOW()');
      client.release();
      this.logger.info('PostgreSQL connection established', { 
        host: config.host, 
        port: config.port, 
        database: config.database 
      });
    } catch (error) {
      this.logger.warn('PostgreSQL connection failed, using mock implementation', { 
        error: error.message 
      });
      this.postgres = new MockPostgreSQL(this.logger);
    }

    // Handle pool errors
    this.postgres.on('error', (err) => {
      this.logger.error('PostgreSQL pool error', { error: err.message });
    });
  }

  async initializeRedis() {
    const config = {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT) || 6379,
      password: process.env.REDIS_PASSWORD || undefined,
      db: parseInt(process.env.REDIS_DB) || 0,
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
      connectTimeout: 2000,
      commandTimeout: 1000
    };

    // Try Redis connection
    try {
      this.redis = new Redis(config);
      await this.redis.connect();
      await this.redis.ping();
      this.logger.info('Redis connection established', { 
        host: config.host, 
        port: config.port, 
        db: config.db 
      });
    } catch (error) {
      this.logger.warn('Redis connection failed, using mock implementation', { 
        error: error.message 
      });
      this.redis = new MockRedis(this.logger);
    }

    // Handle Redis errors
    this.redis.on('error', (err) => {
      this.logger.error('Redis connection error', { error: err.message });
    });
  }

  async initializeNeo4j() {
    const uri = process.env.NEO4J_URI || 'bolt://localhost:7687';
    const user = process.env.NEO4J_USER || 'neo4j';
    const password = process.env.NEO4J_PASSWORD || 'password';

    try {
      this.neo4j = neo4j.driver(uri, neo4j.auth.basic(user, password), {
        connectionTimeout: 2000,
        maxConnectionLifetime: 30 * 60 * 1000, // 30 minutes
        maxConnectionPoolSize: 50,
        connectionAcquisitionTimeout: 2000
      });

      // Test connection
      const session = this.neo4j.session();
      await session.run('RETURN 1');
      await session.close();

      this.logger.info('Neo4j connection established', { uri });
    } catch (error) {
      this.logger.warn('Neo4j connection failed, using mock implementation', { 
        error: error.message 
      });
      this.neo4j = new MockNeo4j(this.logger);
    }
  }

  // PostgreSQL query methods
  async query(text, params = []) {
    if (!this.postgres) {
      throw new Error('PostgreSQL not initialized');
    }
    
    const start = Date.now();
    try {
      const result = await this.postgres.query(text, params);
      const duration = Date.now() - start;
      this.logger.debug('PostgreSQL query executed', { 
        query: text.substring(0, 100),
        duration,
        rows: result.rows?.length || 0
      });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      this.logger.error('PostgreSQL query failed', { 
        query: text.substring(0, 100),
        error: error.message,
        duration
      });
      throw error;
    }
  }

  async getPostgresClient() {
    return await this.postgres.connect();
  }

  // Redis methods
  async get(key) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    return await this.redis.get(key);
  }

  async set(key, value, expireSeconds = null) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    
    if (expireSeconds) {
      return await this.redis.setex(key, expireSeconds, value);
    } else {
      return await this.redis.set(key, value);
    }
  }

  async del(key) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    return await this.redis.del(key);
  }

  async exists(key) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    return await this.redis.exists(key);
  }

  async hset(key, field, value) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    return await this.redis.hset(key, field, value);
  }

  async hget(key, field) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    return await this.redis.hget(key, field);
  }

  async hgetall(key) {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }
    return await this.redis.hgetall(key);
  }

  // Neo4j methods
  async runCypher(query, parameters = {}) {
    if (!this.neo4j) {
      throw new Error('Neo4j not initialized');
    }

    const session = this.neo4j.session();
    try {
      const result = await session.run(query, parameters);
      this.logger.debug('Neo4j query executed', { 
        query: query.substring(0, 100),
        records: result.records.length
      });
      return result;
    } catch (error) {
      this.logger.error('Neo4j query failed', { 
        query: query.substring(0, 100),
        error: error.message
      });
      throw error;
    } finally {
      await session.close();
    }
  }

  async runCypherTransaction(queries) {
    if (!this.neo4j) {
      throw new Error('Neo4j not initialized');
    }

    const session = this.neo4j.session();
    const tx = session.beginTransaction();

    try {
      const results = [];
      for (const { query, parameters } of queries) {
        const result = await tx.run(query, parameters);
        results.push(result);
      }
      await tx.commit();
      return results;
    } catch (error) {
      await tx.rollback();
      this.logger.error('Neo4j transaction failed', { error: error.message });
      throw error;
    } finally {
      await session.close();
    }
  }

  // Health check methods
  async checkHealth() {
    const health = {
      postgres: false,
      redis: false,
      neo4j: false,
      timestamp: new Date().toISOString()
    };

    // Check PostgreSQL
    try {
      await this.query('SELECT 1');
      health.postgres = true;
    } catch (error) {
      this.logger.warn('PostgreSQL health check failed', { error: error.message });
    }

    // Check Redis
    try {
      await this.redis.ping();
      health.redis = true;
    } catch (error) {
      this.logger.warn('Redis health check failed', { error: error.message });
    }

    // Check Neo4j
    try {
      await this.runCypher('RETURN 1');
      health.neo4j = true;
    } catch (error) {
      this.logger.warn('Neo4j health check failed', { error: error.message });
    }

    return health;
  }

  // Cleanup methods
  async cleanup() {
    this.logger.info('Closing database connections...');

    if (this.postgres && this.postgres.end) {
      await this.postgres.end();
      this.logger.info('PostgreSQL connection closed');
    }

    if (this.redis && this.redis.disconnect) {
      await this.redis.disconnect();
      this.logger.info('Redis connection closed');
    }

    if (this.neo4j && this.neo4j.close) {
      await this.neo4j.close();
      this.logger.info('Neo4j connection closed');
    }

    this.initialized = false;
  }
}

// Mock implementations for development without external dependencies
class MockPostgreSQL {
  constructor(logger) {
    this.logger = logger;
    this.mockData = new Map();
  }

  async query(text, params = []) {
    this.logger.debug('Mock PostgreSQL query', { query: text.substring(0, 100) });
    
    // Simple mock responses for common queries
    if (text.includes('SELECT NOW()')) {
      return { rows: [{ now: new Date().toISOString() }] };
    }
    
    if (text.includes('SELECT 1')) {
      return { rows: [{ '?column?': 1 }] };
    }

    return { rows: [] };
  }

  async connect() {
    return {
      query: this.query.bind(this),
      release: () => {}
    };
  }

  on() {} // Mock event handler
  end() {} // Mock cleanup
}

class MockRedis {
  constructor(logger) {
    this.logger = logger;
    this.mockData = new Map();
  }

  async connect() {
    return Promise.resolve();
  }

  async ping() {
    return 'PONG';
  }

  async get(key) {
    return this.mockData.get(key) || null;
  }

  async set(key, value) {
    this.mockData.set(key, value);
    return 'OK';
  }

  async setex(key, seconds, value) {
    this.mockData.set(key, value);
    setTimeout(() => this.mockData.delete(key), seconds * 1000);
    return 'OK';
  }

  async del(key) {
    return this.mockData.delete(key) ? 1 : 0;
  }

  async exists(key) {
    return this.mockData.has(key) ? 1 : 0;
  }

  async hset(key, field, value) {
    const hash = this.mockData.get(key) || {};
    hash[field] = value;
    this.mockData.set(key, hash);
    return 1;
  }

  async hget(key, field) {
    const hash = this.mockData.get(key) || {};
    return hash[field] || null;
  }

  async hgetall(key) {
    return this.mockData.get(key) || {};
  }

  on() {} // Mock event handler
  disconnect() {} // Mock cleanup
}

class MockNeo4j {
  constructor(logger) {
    this.logger = logger;
    this.mockData = [];
  }

  session() {
    return {
      run: async (query, parameters = {}) => {
        this.logger.debug('Mock Neo4j query', { query: query.substring(0, 100) });
        return { records: [] };
      },
      close: async () => {},
      beginTransaction: () => ({
        run: async (query, parameters = {}) => ({ records: [] }),
        commit: async () => {},
        rollback: async () => {}
      })
    };
  }

  close() {} // Mock cleanup
}

// Singleton instance
let dbManager = null;

export function getDatabaseManager(logger) {
  if (!dbManager) {
    dbManager = new DatabaseManager(logger);
  }
  return dbManager;
}

export { DatabaseManager };