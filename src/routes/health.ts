import { Router, Request, Response } from 'express';
import { HealthMetrics } from '../middleware/metrics';

const router = Router();

interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
  uptime: number;
  environment: string;
  services: {
    database: ServiceHealth;
    redis: ServiceHealth;
    neo4j: ServiceHealth;
    storage: ServiceHealth;
  };
  metrics: {
    memory: MemoryMetrics;
    cpu: CpuMetrics;
    disk: DiskMetrics;
  };
}

interface ServiceHealth {
  status: 'healthy' | 'unhealthy' | 'degraded';
  latency?: number;
  error?: string;
  lastCheck: string;
}

interface MemoryMetrics {
  used: number;
  total: number;
  percentage: number;
}

interface CpuMetrics {
  percentage: number;
  loadAverage: number[];
}

interface DiskMetrics {
  used: number;
  total: number;
  percentage: number;
}

// Health check endpoint
router.get('/', async (req: Request, res: Response) => {
  try {
    const health = await performHealthCheck();
    
    // Record health check metrics
    HealthMetrics.recordHealthCheck('overall', health.status);
    
    const statusCode = health.status === 'healthy' ? 200 : 503;
    res.status(statusCode).json(health);
  } catch (error) {
    HealthMetrics.recordHealthCheck('overall', 'unhealthy');
    res.status(500).json({
      status: 'unhealthy',
      error: 'Health check failed',
      timestamp: new Date().toISOString(),
    });
  }
});

// Readiness probe
router.get('/ready', async (req: Request, res: Response) => {
  try {
    const services = await checkServices();
    const isReady = Object.values(services).every(service => service.status === 'healthy');
    
    res.status(isReady ? 200 : 503).json({
      ready: isReady,
      services,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(503).json({
      ready: false,
      error: 'Readiness check failed',
      timestamp: new Date().toISOString(),
    });
  }
});

// Liveness probe
router.get('/live', (req: Request, res: Response) => {
  res.status(200).json({
    alive: true,
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// Detailed health check
router.get('/detailed', async (req: Request, res: Response) => {
  try {
    const health = await performDetailedHealthCheck();
    const statusCode = health.status === 'healthy' ? 200 : 503;
    res.status(statusCode).json(health);
  } catch (error) {
    res.status(500).json({
      status: 'unhealthy',
      error: 'Detailed health check failed',
      timestamp: new Date().toISOString(),
    });
  }
});

async function performHealthCheck(): Promise<HealthStatus> {
  const services = await checkServices();
  const metrics = await getSystemMetrics();
  
  const overallStatus = Object.values(services).every(service => service.status === 'healthy')
    ? 'healthy'
    : 'unhealthy';

  return {
    status: overallStatus,
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0',
    uptime: process.uptime(),
    environment: process.env.NODE_ENV || 'development',
    services,
    metrics,
  };
}

async function performDetailedHealthCheck(): Promise<any> {
  const basicHealth = await performHealthCheck();
  
  // Add more detailed checks
  const additionalChecks = {
    dependencies: await checkDependencies(),
    configuration: await checkConfiguration(),
    resources: await checkResources(),
  };
  
  return {
    ...basicHealth,
    ...additionalChecks,
  };
}

async function checkServices(): Promise<HealthStatus['services']> {
  const services = {
    database: await checkDatabase(),
    redis: await checkRedis(),
    neo4j: await checkNeo4j(),
    storage: await checkStorage(),
  };
  
  // Record individual service health
  Object.entries(services).forEach(([service, health]) => {
    HealthMetrics.recordHealthCheck(service, health.status === 'healthy' ? 'healthy' : 'unhealthy');
  });
  
  return services;
}

async function checkDatabase(): Promise<ServiceHealth> {
  try {
    const start = Date.now();
    // Mock database check - replace with actual database ping
    await new Promise(resolve => setTimeout(resolve, 10));
    const latency = Date.now() - start;
    
    return {
      status: 'healthy',
      latency,
      lastCheck: new Date().toISOString(),
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      error: error instanceof Error ? error.message : 'Database connection failed',
      lastCheck: new Date().toISOString(),
    };
  }
}

async function checkRedis(): Promise<ServiceHealth> {
  try {
    const start = Date.now();
    // Mock Redis check - replace with actual Redis ping
    await new Promise(resolve => setTimeout(resolve, 5));
    const latency = Date.now() - start;
    
    return {
      status: 'healthy',
      latency,
      lastCheck: new Date().toISOString(),
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      error: error instanceof Error ? error.message : 'Redis connection failed',
      lastCheck: new Date().toISOString(),
    };
  }
}

async function checkNeo4j(): Promise<ServiceHealth> {
  try {
    const start = Date.now();
    // Mock Neo4j check - replace with actual Neo4j ping
    await new Promise(resolve => setTimeout(resolve, 15));
    const latency = Date.now() - start;
    
    return {
      status: 'healthy',
      latency,
      lastCheck: new Date().toISOString(),
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      error: error instanceof Error ? error.message : 'Neo4j connection failed',
      lastCheck: new Date().toISOString(),
    };
  }
}

async function checkStorage(): Promise<ServiceHealth> {
  try {
    const start = Date.now();
    // Mock storage check - replace with actual storage ping
    await new Promise(resolve => setTimeout(resolve, 8));
    const latency = Date.now() - start;
    
    return {
      status: 'healthy',
      latency,
      lastCheck: new Date().toISOString(),
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      error: error instanceof Error ? error.message : 'Storage connection failed',
      lastCheck: new Date().toISOString(),
    };
  }
}

async function getSystemMetrics(): Promise<HealthStatus['metrics']> {
  const memUsage = process.memoryUsage();
  const cpuUsage = process.cpuUsage();
  
  return {
    memory: {
      used: memUsage.heapUsed,
      total: memUsage.heapTotal,
      percentage: (memUsage.heapUsed / memUsage.heapTotal) * 100,
    },
    cpu: {
      percentage: (cpuUsage.user + cpuUsage.system) / 1000000, // Convert to percentage
      loadAverage: process.platform !== 'win32' ? require('os').loadavg() : [0, 0, 0],
    },
    disk: {
      used: 0, // Would need additional logic to check disk usage
      total: 0,
      percentage: 0,
    },
  };
}

async function checkDependencies(): Promise<any> {
  return {
    nodeVersion: process.version,
    dependencies: {
      express: '4.18.2', // These would be read from package.json
      prometheus: '15.1.0',
      // Add other critical dependencies
    },
  };
}

async function checkConfiguration(): Promise<any> {
  return {
    environment: process.env.NODE_ENV,
    port: process.env.PORT,
    databaseUrl: process.env.DATABASE_URL ? 'configured' : 'not configured',
    redisUrl: process.env.REDIS_URL ? 'configured' : 'not configured',
    neo4jUri: process.env.NEO4J_URI ? 'configured' : 'not configured',
  };
}

async function checkResources(): Promise<any> {
  return {
    maxMemory: process.env.NODE_OPTIONS?.includes('--max-old-space-size') || 'default',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    locale: Intl.DateTimeFormat().resolvedOptions().locale,
  };
}

export default router;