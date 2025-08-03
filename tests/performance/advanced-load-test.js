/**
 * Advanced Load Testing Configuration for Synthetic Data Guardian
 * Comprehensive performance testing with multiple scenarios and metrics
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// Custom metrics
const errorRate = new Rate('error_rate');
const generationDuration = new Trend('generation_duration');
const validationDuration = new Trend('validation_duration');
const dataQualityScore = new Trend('data_quality_score');
const privacyScore = new Trend('privacy_score');
const apiResponseTime = new Trend('api_response_time');
const successfulGenerations = new Counter('successful_generations');
const failedGenerations = new Counter('failed_generations');

// Test configuration
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Load test scenarios
export const options = {
  scenarios: {
    // Smoke test - verify basic functionality
    smoke_test: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      tags: { test_type: 'smoke' },
    },
    
    // Load test - normal expected load
    load_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },  // Ramp up
        { duration: '5m', target: 10 },  // Stay at 10 users
        { duration: '2m', target: 20 },  // Ramp up to 20
        { duration: '5m', target: 20 },  // Stay at 20 users
        { duration: '2m', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'load' },
    },
    
    // Stress test - beyond normal capacity
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },  // Ramp up to normal load
        { duration: '5m', target: 20 },  // Stay at normal load
        { duration: '2m', target: 50 },  // Ramp up to stress level
        { duration: '5m', target: 50 },  // Stay at stress level
        { duration: '2m', target: 100 }, // Ramp up to breaking point
        { duration: '5m', target: 100 }, // Stay at breaking point
        { duration: '2m', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'stress' },
    },
    
    // Spike test - sudden load increase
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 10 },  // Normal load
        { duration: '30s', target: 100 }, // Spike
        { duration: '1m', target: 10 },  // Back to normal
        { duration: '30s', target: 100 }, // Another spike
        { duration: '1m', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'spike' },
    },
    
    // Volume test - large data generation
    volume_test: {
      executor: 'constant-vus',
      vus: 5,
      duration: '10m',
      tags: { test_type: 'volume' },
    },
    
    // Endurance test - extended duration
    endurance_test: {
      executor: 'constant-vus',
      vus: 10,
      duration: '30m',
      tags: { test_type: 'endurance' },
    }
  },
  
  // Thresholds - SLA requirements
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.05'],    // Error rate under 5%
    error_rate: ['rate<0.05'],
    generation_duration: ['p(95)<30000'], // 95% of generations under 30s
    validation_duration: ['p(95)<5000'],  // 95% of validations under 5s
    data_quality_score: ['avg>0.9'],      // Average quality score > 90%
    privacy_score: ['avg>0.95'],          // Average privacy score > 95%
  },
  
  // Resource limits
  noConnectionReuse: false,
  userAgent: 'K6-SyntheticGuardian-LoadTest/1.0',
  throw: true,
  
  // Test data management
  setupTimeout: '60s',
  teardownTimeout: '60s',
};

// Setup function - runs once before all tests
export function setup() {
  console.log('Setting up load test environment...');
  
  // Verify API is accessible
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error(`API health check failed: ${healthCheck.status}`);
  }
  
  // Setup test data
  const setupData = {
    timestamp: new Date().toISOString(),
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    testSchemas: generateTestSchemas(),
  };
  
  console.log('Load test setup complete');
  return setupData;
}

// Main test function
export default function(data) {
  const scenario = __ENV.TEST_SCENARIO || 'mixed';
  
  group('API Health Check', () => {
    testHealthEndpoint();
  });
  
  group('Data Generation Tests', () => {
    switch(scenario) {
      case 'tabular':
        testTabularGeneration(data);
        break;
      case 'timeseries':
        testTimeSeriesGeneration(data);
        break;
      case 'text':
        testTextGeneration(data);
        break;
      case 'image':
        testImageGeneration(data);
        break;
      default:
        testMixedGeneration(data);
    }
  });
  
  group('Validation Tests', () => {
    testDataValidation(data);
  });
  
  group('Lineage Tests', () => {
    testLineageTracking(data);
  });
  
  // Random sleep between 1-3 seconds to simulate user behavior
  sleep(Math.random() * 2 + 1);
}

// Test functions
function testHealthEndpoint() {
  const response = http.get(`${BASE_URL}/health`, {
    headers: { 'Authorization': `Bearer ${API_KEY}` },
  });
  
  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  apiResponseTime.add(response.timings.duration);
  errorRate.add(response.status !== 200);
}

function testTabularGeneration(data) {
  const payload = {
    generator: 'sdv',
    dataType: 'tabular',
    schema: data.testSchemas.tabular,
    numRecords: Math.floor(Math.random() * 1000) + 100, // 100-1100 records
    seed: Math.floor(Math.random() * 10000),
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/v1/generate`, JSON.stringify(payload), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'generation status is 200 or 202': (r) => [200, 202].includes(r.status),
    'generation response has data': (r) => r.json() && r.json().data,
    'generation quality score > 0.8': (r) => {
      const result = r.json();
      return result.qualityScore && result.qualityScore > 0.8;
    },
    'generation privacy score > 0.9': (r) => {
      const result = r.json();
      return result.privacyScore && result.privacyScore > 0.9;
    },
  });
  
  if (success) {
    successfulGenerations.add(1);
    const result = response.json();
    if (result.qualityScore) dataQualityScore.add(result.qualityScore);
    if (result.privacyScore) privacyScore.add(result.privacyScore);
  } else {
    failedGenerations.add(1);
  }
  
  generationDuration.add(duration);
  errorRate.add(!success);
}

function testTimeSeriesGeneration(data) {
  const payload = {
    generator: 'timegan',
    dataType: 'timeseries',
    schema: data.testSchemas.timeseries,
    sequenceLength: Math.floor(Math.random() * 50) + 50, // 50-100 time steps
    numSequences: Math.floor(Math.random() * 50) + 10,   // 10-60 sequences
    seed: Math.floor(Math.random() * 10000),
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/v1/generate`, JSON.stringify(payload), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
  });
  const duration = Date.now() - startTime;
  
  generationDuration.add(duration);
  
  const success = check(response, {
    'timeseries generation successful': (r) => [200, 202].includes(r.status),
    'timeseries has temporal structure': (r) => {
      const result = r.json();
      return result.data && Array.isArray(result.data);
    },
  });
  
  if (success) successfulGenerations.add(1);
  else failedGenerations.add(1);
  
  errorRate.add(!success);
}

function testTextGeneration(data) {
  const payload = {
    generator: 'gpt',
    dataType: 'text',
    template: 'customer_review',
    numDocuments: Math.floor(Math.random() * 20) + 5, // 5-25 documents
    privacyMode: 'differential',
    seed: Math.floor(Math.random() * 10000),
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/v1/generate`, JSON.stringify(payload), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
  });
  const duration = Date.now() - startTime;
  
  generationDuration.add(duration);
  
  const success = check(response, {
    'text generation successful': (r) => [200, 202].includes(r.status),
    'text contains no PII': (r) => {
      const result = r.json();
      return result.privacyScore && result.privacyScore > 0.95;
    },
  });
  
  if (success) successfulGenerations.add(1);
  else failedGenerations.add(1);
  
  errorRate.add(!success);
}

function testImageGeneration(data) {
  const payload = {
    generator: 'stable_diffusion',
    dataType: 'image',
    prompt: 'medical scan, anonymized',
    numImages: Math.floor(Math.random() * 5) + 1, // 1-5 images
    watermark: true,
    size: [512, 512],
    seed: Math.floor(Math.random() * 10000),
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/v1/generate`, JSON.stringify(payload), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
  });
  const duration = Date.now() - startTime;
  
  generationDuration.add(duration);
  
  const success = check(response, {
    'image generation successful': (r) => [200, 202].includes(r.status),
    'images are watermarked': (r) => {
      const result = r.json();
      return result.watermarked === true;
    },
  });
  
  if (success) successfulGenerations.add(1);
  else failedGenerations.add(1);
  
  errorRate.add(!success);
}

function testMixedGeneration(data) {
  const generators = ['testTabularGeneration', 'testTimeSeriesGeneration', 'testTextGeneration'];
  const randomGenerator = generators[Math.floor(Math.random() * generators.length)];
  
  switch(randomGenerator) {
    case 'testTabularGeneration':
      testTabularGeneration(data);
      break;
    case 'testTimeSeriesGeneration':
      testTimeSeriesGeneration(data);
      break;
    case 'testTextGeneration':
      testTextGeneration(data);
      break;
  }
}

function testDataValidation(data) {
  const payload = {
    validatorType: 'statistical',
    threshold: 0.9,
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/v1/validate`, JSON.stringify(payload), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
  });
  const duration = Date.now() - startTime;
  
  validationDuration.add(duration);
  
  check(response, {
    'validation successful': (r) => r.status === 200,
    'validation response time < 5s': (r) => r.timings.duration < 5000,
  });
}

function testLineageTracking(data) {
  const response = http.get(`${BASE_URL}/api/v1/lineage/recent`, {
    headers: { 'Authorization': `Bearer ${API_KEY}` },
  });
  
  check(response, {
    'lineage tracking accessible': (r) => r.status === 200,
    'lineage has graph structure': (r) => {
      const result = r.json();
      return result.nodes && result.edges;
    },
  });
}

// Helper functions
function generateTestSchemas() {
  return {
    tabular: {
      age: 'integer[18:80]',
      income: 'float[20000:200000]',
      email: 'email',
      category: 'categorical[A,B,C,D]',
    },
    timeseries: {
      timestamp: 'datetime',
      value: 'float[0:100]',
      trend: 'float[-1:1]',
    },
  };
}

// Teardown function - runs once after all tests
export function teardown(data) {
  console.log('Cleaning up load test environment...');
  
  // Cleanup test data if needed
  const cleanup = http.delete(`${BASE_URL}/api/v1/test-data`, {
    headers: { 'Authorization': `Bearer ${API_KEY}` },
  });
  
  console.log('Load test teardown complete');
}

// Custom report generation
export function handleSummary(data) {
  return {
    'test-results/load-test-report.html': htmlReport(data),
    'test-results/load-test-summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}