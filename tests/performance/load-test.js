import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time', true);
const generationTime = new Trend('generation_time', true);

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 }, // Ramp up to 10 users over 2 minutes
    { duration: '5m', target: 10 }, // Stay at 10 users for 5 minutes
    { duration: '2m', target: 20 }, // Ramp up to 20 users over 2 minutes
    { duration: '5m', target: 20 }, // Stay at 20 users for 5 minutes
    { duration: '2m', target: 0 },  // Ramp down to 0 users over 2 minutes
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate must be below 10%
    response_time: ['p(95)<3000'],     // 95% of API responses below 3s
    generation_time: ['p(95)<30000'],  // 95% of generations below 30s
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || 'test-api-key';

export default function () {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
  };

  // Test health endpoint
  testHealthEndpoint(headers);
  
  // Test authentication
  testAuthentication(headers);
  
  // Test data generation
  testDataGeneration(headers);
  
  // Test validation
  testValidation(headers);
  
  // Test lineage tracking
  testLineageTracking(headers);
  
  sleep(1);
}

function testHealthEndpoint(headers) {
  const response = http.get(`${BASE_URL}/api/v1/health`, { headers });
  
  const success = check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  responseTime.add(response.timings.duration);
  errorRate.add(!success);
}

function testAuthentication(headers) {
  const response = http.get(`${BASE_URL}/api/v1/auth/verify`, { headers });
  
  const success = check(response, {
    'auth verification status is 200': (r) => r.status === 200,
    'auth response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  responseTime.add(response.timings.duration);
  errorRate.add(!success);
}

function testDataGeneration(headers) {
  const payload = JSON.stringify({
    pipeline: 'test_pipeline',
    generator: 'mock',
    num_records: 100,
    schema: {
      id: 'integer',
      name: 'string',
      value: 'float',
    },
  });

  const response = http.post(`${BASE_URL}/api/v1/generate`, payload, { headers });
  
  const success = check(response, {
    'generation request status is 202': (r) => r.status === 202,
    'generation response contains job_id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.job_id !== undefined;
      } catch {
        return false;
      }
    },
  });
  
  responseTime.add(response.timings.duration);
  errorRate.add(!success);
  
  if (success) {
    const jobId = JSON.parse(response.body).job_id;
    pollGenerationStatus(jobId, headers);
  }
}

function pollGenerationStatus(jobId, headers) {
  let attempts = 0;
  const maxAttempts = 30; // 30 seconds max wait time
  
  while (attempts < maxAttempts) {
    const response = http.get(`${BASE_URL}/api/v1/jobs/${jobId}/status`, { headers });
    
    const success = check(response, {
      'job status request successful': (r) => r.status === 200,
    });
    
    if (success) {
      const body = JSON.parse(response.body);
      if (body.status === 'completed') {
        generationTime.add((attempts + 1) * 1000); // Convert to milliseconds
        break;
      } else if (body.status === 'failed') {
        errorRate.add(1);
        break;
      }
    }
    
    sleep(1);
    attempts++;
  }
  
  if (attempts >= maxAttempts) {
    errorRate.add(1); // Timeout is considered an error
  }
}

function testValidation(headers) {
  const payload = JSON.stringify({
    data_url: 'test://mock-data.csv',
    validators: ['statistical', 'privacy'],
    reference_data: 'test://reference-data.csv',
  });

  const response = http.post(`${BASE_URL}/api/v1/validate`, payload, { headers });
  
  const success = check(response, {
    'validation request status is 202': (r) => r.status === 202,
    'validation response contains job_id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.job_id !== undefined;
      } catch {
        return false;
      }
    },
  });
  
  responseTime.add(response.timings.duration);
  errorRate.add(!success);
}

function testLineageTracking(headers) {
  const mockDatasetId = 'test-dataset-123';
  const response = http.get(`${BASE_URL}/api/v1/lineage/${mockDatasetId}`, { headers });
  
  const success = check(response, {
    'lineage request status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'lineage response time < 2000ms': (r) => r.timings.duration < 2000,
  });
  
  responseTime.add(response.timings.duration);
  errorRate.add(!success);
}

// Setup function
export function setup() {
  console.log('Starting performance tests...');
  console.log(`Target URL: ${BASE_URL}`);
  
  // Verify the service is running
  const response = http.get(`${BASE_URL}/api/v1/health`);
  if (response.status !== 200) {
    throw new Error(`Service is not running. Health check returned: ${response.status}`);
  }
  
  return { baseUrl: BASE_URL };
}

// Teardown function
export function teardown(data) {
  console.log('Performance tests completed.');
  console.log(`Tested against: ${data.baseUrl}`);
}