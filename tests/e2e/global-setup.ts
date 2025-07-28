import { chromium, FullConfig } from '@playwright/test';
import path from 'path';
import fs from 'fs/promises';

/**
 * Global setup for E2E tests
 * Runs once before all tests
 */
async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting E2E test global setup...');
  
  // Create test results directory
  const testResultsDir = path.join(process.cwd(), 'test-results');
  try {
    await fs.mkdir(testResultsDir, { recursive: true });
  } catch (error) {
    // Directory already exists
  }
  
  // Set up test environment variables
  process.env.NODE_ENV = 'test';
  process.env.LOG_LEVEL = 'error';
  process.env.TESTING_MODE = 'true';
  process.env.MOCK_EXTERNAL_APIS = 'true';
  process.env.E2E_TEST_MODE = 'true';
  
  // Wait for the application to be ready
  const baseURL = config.projects[0].use?.baseURL || 'http://localhost:8080';
  console.log(`⏳ Waiting for application at ${baseURL}...`);
  
  await waitForApplication(baseURL);
  
  // Create a test admin user for E2E tests
  await setupTestUser(baseURL);
  
  // Setup test data
  await setupTestData(baseURL);
  
  console.log('✅ E2E test global setup completed');
}

/**
 * Wait for the application to be ready
 */
async function waitForApplication(baseURL: string, timeout: number = 120000) {
  const start = Date.now();
  const healthEndpoint = `${baseURL}/api/v1/health`;
  
  while (Date.now() - start < timeout) {
    try {
      const response = await fetch(healthEndpoint);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'healthy') {
          console.log('✅ Application is ready');
          return;
        }
      }
    } catch (error) {
      // Application not ready yet
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  throw new Error(`Application at ${baseURL} did not become ready within ${timeout}ms`);
}

/**
 * Create a test admin user for E2E tests
 */
async function setupTestUser(baseURL: string) {
  try {
    const response = await fetch(`${baseURL}/api/v1/test/setup-user`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Test-Mode': 'true',
      },
      body: JSON.stringify({
        email: 'e2e-test@example.com',
        role: 'admin',
        permissions: ['generate', 'validate', 'admin'],
      }),
    });
    
    if (response.ok) {
      const user = await response.json();
      
      // Store test credentials in environment
      process.env.E2E_TEST_USER_EMAIL = user.email;
      process.env.E2E_TEST_USER_ID = user.id;
      process.env.E2E_TEST_API_KEY = user.apiKey;
      
      console.log('✅ Test user created');
    } else {
      console.warn('⚠️ Failed to create test user:', response.statusText);
    }
  } catch (error) {
    console.warn('⚠️ Error creating test user:', error);
  }
}

/**
 * Setup test data for E2E tests
 */
async function setupTestData(baseURL: string) {
  try {
    const response = await fetch(`${baseURL}/api/v1/test/setup-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Test-Mode': 'true',
        'Authorization': `Bearer ${process.env.E2E_TEST_API_KEY}`,
      },
      body: JSON.stringify({
        datasets: [
          {
            id: 'e2e-test-dataset-1',
            name: 'E2E Test Dataset',
            schema: {
              id: 'integer',
              name: 'string',
              age: 'integer[18:80]',
              email: 'email',
            },
            size: 1000,
          },
        ],
        pipelines: [
          {
            id: 'e2e-test-pipeline-1',
            name: 'E2E Test Pipeline',
            generator: 'mock',
            validators: ['statistical', 'privacy'],
          },
        ],
      }),
    });
    
    if (response.ok) {
      console.log('✅ Test data setup completed');
    } else {
      console.warn('⚠️ Failed to setup test data:', response.statusText);
    }
  } catch (error) {
    console.warn('⚠️ Error setting up test data:', error);
  }
}

/**
 * Authentication helper for E2E tests
 */
export async function authenticateUser(page: any) {
  const baseURL = process.env.E2E_BASE_URL || 'http://localhost:8080';
  const apiKey = process.env.E2E_TEST_API_KEY;
  
  if (!apiKey) {
    throw new Error('E2E_TEST_API_KEY not found. Global setup may have failed.');
  }
  
  // Set authentication token in local storage
  await page.addInitScript((token: string) => {
    localStorage.setItem('auth_token', token);
  }, apiKey);
  
  // Set authorization header for API requests
  await page.setExtraHTTPHeaders({
    'Authorization': `Bearer ${apiKey}`,
  });
}

/**
 * Data cleanup helper for E2E tests
 */
export async function cleanupTestData() {
  const baseURL = process.env.E2E_BASE_URL || 'http://localhost:8080';
  const apiKey = process.env.E2E_TEST_API_KEY;
  
  if (!apiKey) {
    console.warn('⚠️ No API key found for cleanup');
    return;
  }
  
  try {
    await fetch(`${baseURL}/api/v1/test/cleanup`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'X-Test-Mode': 'true',
      },
    });
    
    console.log('✅ Test data cleanup completed');
  } catch (error) {
    console.warn('⚠️ Error during test data cleanup:', error);
  }
}

export default globalSetup;