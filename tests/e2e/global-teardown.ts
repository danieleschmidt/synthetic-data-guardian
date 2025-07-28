import { FullConfig } from '@playwright/test';
import { cleanupTestData } from './global-setup';

/**
 * Global teardown for E2E tests
 * Runs once after all tests complete
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting E2E test global teardown...');
  
  // Clean up test data
  await cleanupTestData();
  
  // Clean up test user
  await cleanupTestUser();
  
  // Archive test results if in CI
  if (process.env.CI) {
    await archiveTestResults();
  }
  
  console.log('✅ E2E test global teardown completed');
}

/**
 * Clean up the test user created during setup
 */
async function cleanupTestUser() {
  const baseURL = process.env.E2E_BASE_URL || 'http://localhost:8080';
  const apiKey = process.env.E2E_TEST_API_KEY;
  const userId = process.env.E2E_TEST_USER_ID;
  
  if (!apiKey || !userId) {
    console.warn('⚠️ No test user credentials found for cleanup');
    return;
  }
  
  try {
    const response = await fetch(`${baseURL}/api/v1/test/cleanup-user`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'X-Test-Mode': 'true',
      },
      body: JSON.stringify({ userId }),
    });
    
    if (response.ok) {
      console.log('✅ Test user cleaned up');
    } else {
      console.warn('⚠️ Failed to cleanup test user:', response.statusText);
    }
  } catch (error) {
    console.warn('⚠️ Error cleaning up test user:', error);
  }
}

/**
 * Archive test results for CI/CD pipeline
 */
async function archiveTestResults() {
  const fs = await import('fs/promises');
  const path = await import('path');
  
  try {
    const testResultsDir = path.join(process.cwd(), 'test-results');
    const archiveDir = path.join(process.cwd(), 'archived-test-results');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const targetDir = path.join(archiveDir, `e2e-results-${timestamp}`);
    
    // Create archive directory
    await fs.mkdir(targetDir, { recursive: true });
    
    // Copy test results
    const entries = await fs.readdir(testResultsDir, { withFileTypes: true });
    
    for (const entry of entries) {
      const sourcePath = path.join(testResultsDir, entry.name);
      const targetPath = path.join(targetDir, entry.name);
      
      if (entry.isDirectory()) {
        await fs.cp(sourcePath, targetPath, { recursive: true });
      } else {
        await fs.copyFile(sourcePath, targetPath);
      }
    }
    
    console.log(`✅ Test results archived to ${targetDir}`);
  } catch (error) {
    console.warn('⚠️ Error archiving test results:', error);
  }
}

export default globalTeardown;