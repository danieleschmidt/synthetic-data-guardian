/**
 * Example E2E test to demonstrate testing infrastructure
 */

import { test, expect, Page } from '@playwright/test';
import { authenticateUser, cleanupTestData } from './global-setup';

test.describe('Synthetic Data Guardian E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Authenticate user for each test
    await authenticateUser(page);
  });

  test.afterEach(async () => {
    // Clean up test data after each test
    await cleanupTestData();
  });

  test('should load the application homepage', async ({ page }) => {
    await page.goto('/');
    
    // Check that the page loads
    await expect(page).toHaveTitle(/Synthetic Data Guardian/);
    
    // Check for main navigation
    await expect(page.locator('nav')).toBeVisible();
    
    // Check for key sections
    await expect(page.locator('text=Generate')).toBeVisible();
    await expect(page.locator('text=Validate')).toBeVisible();
    await expect(page.locator('text=Lineage')).toBeVisible();
  });

  test('should navigate to data generation page', async ({ page }) => {
    await page.goto('/');
    
    // Click on Generate link/button
    await page.click('text=Generate');
    
    // Verify we're on the generation page
    await expect(page).toHaveURL(/.*\/generate/);
    await expect(page.locator('h1')).toContainText('Generate Synthetic Data');
    
    // Check for key form elements
    await expect(page.locator('select[name="generator"]')).toBeVisible();
    await expect(page.locator('input[name="num_records"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('should create a new data generation job', async ({ page }) => {
    await page.goto('/generate');
    
    // Fill out the generation form
    await page.selectOption('select[name="generator"]', 'sdv');
    await page.fill('input[name="num_records"]', '100');
    await page.fill('input[name="pipeline_name"]', 'E2E Test Pipeline');
    
    // Upload a sample CSV file (if file upload is supported)
    // await page.setInputFiles('input[type="file"]', 'tests/fixtures/sample.csv');
    
    // Add validation settings
    await page.check('input[name="enable_statistical_validation"]');
    await page.check('input[name="enable_privacy_validation"]');
    
    // Submit the form
    await page.click('button[type="submit"]');
    
    // Wait for job creation confirmation
    await expect(page.locator('.success-message')).toBeVisible();
    await expect(page.locator('.success-message')).toContainText('Generation job created successfully');
    
    // Check that we're redirected to job status page
    await expect(page).toHaveURL(/.*\/jobs\/[\w-]+/);
    
    // Verify job details are displayed
    await expect(page.locator('.job-status')).toContainText('pending');
    await expect(page.locator('.pipeline-name')).toContainText('E2E Test Pipeline');
  });

  test('should monitor job progress and completion', async ({ page }) => {
    // Start by creating a job (using API to speed up the test)
    const response = await page.request.post('/api/v1/generate', {
      headers: {
        'Authorization': `Bearer ${process.env.E2E_TEST_API_KEY}`,
        'Content-Type': 'application/json',
      },
      data: {
        pipeline: 'e2e-test-pipeline-1',
        generator: 'mock',
        num_records: 50,
      },
    });
    
    const jobData = await response.json();
    const jobId = jobData.job_id;
    
    // Navigate to job status page
    await page.goto(`/jobs/${jobId}`);
    
    // Wait for job to complete (with timeout)
    await page.waitForFunction(
      () => {
        const statusElement = document.querySelector('.job-status');
        return statusElement && (statusElement.textContent === 'completed' || statusElement.textContent === 'failed');
      },
      { timeout: 30000 }
    );
    
    // Check final status
    const finalStatus = await page.locator('.job-status').textContent();
    expect(['completed', 'failed']).toContain(finalStatus);
    
    if (finalStatus === 'completed') {
      // Verify completion details
      await expect(page.locator('.quality-score')).toBeVisible();
      await expect(page.locator('.privacy-score')).toBeVisible();
      await expect(page.locator('.download-link')).toBeVisible();
      
      // Check that lineage information is available
      await expect(page.locator('.lineage-id')).toBeVisible();
    }
  });

  test('should validate synthetic data', async ({ page }) => {
    await page.goto('/validate');
    
    // Fill out validation form
    await page.fill('input[name="dataset_url"]', 'test://synthetic-data.csv');
    await page.fill('input[name="reference_url"]', 'test://reference-data.csv');
    
    // Select validation types
    await page.check('input[name="statistical_validation"]');
    await page.check('input[name="privacy_validation"]');
    await page.check('input[name="bias_validation"]');
    
    // Set validation thresholds
    await page.fill('input[name="statistical_threshold"]', '0.9');
    await page.fill('input[name="privacy_epsilon"]', '1.0');
    
    // Submit validation request
    await page.click('button[type="submit"]');
    
    // Wait for validation results
    await expect(page.locator('.validation-results')).toBeVisible({ timeout: 30000 });
    
    // Check that results are displayed
    await expect(page.locator('.overall-score')).toBeVisible();
    await expect(page.locator('.statistical-score')).toBeVisible();
    await expect(page.locator('.privacy-score')).toBeVisible();
  });

  test('should display lineage information', async ({ page }) => {
    // First create some test data with lineage
    const response = await page.request.post('/api/v1/generate', {
      headers: {
        'Authorization': `Bearer ${process.env.E2E_TEST_API_KEY}`,
        'Content-Type': 'application/json',
      },
      data: {
        pipeline: 'e2e-test-pipeline-1',
        generator: 'mock',
        num_records: 10,
      },
    });
    
    const jobData = await response.json();
    
    // Wait for job completion
    await page.waitForTimeout(5000);
    
    // Navigate to lineage page
    await page.goto('/lineage');
    
    // Search for our dataset
    await page.fill('input[name="search"]', jobData.job_id);
    await page.click('button[name="search_submit"]');
    
    // Wait for search results
    await expect(page.locator('.lineage-graph')).toBeVisible({ timeout: 10000 });
    
    // Check that lineage nodes are displayed
    await expect(page.locator('.lineage-node')).toHaveCount.greaterThan(0);
    
    // Check for dataset details
    await page.click('.lineage-node:first-child');
    await expect(page.locator('.node-details')).toBeVisible();
    await expect(page.locator('.node-details')).toContainText('Dataset');
  });

  test('should handle errors gracefully', async ({ page }) => {
    await page.goto('/generate');
    
    // Try to submit form without required fields
    await page.click('button[type="submit"]');
    
    // Check for validation errors
    await expect(page.locator('.error-message')).toBeVisible();
    await expect(page.locator('.field-error')).toHaveCount.greaterThan(0);
    
    // Fill in some fields but leave others invalid
    await page.selectOption('select[name="generator"]', 'sdv');
    await page.fill('input[name="num_records"]', 'invalid_number');
    
    await page.click('button[type="submit"]');
    
    // Check for specific field validation
    await expect(page.locator('.field-error')).toContainText('must be a number');
  });

  test('should be responsive on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.goto('/');
    
    // Check that mobile navigation works
    const mobileMenuButton = page.locator('.mobile-menu-button');
    if (await mobileMenuButton.isVisible()) {
      await mobileMenuButton.click();
      await expect(page.locator('.mobile-menu')).toBeVisible();
    }
    
    // Check that content is properly responsive
    await expect(page.locator('.main-content')).toBeVisible();
    
    // Test form interactions on mobile
    await page.goto('/generate');
    await expect(page.locator('select[name="generator"]')).toBeVisible();
    await expect(page.locator('input[name="num_records"]')).toBeVisible();
  });

  test('should support accessibility features', async ({ page }) => {
    await page.goto('/');
    
    // Check for proper ARIA labels
    const navElement = page.locator('nav');
    await expect(navElement).toHaveAttribute('role', 'navigation');
    
    // Check for proper heading structure
    const headings = page.locator('h1, h2, h3, h4, h5, h6');
    const headingCount = await headings.count();
    expect(headingCount).toBeGreaterThan(0);
    
    // Check for alt text on images
    const images = page.locator('img');
    const imageCount = await images.count();
    
    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      await expect(img).toHaveAttribute('alt');
    }
    
    // Test keyboard navigation
    await page.keyboard.press('Tab');
    const focusedElement = await page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });
});

// Performance tests
test.describe('Performance Tests', () => {
  test('should load pages within acceptable time limits', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/');
    
    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(3000); // 3 second load time limit
    
    // Check Core Web Vitals
    const metrics = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          resolve(entries);
        }).observe({ entryTypes: ['navigation'] });
      });
    });
    
    // Basic performance assertions
    expect(metrics).toBeDefined();
  });

  test('should handle concurrent users', async ({ browser }) => {
    const contexts = await Promise.all([
      browser.newContext(),
      browser.newContext(),
      browser.newContext(),
    ]);
    
    const pages = await Promise.all(
      contexts.map(context => context.newPage())
    );
    
    // Simulate concurrent access
    const promises = pages.map(async (page, index) => {
      await authenticateUser(page);
      await page.goto('/');
      await expect(page.locator('nav')).toBeVisible();
      return `User ${index + 1} loaded successfully`;
    });
    
    const results = await Promise.all(promises);
    expect(results).toHaveLength(3);
    
    // Cleanup
    await Promise.all(contexts.map(context => context.close()));
  });
});