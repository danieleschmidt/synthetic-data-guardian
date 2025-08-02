# 🧪 Testing Guide

Comprehensive testing framework for the Synthetic Data Guardian project.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Utilities](#test-utilities)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

## 🎯 Overview

Our testing strategy ensures comprehensive coverage across all components:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Full workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability and penetration testing
- **Contract Tests**: API contract validation

### Coverage Goals

- **Unit Tests**: >90% line coverage
- **Integration Tests**: All critical paths
- **E2E Tests**: Complete user workflows
- **Performance Tests**: All API endpoints
- **Security Tests**: All input vectors

## 📚 Test Categories

### Unit Tests

Test individual functions and classes in isolation.

**Location**: `tests/unit/`

**Frameworks**:
- Python: `pytest`
- TypeScript: `Jest`

**Examples**:
```bash
# Python unit tests
pytest tests/unit/test_generators.py -v

# TypeScript unit tests
npm run test -- --testPathPattern="unit"
```

### Integration Tests

Test component interactions and external service integrations.

**Location**: `tests/integration/`

**Coverage**:
- Database operations
- API integrations
- Service communications
- External dependencies

**Examples**:
```bash
# Python integration tests
pytest tests/integration/ -m integration

# TypeScript integration tests
npm run test:integration
```

### End-to-End Tests

Test complete user workflows and system behavior.

**Location**: `tests/e2e/`

**Framework**: Playwright

**Coverage**:
- Full generation pipelines
- Validation workflows
- User interfaces
- Multi-service interactions

**Examples**:
```bash
# Run all E2E tests
npm run test:e2e

# Run specific test
npx playwright test generation-workflow.spec.ts
```

### Performance Tests

Test system performance under various loads.

**Location**: `tests/performance/`

**Frameworks**:
- k6 (load testing)
- pytest-benchmark (Python)
- Custom utilities

**Coverage**:
- API response times
- Throughput testing
- Memory usage
- Concurrency limits

**Examples**:
```bash
# Run performance tests
npm run test:performance

# Python benchmarks
pytest tests/performance/ --benchmark-only
```

### Security Tests

Test for security vulnerabilities and compliance.

**Location**: `tests/security/`

**Coverage**:
- Input validation
- SQL injection
- XSS attacks
- Authentication/authorization
- Data privacy

**Examples**:
```bash
# Run security tests
pytest tests/security/ -m security

# Manual security scan
make security-scan
```

### Contract Tests

Test API contracts between services.

**Location**: `tests/contract/`

**Framework**: Pact

**Coverage**:
- API request/response formats
- Service contracts
- Version compatibility

## 🚀 Running Tests

### Prerequisites

```bash
# Install dependencies
make install

# Start test services
docker-compose -f docker-compose.test.yml up -d
```

### Quick Commands

```bash
# Run all tests
make test

# Run specific test types
make test-python          # Python tests only
make test-node            # Node.js tests only
make test-unit            # Unit tests only
make test-integration     # Integration tests only
make test-e2e            # End-to-end tests only
make test-performance    # Performance tests only

# Generate coverage report
make test-coverage
```

### Detailed Commands

#### Python Tests

```bash
# All Python tests
pytest

# Specific test file
pytest tests/unit/test_generators.py

# With coverage
pytest --cov=src --cov-report=html

# Specific markers
pytest -m "unit and not slow"
pytest -m "integration"
pytest -m "security"

# Parallel execution
pytest -n auto

# Verbose output
pytest -v -s

# Stop on first failure
pytest -x
```

#### TypeScript Tests

```bash
# All TypeScript tests
npm run test

# Watch mode
npm run test:watch

# Specific test pattern
npm test -- --testNamePattern="generator"

# Coverage report
npm run test:coverage

# Debug mode
npm test -- --runInBand --detectOpenHandles
```

#### E2E Tests

```bash
# All E2E tests
npx playwright test

# Specific browser
npx playwright test --project=chromium

# Headed mode (visible browser)
npx playwright test --headed

# Debug mode
npx playwright test --debug

# Generate test report
npx playwright show-report
```

### Environment Configuration

#### Test Environment Variables

```bash
# Test database
TEST_DATABASE_URL=postgresql://test:test@localhost:5433/test_db

# Test Redis
TEST_REDIS_URL=redis://localhost:6380

# Test API
TEST_API_URL=http://localhost:8080

# Performance settings
RUN_PERFORMANCE_TESTS=true
PERFORMANCE_THRESHOLD_MS=500

# Security settings
RUN_SECURITY_TESTS=true
SECURITY_SCAN_DEPTH=thorough
```

## ✍️ Writing Tests

### Python Test Structure

```python
import pytest
from tests.python.test_utils import TestDataFactory, ValidationHelper

class TestDataGenerator:
    """Test class for data generation functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data"""
        return TestDataFactory.create_sample_dataset(100)
    
    def test_generator_initialization(self):
        """Test generator can be initialized"""
        from synthetic_guardian.generators import SDVGenerator
        generator = SDVGenerator()
        assert generator is not None
    
    def test_data_generation(self, sample_data):
        """Test data generation produces valid output"""
        from synthetic_guardian.generators import SDVGenerator
        
        generator = SDVGenerator()
        generator.fit(sample_data)
        
        synthetic_data = generator.generate(num_rows=50)
        
        # Validate output
        ValidationHelper.assert_dataframe_shape(synthetic_data, 50, len(sample_data.columns))
        ValidationHelper.assert_no_null_values(synthetic_data)
    
    @pytest.mark.performance
    def test_generation_performance(self, sample_data, performance_tester):
        """Test generation performance meets requirements"""
        from synthetic_guardian.generators import SDVGenerator
        
        generator = SDVGenerator()
        generator.fit(sample_data)
        
        def generate_data():
            return generator.generate(num_rows=1000)
        
        result, execution_time = performance_tester.measure_execution_time(generate_data)
        ValidationHelper.assert_response_time(execution_time, 5000)  # 5 second threshold
    
    @pytest.mark.security
    def test_input_validation(self, security_tester):
        """Test input validation against malicious payloads"""
        from synthetic_guardian.api.endpoints import generate_data
        
        test_data = {
            'pipeline': 'test_pipeline',
            'num_records': 1000,
            'format': 'json'
        }
        
        results = security_tester.test_input_validation(generate_data, test_data)
        
        # Assert no vulnerabilities found
        for category, tests in results.items():
            vulnerable_tests = [t for t in tests if t['result'] == 'vulnerable']
            assert len(vulnerable_tests) == 0, f"Found {category} vulnerabilities: {vulnerable_tests}"
```

### TypeScript Test Structure

```typescript
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { TestDataGenerator, APITestHelper, PerformanceTestHelper } from '../utils/test-utils';

describe('DataGenerationService', () => {
  let apiHelper: APITestHelper;
  
  beforeEach(() => {
    apiHelper = new APITestHelper();
  });
  
  afterEach(() => {
    // Cleanup
  });
  
  describe('generateSyntheticData', () => {
    it('should generate synthetic data successfully', async () => {
      const request = TestDataGenerator.generateMockGenerationRequest();
      
      const response = await apiHelper.post('/api/v1/generate', request);
      
      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('id');
      expect(response.data).toHaveProperty('status', 'completed');
      expect(response.data.quality_score).toBeGreaterThan(0.8);
    });
    
    it('should handle invalid input gracefully', async () => {
      const invalidRequest = { invalid: 'data' };
      
      const response = await apiHelper.post('/api/v1/generate', invalidRequest);
      
      expect(response.status).toBe(400);
      expect(response.data).toHaveProperty('error');
    });
    
    it('should meet performance requirements', async () => {
      const request = TestDataGenerator.generateMockGenerationRequest();
      
      const { result, duration } = await PerformanceTestHelper.measureExecutionTime(
        () => apiHelper.post('/api/v1/generate', request)
      );
      
      expect(duration).toBeLessThan(500); // 500ms threshold
      expect(result.status).toBe(200);
    });
  });
  
  describe('load testing', () => {
    it('should handle concurrent requests', async () => {
      const request = TestDataGenerator.generateMockGenerationRequest();
      
      const testFunction = () => apiHelper.post('/api/v1/generate', request);
      
      const results = await PerformanceTestHelper.loadTest(testFunction, {
        concurrency: 10,
        duration: 30000, // 30 seconds
        rampUp: 5000     // 5 second ramp-up
      });
      
      expect(results.successRate).toBeGreaterThan(95);
      expect(results.averageResponseTime).toBeLessThan(1000);
      expect(results.throughput).toBeGreaterThan(10); // 10 req/s
    });
  });
});
```

### E2E Test Structure

```typescript
import { test, expect } from '@playwright/test';

test.describe('Synthetic Data Generation Workflow', () => {
  test('complete generation workflow', async ({ page }) => {
    // Navigate to application
    await page.goto('/');
    
    // Create new generation pipeline
    await page.click('[data-testid="new-pipeline"]');
    await page.fill('[data-testid="pipeline-name"]', 'test-pipeline');
    await page.selectOption('[data-testid="generator-type"]', 'sdv');
    
    // Configure pipeline
    await page.fill('[data-testid="num-records"]', '1000');
    await page.selectOption('[data-testid="output-format"]', 'json');
    
    // Add validation
    await page.check('[data-testid="statistical-validation"]');
    await page.check('[data-testid="privacy-validation"]');
    
    // Start generation
    await page.click('[data-testid="generate-button"]');
    
    // Wait for completion
    await page.waitForSelector('[data-testid="generation-complete"]', { timeout: 60000 });
    
    // Verify results
    const qualityScore = await page.textContent('[data-testid="quality-score"]');
    expect(parseFloat(qualityScore!)).toBeGreaterThan(0.8);
    
    const privacyScore = await page.textContent('[data-testid="privacy-score"]');
    expect(parseFloat(privacyScore!)).toBeGreaterThan(0.9);
    
    // Download results
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="download-button"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toMatch(/synthetic_data_.*\.json/);
  });
});
```

## 🛠️ Test Utilities

### Available Utilities

#### Python Utilities (`tests/python/test_utils.py`)

- `TestDataFactory`: Generate test datasets
- `MockHelper`: Create mocks for external services
- `PerformanceTester`: Performance testing utilities
- `SecurityTester`: Security testing utilities
- `ValidationHelper`: Common validation assertions

#### TypeScript Utilities (`tests/utils/test-utils.ts`)

- `TestDataGenerator`: Generate test data
- `APITestHelper`: HTTP API testing
- `PerformanceTestHelper`: Load testing utilities
- `SecurityTestHelper`: Security testing utilities
- `MockFactory`: Create mocks and stubs

### Configuration

Test configuration is centralized in:
- `tests/config/test-config.ts` (TypeScript)
- `tests/python/test_utils.py` (Python)

## ⚡ Performance Testing

### Load Testing with k6

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.01'],   // Error rate under 1%
  },
};

export default function () {
  const payload = JSON.stringify({
    pipeline: 'test_pipeline',
    num_records: 1000,
    format: 'json'
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const response = http.post('http://localhost:8080/api/v1/generate', payload, params);
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

### Python Performance Testing

```python
import pytest
from tests.python.test_utils import PerformanceTester

def test_generation_performance():
    """Test data generation performance"""
    def generate_data():
        # Your generation logic here
        pass
    
    # Single execution test
    result, execution_time = PerformanceTester.measure_execution_time(generate_data)
    assert execution_time < 1000  # 1 second threshold
    
    # Load test
    load_results = PerformanceTester.load_test(
        generate_data,
        num_requests=100,
        concurrency=10
    )
    
    assert load_results['success_rate'] > 95
    assert load_results['average_response_time'] < 500
```

## 🔒 Security Testing

### Automated Security Tests

```python
import pytest
from tests.python.test_utils import SecurityTester

def test_sql_injection_protection():
    """Test SQL injection protection"""
    def vulnerable_function(user_input):
        # Function that might be vulnerable
        return process_user_input(user_input)
    
    payloads = SecurityTester.generate_sql_injection_payloads()
    
    for payload in payloads:
        try:
            result = vulnerable_function(payload)
            # Should not execute malicious SQL
            assert not contains_sensitive_data(result)
        except Exception:
            # Exception is acceptable - indicates protection
            pass

def test_input_validation():
    """Test comprehensive input validation"""
    test_data = {'name': 'test', 'email': 'test@example.com'}
    
    results = SecurityTester.test_input_validation(api_endpoint, test_data)
    
    # Assert no vulnerabilities found
    for category, tests in results.items():
        vulnerable = [t for t in tests if t['result'] == 'vulnerable']
        assert len(vulnerable) == 0, f"Found {category} vulnerabilities"
```

### Manual Security Testing

```bash
# Run OWASP ZAP scan
zap-baseline.py -t http://localhost:8080

# Run Bandit security scan
bandit -r src/

# Run npm audit
npm audit

# Run Snyk scan
snyk test
```

## 🔄 CI/CD Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: make ci-install
      
      - name: Run tests
        run: make ci-test
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Reporting

Test results are reported to:
- **Coverage**: Codecov
- **Performance**: Custom dashboards
- **Security**: GitHub Security tab

## 🎯 Best Practices

### General Guidelines

1. **Test Pyramid**: More unit tests, fewer E2E tests
2. **Fast Feedback**: Unit tests run quickly
3. **Isolated Tests**: Tests don't depend on each other
4. **Deterministic**: Tests produce consistent results
5. **Readable**: Tests are self-documenting

### Writing Good Tests

#### DO ✅

- Write descriptive test names
- Use fixtures for common setup
- Test both happy path and edge cases
- Mock external dependencies
- Assert specific behaviors, not implementation

#### DON'T ❌

- Write flaky tests
- Test implementation details
- Use real external services in unit tests
- Share state between tests
- Write overly complex tests

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── e2e/           # Full workflow tests
├── performance/   # Load and stress tests
├── security/      # Security and vulnerability tests
├── contract/      # API contract tests
├── fixtures/      # Test data and mocks
├── utils/         # Test utilities and helpers
└── config/        # Test configuration
```

### Performance Considerations

- **Unit tests**: < 1 second each
- **Integration tests**: < 10 seconds each
- **E2E tests**: < 5 minutes each
- **Total test suite**: < 15 minutes

### Continuous Improvement

- Monitor test execution times
- Review flaky tests weekly
- Update test data regularly
- Refactor slow tests
- Add tests for new bugs

## 📚 Additional Resources

- [Jest Documentation](https://jestjs.io/docs)
- [Pytest Documentation](https://docs.pytest.org/)
- [Playwright Documentation](https://playwright.dev/)
- [k6 Documentation](https://k6.io/docs/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)

## 🆘 Troubleshooting

### Common Issues

#### Tests are slow
- Check database connections
- Review test isolation
- Use appropriate test doubles

#### Flaky tests
- Check for race conditions
- Review async handling
- Add proper waits/timeouts

#### Coverage gaps
- Review test coverage reports
- Add tests for uncovered branches
- Test error conditions

#### CI failures
- Check environment differences
- Review service dependencies
- Verify test data consistency

### Getting Help

1. Check test logs: `make test-verbose`
2. Run specific test: `pytest tests/unit/test_specific.py -v`
3. Debug mode: `pytest --pdb`
4. Create GitHub issue with:
   - Test output
   - Environment details
   - Steps to reproduce