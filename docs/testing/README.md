# Testing Framework Documentation

This document describes the comprehensive testing framework for the Synthetic Data Guardian project.

## Testing Strategy

Our testing strategy follows a multi-layered approach to ensure data quality, security, and reliability:

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete user workflows
4. **Performance Tests** - Validate performance requirements
5. **Security Tests** - Verify security controls
6. **Contract Tests** - Ensure API compatibility
7. **Mutation Tests** - Validate test quality

## Test Categories

### Data Generation Tests
- Generator functionality (SDV, CTGAN, etc.)
- Schema validation
- Data quality metrics
- Error handling

### Privacy & Security Tests
- Watermarking verification
- Privacy preservation validation
- Re-identification risk assessment
- Cryptographic operations

### Compliance Tests
- GDPR compliance validation
- HIPAA safe harbor verification
- Audit trail completeness
- Data lineage accuracy

### Performance Tests
- Generation throughput
- Memory usage
- Scalability limits
- Response time benchmarks

## Running Tests

### All Tests
```bash
make test
```

### Unit Tests
```bash
# Python
pytest tests/python -m unit
# JavaScript/TypeScript
npm run test:unit
```

### Integration Tests
```bash
# Python
pytest tests/python -m integration
# JavaScript/TypeScript
npm run test:integration
```

### End-to-End Tests
```bash
npm run test:e2e
```

### Performance Tests
```bash
pytest tests/python -m performance
npm run test:performance
```

### Security Tests
```bash
pytest tests/python -m security
```

### Contract Tests
```bash
npm run test:contract
```

### Mutation Tests
```bash
npm run test:mutation
```

## Test Configuration

### Jest (JavaScript/TypeScript)
- **Unit Tests**: `jest.config.js`
- **Integration Tests**: `jest.integration.config.js`
- **Mutation Tests**: `jest.mutation.config.js`

### Pytest (Python)
- **Configuration**: `pytest.ini`
- **Fixtures**: `tests/python/conftest.py`

### Playwright (E2E)
- **Configuration**: `playwright.config.ts`
- **Setup**: `tests/e2e/global-setup.ts`

## Test Markers

### Python Test Markers
```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.security      # Security tests
@pytest.mark.privacy       # Privacy tests
@pytest.mark.performance   # Performance tests
@pytest.mark.compliance    # Compliance tests
@pytest.mark.slow          # Slow running tests
@pytest.mark.requires_gpu  # Tests requiring GPU
@pytest.mark.requires_network  # Tests requiring network
```

### JavaScript Test Categories
```javascript
describe.unit()       // Unit tests
describe.integration() // Integration tests
describe.e2e()        // End-to-end tests
```

## Coverage Requirements

### Minimum Coverage Thresholds
- **Unit Tests**: 90%
- **Integration Tests**: 80%
- **Overall**: 85%

### Coverage Reports
- HTML: `coverage/html/index.html`
- XML: `coverage/coverage.xml`
- JSON: `coverage/coverage.json`

## Test Data Management

### Fixtures
- **Static Data**: `tests/fixtures/`
- **Generated Data**: Created dynamically in tests
- **Cleanup**: Automatic cleanup after test completion

### Test Databases
- **Unit Tests**: In-memory SQLite
- **Integration Tests**: Docker containers
- **E2E Tests**: Full test environment

## Performance Benchmarks

### Generation Performance
- **1K Records**: < 5 seconds
- **10K Records**: < 30 seconds
- **100K Records**: < 5 minutes

### API Performance
- **Response Time**: < 500ms (95th percentile)
- **Throughput**: > 100 requests/second
- **Memory Usage**: < 2GB per worker

## Security Testing

### Automated Scans
- **Dependency Scanning**: Snyk, Safety
- **SAST**: Bandit, ESLint Security
- **Container Scanning**: Trivy
- **Secret Detection**: detect-secrets

### Manual Testing
- **Penetration Testing**: Quarterly
- **Security Code Review**: All PRs
- **Threat Modeling**: Architecture changes

## Continuous Integration

### GitHub Actions
- **PR Validation**: All test suites
- **Nightly Tests**: Full regression suite
- **Performance Tests**: Weekly benchmarks
- **Security Scans**: Daily automated scans

### Test Reporting
- **JUnit XML**: CI integration
- **Coverage Reports**: Codecov
- **Performance Reports**: Dashboard
- **Security Reports**: GitHub Security tab

## Best Practices

### Writing Tests
1. **Arrange-Act-Assert** pattern
2. **Descriptive test names**
3. **Independent tests** (no order dependency)
4. **Proper cleanup** (fixtures, databases)
5. **Mock external dependencies**

### Test Organization
1. **Mirror source structure**
2. **Group related tests**
3. **Use descriptive describe blocks**
4. **Separate unit/integration/e2e**

### Data Generation Testing
1. **Test with synthetic data**
2. **Validate statistical properties**
3. **Check privacy preservation**
4. **Verify watermark integrity**
5. **Test error conditions**

## Troubleshooting

### Common Issues
1. **Flaky Tests**: Use proper waits, not sleeps
2. **Memory Leaks**: Ensure proper cleanup
3. **Timeout Issues**: Increase timeouts for slow operations
4. **Docker Issues**: Check container status

### Debug Mode
```bash
# Python
pytest --pdb -s
# JavaScript
npm run test:debug
```

### Logging
- **Test Logs**: `logs/test.log`
- **Debug Level**: Set `LOG_LEVEL=debug`
- **Verbose Output**: Use `-v` flag

## Metrics and Monitoring

### Test Metrics
- **Test Execution Time**
- **Test Success Rate**
- **Coverage Trends**
- **Flaky Test Detection**

### Performance Metrics
- **Generation Throughput**
- **Memory Usage Patterns**
- **API Response Times**
- **Error Rates**

## Contributing

### Adding New Tests
1. Follow existing patterns
2. Include appropriate markers
3. Add documentation for complex tests
4. Ensure proper cleanup
5. Update this documentation

### Test Reviews
1. **Functionality**: Does it test the right thing?
2. **Coverage**: Are all paths tested?
3. **Performance**: Does it run efficiently?
4. **Reliability**: Is it deterministic?
5. **Maintainability**: Is it easy to understand?