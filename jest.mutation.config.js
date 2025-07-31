// Mutation Testing Configuration for Synthetic Data Guardian
// Helps identify untested code paths and improve test quality

module.exports = {
  testEnvironment: 'node',
  
  // Mutation testing specific configuration
  mutate: [
    'src/**/*.ts',
    'src/**/*.js',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts',
    '!src/**/*.spec.ts',
    '!src/**/index.ts'
  ],
  
  // Test files to run against mutations
  testRunner: 'jest',
  
  // Coverage thresholds for mutations
  thresholds: {
    high: 90,
    low: 70,
    break: 60
  },
  
  // Mutation operators to apply
  mutator: {
    plugins: [
      '@stryker-mutator/javascript-mutator',
      '@stryker-mutator/typescript-mutator'
    ],
    excludedMutations: [
      'StringLiteral',  // Avoid breaking error messages
      'BooleanLiteral', // Avoid breaking feature flags
      'ObjectLiteral'   // Avoid breaking configuration objects
    ]
  },
  
  // Performance configuration
  concurrency: 4,
  timeoutMS: 5000,
  timeoutFactor: 1.5,
  
  // Reporting
  reporters: [
    'progress',
    'clear-text',
    'html',
    'json'
  ],
  
  // File patterns to ignore
  ignore: [
    'node_modules/**/*',
    'dist/**/*',
    'coverage/**/*',
    'test-results/**/*',
    '**/*.config.js',
    '**/*.d.ts'
  ],
  
  // Incremental mutation testing
  incremental: true,
  incrementalFile: '.stryker-tmp/incremental.json'
};