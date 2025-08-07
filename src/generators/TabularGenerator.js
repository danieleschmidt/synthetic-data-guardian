/**
 * TabularGenerator - Generates synthetic tabular data using various backends
 */

import crypto from 'crypto';

export class TabularGenerator {
  constructor(backend, params = {}, logger) {
    this.backend = backend;
    this.params = params;
    this.logger = logger;
    this.initialized = false;
    this.model = null;
    this.schema = null;
  }

  async initialize(config) {
    try {
      this.logger.info('Initializing tabular generator', { backend: this.backend });

      this.schema = config.schema || {};

      // Initialize specific backend
      switch (this.backend.toLowerCase()) {
        case 'sdv':
        case 'gaussian_copula':
          await this.initializeSDV();
          break;
        case 'ctgan':
          await this.initializeCTGAN();
          break;
        case 'synthetic':
        case 'basic':
          await this.initializeBasicGenerator();
          break;
        default:
          throw new Error(`Unsupported tabular generator backend: ${this.backend}`);
      }

      this.initialized = true;
      this.logger.info('Tabular generator initialized successfully', { backend: this.backend });
    } catch (error) {
      this.logger.error('Tabular generator initialization failed', {
        backend: this.backend,
        error: error.message,
      });
      throw error;
    }
  }

  async initializeSDV() {
    // In a real implementation, this would initialize SDV/Gaussian Copula
    // For now, we'll use a statistical approach that mimics SDV behavior
    this.model = {
      type: 'gaussian_copula',
      correlations: {},
      distributions: {},
      constraints: [],
    };

    this.logger.info('SDV/Gaussian Copula model initialized');
  }

  async initializeCTGAN() {
    // In a real implementation, this would initialize CTGAN
    // For now, we'll use a GAN-like statistical approach
    this.model = {
      type: 'ctgan',
      epochs: this.params.epochs || 300,
      batchSize: this.params.batchSize || 500,
      generator: null,
      discriminator: null,
    };

    this.logger.info('CTGAN model initialized', {
      epochs: this.model.epochs,
      batchSize: this.model.batchSize,
    });
  }

  async initializeBasicGenerator() {
    // Basic statistical generator for development/testing
    this.model = {
      type: 'basic',
      fieldGenerators: {},
      correlations: {},
    };

    // Set up field generators based on schema
    this.setupFieldGenerators();

    this.logger.info('Basic generator initialized');
  }

  setupFieldGenerators() {
    for (const [fieldName, fieldDef] of Object.entries(this.schema)) {
      this.model.fieldGenerators[fieldName] = this.createFieldGenerator(fieldName, fieldDef);
    }
  }

  createFieldGenerator(fieldName, fieldDef) {
    const definition = typeof fieldDef === 'string' ? { type: fieldDef } : fieldDef;
    const type = definition.type || 'string';

    switch (type.toLowerCase()) {
      case 'integer':
      case 'int':
        return {
          type: 'integer',
          min: definition.min || 0,
          max: definition.max || 100,
          generator: () => Math.floor(Math.random() * (definition.max - definition.min + 1)) + definition.min,
        };

      case 'float':
      case 'number':
        return {
          type: 'float',
          min: definition.min || 0,
          max: definition.max || 100,
          precision: definition.precision || 2,
          generator: () => {
            const value = Math.random() * (definition.max - definition.min) + definition.min;
            return Math.round(value * Math.pow(10, definition.precision)) / Math.pow(10, definition.precision);
          },
        };

      case 'string':
      case 'text':
        return {
          type: 'string',
          length: definition.length || 10,
          pattern: definition.pattern,
          generator: () => {
            if (definition.enum) {
              return definition.enum[Math.floor(Math.random() * definition.enum.length)];
            }
            return this.generateRandomString(definition.length || 10);
          },
        };

      case 'email':
        const domains = definition.domains || ['example.com', 'test.org', 'demo.net'];
        return {
          type: 'email',
          domains: domains,
          generator: () => {
            const username = this.generateRandomString(8);
            const domain = domains[Math.floor(Math.random() * domains.length)];
            return `${username}@${domain}`;
          },
        };

      case 'boolean':
      case 'bool':
        const probability = definition.probability || 0.5;
        return {
          type: 'boolean',
          probability: probability,
          generator: () => Math.random() < probability,
        };

      case 'date':
      case 'datetime':
        return {
          type: 'datetime',
          start: definition.start ? new Date(definition.start) : new Date('2020-01-01'),
          end: definition.end ? new Date(definition.end) : new Date(),
          generator: () => {
            const start = definition.start ? new Date(definition.start) : new Date('2020-01-01');
            const end = definition.end ? new Date(definition.end) : new Date();
            const randomTime = start.getTime() + Math.random() * (end.getTime() - start.getTime());
            return new Date(randomTime).toISOString();
          },
        };

      case 'uuid':
        return {
          type: 'uuid',
          generator: () => crypto.randomUUID(),
        };

      case 'categorical':
        const categories = definition.categories || definition.enum || ['A', 'B', 'C'];
        const weights = definition.weights;
        return {
          type: 'categorical',
          categories: categories,
          weights: weights,
          generator: () => {
            if (weights && weights.length === categories.length) {
              return this.weightedRandomChoice(categories, weights);
            }
            return categories[Math.floor(Math.random() * categories.length)];
          },
        };

      case 'json':
        return {
          type: 'json',
          template: definition.template || {},
          generator: () => {
            if (definition.template) {
              return this.generateFromTemplate(definition.template);
            }
            return { id: Math.floor(Math.random() * 1000), value: this.generateRandomString(5) };
          },
        };

      default:
        return {
          type: 'string',
          generator: () => this.generateRandomString(10),
        };
    }
  }

  generateRandomString(length) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }

  weightedRandomChoice(choices, weights) {
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    let random = Math.random() * totalWeight;

    for (let i = 0; i < choices.length; i++) {
      random -= weights[i];
      if (random <= 0) {
        return choices[i];
      }
    }

    return choices[choices.length - 1];
  }

  generateFromTemplate(template) {
    const result = {};

    for (const [key, value] of Object.entries(template)) {
      if (typeof value === 'string' && value.startsWith('$')) {
        // Template variable
        const varType = value.substring(1);
        result[key] = this.generateTemplateValue(varType);
      } else if (typeof value === 'object' && value !== null) {
        result[key] = this.generateFromTemplate(value);
      } else {
        result[key] = value;
      }
    }

    return result;
  }

  generateTemplateValue(type) {
    switch (type) {
      case 'int':
        return Math.floor(Math.random() * 1000);
      case 'float':
        return Math.round(Math.random() * 1000 * 100) / 100;
      case 'string':
        return this.generateRandomString(8);
      case 'bool':
        return Math.random() < 0.5;
      case 'date':
        return new Date().toISOString();
      default:
        return null;
    }
  }

  async generate(options = {}) {
    if (!this.initialized) {
      throw new Error('Generator not initialized');
    }

    const { numRecords, seed, conditions = {}, onProgress = () => {} } = options;

    this.logger.info('Starting tabular data generation', {
      numRecords,
      seed,
      backend: this.backend,
    });

    // Set random seed if provided
    if (seed !== undefined) {
      this.setSeed(seed);
    }

    const data = [];
    const batchSize = Math.min(1000, numRecords);
    let generated = 0;

    while (generated < numRecords) {
      const currentBatchSize = Math.min(batchSize, numRecords - generated);
      const batch = await this.generateBatch(currentBatchSize, conditions);

      data.push(...batch);
      generated += currentBatchSize;

      // Report progress
      const progress = (generated / numRecords) * 100;
      onProgress(progress);

      // Allow other operations in event loop
      await new Promise(resolve => setImmediate(resolve));
    }

    // Apply post-generation processing
    const processedData = this.postProcessData(data, options);

    this.logger.info('Tabular data generation completed', {
      recordsGenerated: processedData.length,
      backend: this.backend,
    });

    return processedData;
  }

  setSeed(seed) {
    // For deterministic generation
    this.randomState = this.createSeededRandom(seed);
  }

  createSeededRandom(seed) {
    // Simple seeded random number generator (Linear Congruential Generator)
    let current = seed;
    return {
      next: () => {
        current = (current * 1103515245 + 12345) & 0x7fffffff;
        return current / 0x7fffffff;
      },
    };
  }

  async generateBatch(batchSize, conditions) {
    const batch = [];

    for (let i = 0; i < batchSize; i++) {
      const record = this.generateRecord(conditions);
      batch.push(record);
    }

    return batch;
  }

  generateRecord(conditions = {}) {
    const record = {};

    // Generate base record
    for (const [fieldName, generator] of Object.entries(this.model.fieldGenerators)) {
      if (conditions[fieldName] !== undefined) {
        // Apply condition
        record[fieldName] = this.applyCondition(conditions[fieldName], generator);
      } else {
        // Generate normally
        record[fieldName] = this.randomState
          ? this.generateWithSeed(generator, this.randomState)
          : generator.generator();
      }
    }

    // Apply correlations if defined
    this.applyCorrelations(record);

    return record;
  }

  generateWithSeed(generator, randomState) {
    // Use seeded random for deterministic generation
    const originalRandom = Math.random;
    Math.random = randomState.next;

    try {
      return generator.generator();
    } finally {
      Math.random = originalRandom;
    }
  }

  applyCondition(condition, generator) {
    if (typeof condition === 'function') {
      return condition(generator);
    } else if (typeof condition === 'object') {
      // Handle range conditions
      if (condition.min !== undefined || condition.max !== undefined) {
        const min = condition.min !== undefined ? condition.min : generator.min;
        const max = condition.max !== undefined ? condition.max : generator.max;

        if (generator.type === 'integer') {
          return Math.floor(Math.random() * (max - min + 1)) + min;
        } else if (generator.type === 'float') {
          return Math.random() * (max - min) + min;
        }
      }

      // Handle enum conditions
      if (condition.enum) {
        return condition.enum[Math.floor(Math.random() * condition.enum.length)];
      }
    } else {
      // Direct value
      return condition;
    }

    return generator.generator();
  }

  applyCorrelations(record) {
    // Apply simple correlation logic
    // In a real implementation, this would use proper statistical correlations

    if (this.model.correlations) {
      for (const [field1, correlatedFields] of Object.entries(this.model.correlations)) {
        if (record[field1] !== undefined) {
          for (const [field2, correlation] of Object.entries(correlatedFields)) {
            if (record[field2] !== undefined && Math.abs(correlation) > 0.5) {
              // Simple correlation: adjust field2 based on field1
              if (typeof record[field1] === 'number' && typeof record[field2] === 'number') {
                const adjustment = (record[field1] - 50) * correlation * 0.1;
                record[field2] = Math.max(0, record[field2] + adjustment);
              }
            }
          }
        }
      }
    }
  }

  postProcessData(data, options) {
    // Apply any post-generation processing
    let processedData = data;

    // Remove duplicates if requested
    if (options.removeDuplicates) {
      processedData = this.removeDuplicates(processedData);
    }

    // Apply sampling if requested
    if (options.sampleRatio && options.sampleRatio < 1) {
      const sampleSize = Math.floor(processedData.length * options.sampleRatio);
      processedData = this.sampleData(processedData, sampleSize);
    }

    // Sort if requested
    if (options.sortBy) {
      processedData = this.sortData(processedData, options.sortBy);
    }

    return processedData;
  }

  removeDuplicates(data) {
    const seen = new Set();
    return data.filter(record => {
      const key = JSON.stringify(record);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  sampleData(data, sampleSize) {
    if (sampleSize >= data.length) {
      return data;
    }

    const sampled = [];
    const indices = new Set();

    while (sampled.length < sampleSize) {
      const index = Math.floor(Math.random() * data.length);
      if (!indices.has(index)) {
        indices.add(index);
        sampled.push(data[index]);
      }
    }

    return sampled;
  }

  sortData(data, sortBy) {
    if (typeof sortBy === 'string') {
      return data.sort((a, b) => {
        const aVal = a[sortBy];
        const bVal = b[sortBy];
        if (aVal < bVal) return -1;
        if (aVal > bVal) return 1;
        return 0;
      });
    } else if (Array.isArray(sortBy)) {
      return data.sort((a, b) => {
        for (const field of sortBy) {
          const aVal = a[field];
          const bVal = b[field];
          if (aVal < bVal) return -1;
          if (aVal > bVal) return 1;
        }
        return 0;
      });
    }

    return data;
  }

  async fit(trainingData) {
    // Train the model on real data (for SDV/CTGAN)
    this.logger.info('Training tabular generator on real data', {
      records: trainingData.length,
      backend: this.backend,
    });

    switch (this.backend.toLowerCase()) {
      case 'sdv':
      case 'gaussian_copula':
        await this.fitSDV(trainingData);
        break;
      case 'ctgan':
        await this.fitCTGAN(trainingData);
        break;
      default:
        // Basic generator learns simple statistics
        await this.fitBasic(trainingData);
    }

    this.logger.info('Model training completed');
  }

  async fitSDV(trainingData) {
    // Learn correlations and distributions
    if (trainingData.length === 0) return;

    const fields = Object.keys(trainingData[0]);

    // Calculate correlations
    for (const field1 of fields) {
      if (typeof trainingData[0][field1] === 'number') {
        this.model.correlations[field1] = {};

        for (const field2 of fields) {
          if (field1 !== field2 && typeof trainingData[0][field2] === 'number') {
            this.model.correlations[field1][field2] = this.calculateCorrelation(
              trainingData.map(r => r[field1]),
              trainingData.map(r => r[field2]),
            );
          }
        }
      }
    }

    // Learn distributions
    for (const field of fields) {
      const values = trainingData.map(r => r[field]);
      this.model.distributions[field] = this.analyzeDistribution(values);
    }
  }

  async fitCTGAN(trainingData) {
    // In a real implementation, this would train a CTGAN model
    // For now, we'll learn basic patterns
    await this.fitBasic(trainingData);
  }

  async fitBasic(trainingData) {
    if (trainingData.length === 0) return;

    const fields = Object.keys(trainingData[0]);

    // Update field generators based on training data
    for (const field of fields) {
      const values = trainingData.map(r => r[field]);
      const nonNullValues = values.filter(v => v !== null && v !== undefined);

      if (nonNullValues.length === 0) continue;

      const sampleValue = nonNullValues[0];

      if (typeof sampleValue === 'number') {
        this.model.fieldGenerators[field] = {
          type: Number.isInteger(sampleValue) ? 'integer' : 'float',
          min: Math.min(...nonNullValues),
          max: Math.max(...nonNullValues),
          mean: nonNullValues.reduce((sum, v) => sum + v, 0) / nonNullValues.length,
          generator: () => {
            const min = Math.min(...nonNullValues);
            const max = Math.max(...nonNullValues);
            return Number.isInteger(sampleValue)
              ? Math.floor(Math.random() * (max - min + 1)) + min
              : Math.random() * (max - min) + min;
          },
        };
      } else if (typeof sampleValue === 'string') {
        const uniqueValues = [...new Set(nonNullValues)];
        this.model.fieldGenerators[field] = {
          type: 'categorical',
          categories: uniqueValues,
          generator: () => uniqueValues[Math.floor(Math.random() * uniqueValues.length)],
        };
      } else if (typeof sampleValue === 'boolean') {
        const trueCount = nonNullValues.filter(v => v === true).length;
        const probability = trueCount / nonNullValues.length;
        this.model.fieldGenerators[field] = {
          type: 'boolean',
          probability: probability,
          generator: () => Math.random() < probability,
        };
      }
    }
  }

  calculateCorrelation(x, y) {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;

    const meanX = x.slice(0, n).reduce((sum, val) => sum + val, 0) / n;
    const meanY = y.slice(0, n).reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let sumSqX = 0;
    let sumSqY = 0;

    for (let i = 0; i < n; i++) {
      const deltaX = x[i] - meanX;
      const deltaY = y[i] - meanY;
      numerator += deltaX * deltaY;
      sumSqX += deltaX * deltaX;
      sumSqY += deltaY * deltaY;
    }

    const denominator = Math.sqrt(sumSqX * sumSqY);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  analyzeDistribution(values) {
    const nonNullValues = values.filter(v => v !== null && v !== undefined);

    if (nonNullValues.length === 0) {
      return { type: 'empty' };
    }

    const sampleValue = nonNullValues[0];

    if (typeof sampleValue === 'number') {
      const mean = nonNullValues.reduce((sum, v) => sum + v, 0) / nonNullValues.length;
      const variance = nonNullValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nonNullValues.length;

      return {
        type: 'numeric',
        mean: mean,
        variance: variance,
        min: Math.min(...nonNullValues),
        max: Math.max(...nonNullValues),
      };
    } else {
      const uniqueValues = [...new Set(nonNullValues)];
      const frequencies = {};

      for (const value of nonNullValues) {
        frequencies[value] = (frequencies[value] || 0) + 1;
      }

      return {
        type: 'categorical',
        uniqueValues: uniqueValues,
        frequencies: frequencies,
        cardinality: uniqueValues.length,
      };
    }
  }

  getModelInfo() {
    return {
      backend: this.backend,
      initialized: this.initialized,
      modelType: this.model?.type,
      fieldCount: Object.keys(this.model?.fieldGenerators || {}).length,
      hasCorrelations: Object.keys(this.model?.correlations || {}).length > 0,
    };
  }

  async cleanup() {
    this.logger.info('Cleaning up tabular generator');
    this.model = null;
    this.initialized = false;
  }
}
