/**
 * GenerationPipeline - Orchestrates synthetic data generation workflows
 */

import crypto from 'crypto';
import { EventEmitter } from 'events';
import { TabularGenerator } from '../generators/TabularGenerator.js';
import { TimeSeriesGenerator } from '../generators/TimeSeriesGenerator.js';
import { TextGenerator } from '../generators/TextGenerator.js';
import { ImageGenerator } from '../generators/ImageGenerator.js';
import { GraphGenerator } from '../generators/GraphGenerator.js';

export class GenerationPipeline extends EventEmitter {
  constructor(id, logger) {
    super();
    this.id = id || crypto.randomUUID();
    this.logger = logger;
    this.config = {};
    this.generator = null;
    this.validators = [];
    this.watermarkConfig = null;
    this.created = new Date().toISOString();
    this.lastUsed = null;
    this.initialized = false;
  }

  async configure(config) {
    try {
      this.logger.info('Configuring generation pipeline', { pipelineId: this.id, config: { ...config, data: '[REDACTED]' } });

      // Validate configuration
      this.validateConfig(config);

      // Store configuration
      this.config = { ...config };
      this.config.id = this.id;

      // Initialize generator
      await this.initializeGenerator();

      // Setup validators
      this.setupValidators();

      // Setup watermarking
      this.setupWatermarking();

      this.initialized = true;
      this.logger.info('Pipeline configuration completed', { pipelineId: this.id });

      this.emit('configured', this.id);

    } catch (error) {
      this.logger.error('Pipeline configuration failed', { 
        pipelineId: this.id, 
        error: error.message 
      });
      throw error;
    }
  }

  validateConfig(config) {
    if (!config) {
      throw new Error('Pipeline configuration is required');
    }

    if (!config.generator) {
      throw new Error('Generator type is required');
    }

    if (!config.dataType) {
      throw new Error('Data type is required');
    }

    const supportedDataTypes = ['tabular', 'timeseries', 'text', 'image', 'graph'];
    if (!supportedDataTypes.includes(config.dataType)) {
      throw new Error(`Unsupported data type: ${config.dataType}. Supported types: ${supportedDataTypes.join(', ')}`);
    }

    if (config.dataType === 'tabular' && !config.schema) {
      throw new Error('Schema is required for tabular data generation');
    }
  }

  async initializeGenerator() {
    const { generator, dataType, params = {} } = this.config;

    this.logger.info('Initializing generator', { 
      pipelineId: this.id, 
      generator, 
      dataType 
    });

    switch (dataType) {
      case 'tabular':
        this.generator = new TabularGenerator(generator, params, this.logger);
        break;
      case 'timeseries':
        this.generator = new TimeSeriesGenerator(generator, params, this.logger);
        break;
      case 'text':
        this.generator = new TextGenerator(generator, params, this.logger);
        break;
      case 'image':
        this.generator = new ImageGenerator(generator, params, this.logger);
        break;
      case 'graph':
        this.generator = new GraphGenerator(generator, params, this.logger);
        break;
      default:
        throw new Error(`Unsupported data type: ${dataType}`);
    }

    await this.generator.initialize(this.config);
  }

  setupValidators() {
    this.validators = [];

    if (this.config.validation?.enabled) {
      const validatorConfigs = this.config.validation.validators || [];
      
      for (const validatorConfig of validatorConfigs) {
        this.validators.push({
          type: validatorConfig.type || validatorConfig,
          config: validatorConfig.config || {},
          threshold: validatorConfig.threshold || 0.8
        });
      }
    }

    this.logger.info('Validators configured', { 
      pipelineId: this.id, 
      count: this.validators.length 
    });
  }

  setupWatermarking() {
    if (this.config.watermarking?.enabled) {
      this.watermarkConfig = {
        method: this.config.watermarking.method || 'statistical',
        strength: this.config.watermarking.strength || 0.8,
        key: this.config.watermarking.key || crypto.randomBytes(32).toString('hex'),
        message: this.config.watermarking.message || `pipeline:${this.id}`
      };

      this.logger.info('Watermarking configured', { 
        pipelineId: this.id, 
        method: this.watermarkConfig.method 
      });
    }
  }

  async generate(options = {}) {
    if (!this.initialized) {
      throw new Error('Pipeline not initialized. Call configure() first.');
    }

    const startTime = Date.now();
    this.lastUsed = new Date().toISOString();

    try {
      this.logger.info('Starting data generation', { 
        pipelineId: this.id,
        numRecords: options.numRecords,
        seed: options.seed
      });

      // Validate generation options
      this.validateGenerationOptions(options);

      // Setup progress tracking
      const progressCallback = options.onProgress || (() => {});
      progressCallback(0);

      // Prepare generation parameters
      const generationParams = {
        numRecords: options.numRecords,
        seed: options.seed,
        conditions: options.conditions,
        schema: this.config.schema,
        constraints: this.config.constraints,
        onProgress: (progress) => {
          progressCallback(Math.floor(progress * 0.8)); // Reserve 20% for post-processing
        }
      };

      progressCallback(5);

      // Generate synthetic data
      const rawData = await this.generator.generate(generationParams);

      progressCallback(80);

      // Post-process data
      const processedData = await this.postProcessData(rawData, options);

      progressCallback(90);

      // Create generation result
      const result = {
        data: processedData,
        metadata: {
          pipelineId: this.id,
          generator: this.config.generator,
          dataType: this.config.dataType,
          generationTime: Date.now() - startTime,
          recordCount: processedData?.length || 0,
          seed: options.seed,
          schema: this.config.schema
        },
        qualityScore: null, // Will be set by Guardian
        privacyScore: null, // Will be set by Guardian
        watermarkInfo: null // Will be set by Guardian if watermarking is enabled
      };

      progressCallback(100);

      this.logger.info('Data generation completed', {
        pipelineId: this.id,
        recordCount: result.metadata.recordCount,
        executionTime: result.metadata.generationTime
      });

      this.emit('generation.completed', this.id, result);

      return result;

    } catch (error) {
      this.logger.error('Data generation failed', {
        pipelineId: this.id,
        error: error.message,
        stack: error.stack
      });

      this.emit('generation.failed', this.id, error);
      throw error;
    }
  }

  validateGenerationOptions(options) {
    if (!options.numRecords || typeof options.numRecords !== 'number' || options.numRecords <= 0) {
      throw new Error('numRecords must be a positive number');
    }

    if (options.numRecords > 10000000) {
      throw new Error('numRecords cannot exceed 10,000,000');
    }

    if (options.seed !== undefined && (typeof options.seed !== 'number' || options.seed < 0)) {
      throw new Error('seed must be a non-negative number');
    }
  }

  async postProcessData(data, options) {
    if (!data) return data;

    let processedData = data;

    // Apply schema validation and type conversion
    if (this.config.schema && this.config.dataType === 'tabular') {
      processedData = this.applySchemaValidation(processedData);
    }

    // Apply business rules
    if (this.config.businessRules) {
      processedData = this.applyBusinessRules(processedData);
    }

    // Apply data transformations
    if (this.config.transformations) {
      processedData = this.applyTransformations(processedData);
    }

    // Apply output formatting
    if (this.config.outputFormat) {
      processedData = this.formatOutput(processedData);
    }

    return processedData;
  }

  applySchemaValidation(data) {
    if (!Array.isArray(data) || data.length === 0) {
      return data;
    }

    const schema = this.config.schema;
    const validatedData = [];

    for (const record of data) {
      const validatedRecord = {};

      for (const [field, definition] of Object.entries(schema)) {
        const value = record[field];
        
        try {
          validatedRecord[field] = this.validateAndConvertField(value, definition, field);
        } catch (error) {
          this.logger.warn('Field validation failed', {
            pipelineId: this.id,
            field,
            value,
            definition,
            error: error.message
          });
          validatedRecord[field] = this.getDefaultValue(definition);
        }
      }

      validatedData.push(validatedRecord);
    }

    return validatedData;
  }

  validateAndConvertField(value, definition, fieldName) {
    if (typeof definition === 'string') {
      // Simple type definition (e.g., "integer", "string", "email")
      return this.convertSimpleType(value, definition);
    } else if (typeof definition === 'object') {
      // Complex definition with constraints
      const type = definition.type || 'string';
      let convertedValue = this.convertSimpleType(value, type);

      // Apply constraints
      if (definition.min !== undefined && convertedValue < definition.min) {
        convertedValue = definition.min;
      }
      if (definition.max !== undefined && convertedValue > definition.max) {
        convertedValue = definition.max;
      }
      if (definition.enum && !definition.enum.includes(convertedValue)) {
        convertedValue = definition.enum[0];
      }

      return convertedValue;
    }

    return value;
  }

  convertSimpleType(value, type) {
    switch (type.toLowerCase()) {
      case 'integer':
      case 'int':
        return Math.floor(Number(value)) || 0;
      
      case 'float':
      case 'number':
        return Number(value) || 0;
      
      case 'string':
      case 'text':
        return String(value || '');
      
      case 'boolean':
      case 'bool':
        return Boolean(value);
      
      case 'date':
      case 'datetime':
        return new Date(value).toISOString();
      
      case 'email':
        // Basic email validation
        const email = String(value || '');
        return email.includes('@') ? email : `user${Math.floor(Math.random() * 10000)}@example.com`;
      
      case 'uuid':
        return crypto.randomUUID();
      
      default:
        return value;
    }
  }

  getDefaultValue(definition) {
    const type = typeof definition === 'string' ? definition : definition.type;
    
    switch (type?.toLowerCase()) {
      case 'integer':
      case 'int':
      case 'float':
      case 'number':
        return 0;
      case 'boolean':
      case 'bool':
        return false;
      case 'date':
      case 'datetime':
        return new Date().toISOString();
      case 'email':
        return 'user@example.com';
      case 'uuid':
        return crypto.randomUUID();
      default:
        return '';
    }
  }

  applyBusinessRules(data) {
    // Apply custom business logic rules
    if (!this.config.businessRules || !Array.isArray(data)) {
      return data;
    }

    return data.map(record => {
      let processedRecord = { ...record };

      for (const rule of this.config.businessRules) {
        try {
          processedRecord = this.applyBusinessRule(processedRecord, rule);
        } catch (error) {
          this.logger.warn('Business rule application failed', {
            pipelineId: this.id,
            rule,
            error: error.message
          });
        }
      }

      return processedRecord;
    });
  }

  applyBusinessRule(record, rule) {
    // Simple rule engine implementation
    switch (rule.type) {
      case 'conditional_field':
        if (this.evaluateCondition(record, rule.condition)) {
          record[rule.field] = rule.value;
        }
        break;
      
      case 'calculated_field':
        record[rule.field] = this.calculateFieldValue(record, rule.expression);
        break;
      
      case 'data_masking':
        if (rule.fields) {
          for (const field of rule.fields) {
            if (record[field]) {
              record[field] = this.maskValue(record[field], rule.method);
            }
          }
        }
        break;
      
      default:
        this.logger.warn('Unknown business rule type', { type: rule.type });
    }

    return record;
  }

  evaluateCondition(record, condition) {
    // Simple condition evaluation
    const { field, operator, value } = condition;
    const recordValue = record[field];

    switch (operator) {
      case 'equals':
        return recordValue === value;
      case 'not_equals':
        return recordValue !== value;
      case 'greater_than':
        return recordValue > value;
      case 'less_than':
        return recordValue < value;
      case 'contains':
        return String(recordValue).includes(String(value));
      default:
        return false;
    }
  }

  calculateFieldValue(record, expression) {
    // Simple expression evaluation (for security, only basic math operations)
    try {
      const sanitizedExpression = expression.replace(/[^0-9+\-*/.() ]/g, '');
      return eval(sanitizedExpression);
    } catch (error) {
      this.logger.warn('Expression evaluation failed', { expression, error: error.message });
      return 0;
    }
  }

  maskValue(value, method) {
    const str = String(value);
    
    switch (method) {
      case 'full':
        return '*'.repeat(str.length);
      case 'partial':
        if (str.length <= 4) return str;
        return str.substring(0, 2) + '*'.repeat(str.length - 4) + str.substring(str.length - 2);
      case 'email':
        const atIndex = str.indexOf('@');
        if (atIndex > 0) {
          return str[0] + '*'.repeat(atIndex - 1) + str.substring(atIndex);
        }
        return str;
      default:
        return str;
    }
  }

  applyTransformations(data) {
    // Apply data transformations
    if (!this.config.transformations || !Array.isArray(data)) {
      return data;
    }

    let transformedData = data;

    for (const transformation of this.config.transformations) {
      try {
        transformedData = this.applyTransformation(transformedData, transformation);
      } catch (error) {
        this.logger.warn('Transformation failed', {
          pipelineId: this.id,
          transformation,
          error: error.message
        });
      }
    }

    return transformedData;
  }

  applyTransformation(data, transformation) {
    switch (transformation.type) {
      case 'filter':
        return data.filter(record => this.evaluateCondition(record, transformation.condition));
      
      case 'sort':
        return data.sort((a, b) => {
          const aVal = a[transformation.field];
          const bVal = b[transformation.field];
          return transformation.order === 'desc' ? bVal - aVal : aVal - bVal;
        });
      
      case 'sample':
        const sampleSize = Math.min(transformation.size, data.length);
        return data.slice(0, sampleSize);
      
      default:
        return data;
    }
  }

  formatOutput(data) {
    const format = this.config.outputFormat;
    
    if (!format || format === 'json') {
      return data;
    }

    // For now, we'll keep data as-is since we're working with JSON internally
    // Actual format conversion would happen during export
    return data;
  }

  addValidator(type, config = {}) {
    this.validators.push({ type, config });
    this.logger.info('Validator added', { pipelineId: this.id, type });
  }

  removeValidator(type) {
    this.validators = this.validators.filter(v => v.type !== type);
    this.logger.info('Validator removed', { pipelineId: this.id, type });
  }

  updateConfig(updates) {
    this.config = { ...this.config, ...updates };
    this.logger.info('Pipeline config updated', { pipelineId: this.id });
    this.emit('config.updated', this.id, updates);
  }

  getConfig() {
    return { ...this.config };
  }

  getStatus() {
    return {
      id: this.id,
      initialized: this.initialized,
      generator: this.config.generator,
      dataType: this.config.dataType,
      created: this.created,
      lastUsed: this.lastUsed,
      validators: this.validators.length,
      watermarkEnabled: !!this.watermarkConfig
    };
  }

  async cleanup() {
    this.logger.info('Cleaning up pipeline', { pipelineId: this.id });

    if (this.generator) {
      await this.generator.cleanup();
    }

    this.removeAllListeners();
    this.logger.info('Pipeline cleanup completed', { pipelineId: this.id });
  }
}