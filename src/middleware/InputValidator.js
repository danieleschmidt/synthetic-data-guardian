/**
 * Comprehensive Input Validation and Sanitization
 */

import { ValidationError } from './ErrorHandler.js';

// Simple built-in validators to avoid external dependencies
const builtInValidators = {
  isEmail: email => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email),
  isURL: url => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  },
  isUUID: uuid => /^[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(uuid),
  isISO8601: date => /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$/.test(date),
  trim: str => String(str).trim(),
  escape: str =>
    String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;'),
  normalizeEmail: email => String(email).toLowerCase().trim(),
};

export class InputValidator {
  constructor(logger) {
    this.logger = logger;
    this.schemas = new Map();
    this.sanitizers = new Map();
    this.setupDefaultSanitizers();
  }

  setupDefaultSanitizers() {
    this.sanitizers.set('string', value => {
      if (typeof value !== 'string') return value;
      return builtInValidators.escape(builtInValidators.trim(value));
    });

    this.sanitizers.set('email', value => {
      if (typeof value !== 'string') return value;
      return builtInValidators.normalizeEmail(builtInValidators.trim(value)) || value;
    });

    this.sanitizers.set('number', value => {
      const num = Number(value);
      return isNaN(num) ? value : num;
    });

    this.sanitizers.set('boolean', value => {
      if (typeof value === 'boolean') return value;
      if (typeof value === 'string') {
        return value.toLowerCase() === 'true';
      }
      return Boolean(value);
    });

    this.sanitizers.set('array', value => {
      return Array.isArray(value) ? value : [value];
    });

    this.sanitizers.set('object', value => {
      if (typeof value === 'object' && value !== null) {
        return value;
      }
      try {
        return JSON.parse(value);
      } catch {
        return value;
      }
    });
  }

  registerSchema(name, schema) {
    this.schemas.set(name, schema);
    this.logger.debug('Validation schema registered', { name });
  }

  validate(data, schemaName, options = {}) {
    const schema = this.schemas.get(schemaName);
    if (!schema) {
      throw new Error(`Validation schema not found: ${schemaName}`);
    }

    const result = {
      valid: true,
      data: {},
      errors: [],
      warnings: [],
    };

    try {
      result.data = this.validateObject(data, schema, options);
    } catch (error) {
      result.valid = false;
      if (error instanceof ValidationError) {
        result.errors.push({
          field: error.field,
          message: error.message,
        });
      } else {
        result.errors.push({
          field: 'general',
          message: error.message,
        });
      }
    }

    return result;
  }

  validateObject(data, schema, options = {}) {
    if (!data || typeof data !== 'object') {
      throw new ValidationError('Input must be an object');
    }

    const result = {};
    const errors = [];

    // Validate required fields
    for (const [field, rules] of Object.entries(schema)) {
      if (rules.required && (data[field] === undefined || data[field] === null)) {
        errors.push(new ValidationError(`Field '${field}' is required`, field));
        continue;
      }

      if (data[field] !== undefined) {
        try {
          result[field] = this.validateField(data[field], rules, field, options);
        } catch (error) {
          errors.push(error);
        }
      } else if (rules.default !== undefined) {
        result[field] = rules.default;
      }
    }

    // Check for unexpected fields if strict mode
    if (options.strict) {
      for (const field of Object.keys(data)) {
        if (!schema[field]) {
          errors.push(new ValidationError(`Unexpected field '${field}'`, field));
        }
      }
    }

    if (errors.length > 0) {
      const error = new ValidationError('Validation failed');
      error.errors = errors;
      throw error;
    }

    return result;
  }

  validateField(value, rules, fieldName = 'field', options = {}) {
    let processedValue = value;

    // Apply sanitization first
    if (rules.sanitize !== false && rules.type) {
      const sanitizer = this.sanitizers.get(rules.type);
      if (sanitizer) {
        processedValue = sanitizer(processedValue);
      }
    }

    // Type validation
    if (rules.type && !this.validateType(processedValue, rules.type)) {
      throw new ValidationError(`Field '${fieldName}' must be of type ${rules.type}`, fieldName);
    }

    // Length validation for strings and arrays
    if (rules.minLength !== undefined || rules.maxLength !== undefined) {
      this.validateLength(processedValue, rules, fieldName);
    }

    // Range validation for numbers
    if (rules.min !== undefined || rules.max !== undefined) {
      this.validateRange(processedValue, rules, fieldName);
    }

    // Pattern validation for strings
    if (rules.pattern && typeof processedValue === 'string') {
      const regex = new RegExp(rules.pattern);
      if (!regex.test(processedValue)) {
        throw new ValidationError(`Field '${fieldName}' does not match required pattern`, fieldName);
      }
    }

    // Enum validation
    if (rules.enum && !rules.enum.includes(processedValue)) {
      throw new ValidationError(`Field '${fieldName}' must be one of: ${rules.enum.join(', ')}`, fieldName);
    }

    // Custom validation function
    if (rules.validate && typeof rules.validate === 'function') {
      const customResult = rules.validate(processedValue, fieldName);
      if (customResult !== true) {
        const message = typeof customResult === 'string' ? customResult : `Invalid value for field '${fieldName}'`;
        throw new ValidationError(message, fieldName);
      }
    }

    // Security checks
    if (rules.security !== false) {
      this.performSecurityChecks(processedValue, fieldName);
    }

    return processedValue;
  }

  validateType(value, expectedType) {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value);
      case 'integer':
        return typeof value === 'number' && Number.isInteger(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'array':
        return Array.isArray(value);
      case 'object':
        return typeof value === 'object' && value !== null && !Array.isArray(value);
      case 'email':
        return typeof value === 'string' && builtInValidators.isEmail(value);
      case 'url':
        return typeof value === 'string' && builtInValidators.isURL(value);
      case 'uuid':
        return typeof value === 'string' && builtInValidators.isUUID(value);
      case 'date':
        return value instanceof Date || (typeof value === 'string' && builtInValidators.isISO8601(value));
      default:
        return true;
    }
  }

  validateLength(value, rules, fieldName) {
    let length;

    if (typeof value === 'string' || Array.isArray(value)) {
      length = value.length;
    } else {
      return; // Skip length validation for non-string/array types
    }

    if (rules.minLength !== undefined && length < rules.minLength) {
      throw new ValidationError(
        `Field '${fieldName}' must be at least ${rules.minLength} characters/items long`,
        fieldName,
      );
    }

    if (rules.maxLength !== undefined && length > rules.maxLength) {
      throw new ValidationError(
        `Field '${fieldName}' must be at most ${rules.maxLength} characters/items long`,
        fieldName,
      );
    }
  }

  validateRange(value, rules, fieldName) {
    if (typeof value !== 'number') {
      return; // Skip range validation for non-numeric types
    }

    if (rules.min !== undefined && value < rules.min) {
      throw new ValidationError(`Field '${fieldName}' must be at least ${rules.min}`, fieldName);
    }

    if (rules.max !== undefined && value > rules.max) {
      throw new ValidationError(`Field '${fieldName}' must be at most ${rules.max}`, fieldName);
    }
  }

  performSecurityChecks(value, fieldName) {
    if (typeof value !== 'string') {
      return; // Only perform security checks on strings
    }

    // Check for SQL injection patterns
    const sqlPatterns = [
      /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/gi,
      /(;|\-\-|\||\/\*|\*\/)/g,
      /((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))/gi,
    ];

    for (const pattern of sqlPatterns) {
      if (pattern.test(value)) {
        throw new ValidationError(`Field '${fieldName}' contains potentially dangerous content`, fieldName);
      }
    }

    // Check for XSS patterns
    const xssPatterns = [
      /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,
      /<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi,
    ];

    for (const pattern of xssPatterns) {
      if (pattern.test(value)) {
        throw new ValidationError(`Field '${fieldName}' contains potentially dangerous content`, fieldName);
      }
    }

    // Check for directory traversal
    if (value.includes('../') || value.includes('..\\')) {
      throw new ValidationError(`Field '${fieldName}' contains potentially dangerous path traversal`, fieldName);
    }

    // Check for excessively long strings (DoS prevention)
    if (value.length > 10000) {
      throw new ValidationError(`Field '${fieldName}' is too long (max 10000 characters)`, fieldName);
    }
  }

  // Middleware factory
  middleware(schemaName, options = {}) {
    return (req, res, next) => {
      try {
        // Validate different parts of the request
        const validationResults = {};

        if (req.body && Object.keys(req.body).length > 0) {
          validationResults.body = this.validate(req.body, schemaName, options);
          if (!validationResults.body.valid) {
            return res.status(400).json({
              success: false,
              error: {
                type: 'VALIDATION_ERROR',
                message: 'Request body validation failed',
                details: validationResults.body.errors,
              },
            });
          }
          req.body = validationResults.body.data;
        }

        // Validate query parameters if schema provided
        if (options.validateQuery && req.query) {
          const querySchema = schemaName + '_query';
          if (this.schemas.has(querySchema)) {
            validationResults.query = this.validate(req.query, querySchema, options);
            if (!validationResults.query.valid) {
              return res.status(400).json({
                success: false,
                error: {
                  type: 'VALIDATION_ERROR',
                  message: 'Query parameters validation failed',
                  details: validationResults.query.errors,
                },
              });
            }
            req.query = validationResults.query.data;
          }
        }

        // Validate path parameters if schema provided
        if (options.validateParams && req.params) {
          const paramsSchema = schemaName + '_params';
          if (this.schemas.has(paramsSchema)) {
            validationResults.params = this.validate(req.params, paramsSchema, options);
            if (!validationResults.params.valid) {
              return res.status(400).json({
                success: false,
                error: {
                  type: 'VALIDATION_ERROR',
                  message: 'Path parameters validation failed',
                  details: validationResults.params.errors,
                },
              });
            }
            req.params = validationResults.params.data;
          }
        }

        req.validationResults = validationResults;
        next();
      } catch (error) {
        this.logger.error('Input validation error', {
          error: error.message,
          schema: schemaName,
          url: req.url,
          method: req.method,
        });

        res.status(400).json({
          success: false,
          error: {
            type: 'VALIDATION_ERROR',
            message: error.message || 'Input validation failed',
          },
        });
      }
    };
  }

  // Register common validation schemas
  registerCommonSchemas() {
    // Generation request schema
    this.registerSchema('generation', {
      pipeline: {
        type: 'object',
        required: true,
        validate: value => {
          if (typeof value === 'string') return true;
          if (typeof value === 'object' && value.generator) return true;
          return 'Pipeline must be a string ID or configuration object with generator';
        },
      },
      numRecords: {
        type: 'integer',
        required: true,
        min: 1,
        max: 1000000,
      },
      seed: {
        type: 'integer',
        required: false,
      },
      conditions: {
        type: 'object',
        required: false,
      },
      validate: {
        type: 'boolean',
        required: false,
        default: true,
      },
      assessQuality: {
        type: 'boolean',
        required: false,
        default: true,
      },
      analyzePrivacy: {
        type: 'boolean',
        required: false,
        default: true,
      },
    });

    // Validation request schema
    this.registerSchema('validation', {
      data: {
        type: 'array',
        required: true,
        minLength: 1,
        maxLength: 100000,
      },
      validators: {
        type: 'array',
        required: false,
        default: ['statistical_fidelity'],
      },
      thresholds: {
        type: 'object',
        required: false,
      },
      referenceData: {
        type: 'array',
        required: false,
      },
    });

    // Watermark request schema
    this.registerSchema('watermark', {
      data: {
        type: 'array',
        required: true,
        minLength: 1,
      },
      method: {
        type: 'string',
        required: false,
        enum: ['statistical', 'steganographic', 'frequency'],
        default: 'statistical',
      },
      strength: {
        type: 'number',
        required: false,
        min: 0.1,
        max: 1.0,
        default: 0.8,
      },
      message: {
        type: 'string',
        required: false,
        maxLength: 1000,
      },
    });

    this.logger.info('Common validation schemas registered');
  }
}

export function createInputValidator(logger) {
  const validator = new InputValidator(logger);
  validator.registerCommonSchemas();
  return validator;
}
