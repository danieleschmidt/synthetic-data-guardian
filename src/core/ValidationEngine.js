/**
 * ValidationEngine - Core validation orchestrator with comprehensive validation capabilities
 */

import crypto from 'crypto';

export class ValidationEngine {
  constructor(logger) {
    this.logger = logger;
    this.validators = new Map();
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    this.logger.info('Initializing Validation Engine...');
    this.registerBuiltInValidators();
    this.initialized = true;
    this.logger.info('Validation Engine initialized');
  }

  registerBuiltInValidators() {
    // Statistical fidelity validator
    this.validators.set('statistical_fidelity', {
      validate: async (data, options) => {
        return this.validateStatisticalFidelity(data, options);
      },
    });

    // Privacy preservation validator
    this.validators.set('privacy_preservation', {
      validate: async (data, options) => {
        return this.validatePrivacyPreservation(data, options);
      },
    });

    // Bias detection validator
    this.validators.set('bias_detection', {
      validate: async (data, options) => {
        return this.validateBiasDetection(data, options);
      },
    });

    // Quality assessment validator
    this.validators.set('quality_assessment', {
      validate: async (data, options) => {
        return this.validateQuality(data, options);
      },
    });
  }

  async validate(data, options = {}) {
    if (!this.initialized) {
      throw new Error('ValidationEngine not initialized');
    }

    const { validators = [], thresholds = {}, referenceData } = options;
    const results = {};
    let overallScore = 0;
    let validatorCount = 0;

    this.logger.info('Starting validation', {
      validators: validators.length,
      dataSize: data?.length || 0,
    });

    for (const validator of validators) {
      const validatorName = typeof validator === 'string' ? validator : validator.type;
      const validatorConfig = typeof validator === 'object' ? validator.config : {};

      try {
        const validatorInstance = this.validators.get(validatorName);
        if (!validatorInstance) {
          this.logger.warn('Unknown validator', { validator: validatorName });
          continue;
        }

        const result = await validatorInstance.validate(data, {
          ...validatorConfig,
          referenceData,
          threshold: thresholds[validatorName] || 0.8,
        });

        results[validatorName] = result;
        overallScore += result.score || 0;
        validatorCount++;
      } catch (error) {
        this.logger.error('Validator failed', {
          validator: validatorName,
          error: error.message,
        });
        results[validatorName] = {
          passed: false,
          score: 0,
          error: error.message,
        };
      }
    }

    const finalOverallScore = validatorCount > 0 ? overallScore / validatorCount : 1;
    const passed = Object.values(results).every(r => r.passed !== false);

    return {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      overallScore: finalOverallScore,
      passed: passed,
      results: results,
      dataSize: data?.length || 0,
      validatorCount: validatorCount,
    };
  }

  async validateStatisticalFidelity(data, options = {}) {
    const { referenceData, threshold = 0.8 } = options;

    if (!referenceData || !Array.isArray(referenceData) || referenceData.length === 0) {
      return {
        passed: false,
        score: 0,
        message: 'Reference data is required for statistical fidelity validation',
      };
    }

    const syntheticStats = this.calculateStatistics(data);
    const referenceStats = this.calculateStatistics(referenceData);

    const fidelityScores = {};
    let totalScore = 0;
    let fieldCount = 0;

    for (const field of Object.keys(referenceStats)) {
      if (syntheticStats[field]) {
        const similarity = this.calculateFieldSimilarity(referenceStats[field], syntheticStats[field]);
        fidelityScores[field] = similarity;
        totalScore += similarity;
        fieldCount++;
      }
    }

    const averageScore = fieldCount > 0 ? totalScore / fieldCount : 0;
    const passed = averageScore >= threshold;

    return {
      passed,
      score: averageScore,
      message: passed ? 'Statistical fidelity validation passed' : 'Statistical fidelity below threshold',
      details: {
        fieldScores: fidelityScores,
        threshold,
      },
    };
  }

  async validatePrivacyPreservation(data, options = {}) {
    const { sensitiveColumns = [], threshold = 0.8 } = options;

    const privacyChecks = {
      hasSensitiveData: this.checkSensitiveDataLeakage(data, sensitiveColumns),
      hasDirectIdentifiers: this.checkDirectIdentifiers(data),
      hasQuasiIdentifiers: this.checkQuasiIdentifiers(data),
    };

    const passedChecks = Object.values(privacyChecks).filter(check => check.passed).length;
    const totalChecks = Object.keys(privacyChecks).length;
    const score = totalChecks > 0 ? passedChecks / totalChecks : 1;
    const passed = score >= threshold;

    return {
      passed,
      score,
      message: passed ? 'Privacy preservation validation passed' : 'Privacy concerns detected',
      details: { checks: privacyChecks },
    };
  }

  async validateBiasDetection(data, options = {}) {
    const { protectedAttributes = [], threshold = 0.8 } = options;

    if (protectedAttributes.length === 0) {
      return {
        passed: true,
        score: 1,
        message: 'No protected attributes specified for bias detection',
      };
    }

    const biasMetrics = {};
    let totalScore = 0;
    let attributeCount = 0;

    for (const attribute of protectedAttributes) {
      if (data.some(record => record[attribute] !== undefined)) {
        const bias = this.calculateBiasMetric(data, attribute);
        biasMetrics[attribute] = bias;
        totalScore += bias.score;
        attributeCount++;
      }
    }

    const averageScore = attributeCount > 0 ? totalScore / attributeCount : 1;
    const passed = averageScore >= threshold;

    return {
      passed,
      score: averageScore,
      message: passed ? 'Bias detection validation passed' : 'Bias detected in protected attributes',
      details: { attributeMetrics: biasMetrics },
    };
  }

  async validateQuality(data, options = {}) {
    const { threshold = 0.8 } = options;

    const qualityChecks = {
      completeness: this.checkCompleteness(data),
      consistency: this.checkConsistency(data),
      validity: this.checkValidity(data),
      uniqueness: this.checkUniqueness(data),
    };

    const scores = Object.values(qualityChecks).map(check => check.score);
    const averageScore = scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
    const passed = averageScore >= threshold;

    return {
      passed,
      score: averageScore,
      message: passed ? 'Quality validation passed' : 'Quality issues detected',
      details: { checks: qualityChecks },
    };
  }

  calculateStatistics(data) {
    if (!Array.isArray(data) || data.length === 0) return {};

    const stats = {};
    const fields = Object.keys(data[0] || {});

    for (const field of fields) {
      const values = data.map(record => record[field]).filter(v => v !== null && v !== undefined);

      if (values.length === 0) continue;

      const sampleValue = values[0];

      if (typeof sampleValue === 'number') {
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;

        stats[field] = {
          type: 'numeric',
          mean,
          variance,
          min: Math.min(...values),
          max: Math.max(...values),
        };
      } else {
        const unique = [...new Set(values)];
        stats[field] = {
          type: 'categorical',
          unique: unique.length,
          distribution: unique.reduce((dist, val) => {
            dist[val] = values.filter(v => v === val).length / values.length;
            return dist;
          }, {}),
        };
      }
    }

    return stats;
  }

  calculateFieldSimilarity(referenceStats, syntheticStats) {
    if (referenceStats.type !== syntheticStats.type) return 0;

    if (referenceStats.type === 'numeric') {
      const meanSimilarity =
        1 -
        Math.abs(referenceStats.mean - syntheticStats.mean) /
          Math.max(Math.abs(referenceStats.mean), Math.abs(syntheticStats.mean), 1);
      const varianceSimilarity =
        1 -
        Math.abs(referenceStats.variance - syntheticStats.variance) /
          Math.max(referenceStats.variance, syntheticStats.variance, 1);

      return (meanSimilarity + varianceSimilarity) / 2;
    } else {
      // Compare distributions for categorical data
      const refDist = referenceStats.distribution;
      const synDist = syntheticStats.distribution;
      const allKeys = new Set([...Object.keys(refDist), ...Object.keys(synDist)]);

      let similarity = 0;
      for (const key of allKeys) {
        similarity += Math.min(refDist[key] || 0, synDist[key] || 0);
      }

      return similarity;
    }
  }

  checkSensitiveDataLeakage(data, sensitiveColumns) {
    const leakageFound = data.some(record =>
      sensitiveColumns.some(col => record[col] && this.containsSensitivePattern(record[col])),
    );

    return {
      passed: !leakageFound,
      score: leakageFound ? 0 : 1,
    };
  }

  containsSensitivePattern(value) {
    const str = String(value);
    const patterns = [
      /\d{3}-\d{2}-\d{4}/, // SSN
      /\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}/, // Credit card
    ];

    return patterns.some(pattern => pattern.test(str));
  }

  checkDirectIdentifiers(data) {
    const identifierFields = ['ssn', 'social_security', 'passport', 'license'];
    const hasDirectIdentifiers = data.some(record => identifierFields.some(field => record[field]));

    return { passed: !hasDirectIdentifiers, score: hasDirectIdentifiers ? 0 : 1 };
  }

  checkQuasiIdentifiers(data) {
    const quasiFields = ['zip_code', 'age', 'gender', 'birth_date'];
    const hasQuasiCombination = data.some(record => quasiFields.filter(field => record[field]).length >= 3);

    return { passed: !hasQuasiCombination, score: hasQuasiCombination ? 0.5 : 1 };
  }

  calculateBiasMetric(data, attribute) {
    const values = data.map(record => record[attribute]).filter(v => v !== null && v !== undefined);
    const distribution = {};

    values.forEach(value => {
      distribution[value] = (distribution[value] || 0) + 1;
    });

    // Calculate entropy as bias metric
    const total = values.length;
    let entropy = 0;

    for (const count of Object.values(distribution)) {
      const probability = count / total;
      entropy -= probability * Math.log2(probability);
    }

    const maxEntropy = Math.log2(Object.keys(distribution).length);
    const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 1;

    return { score: normalizedEntropy, distribution };
  }

  checkCompleteness(data) {
    if (!data || data.length === 0) return { score: 0 };

    const fields = Object.keys(data[0] || {});
    let totalFields = 0;
    let completeFields = 0;

    for (const record of data) {
      for (const field of fields) {
        totalFields++;
        if (record[field] !== null && record[field] !== undefined && record[field] !== '') {
          completeFields++;
        }
      }
    }

    return { score: totalFields > 0 ? completeFields / totalFields : 0 };
  }

  checkConsistency(data) {
    if (!data || data.length === 0) return { score: 0 };

    const fields = Object.keys(data[0] || {});
    let consistentFields = 0;

    for (const field of fields) {
      const types = new Set();
      data.forEach(record => {
        if (record[field] !== null && record[field] !== undefined) {
          types.add(typeof record[field]);
        }
      });

      if (types.size <= 1) consistentFields++;
    }

    return { score: fields.length > 0 ? consistentFields / fields.length : 0 };
  }

  checkValidity(data) {
    if (!data || data.length === 0) return { score: 0 };

    let validRecords = 0;
    for (const record of data) {
      if (this.isValidRecord(record)) validRecords++;
    }

    return { score: data.length > 0 ? validRecords / data.length : 0 };
  }

  isValidRecord(record) {
    if (!record || typeof record !== 'object') return false;

    for (const [key, value] of Object.entries(record)) {
      if (typeof value === 'string' && value.length > 1000) return false;
      if (typeof value === 'number' && !isFinite(value)) return false;
    }

    return true;
  }

  checkUniqueness(data) {
    if (!data || data.length === 0) return { score: 0 };

    const seen = new Set();
    let uniqueRecords = 0;

    for (const record of data) {
      const key = JSON.stringify(record);
      if (!seen.has(key)) {
        seen.add(key);
        uniqueRecords++;
      }
    }

    return { score: data.length > 0 ? uniqueRecords / data.length : 0 };
  }

  async cleanup() {
    this.logger.info('Cleaning up Validation Engine');
    this.validators.clear();
    this.initialized = false;
  }
}
