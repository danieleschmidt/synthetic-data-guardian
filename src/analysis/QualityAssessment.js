/**
 * QualityAssessment - Comprehensive synthetic data quality analysis
 */

export class QualityAssessment {
  constructor(logger) {
    this.logger = logger;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    this.logger.info('Initializing Quality Assessment...');
    this.initialized = true;
    this.logger.info('Quality Assessment initialized');
  }

  async assess(data, options = {}) {
    if (!this.initialized) {
      throw new Error('QualityAssessment not initialized');
    }

    const { referenceData, metrics = ['statistical_fidelity'] } = options;

    this.logger.info('Starting quality assessment', {
      dataSize: data?.length || 0,
      metrics: metrics.length,
    });

    const assessmentResults = {
      timestamp: new Date().toISOString(),
      dataSize: data?.length || 0,
      metrics: {},
      overallScore: 0,
    };

    let totalScore = 0;
    let metricCount = 0;

    for (const metric of metrics) {
      try {
        const result = await this.runMetric(metric, data, { referenceData, ...options });
        assessmentResults.metrics[metric] = result;
        totalScore += result.score || 0;
        metricCount++;
      } catch (error) {
        this.logger.error('Quality metric failed', {
          metric,
          error: error.message,
        });
        assessmentResults.metrics[metric] = {
          score: 0,
          error: error.message,
          passed: false,
        };
      }
    }

    assessmentResults.overallScore = metricCount > 0 ? totalScore / metricCount : 0;

    this.logger.info('Quality assessment completed', {
      overallScore: assessmentResults.overallScore,
      metricCount,
    });

    return assessmentResults;
  }

  async runMetric(metricName, data, options = {}) {
    switch (metricName) {
      case 'statistical_fidelity':
        return this.assessStatisticalFidelity(data, options);
      case 'data_completeness':
        return this.assessDataCompleteness(data, options);
      case 'data_consistency':
        return this.assessDataConsistency(data, options);
      case 'data_validity':
        return this.assessDataValidity(data, options);
      case 'correlation_preservation':
        return this.assessCorrelationPreservation(data, options);
      default:
        throw new Error(`Unknown quality metric: ${metricName}`);
    }
  }

  async assessStatisticalFidelity(data, options = {}) {
    const { referenceData, threshold = 0.8 } = options;

    if (!referenceData || !Array.isArray(referenceData) || referenceData.length === 0) {
      return {
        score: 0,
        passed: false,
        message: 'Reference data is required for statistical fidelity assessment',
      };
    }

    const syntheticStats = this.calculateBasicStatistics(data);
    const referenceStats = this.calculateBasicStatistics(referenceData);

    const fidelityScores = {};
    let totalScore = 0;
    let fieldCount = 0;

    for (const field of Object.keys(referenceStats)) {
      if (syntheticStats[field]) {
        const similarity = this.calculateStatisticalSimilarity(referenceStats[field], syntheticStats[field]);
        fidelityScores[field] = similarity;
        totalScore += similarity;
        fieldCount++;
      }
    }

    const averageScore = fieldCount > 0 ? totalScore / fieldCount : 0;
    const passed = averageScore >= threshold;

    return {
      score: averageScore,
      passed,
      message: passed ? 'Statistical fidelity is acceptable' : 'Statistical fidelity below threshold',
      details: {
        fieldScores: fidelityScores,
        threshold,
        fieldCount,
      },
    };
  }

  async assessDataCompleteness(data, options = {}) {
    const { threshold = 0.95 } = options;

    if (!Array.isArray(data) || data.length === 0) {
      return { score: 0, passed: false, message: 'No data to assess completeness' };
    }

    const fields = Object.keys(data[0] || {});
    let totalFields = 0;
    let completeFields = 0;

    for (const record of data) {
      for (const field of fields) {
        totalFields++;
        if (this.isComplete(record[field])) {
          completeFields++;
        }
      }
    }

    const completenessScore = totalFields > 0 ? completeFields / totalFields : 0;
    const passed = completenessScore >= threshold;

    return {
      score: completenessScore,
      passed,
      message: `Data is ${Math.round(completenessScore * 100)}% complete`,
      details: { totalFields, completeFields, threshold },
    };
  }

  async assessDataConsistency(data, options = {}) {
    const { threshold = 0.9 } = options;

    if (!Array.isArray(data) || data.length === 0) {
      return { score: 0, passed: false, message: 'No data to assess consistency' };
    }

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

    const consistencyScore = fields.length > 0 ? consistentFields / fields.length : 0;
    const passed = consistencyScore >= threshold;

    return {
      score: consistencyScore,
      passed,
      message: `Data consistency: ${Math.round(consistencyScore * 100)}%`,
      details: { consistentFields, totalFields: fields.length, threshold },
    };
  }

  async assessDataValidity(data, options = {}) {
    const { schema, threshold = 0.9 } = options;

    if (!Array.isArray(data) || data.length === 0) {
      return { score: 0, passed: false, message: 'No data to assess validity' };
    }

    let validRecords = 0;

    for (const record of data) {
      if (this.isValidRecord(record, schema)) {
        validRecords++;
      }
    }

    const validityScore = data.length > 0 ? validRecords / data.length : 0;
    const passed = validityScore >= threshold;

    return {
      score: validityScore,
      passed,
      message: `${Math.round(validityScore * 100)}% of records are valid`,
      details: { validRecords, totalRecords: data.length, threshold },
    };
  }

  async assessCorrelationPreservation(data, options = {}) {
    const { referenceData, threshold = 0.8 } = options;

    if (!referenceData) {
      return {
        score: 0,
        passed: false,
        message: 'Reference data required for correlation assessment',
      };
    }

    const syntheticCorrelations = this.calculateCorrelationMatrix(data);
    const referenceCorrelations = this.calculateCorrelationMatrix(referenceData);

    const correlationSimilarity = this.compareCorrelationMatrices(referenceCorrelations, syntheticCorrelations);

    const passed = correlationSimilarity >= threshold;

    return {
      score: correlationSimilarity,
      passed,
      message: passed ? 'Correlations preserved well' : 'Correlation preservation below threshold',
      details: { similarity: correlationSimilarity, threshold },
    };
  }

  calculateBasicStatistics(data) {
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
          std: Math.sqrt(variance),
          min: Math.min(...values),
          max: Math.max(...values),
          count: values.length,
        };
      } else {
        const unique = [...new Set(values)];
        const frequencies = {};
        values.forEach(v => (frequencies[v] = (frequencies[v] || 0) + 1));

        stats[field] = {
          type: 'categorical',
          unique: unique.length,
          frequencies,
          count: values.length,
        };
      }
    }

    return stats;
  }

  calculateStatisticalSimilarity(referenceStats, syntheticStats) {
    if (referenceStats.type !== syntheticStats.type) return 0;

    if (referenceStats.type === 'numeric') {
      const meanSimilarity =
        1 -
        Math.abs(referenceStats.mean - syntheticStats.mean) /
          Math.max(Math.abs(referenceStats.mean), Math.abs(syntheticStats.mean), 1);
      const stdSimilarity =
        1 - Math.abs(referenceStats.std - syntheticStats.std) / Math.max(referenceStats.std, syntheticStats.std, 1);

      return (meanSimilarity + stdSimilarity) / 2;
    } else {
      const refFreqs = referenceStats.frequencies;
      const synFreqs = syntheticStats.frequencies;
      const allKeys = new Set([...Object.keys(refFreqs), ...Object.keys(synFreqs)]);

      let similarity = 0;
      for (const key of allKeys) {
        const refProb = (refFreqs[key] || 0) / referenceStats.count;
        const synProb = (synFreqs[key] || 0) / syntheticStats.count;
        similarity += Math.min(refProb, synProb);
      }

      return similarity;
    }
  }

  calculateCorrelationMatrix(data) {
    if (!Array.isArray(data) || data.length === 0) return {};

    const numericFields = [];
    const fields = Object.keys(data[0] || {});

    for (const field of fields) {
      const values = data.map(record => record[field]);
      if (values.every(v => typeof v === 'number' && isFinite(v))) {
        numericFields.push(field);
      }
    }

    const correlations = {};

    for (const field1 of numericFields) {
      correlations[field1] = {};
      for (const field2 of numericFields) {
        if (field1 === field2) {
          correlations[field1][field2] = 1;
        } else {
          const values1 = data.map(record => record[field1]);
          const values2 = data.map(record => record[field2]);
          correlations[field1][field2] = this.calculateCorrelation(values1, values2);
        }
      }
    }

    return correlations;
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

  compareCorrelationMatrices(reference, synthetic) {
    const commonFields = Object.keys(reference).filter(field => synthetic[field]);
    if (commonFields.length === 0) return 0;

    let totalSimilarity = 0;
    let comparisonCount = 0;

    for (const field1 of commonFields) {
      for (const field2 of commonFields) {
        const refCorr = reference[field1][field2] || 0;
        const synCorr = synthetic[field1][field2] || 0;

        const similarity = 1 - Math.abs(refCorr - synCorr) / 2;
        totalSimilarity += similarity;
        comparisonCount++;
      }
    }

    return comparisonCount > 0 ? totalSimilarity / comparisonCount : 0;
  }

  isComplete(value) {
    return value !== null && value !== undefined && value !== '' && !(typeof value === 'number' && isNaN(value));
  }

  isValidRecord(record, schema) {
    if (!record || typeof record !== 'object') return false;

    for (const [key, value] of Object.entries(record)) {
      if (typeof value === 'string' && value.length > 10000) return false;
      if (typeof value === 'number' && !isFinite(value)) return false;
    }

    return true;
  }

  async cleanup() {
    this.logger.info('Cleaning up Quality Assessment');
    this.initialized = false;
  }
}
