/**
 * PrivacyAnalysis - Comprehensive privacy analysis for synthetic data
 */

export class PrivacyAnalysis {
  constructor(logger) {
    this.logger = logger;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    this.logger.info('Initializing Privacy Analysis...');
    this.initialized = true;
    this.logger.info('Privacy Analysis initialized');
  }

  async analyze(data, options = {}) {
    if (!this.initialized) {
      throw new Error('PrivacyAnalysis not initialized');
    }

    const { sensitiveColumns = [], epsilon = 1.0 } = options;

    this.logger.info('Starting privacy analysis', {
      dataSize: data?.length || 0,
      sensitiveColumns: sensitiveColumns.length,
    });

    const analysisResults = {
      timestamp: new Date().toISOString(),
      dataSize: data?.length || 0,
      sensitiveColumns,
      epsilon,
      checks: {},
      overallScore: 0,
    };

    // Run privacy checks
    const checks = {
      reidentificationRisk: await this.assessReidentificationRisk(data, options),
      attributeInference: await this.assessAttributeInference(data, options),
      membershipInference: await this.assessMembershipInference(data, options),
      sensitiveDataLeakage: await this.assessSensitiveDataLeakage(data, options),
      kAnonymity: await this.assessKAnonymity(data, options),
    };

    analysisResults.checks = checks;

    // Calculate overall privacy score
    const scores = Object.values(checks).map(check => check.score || 0);
    analysisResults.overallScore =
      scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;

    this.logger.info('Privacy analysis completed', {
      overallScore: analysisResults.overallScore,
      checksPerformed: Object.keys(checks).length,
    });

    return analysisResults;
  }

  async assessReidentificationRisk(data, options = {}) {
    const { sensitiveColumns = [], threshold = 0.05 } = options;

    if (!Array.isArray(data) || data.length === 0) {
      return {
        score: 1,
        passed: true,
        risk: 0,
        message: 'No data to analyze for reidentification risk',
      };
    }

    // Simple reidentification risk assessment
    const quasiIdentifiers = this.identifyQuasiIdentifiers(data, sensitiveColumns);
    const uniqueCombinations = this.countUniqueCombinations(data, quasiIdentifiers);

    const totalRecords = data.length;
    const uniqueRatio = uniqueCombinations / totalRecords;
    const reidentificationRisk = Math.min(1, uniqueRatio * 2);

    const score = Math.max(0, 1 - reidentificationRisk);
    const passed = reidentificationRisk <= threshold;

    return {
      score,
      passed,
      risk: reidentificationRisk,
      message: passed ? 'Low reidentification risk' : 'High reidentification risk detected',
      details: { quasiIdentifiers, uniqueCombinations, totalRecords, threshold },
    };
  }

  async assessAttributeInference(data, options = {}) {
    const { sensitiveColumns = [], threshold = 0.8 } = options;

    if (sensitiveColumns.length === 0) {
      return {
        score: 1,
        passed: true,
        message: 'No sensitive attributes specified',
      };
    }

    let totalInferenceRisk = 0;
    const attributeRisks = {};

    for (const sensitiveAttr of sensitiveColumns) {
      if (!data.some(record => record[sensitiveAttr] !== undefined)) {
        continue;
      }

      const inferenceRisk = this.calculateAttributeInferenceRisk(data, sensitiveAttr);
      attributeRisks[sensitiveAttr] = inferenceRisk;
      totalInferenceRisk += inferenceRisk;
    }

    const averageRisk = sensitiveColumns.length > 0 ? totalInferenceRisk / sensitiveColumns.length : 0;
    const score = Math.max(0, 1 - averageRisk);
    const passed = score >= threshold;

    return {
      score,
      passed,
      risk: averageRisk,
      message: passed ? 'Low attribute inference risk' : 'High attribute inference risk',
      details: { attributeRisks, threshold },
    };
  }

  async assessMembershipInference(data, options = {}) {
    const { referenceData, threshold = 0.8 } = options;

    if (!referenceData) {
      return {
        score: 0.8,
        passed: true,
        message: 'Reference data required for membership inference assessment',
      };
    }

    const overlapRatio = this.calculateDataOverlap(data, referenceData);
    const membershipRisk = Math.min(1, overlapRatio * 3);

    const score = Math.max(0, 1 - membershipRisk);
    const passed = score >= threshold;

    return {
      score,
      passed,
      risk: membershipRisk,
      message: passed ? 'Low membership inference risk' : 'High membership inference risk',
      details: { overlapRatio, threshold },
    };
  }

  async assessSensitiveDataLeakage(data, options = {}) {
    const { sensitiveColumns = [], sensitivePatterns = [] } = options;

    if (!Array.isArray(data) || data.length === 0) {
      return {
        score: 1,
        passed: true,
        message: 'No data to assess for sensitive data leakage',
      };
    }

    let totalChecked = 0;
    let leakagesFound = 0;

    for (const record of data) {
      for (const [field, value] of Object.entries(record)) {
        totalChecked++;

        if (
          this.containsSensitivePattern(value, sensitivePatterns) ||
          (sensitiveColumns.includes(field) && this.isDirectlyIdentifiable(value))
        ) {
          leakagesFound++;
        }
      }
    }

    const leakageRate = totalChecked > 0 ? leakagesFound / totalChecked : 0;
    const score = Math.max(0, 1 - leakageRate * 10);
    const passed = leakageRate === 0;

    return {
      score,
      passed,
      risk: leakageRate,
      message: passed ? 'No sensitive data leakage detected' : 'Sensitive data leakage detected',
      details: { totalChecked, leakagesFound },
    };
  }

  async assessKAnonymity(data, options = {}) {
    const { quasiIdentifiers = [], k = 5 } = options;

    if (!Array.isArray(data) || data.length === 0 || quasiIdentifiers.length === 0) {
      return {
        score: 0.8,
        passed: true,
        message: 'Cannot assess k-anonymity without quasi-identifiers',
      };
    }

    const equivalenceClasses = this.calculateEquivalenceClasses(data, quasiIdentifiers);
    const minGroupSize = Math.min(...Object.values(equivalenceClasses));

    const score = Math.min(1, minGroupSize / k);
    const passed = minGroupSize >= k;

    return {
      score,
      passed,
      anonymityLevel: minGroupSize,
      message: passed ? `Achieves ${k}-anonymity` : `Does not achieve ${k}-anonymity`,
      details: { k, minGroupSize, quasiIdentifiers },
    };
  }

  identifyQuasiIdentifiers(data, sensitiveColumns = []) {
    if (!Array.isArray(data) || data.length === 0) return [];

    const fields = Object.keys(data[0] || {});
    const quasiIdentifiers = [];

    const potentialQuasiFields = ['age', 'gender', 'zip_code', 'postal_code', 'birth_date'];

    for (const field of fields) {
      if (sensitiveColumns.includes(field)) continue;

      if (potentialQuasiFields.some(qf => field.toLowerCase().includes(qf))) {
        quasiIdentifiers.push(field);
        continue;
      }

      const values = data.map(record => record[field]).filter(v => v !== null && v !== undefined);
      const uniqueValues = new Set(values);
      const uniqueRatio = uniqueValues.size / values.length;

      if (uniqueRatio > 0.1 && uniqueRatio < 0.8) {
        quasiIdentifiers.push(field);
      }
    }

    return quasiIdentifiers;
  }

  countUniqueCombinations(data, quasiIdentifiers) {
    if (!Array.isArray(data) || data.length === 0 || quasiIdentifiers.length === 0) {
      return 0;
    }

    const combinations = new Set();
    for (const record of data) {
      const combination = quasiIdentifiers.map(field => record[field]).join('|');
      combinations.add(combination);
    }

    return combinations.size;
  }

  calculateAttributeInferenceRisk(data, sensitiveAttribute) {
    if (!Array.isArray(data) || data.length === 0) return 0;

    const sensitiveValues = data.map(record => record[sensitiveAttribute]).filter(v => v !== null && v !== undefined);

    if (sensitiveValues.length === 0) return 0;

    const valueCounts = {};
    sensitiveValues.forEach(value => {
      valueCounts[value] = (valueCounts[value] || 0) + 1;
    });

    let entropy = 0;
    const total = sensitiveValues.length;

    for (const count of Object.values(valueCounts)) {
      const probability = count / total;
      entropy -= probability * Math.log2(probability);
    }

    const maxEntropy = Math.log2(Object.keys(valueCounts).length);
    const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 1;

    return Math.max(0, 1 - normalizedEntropy);
  }

  calculateDataOverlap(syntheticData, referenceData) {
    if (
      !Array.isArray(syntheticData) ||
      !Array.isArray(referenceData) ||
      syntheticData.length === 0 ||
      referenceData.length === 0
    ) {
      return 0;
    }

    const syntheticRecords = new Set(syntheticData.map(record => JSON.stringify(record)));
    const referenceRecords = new Set(referenceData.map(record => JSON.stringify(record)));

    let overlapCount = 0;
    for (const synRecord of syntheticRecords) {
      if (referenceRecords.has(synRecord)) {
        overlapCount++;
      }
    }

    return overlapCount / syntheticRecords.size;
  }

  containsSensitivePattern(value, customPatterns = []) {
    if (value === null || value === undefined) return false;

    const str = String(value);
    const patterns = [
      /\d{3}-\d{2}-\d{4}/, // SSN
      /\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}/, // Credit card
      /\([0-9]{3}\)[\s-]?[0-9]{3}[\s-]?[0-9]{4}/, // Phone number
      ...customPatterns,
    ];

    return patterns.some(pattern => pattern.test(str));
  }

  isDirectlyIdentifiable(value) {
    if (value === null || value === undefined) return false;

    const str = String(value).toLowerCase();
    const directIdentifiers = ['ssn', 'social_security', 'passport', 'license'];

    return directIdentifiers.some(id => str.includes(id));
  }

  calculateEquivalenceClasses(data, quasiIdentifiers) {
    const equivalenceClasses = {};

    for (const record of data) {
      const key = quasiIdentifiers.map(field => record[field]).join('|');
      equivalenceClasses[key] = (equivalenceClasses[key] || 0) + 1;
    }

    return equivalenceClasses;
  }

  async cleanup() {
    this.logger.info('Cleaning up Privacy Analysis');
    this.initialized = false;
  }
}
