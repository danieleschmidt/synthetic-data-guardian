/**
 * Test fixtures and sample data for Synthetic Data Guardian tests
 */

export const mockUsers = {
  dataScientist: {
    id: 'user-data-scientist-1',
    email: 'scientist@example.com',
    role: 'data_scientist',
    permissions: ['generate', 'validate', 'view_lineage'],
    apiKey: 'test-api-key-scientist-123',
  },
  privacyOfficer: {
    id: 'user-privacy-officer-1',
    email: 'privacy@example.com',
    role: 'privacy_officer',
    permissions: ['validate', 'view_lineage', 'compliance_report'],
    apiKey: 'test-api-key-privacy-456',
  },
  admin: {
    id: 'user-admin-1',
    email: 'admin@example.com',
    role: 'administrator',
    permissions: ['generate', 'validate', 'view_lineage', 'admin', 'compliance_report'],
    apiKey: 'test-api-key-admin-789',
  },
};

export const mockDatasets = {
  customerData: {
    id: 'dataset-customers-1',
    name: 'Customer Dataset',
    schema: {
      customer_id: 'integer',
      first_name: 'string',
      last_name: 'string',
      email: 'email',
      age: 'integer[18:95]',
      income: 'float[20000:200000]',
      city: 'string',
      state: 'categorical[CA,NY,TX,FL,WA]',
      signup_date: 'datetime',
      is_premium: 'boolean',
    },
    size: 10000,
    sensitiveFields: ['email', 'income'],
  },
  transactionData: {
    id: 'dataset-transactions-1',
    name: 'Transaction Dataset',
    schema: {
      transaction_id: 'uuid',
      customer_id: 'integer',
      amount: 'float[0.01:10000]',
      merchant_category: 'categorical[retail,food,transport,utilities,entertainment]',
      transaction_date: 'datetime',
      payment_method: 'categorical[credit_card,debit_card,cash,digital_wallet]',
      location: 'geo_point',
      is_fraud: 'boolean',
    },
    size: 50000,
    sensitiveFields: ['customer_id', 'amount', 'location'],
  },
  medicalData: {
    id: 'dataset-medical-1',
    name: 'Medical Records Dataset',
    schema: {
      patient_id: 'uuid',
      age: 'integer[0:120]',
      gender: 'categorical[M,F,O]',
      diagnosis_code: 'string',
      treatment: 'string',
      admission_date: 'datetime',
      discharge_date: 'datetime',
      hospital_id: 'integer',
      insurance_type: 'categorical[private,medicare,medicaid,uninsured]',
      cost: 'float[100:100000]',
    },
    size: 25000,
    sensitiveFields: ['patient_id', 'diagnosis_code', 'treatment', 'cost'],
    complianceRequirements: ['hipaa'],
  },
};

export const mockPipelines = {
  basicGeneration: {
    id: 'pipeline-basic-1',
    name: 'Basic Data Generation',
    description: 'Simple synthetic data generation pipeline',
    generator: 'sdv',
    generatorConfig: {
      model: 'gaussian_copula',
      epochs: 100,
      batch_size: 500,
    },
    validators: [
      {
        type: 'statistical_fidelity',
        config: { threshold: 0.9 },
      },
    ],
    watermarking: {
      enabled: true,
      method: 'statistical',
      strength: 0.8,
    },
  },
  privacyPreserving: {
    id: 'pipeline-privacy-1',
    name: 'Privacy-Preserving Generation',
    description: 'High-privacy synthetic data generation',
    generator: 'dpgan',
    generatorConfig: {
      epsilon: 1.0,
      delta: 1e-5,
      noise_multiplier: 1.1,
      max_grad_norm: 1.0,
    },
    validators: [
      {
        type: 'statistical_fidelity',
        config: { threshold: 0.85 },
      },
      {
        type: 'privacy_preservation',
        config: { 
          epsilon: 1.0,
          attack_models: ['linkage', 'inference'],
        },
      },
    ],
    watermarking: {
      enabled: true,
      method: 'cryptographic',
      strength: 0.9,
    },
  },
  enterpriseCompliance: {
    id: 'pipeline-enterprise-1',
    name: 'Enterprise Compliance Pipeline',
    description: 'Full compliance and audit trail generation',
    generator: 'ctgan',
    generatorConfig: {
      epochs: 300,
      batch_size: 500,
      pac: 10,
      generator_lr: 2e-4,
      discriminator_lr: 2e-4,
    },
    validators: [
      {
        type: 'statistical_fidelity',
        config: { threshold: 0.95 },
      },
      {
        type: 'privacy_preservation',
        config: { epsilon: 0.5 },
      },
      {
        type: 'bias_detection',
        config: {
          protected_attributes: ['age', 'gender'],
          fairness_metrics: ['demographic_parity', 'equalized_odds'],
        },
      },
    ],
    watermarking: {
      enabled: true,
      method: 'stegastamp',
      strength: 0.95,
    },
    compliance: {
      frameworks: ['gdpr', 'hipaa'],
      auditLevel: 'full',
      reportingEnabled: true,
    },
  },
};

export const mockGenerationJobs = {
  completed: {
    id: 'job-completed-1',
    pipelineId: 'pipeline-basic-1',
    status: 'completed',
    startTime: new Date('2024-01-15T10:00:00Z'),
    endTime: new Date('2024-01-15T10:05:30Z'),
    numRecords: 1000,
    qualityScore: 0.94,
    privacyScore: 0.87,
    lineageId: 'lineage-123',
    outputPath: 's3://test-bucket/synthetic/job-completed-1.parquet',
  },
  running: {
    id: 'job-running-1',
    pipelineId: 'pipeline-privacy-1',
    status: 'running',
    startTime: new Date('2024-01-15T11:00:00Z'),
    progress: 0.65,
    estimatedCompletion: new Date('2024-01-15T11:08:00Z'),
  },
  failed: {
    id: 'job-failed-1',
    pipelineId: 'pipeline-enterprise-1',
    status: 'failed',
    startTime: new Date('2024-01-15T09:00:00Z'),
    endTime: new Date('2024-01-15T09:02:15Z'),
    error: {
      code: 'VALIDATION_FAILED',
      message: 'Statistical fidelity threshold not met',
      details: {
        actualScore: 0.78,
        requiredScore: 0.95,
        failedColumns: ['age', 'income'],
      },
    },
  },
};

export const mockValidationResults = {
  passed: {
    id: 'validation-passed-1',
    jobId: 'job-completed-1',
    overallScore: 0.92,
    results: {
      statistical_fidelity: {
        score: 0.94,
        passed: true,
        metrics: {
          ks_test: 0.96,
          wasserstein: 0.91,
          correlation: 0.95,
        },
      },
      privacy_preservation: {
        score: 0.89,
        passed: true,
        reidentificationRisk: 0.02,
        membershipInferenceAccuracy: 0.51,
      },
    },
  },
  warning: {
    id: 'validation-warning-1',
    jobId: 'job-completed-1',
    overallScore: 0.81,
    results: {
      statistical_fidelity: {
        score: 0.85,
        passed: true,
        warnings: ['Low correlation in age-income relationship'],
      },
      bias_detection: {
        score: 0.76,
        passed: false,
        issues: [
          {
            attribute: 'gender',
            metric: 'demographic_parity',
            value: 0.12,
            threshold: 0.1,
          },
        ],
      },
    },
  },
};

export const mockLineageData = {
  simple: {
    id: 'lineage-123',
    rootDataset: 'dataset-customers-1',
    nodes: [
      {
        id: 'node-source',
        type: 'dataset',
        name: 'Original Customer Data',
        path: 's3://raw-data/customers.csv',
      },
      {
        id: 'node-generator',
        type: 'generator',
        name: 'SDV Gaussian Copula',
        version: '1.2.0',
        parameters: { epochs: 100 },
      },
      {
        id: 'node-synthetic',
        type: 'dataset',
        name: 'Synthetic Customer Data',
        path: 's3://synthetic/customers_synthetic.parquet',
      },
    ],
    edges: [
      {
        from: 'node-source',
        to: 'node-generator',
        type: 'input',
      },
      {
        from: 'node-generator',
        to: 'node-synthetic',
        type: 'output',
      },
    ],
    metadata: {
      createdAt: '2024-01-15T10:00:00Z',
      createdBy: 'user-data-scientist-1',
      tags: ['customer-data', 'synthetic', 'v1'],
    },
  },
};

export const mockComplianceReports = {
  gdpr: {
    id: 'compliance-gdpr-1',
    framework: 'gdpr',
    datasetId: 'dataset-customers-1',
    generatedAt: '2024-01-15T12:00:00Z',
    status: 'compliant',
    checks: {
      dataMinimization: { passed: true, score: 0.95 },
      purposeLimitation: { passed: true, score: 1.0 },
      storageMinimization: { passed: true, score: 0.88 },
      transparency: { passed: true, score: 0.92 },
    },
    recommendations: [
      'Consider reducing data retention period from 365 to 180 days',
      'Implement automated data deletion for expired records',
    ],
  },
  hipaa: {
    id: 'compliance-hipaa-1',
    framework: 'hipaa',
    datasetId: 'dataset-medical-1',
    generatedAt: '2024-01-15T12:30:00Z',
    status: 'compliant',
    checks: {
      safeHarbor: { passed: true, score: 1.0 },
      minimumNecessary: { passed: true, score: 0.94 },
      accessControls: { passed: true, score: 0.96 },
      auditTrails: { passed: true, score: 0.98 },
    },
    deidentificationMethod: 'safe_harbor',
    removedIdentifiers: [
      'patient_id',
      'admission_date',
      'discharge_date',
    ],
  },
};

// Helper functions for generating test data
export const generateMockCsvData = (schema: Record<string, string>, numRows: number = 100): string => {
  const headers = Object.keys(schema);
  const rows = [headers.join(',')];
  
  for (let i = 0; i < numRows; i++) {
    const row = headers.map(header => {
      const type = schema[header];
      return generateMockValue(type, i);
    });
    rows.push(row.join(','));
  }
  
  return rows.join('\n');
};

const generateMockValue = (type: string, index: number): string => {
  if (type === 'integer') {
    return (index + 1).toString();
  }
  if (type.startsWith('integer[')) {
    const [min, max] = type.match(/\[(\d+):(\d+)\]/)!.slice(1).map(Number);
    return (min + (index % (max - min))).toString();
  }
  if (type === 'string') {
    return `"String Value ${index + 1}"`;
  }
  if (type === 'email') {
    return `"user${index + 1}@example.com"`;
  }
  if (type === 'boolean') {
    return (index % 2 === 0).toString();
  }
  if (type === 'datetime') {
    const date = new Date(2024, 0, 1 + (index % 365));
    return `"${date.toISOString()}"`;
  }
  if (type.startsWith('categorical[')) {
    const options = type.match(/\[(.+)\]/)![1].split(',');
    return `"${options[index % options.length]}"`;
  }
  if (type.startsWith('float[')) {
    const [min, max] = type.match(/\[([\d.]+):([\d.]+)\]/)!.slice(1).map(Number);
    return (min + (index / 100) * (max - min)).toFixed(2);
  }
  
  return `"Unknown Type: ${type}"`;
};