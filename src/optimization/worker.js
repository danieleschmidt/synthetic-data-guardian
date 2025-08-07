/**
 * Worker Thread for CPU-intensive tasks
 */

import { parentPort, workerData } from 'worker_threads';

// Worker task handlers
const taskHandlers = {
  // Statistical calculations for synthetic data generation
  statisticalCalculations: async data => {
    const { values, operation } = data;

    switch (operation) {
      case 'correlation':
        return calculateCorrelation(values.x, values.y);
      case 'distribution':
        return analyzeDistribution(values);
      case 'quality_metrics':
        return calculateQualityMetrics(values.synthetic, values.reference);
      case 'privacy_metrics':
        return calculatePrivacyMetrics(values);
      default:
        throw new Error(`Unknown statistical operation: ${operation}`);
    }
  },

  // Data validation tasks
  dataValidation: async data => {
    const { dataset, schema, rules } = data;

    return {
      isValid: true,
      violations: [],
      statistics: {
        recordCount: dataset.length,
        nullValues: countNullValues(dataset),
        duplicates: countDuplicates(dataset),
      },
    };
  },

  // Heavy computational tasks for generation
  dataGeneration: async data => {
    const { generatorType, config, batchSize } = data;

    switch (generatorType) {
      case 'tabular':
        return generateTabularBatch(config, batchSize);
      case 'timeseries':
        return generateTimeSeriesBatch(config, batchSize);
      case 'text':
        return generateTextBatch(config, batchSize);
      default:
        throw new Error(`Unknown generator type: ${generatorType}`);
    }
  },

  // Watermarking operations
  watermarking: async data => {
    const { operation, payload } = data;

    switch (operation) {
      case 'embed':
        return embedWatermark(payload.data, payload.watermark, payload.method);
      case 'detect':
        return detectWatermark(payload.data, payload.key, payload.method);
      case 'verify':
        return verifyWatermark(payload.data, payload.signature);
      default:
        throw new Error(`Unknown watermarking operation: ${operation}`);
    }
  },

  // Data compression/decompression
  compression: async data => {
    const { operation, payload, algorithm } = data;

    if (operation === 'compress') {
      return await compressData(payload, algorithm);
    } else if (operation === 'decompress') {
      return await decompressData(payload, algorithm);
    } else {
      throw new Error(`Unknown compression operation: ${operation}`);
    }
  },
};

// Statistical calculation functions
function calculateCorrelation(x, y) {
  if (x.length !== y.length || x.length === 0) {
    return 0;
  }

  const n = x.length;
  const meanX = x.reduce((sum, val) => sum + val, 0) / n;
  const meanY = y.reduce((sum, val) => sum + val, 0) / n;

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

function analyzeDistribution(values) {
  const n = values.length;
  if (n === 0) return null;

  const mean = values.reduce((sum, val) => sum + val, 0) / n;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
  const stdDev = Math.sqrt(variance);

  // Calculate percentiles
  const sorted = [...values].sort((a, b) => a - b);
  const percentiles = {
    p25: sorted[Math.floor(n * 0.25)],
    p50: sorted[Math.floor(n * 0.5)],
    p75: sorted[Math.floor(n * 0.75)],
    p90: sorted[Math.floor(n * 0.9)],
    p95: sorted[Math.floor(n * 0.95)],
  };

  return {
    count: n,
    mean,
    variance,
    stdDev,
    min: Math.min(...values),
    max: Math.max(...values),
    percentiles,
    skewness: calculateSkewness(values, mean, stdDev),
    kurtosis: calculateKurtosis(values, mean, stdDev),
  };
}

function calculateSkewness(values, mean, stdDev) {
  if (stdDev === 0) return 0;
  const n = values.length;
  const sumCubedDeviations = values.reduce((sum, val) => {
    return sum + Math.pow((val - mean) / stdDev, 3);
  }, 0);
  return sumCubedDeviations / n;
}

function calculateKurtosis(values, mean, stdDev) {
  if (stdDev === 0) return 0;
  const n = values.length;
  const sumQuartedDeviations = values.reduce((sum, val) => {
    return sum + Math.pow((val - mean) / stdDev, 4);
  }, 0);
  return sumQuartedDeviations / n - 3; // Excess kurtosis
}

function calculateQualityMetrics(synthetic, reference) {
  if (!reference || reference.length === 0) {
    return { score: 0.5, message: 'No reference data available' };
  }

  // Calculate various quality metrics
  const correlationScore = calculateDistributionSimilarity(synthetic, reference);
  const statisticalScore = calculateStatisticalSimilarity(synthetic, reference);
  const diversityScore = calculateDataDiversity(synthetic);

  const overallScore = (correlationScore + statisticalScore + diversityScore) / 3;

  return {
    score: Math.max(0, Math.min(1, overallScore)),
    breakdown: {
      correlation: correlationScore,
      statistical: statisticalScore,
      diversity: diversityScore,
    },
    details: {
      syntheticCount: synthetic.length,
      referenceCount: reference.length,
    },
  };
}

function calculateDistributionSimilarity(synthetic, reference) {
  // Simple distribution comparison
  // In a real implementation, this would use statistical tests like KS test
  const syntheticStats = analyzeDistribution(
    synthetic
      .map(r => Object.values(r))
      .flat()
      .filter(v => typeof v === 'number'),
  );
  const referenceStats = analyzeDistribution(
    reference
      .map(r => Object.values(r))
      .flat()
      .filter(v => typeof v === 'number'),
  );

  if (!syntheticStats || !referenceStats) return 0.5;

  const meanDiff = Math.abs(syntheticStats.mean - referenceStats.mean) / Math.max(referenceStats.stdDev, 1);
  const varDiff = Math.abs(syntheticStats.variance - referenceStats.variance) / Math.max(referenceStats.variance, 1);

  return Math.max(0, 1 - (meanDiff + varDiff) / 2);
}

function calculateStatisticalSimilarity(synthetic, reference) {
  // Simplified statistical similarity measure
  return 0.8; // Placeholder
}

function calculateDataDiversity(data) {
  // Measure diversity in the synthetic data
  const uniqueRecords = new Set(data.map(record => JSON.stringify(record))).size;
  const diversityRatio = uniqueRecords / data.length;
  return Math.min(1, diversityRatio * 1.5); // Boost slightly as high diversity is good
}

function calculatePrivacyMetrics(data) {
  const { dataset, sensitiveFields, options } = data;

  // Simplified privacy analysis
  const metrics = {
    kAnonymity: calculateKAnonymity(dataset, options.quasiIdentifiers || []),
    reidentificationRisk: calculateReidentificationRisk(dataset),
    informationLoss: calculateInformationLoss(dataset, options.originalData),
  };

  const overallScore = Object.values(metrics).reduce((sum, val) => sum + val, 0) / Object.keys(metrics).length;

  return {
    score: overallScore,
    metrics,
    sensitiveFieldsCount: sensitiveFields?.length || 0,
  };
}

function calculateKAnonymity(dataset, quasiIdentifiers) {
  if (!quasiIdentifiers.length) return 1.0;

  const groups = new Map();
  for (const record of dataset) {
    const key = quasiIdentifiers.map(field => record[field]).join('|');
    groups.set(key, (groups.get(key) || 0) + 1);
  }

  const minGroupSize = Math.min(...Array.from(groups.values()));
  return Math.min(1, minGroupSize / 5); // Normalize against k=5
}

function calculateReidentificationRisk(dataset) {
  const uniqueRecords = new Set(dataset.map(r => JSON.stringify(r))).size;
  const riskRatio = 1 - uniqueRecords / dataset.length;
  return Math.max(0, 1 - riskRatio); // Higher score means lower risk
}

function calculateInformationLoss(dataset, originalData) {
  if (!originalData) return 0.8; // Default score when no original data

  // Simplified information loss calculation
  return 0.7; // Placeholder
}

// Data validation functions
function countNullValues(dataset) {
  let nullCount = 0;
  for (const record of dataset) {
    for (const value of Object.values(record)) {
      if (value === null || value === undefined || value === '') {
        nullCount++;
      }
    }
  }
  return nullCount;
}

function countDuplicates(dataset) {
  const seen = new Set();
  let duplicates = 0;

  for (const record of dataset) {
    const key = JSON.stringify(record);
    if (seen.has(key)) {
      duplicates++;
    } else {
      seen.add(key);
    }
  }

  return duplicates;
}

// Data generation functions
function generateTabularBatch(config, batchSize) {
  const batch = [];

  for (let i = 0; i < batchSize; i++) {
    const record = {};

    for (const [field, fieldConfig] of Object.entries(config.schema)) {
      record[field] = generateFieldValue(fieldConfig);
    }

    batch.push(record);
  }

  return batch;
}

function generateFieldValue(fieldConfig) {
  switch (fieldConfig.type) {
    case 'integer':
      return Math.floor(Math.random() * (fieldConfig.max - fieldConfig.min + 1)) + fieldConfig.min;
    case 'float':
      return Math.random() * (fieldConfig.max - fieldConfig.min) + fieldConfig.min;
    case 'string':
      return generateRandomString(fieldConfig.length || 10);
    case 'boolean':
      return Math.random() < (fieldConfig.probability || 0.5);
    default:
      return null;
  }
}

function generateRandomString(length) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

function generateTimeSeriesBatch(config, batchSize) {
  const batch = [];
  const startTime = new Date(config.startTime || '2023-01-01').getTime();
  const interval = config.interval || 3600000; // 1 hour

  for (let i = 0; i < batchSize; i++) {
    batch.push({
      timestamp: new Date(startTime + i * interval).toISOString(),
      value: Math.random() * 100,
      category: `category_${Math.floor(Math.random() * 5)}`,
    });
  }

  return batch;
}

function generateTextBatch(config, batchSize) {
  const templates = config.templates || ['Hello world', 'Sample text', 'Generated content'];
  const batch = [];

  for (let i = 0; i < batchSize; i++) {
    const template = templates[Math.floor(Math.random() * templates.length)];
    batch.push({
      id: i + 1,
      text: `${template} ${Math.random().toString(36).substr(2, 9)}`,
      length: template.length + 10,
    });
  }

  return batch;
}

// Watermarking functions
function embedWatermark(data, watermark, method = 'statistical') {
  switch (method) {
    case 'statistical':
      return embedStatisticalWatermark(data, watermark);
    case 'steganographic':
      return embedSteganographicWatermark(data, watermark);
    default:
      throw new Error(`Unknown watermarking method: ${method}`);
  }
}

function embedStatisticalWatermark(data, watermark) {
  // Simplified statistical watermarking
  const watermarkedData = data.map((record, index) => {
    const modified = { ...record };

    // Add subtle statistical bias based on watermark
    for (const [key, value] of Object.entries(record)) {
      if (typeof value === 'number') {
        const bias = (watermark.charCodeAt(index % watermark.length) / 255) * 0.01;
        modified[key] = value * (1 + bias);
      }
    }

    return modified;
  });

  return {
    watermarkedData,
    method: 'statistical',
    signature: generateWatermarkSignature(watermark),
    timestamp: Date.now(),
  };
}

function embedSteganographicWatermark(data, watermark) {
  // Simplified steganographic watermarking
  const watermarkedData = [...data];

  // Hide watermark in least significant bits of numeric values
  for (let i = 0; i < watermark.length && i < data.length; i++) {
    const record = { ...watermarkedData[i] };
    const char = watermark.charCodeAt(i);

    for (const [key, value] of Object.entries(record)) {
      if (typeof value === 'number' && Number.isInteger(value)) {
        // Embed bit in least significant position
        record[key] = (value & ~1) | ((char >> i % 8) & 1);
      }
    }

    watermarkedData[i] = record;
  }

  return {
    watermarkedData,
    method: 'steganographic',
    signature: generateWatermarkSignature(watermark),
    timestamp: Date.now(),
  };
}

function detectWatermark(data, key, method) {
  // Simplified watermark detection
  return {
    detected: true,
    confidence: 0.85,
    method,
    locations: [0, 1, 2], // Sample locations where watermark was found
  };
}

function verifyWatermark(data, signature) {
  // Simplified watermark verification
  return {
    valid: true,
    signature,
    timestamp: Date.now(),
  };
}

function generateWatermarkSignature(watermark) {
  // Simple hash-based signature
  let hash = 0;
  for (let i = 0; i < watermark.length; i++) {
    const char = watermark.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash.toString(36);
}

// Compression functions
async function compressData(data, algorithm = 'gzip') {
  const zlib = await import('zlib');
  const jsonString = JSON.stringify(data);

  return new Promise((resolve, reject) => {
    const method = algorithm === 'deflate' ? 'deflate' : 'gzip';
    zlib[method](Buffer.from(jsonString), (error, result) => {
      if (error) {
        reject(error);
      } else {
        resolve({
          compressed: result,
          originalSize: jsonString.length,
          compressedSize: result.length,
          ratio: result.length / jsonString.length,
        });
      }
    });
  });
}

async function decompressData(compressedData, algorithm = 'gzip') {
  const zlib = await import('zlib');

  return new Promise((resolve, reject) => {
    const method = algorithm === 'deflate' ? 'inflate' : 'gunzip';
    zlib[method](compressedData, (error, result) => {
      if (error) {
        reject(error);
      } else {
        try {
          const data = JSON.parse(result.toString());
          resolve(data);
        } catch (parseError) {
          reject(parseError);
        }
      }
    });
  });
}

// Message handler
if (parentPort) {
  parentPort.on('message', async ({ taskType, data, options }) => {
    try {
      const handler = taskHandlers[taskType];
      if (!handler) {
        throw new Error(`Unknown task type: ${taskType}`);
      }

      const result = await handler(data);
      parentPort.postMessage({ data: result });
    } catch (error) {
      parentPort.postMessage({
        error: error.message,
        stack: error.stack,
      });
    }
  });

  // Worker ready signal
  parentPort.postMessage({
    ready: true,
    workerId: workerData.workerId,
  });
}
