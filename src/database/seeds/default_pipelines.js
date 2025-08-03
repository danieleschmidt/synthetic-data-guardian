/**
 * Default Pipeline Seeds - Sample pipeline configurations
 */

export const defaultPipelines = [
  {
    name: 'customer-profiles-demo',
    description: 'Demo customer profile generator with realistic data patterns',
    generator: 'sdv',
    dataType: 'tabular',
    config: {
      params: {
        model: 'gaussian_copula',
        epochs: 100
      },
      schema: {
        customer_id: { type: 'uuid' },
        first_name: { type: 'string', length: 15 },
        last_name: { type: 'string', length: 20 },
        email: { type: 'email', domains: ['gmail.com', 'yahoo.com', 'company.com'] },
        age: { type: 'integer', min: 18, max: 80 },
        income: { type: 'float', min: 25000, max: 150000 },
        credit_score: { type: 'integer', min: 300, max: 850 },
        registration_date: { 
          type: 'datetime', 
          start: '2020-01-01', 
          end: '2024-01-01' 
        },
        is_premium: { type: 'boolean', probability: 0.3 },
        preferred_category: { 
          type: 'categorical',
          categories: ['electronics', 'clothing', 'books', 'home', 'sports'],
          weights: [0.3, 0.25, 0.15, 0.2, 0.1]
        }
      },
      validation: {
        enabled: true,
        validators: ['statistical_fidelity', 'privacy_preservation'],
        thresholds: {
          statistical_fidelity: 0.9,
          privacy_preservation: 0.8
        }
      },
      watermarking: {
        enabled: true,
        method: 'statistical',
        strength: 0.8
      }
    },
    tags: ['demo', 'customer', 'ecommerce']
  },

  {
    name: 'financial-transactions',
    description: 'Synthetic financial transaction data for fraud detection',
    generator: 'ctgan',
    dataType: 'tabular',
    config: {
      params: {
        epochs: 300,
        batchSize: 500,
        pac: 10
      },
      schema: {
        transaction_id: { type: 'uuid' },
        user_id: { type: 'integer', min: 1, max: 100000 },
        amount: { type: 'float', min: 0.01, max: 10000, precision: 2 },
        timestamp: { 
          type: 'datetime',
          start: '2023-01-01',
          end: '2024-01-01'
        },
        merchant_category: {
          type: 'categorical',
          categories: ['retail', 'food', 'transport', 'utilities', 'entertainment', 'healthcare'],
          weights: [0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
        },
        payment_method: {
          type: 'categorical',
          categories: ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'],
          weights: [0.4, 0.3, 0.2, 0.1]
        },
        is_fraud: { type: 'boolean', probability: 0.02 },
        location: {
          type: 'categorical',
          categories: ['US-CA', 'US-NY', 'US-TX', 'US-FL', 'US-WA'],
          weights: [0.3, 0.25, 0.2, 0.15, 0.1]
        }
      },
      businessRules: [
        {
          type: 'conditional_field',
          condition: { field: 'is_fraud', operator: 'equals', value: true },
          field: 'amount',
          value: 'random(1000, 10000)' // Fraudulent transactions tend to be higher
        }
      ],
      validation: {
        enabled: true,
        validators: ['statistical_fidelity', 'bias_detection'],
        thresholds: {
          statistical_fidelity: 0.85,
          bias_detection: 0.9
        }
      }
    },
    tags: ['financial', 'fraud-detection', 'security']
  },

  {
    name: 'iot-sensor-data',
    description: 'Time series IoT sensor readings for anomaly detection',
    generator: 'basic',
    dataType: 'timeseries',
    config: {
      params: {
        sequenceLength: 1440, // 24 hours of minute-by-minute data
        features: ['temperature', 'humidity', 'pressure', 'light_level']
      },
      schema: {
        device_id: { type: 'string', pattern: 'SENSOR_[0-9]{4}' },
        timestamp: { type: 'datetime' },
        temperature: { type: 'float', min: -10, max: 50 },
        humidity: { type: 'float', min: 0, max: 100 },
        pressure: { type: 'float', min: 980, max: 1050 },
        light_level: { type: 'integer', min: 0, max: 1000 },
        battery_level: { type: 'float', min: 0, max: 100 },
        is_anomaly: { type: 'boolean', probability: 0.05 }
      },
      correlations: {
        temperature: { humidity: -0.6, pressure: 0.3 },
        humidity: { temperature: -0.6, light_level: -0.4 },
        pressure: { temperature: 0.3 }
      },
      validation: {
        enabled: true,
        validators: ['statistical_fidelity'],
        thresholds: {
          statistical_fidelity: 0.9
        }
      }
    },
    tags: ['iot', 'timeseries', 'monitoring']
  },

  {
    name: 'product-reviews',
    description: 'Synthetic product review text data',
    generator: 'basic',
    dataType: 'text',
    config: {
      params: {
        model: 'basic_template',
        maxLength: 500
      },
      schema: {
        review_id: { type: 'uuid' },
        product_id: { type: 'string', pattern: 'PROD_[0-9]{6}' },
        user_id: { type: 'string', pattern: 'USER_[0-9]{8}' },
        rating: { type: 'integer', min: 1, max: 5 },
        title: { type: 'text', length: 50 },
        content: { type: 'text', length: 300 },
        sentiment: {
          type: 'categorical',
          categories: ['positive', 'negative', 'neutral'],
          weights: [0.6, 0.2, 0.2]
        },
        verified_purchase: { type: 'boolean', probability: 0.8 },
        helpful_votes: { type: 'integer', min: 0, max: 100 },
        review_date: {
          type: 'datetime',
          start: '2023-01-01',
          end: '2024-01-01'
        }
      },
      templates: {
        positive: [
          'Great product! Highly recommend.',
          'Excellent quality and fast delivery.',
          'Perfect for my needs, very satisfied.',
          'Outstanding value for money.'
        ],
        negative: [
          'Product did not meet expectations.',
          'Poor quality, would not recommend.',
          'Delivery was delayed and item damaged.',
          'Not worth the price paid.'
        ],
        neutral: [
          'Product is okay, nothing special.',
          'Average quality, does the job.',
          'Decent product for the price.',
          'Standard quality, no complaints.'
        ]
      },
      validation: {
        enabled: true,
        validators: ['statistical_fidelity'],
        thresholds: {
          statistical_fidelity: 0.8
        }
      }
    },
    tags: ['text', 'reviews', 'sentiment']
  },

  {
    name: 'social-network-graph',
    description: 'Synthetic social network graph data',
    generator: 'basic',
    dataType: 'graph',
    config: {
      params: {
        nodeCount: 1000,
        avgDegree: 8,
        communityStructure: true
      },
      nodeSchema: {
        user_id: { type: 'uuid' },
        username: { type: 'string', pattern: 'user_[0-9]{6}' },
        age: { type: 'integer', min: 13, max: 80 },
        location: {
          type: 'categorical',
          categories: ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP'],
          weights: [0.4, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05]
        },
        follower_count: { type: 'integer', min: 0, max: 10000 },
        following_count: { type: 'integer', min: 0, max: 5000 },
        account_type: {
          type: 'categorical',
          categories: ['personal', 'business', 'influencer'],
          weights: [0.8, 0.15, 0.05]
        },
        created_date: {
          type: 'datetime',
          start: '2020-01-01',
          end: '2024-01-01'
        }
      },
      edgeSchema: {
        relationship_type: {
          type: 'categorical',
          categories: ['follows', 'friend', 'blocked'],
          weights: [0.7, 0.25, 0.05]
        },
        connection_strength: { type: 'float', min: 0, max: 1 },
        created_date: {
          type: 'datetime',
          start: '2020-01-01',
          end: '2024-01-01'
        }
      },
      validation: {
        enabled: true,
        validators: ['statistical_fidelity'],
        thresholds: {
          statistical_fidelity: 0.85
        }
      }
    },
    tags: ['graph', 'social-network', 'analytics']
  },

  {
    name: 'medical-records-hipaa',
    description: 'HIPAA-compliant synthetic medical records',
    generator: 'sdv',
    dataType: 'tabular',
    config: {
      params: {
        model: 'gaussian_copula',
        epochs: 200,
        privacyMode: 'differential'
      },
      schema: {
        patient_id: { type: 'uuid' },
        age: { type: 'integer', min: 0, max: 120 },
        gender: {
          type: 'categorical',
          categories: ['M', 'F', 'O'],
          weights: [0.49, 0.49, 0.02]
        },
        diagnosis_code: {
          type: 'categorical',
          categories: ['J44.0', 'I25.9', 'E11.9', 'M54.5', 'F32.9'],
          weights: [0.2, 0.2, 0.2, 0.2, 0.2]
        },
        admission_date: {
          type: 'datetime',
          start: '2023-01-01',
          end: '2024-01-01'
        },
        length_of_stay: { type: 'integer', min: 1, max: 30 },
        total_cost: { type: 'float', min: 500, max: 50000 },
        insurance_type: {
          type: 'categorical',
          categories: ['private', 'medicare', 'medicaid', 'uninsured'],
          weights: [0.6, 0.2, 0.15, 0.05]
        },
        severity_score: { type: 'integer', min: 1, max: 10 }
      },
      validation: {
        enabled: true,
        validators: ['privacy_preservation', 'hipaa_compliance'],
        thresholds: {
          privacy_preservation: 0.95,
          hipaa_compliance: 1.0
        }
      },
      privacy: {
        epsilon: 1.0,
        delta: 1e-5,
        sensitiveColumns: ['patient_id', 'diagnosis_code']
      },
      compliance: {
        framework: 'hipaa',
        safeharbor: true,
        deidentification: 'expert_determination'
      }
    },
    tags: ['medical', 'hipaa', 'privacy', 'healthcare']
  }
];

export async function seedDefaultPipelines(db, logger) {
  logger.info('Seeding default pipelines...');

  for (const pipeline of defaultPipelines) {
    try {
      const query = `
        INSERT INTO pipelines (name, description, generator, data_type, config, tags)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (name) DO UPDATE SET
          description = EXCLUDED.description,
          config = EXCLUDED.config,
          tags = EXCLUDED.tags,
          updated_at = NOW()
        RETURNING id, name
      `;

      const values = [
        pipeline.name,
        pipeline.description,
        pipeline.generator,
        pipeline.dataType,
        JSON.stringify(pipeline.config),
        pipeline.tags
      ];

      const result = await db.query(query, values);
      logger.info('Pipeline seeded', { 
        id: result.rows[0].id, 
        name: result.rows[0].name 
      });

    } catch (error) {
      logger.error('Failed to seed pipeline', { 
        name: pipeline.name, 
        error: error.message 
      });
    }
  }

  logger.info('Default pipeline seeding completed');
}