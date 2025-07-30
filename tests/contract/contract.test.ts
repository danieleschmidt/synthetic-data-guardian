// Contract Testing Configuration for API Validation
// Ensures API contracts are maintained between services

import { pactWith } from 'jest-pact';
import { Interaction, Matchers } from '@pact-foundation/pact';
import axios from 'axios';

const { like, eachLike, term } = Matchers;

// Test consumer contracts for synthetic data API
pactWith(
  {
    consumer: 'SyntheticDataConsumer',
    provider: 'SyntheticDataProvider',
    port: 3001,
    dir: './tests/contract/pacts'
  },
  (provider) => {
    describe('Synthetic Data Generation API', () => {
      describe('POST /api/v1/generate', () => {
        beforeEach(() => {
          const interaction = new Interaction()
            .given('valid generation request')
            .uponReceiving('a request to generate synthetic data')
            .withRequest({
              method: 'POST',
              path: '/api/v1/generate',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': like('Bearer token123')
              },
              body: {
                pipeline: like('customer_profiles'),
                num_records: like(1000),
                format: term({ generate: 'csv|json|parquet', matcher: '^(csv|json|parquet)$' })
              }
            })
            .willRespondWith({
              status: 200,
              headers: {
                'Content-Type': 'application/json'
              },
              body: {
                id: like('gen_123456'),
                status: 'completed',
                records_generated: like(1000),
                quality_score: like(0.95),
                privacy_score: like(0.98),
                lineage_id: like('lineage_789'),
                download_url: like('https://api.example.com/download/gen_123456'),
                expires_at: like('2024-01-15T10:30:00Z')
              }
            });

          return provider.addInteraction(interaction);
        });

        it('should generate synthetic data successfully', async () => {
          const response = await axios.post(
            `${provider.mockService.baseUrl}/api/v1/generate`,
            {
              pipeline: 'customer_profiles',
              num_records: 1000,
              format: 'json'
            },
            {
              headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer token123'
              }
            }
          );

          expect(response.status).toBe(200);
          expect(response.data).toHaveProperty('id');
          expect(response.data).toHaveProperty('status', 'completed');
          expect(response.data).toHaveProperty('quality_score');
          expect(response.data).toHaveProperty('privacy_score');
        });
      });

      describe('GET /api/v1/validate', () => {
        beforeEach(() => {
          const interaction = new Interaction()
            .given('synthetic data exists')
            .uponReceiving('a request to validate synthetic data')
            .withRequest({
              method: 'POST',
              path: '/api/v1/validate',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': like('Bearer token123')
              },
              body: {
                data_url: like('s3://bucket/data.csv'),
                validators: eachLike('statistical'),
                reference_data: like('s3://bucket/real.csv')
              }
            })
            .willRespondWith({
              status: 200,
              headers: {
                'Content-Type': 'application/json'
              },
              body: {
                validation_id: like('val_123456'),
                overall_score: like(0.92),
                validators: eachLike({
                  name: 'statistical_fidelity',
                  score: 0.95,
                  details: {
                    ks_test: 0.98,
                    wasserstein_distance: 0.02,
                    correlation_score: 0.94
                  }
                }),
                recommendations: eachLike('Consider increasing sample size'),
                compliance_status: {
                  gdpr: 'compliant',
                  hipaa: 'compliant'
                }
              }
            });

          return provider.addInteraction(interaction);
        });

        it('should validate synthetic data successfully', async () => {
          const response = await axios.post(
            `${provider.mockService.baseUrl}/api/v1/validate`,
            {
              data_url: 's3://bucket/data.csv',
              validators: ['statistical'],
              reference_data: 's3://bucket/real.csv'
            },
            {
              headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer token123'
              }
            }
          );

          expect(response.status).toBe(200);
          expect(response.data).toHaveProperty('validation_id');
          expect(response.data).toHaveProperty('overall_score');
          expect(response.data.validators).toHaveLength(1);
          expect(response.data.compliance_status).toHaveProperty('gdpr');
        });
      });

      describe('GET /api/v1/lineage/{dataset_id}', () => {
        beforeEach(() => {
          const interaction = new Interaction()
            .given('dataset with lineage exists')
            .uponReceiving('a request for lineage information')
            .withRequest({
              method: 'GET',
              path: term({ generate: '/api/v1/lineage/dataset_123', matcher: '\\/api\\/v1\\/lineage\\/[a-zA-Z0-9_]+' }),
              headers: {
                'Authorization': like('Bearer token123')
              }
            })
            .willRespondWith({
              status: 200,
              headers: {
                'Content-Type': 'application/json'
              },
              body: {
                dataset_id: like('dataset_123'),
                lineage_graph: {
                  nodes: eachLike({
                    id: 'node_1',
                    type: 'source_data',
                    name: 'customers.csv',
                    timestamp: '2024-01-15T09:00:00Z'
                  }),
                  edges: eachLike({
                    source: 'node_1',
                    target: 'node_2',
                    relationship: 'generated_from'
                  })
                },
                provenance: {
                  generator: like('sdv'),
                  version: like('1.0.0'),
                  parameters: like({ model: 'gaussian_copula' }),
                  timestamp: like('2024-01-15T09:30:00Z')
                },
                audit_trail: eachLike({
                  event: 'generation_started',
                  timestamp: '2024-01-15T09:00:00Z',
                  user: 'system',
                  details: { pipeline: 'customer_profiles_v2' }
                })
              }
            });

          return provider.addInteraction(interaction);
        });

        it('should return lineage information', async () => {
          const response = await axios.get(
            `${provider.mockService.baseUrl}/api/v1/lineage/dataset_123`,
            {
              headers: {
                'Authorization': 'Bearer token123'
              }
            }
          );

          expect(response.status).toBe(200);
          expect(response.data).toHaveProperty('dataset_id', 'dataset_123');
          expect(response.data).toHaveProperty('lineage_graph');
          expect(response.data).toHaveProperty('provenance');
          expect(response.data).toHaveProperty('audit_trail');
        });
      });
    });
  }
);