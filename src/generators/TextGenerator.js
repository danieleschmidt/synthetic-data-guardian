/**
 * TextGenerator - Generates synthetic text data
 */

export class TextGenerator {
  constructor(backend, params = {}, logger) {
    this.backend = backend;
    this.params = params;
    this.logger = logger;
    this.initialized = false;
  }

  async initialize(config) {
    this.logger.info('Initializing text generator', { backend: this.backend });
    this.config = config;
    this.initialized = true;
  }

  async generate(options = {}) {
    const { numRecords, onProgress = () => {} } = options;
    
    // Mock text generation
    const templates = [
      'The customer was satisfied with the service.',
      'Product quality exceeded expectations.',
      'Delivery was prompt and efficient.',
      'Support team was helpful and responsive.',
      'Overall experience was positive.'
    ];

    const data = [];
    for (let i = 0; i < numRecords; i++) {
      const template = templates[i % templates.length];
      data.push({
        id: i + 1,
        text: template,
        sentiment: Math.random() > 0.5 ? 'positive' : 'neutral',
        length: template.length,
        words: template.split(' ').length
      });
      
      if (i % 50 === 0) {
        onProgress((i / numRecords) * 100);
      }
    }

    return data;
  }

  async cleanup() {
    this.initialized = false;
  }
}