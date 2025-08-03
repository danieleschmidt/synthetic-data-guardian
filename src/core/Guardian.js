/**
 * Guardian - Core orchestrator for synthetic data generation and validation
 */

import { EventEmitter } from 'events';
import crypto from 'crypto';
import { GenerationPipeline } from './GenerationPipeline.js';
import { ValidationEngine } from './ValidationEngine.js';
import { LineageTracker } from './LineageTracker.js';
import { WatermarkEngine } from './WatermarkEngine.js';
import { QualityAssessment } from '../analysis/QualityAssessment.js';
import { PrivacyAnalysis } from '../analysis/PrivacyAnalysis.js';
import { GenerationResult } from '../models/GenerationResult.js';

export class Guardian extends EventEmitter {
  constructor(logger) {
    super();
    this.logger = logger;
    this.pipelines = new Map();
    this.activeTasks = new Map();
    this.validationEngine = new ValidationEngine(logger);
    this.lineageTracker = new LineageTracker(logger);
    this.watermarkEngine = new WatermarkEngine(logger);
    this.qualityAssessment = new QualityAssessment(logger);
    this.privacyAnalysis = new PrivacyAnalysis(logger);
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    try {
      this.logger.info('Initializing Synthetic Data Guardian...');

      // Initialize core components
      await this.validationEngine.initialize();
      await this.lineageTracker.initialize();
      await this.watermarkEngine.initialize();
      await this.qualityAssessment.initialize();
      await this.privacyAnalysis.initialize();

      // Setup event listeners
      this.setupEventListeners();

      this.initialized = true;
      this.logger.info('Guardian initialization completed');
      
      this.emit('initialized');
    } catch (error) {
      this.logger.error('Guardian initialization failed', { error: error.message });
      throw error;
    }
  }

  setupEventListeners() {
    // Pipeline events
    this.on('pipeline.created', (pipelineId) => {
      this.logger.info('Pipeline created', { pipelineId });
    });

    this.on('generation.started', (taskId, pipelineId) => {
      this.logger.info('Generation started', { taskId, pipelineId });
    });

    this.on('generation.progress', (taskId, progress) => {
      this.logger.debug('Generation progress', { taskId, progress });
    });

    this.on('generation.completed', (taskId, result) => {
      this.logger.info('Generation completed', { 
        taskId, 
        recordsGenerated: result.data?.length || 0,
        qualityScore: result.qualityScore 
      });
    });

    this.on('validation.completed', (taskId, report) => {
      this.logger.info('Validation completed', { 
        taskId, 
        overallScore: report.overallScore,
        passed: report.passed 
      });
    });
  }

  async generate(options) {
    const taskId = crypto.randomUUID();
    const startTime = Date.now();

    try {
      this.logger.info('Starting data generation', { taskId, options: { ...options, data: '[REDACTED]' } });

      // Validate input parameters
      this.validateGenerationOptions(options);

      // Get or create pipeline
      const pipeline = await this.getPipeline(options.pipeline);

      // Create generation task
      const task = {
        id: taskId,
        pipeline: pipeline,
        options: options,
        status: 'running',
        startTime: startTime,
        progress: 0
      };

      this.activeTasks.set(taskId, task);
      this.emit('generation.started', taskId, pipeline.id);

      // Execute generation pipeline
      const generationResult = await this.executeGeneration(task);

      // Apply validation if configured
      let validationReport = null;
      if (options.validate !== false) {
        validationReport = await this.validateResult(generationResult, pipeline);
      }

      // Apply watermarking if configured
      if (pipeline.config.watermarking?.enabled) {
        await this.applyWatermarking(generationResult, pipeline.config.watermarking);
      }

      // Track lineage
      const lineageId = await this.trackLineage(taskId, pipeline, generationResult, options);

      // Create final result
      const result = new GenerationResult({
        taskId: taskId,
        data: generationResult.data,
        metadata: generationResult.metadata,
        qualityScore: generationResult.qualityScore,
        privacyScore: generationResult.privacyScore,
        validationReport: validationReport,
        lineageId: lineageId,
        watermarkInfo: generationResult.watermarkInfo,
        executionTime: Date.now() - startTime,
        recordCount: generationResult.data?.length || 0
      });

      // Update task status
      task.status = 'completed';
      task.result = result;
      task.endTime = Date.now();

      this.emit('generation.completed', taskId, result);

      return result;

    } catch (error) {
      this.logger.error('Generation failed', { taskId, error: error.message, stack: error.stack });
      
      // Update task status
      const task = this.activeTasks.get(taskId);
      if (task) {
        task.status = 'failed';
        task.error = error.message;
        task.endTime = Date.now();
      }

      this.emit('generation.failed', taskId, error);
      throw error;
    } finally {
      // Cleanup task after delay
      setTimeout(() => {
        this.activeTasks.delete(taskId);
      }, 300000); // Keep for 5 minutes
    }
  }

  validateGenerationOptions(options) {
    if (!options) {
      throw new Error('Generation options are required');
    }

    if (!options.pipeline) {
      throw new Error('Pipeline configuration is required');
    }

    if (typeof options.numRecords !== 'number' || options.numRecords <= 0) {
      throw new Error('numRecords must be a positive number');
    }

    if (options.numRecords > 1000000) {
      throw new Error('numRecords cannot exceed 1,000,000 for single generation');
    }
  }

  async getPipeline(pipelineConfig) {
    let pipeline;

    if (typeof pipelineConfig === 'string') {
      // Pipeline ID or name
      pipeline = this.pipelines.get(pipelineConfig);
      if (!pipeline) {
        throw new Error(`Pipeline not found: ${pipelineConfig}`);
      }
    } else if (typeof pipelineConfig === 'object') {
      // Pipeline configuration object
      const pipelineId = pipelineConfig.id || crypto.randomUUID();
      
      pipeline = this.pipelines.get(pipelineId);
      if (!pipeline) {
        pipeline = new GenerationPipeline(pipelineId, this.logger);
        await pipeline.configure(pipelineConfig);
        this.pipelines.set(pipelineId, pipeline);
        this.emit('pipeline.created', pipelineId);
      }
    } else {
      throw new Error('Invalid pipeline configuration');
    }

    return pipeline;
  }

  async executeGeneration(task) {
    const { pipeline, options } = task;

    // Progress tracking
    const updateProgress = (progress) => {
      task.progress = progress;
      this.emit('generation.progress', task.id, progress);
    };

    updateProgress(10);

    // Execute the generation pipeline
    const result = await pipeline.generate({
      numRecords: options.numRecords,
      seed: options.seed,
      conditions: options.conditions,
      onProgress: updateProgress
    });

    updateProgress(80);

    // Quality assessment
    if (options.assessQuality !== false) {
      const qualityScore = await this.qualityAssessment.assess(result.data, {
        referenceData: options.referenceData,
        metrics: pipeline.config.validation?.quality?.metrics || ['statistical_fidelity']
      });
      result.qualityScore = qualityScore.overallScore;
      result.qualityReport = qualityScore;
    }

    updateProgress(90);

    // Privacy analysis
    if (options.analyzePrivacy !== false) {
      const privacyScore = await this.privacyAnalysis.analyze(result.data, {
        sensitiveColumns: pipeline.config.schema?.sensitiveColumns || [],
        epsilon: pipeline.config.validation?.privacy?.epsilon || 1.0
      });
      result.privacyScore = privacyScore.overallScore;
      result.privacyReport = privacyScore;
    }

    updateProgress(100);

    return result;
  }

  async validateResult(result, pipeline) {
    if (!pipeline.config.validation?.enabled) {
      return null;
    }

    const validators = pipeline.config.validation.validators || [];
    
    this.emit('validation.started', result.taskId);

    const validationReport = await this.validationEngine.validate(result.data, {
      validators: validators,
      thresholds: pipeline.config.validation.thresholds || {},
      referenceData: pipeline.config.validation.referenceData
    });

    this.emit('validation.completed', result.taskId, validationReport);

    return validationReport;
  }

  async applyWatermarking(result, watermarkConfig) {
    if (!watermarkConfig.enabled) return;

    const watermarkInfo = await this.watermarkEngine.embed(result.data, {
      method: watermarkConfig.method || 'statistical',
      strength: watermarkConfig.strength || 0.8,
      message: watermarkConfig.message || `synthetic:guardian:${Date.now()}`,
      key: watermarkConfig.key
    });

    result.watermarkInfo = watermarkInfo;
    result.data = watermarkInfo.watermarkedData;
  }

  async trackLineage(taskId, pipeline, result, options) {
    const lineageEvent = {
      eventId: crypto.randomUUID(),
      taskId: taskId,
      eventType: 'generation',
      timestamp: new Date().toISOString(),
      pipeline: {
        id: pipeline.id,
        name: pipeline.config.name || 'unnamed',
        version: pipeline.config.version || '1.0.0',
        generator: pipeline.config.generator
      },
      input: {
        numRecords: options.numRecords,
        seed: options.seed,
        conditions: options.conditions,
        referenceData: options.referenceData
      },
      output: {
        recordCount: result.recordCount,
        qualityScore: result.qualityScore,
        privacyScore: result.privacyScore,
        dataHash: this.calculateDataHash(result.data)
      },
      execution: {
        startTime: result.startTime,
        endTime: result.endTime,
        executionTime: result.executionTime
      }
    };

    return await this.lineageTracker.recordEvent(lineageEvent);
  }

  calculateDataHash(data) {
    const hash = crypto.createHash('sha256');
    hash.update(JSON.stringify(data));
    return hash.digest('hex');
  }

  async getStatus(taskId) {
    const task = this.activeTasks.get(taskId);
    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    return {
      taskId: task.id,
      status: task.status,
      progress: task.progress,
      startTime: task.startTime,
      endTime: task.endTime,
      executionTime: task.endTime ? task.endTime - task.startTime : Date.now() - task.startTime,
      error: task.error
    };
  }

  async getLineage(lineageId) {
    return await this.lineageTracker.getLineage(lineageId);
  }

  async validate(data, options = {}) {
    return await this.validationEngine.validate(data, options);
  }

  async watermark(data, options = {}) {
    return await this.watermarkEngine.embed(data, options);
  }

  async verifyWatermark(data, options = {}) {
    return await this.watermarkEngine.verify(data, options);
  }

  listPipelines() {
    return Array.from(this.pipelines.values()).map(pipeline => ({
      id: pipeline.id,
      name: pipeline.config.name,
      description: pipeline.config.description,
      generator: pipeline.config.generator,
      dataType: pipeline.config.dataType,
      created: pipeline.created,
      lastUsed: pipeline.lastUsed
    }));
  }

  async removePipeline(pipelineId) {
    const pipeline = this.pipelines.get(pipelineId);
    if (!pipeline) {
      throw new Error(`Pipeline not found: ${pipelineId}`);
    }

    await pipeline.cleanup();
    this.pipelines.delete(pipelineId);
    
    this.emit('pipeline.removed', pipelineId);
  }

  async cleanup() {
    this.logger.info('Starting Guardian cleanup...');

    // Cancel active tasks
    for (const [taskId, task] of this.activeTasks) {
      if (task.status === 'running') {
        task.status = 'cancelled';
        this.emit('generation.cancelled', taskId);
      }
    }

    // Cleanup components
    await this.validationEngine.cleanup();
    await this.lineageTracker.cleanup();
    await this.watermarkEngine.cleanup();
    await this.qualityAssessment.cleanup();
    await this.privacyAnalysis.cleanup();

    // Cleanup pipelines
    for (const pipeline of this.pipelines.values()) {
      await pipeline.cleanup();
    }

    this.pipelines.clear();
    this.activeTasks.clear();

    this.logger.info('Guardian cleanup completed');
  }
}