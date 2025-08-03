/**
 * API Routes - RESTful endpoints for synthetic data operations
 */

import express from 'express';
import Joi from 'joi';
import { GenerationPipeline } from '../core/GenerationPipeline.js';

export function apiRouter(guardian, logger) {
  const router = express.Router();

  // Validation schemas
  const generateSchema = Joi.object({
    pipeline: Joi.alternatives().try(
      Joi.string(),
      Joi.object({
        name: Joi.string().required(),
        description: Joi.string(),
        generator: Joi.string().required(),
        dataType: Joi.string().valid('tabular', 'timeseries', 'text', 'image', 'graph').required(),
        schema: Joi.object(),
        params: Joi.object(),
        validation: Joi.object(),
        watermarking: Joi.object()
      })
    ).required(),
    numRecords: Joi.number().integer().min(1).max(1000000).required(),
    seed: Joi.number().integer().min(0),
    conditions: Joi.object(),
    validate: Joi.boolean().default(true),
    assessQuality: Joi.boolean().default(true),
    analyzePrivacy: Joi.boolean().default(true)
  });

  const validateSchema = Joi.object({
    data: Joi.alternatives().try(Joi.array(), Joi.string()).required(),
    validators: Joi.array().items(Joi.string()),
    referenceData: Joi.alternatives().try(Joi.array(), Joi.string()),
    thresholds: Joi.object()
  });

  const watermarkSchema = Joi.object({
    data: Joi.alternatives().try(Joi.array(), Joi.string()).required(),
    method: Joi.string().valid('statistical', 'stegastamp', 'robust').default('statistical'),
    strength: Joi.number().min(0).max(1).default(0.8),
    message: Joi.string(),
    key: Joi.string()
  });

  // Generate synthetic data
  router.post('/generate', async (req, res) => {
    try {
      // Validate request
      const { error, value } = generateSchema.validate(req.body);
      if (error) {
        return res.status(400).json({
          error: 'Validation Error',
          message: error.details[0].message,
          timestamp: new Date().toISOString()
        });
      }

      // Log request
      logger.info('Generation request received', { 
        pipeline: typeof value.pipeline === 'string' ? value.pipeline : value.pipeline.name,
        numRecords: value.numRecords,
        requestId: req.id 
      });

      // Generate data
      const result = await guardian.generate(value);

      // Return result
      res.json({
        success: true,
        result: {
          taskId: result.taskId,
          recordCount: result.recordCount,
          qualityScore: result.qualityScore,
          privacyScore: result.privacyScore,
          lineageId: result.lineageId,
          executionTime: result.executionTime,
          timestamp: result.timestamp,
          data: req.query.includeData === 'true' ? result.data : undefined,
          metadata: result.metadata,
          summary: result.summary
        }
      });

    } catch (error) {
      logger.error('Generation failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Generation Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Validate synthetic data
  router.post('/validate', async (req, res) => {
    try {
      const { error, value } = validateSchema.validate(req.body);
      if (error) {
        return res.status(400).json({
          error: 'Validation Error',
          message: error.details[0].message,
          timestamp: new Date().toISOString()
        });
      }

      logger.info('Validation request received', { requestId: req.id });

      const validationReport = await guardian.validate(value.data, {
        validators: value.validators,
        referenceData: value.referenceData,
        thresholds: value.thresholds
      });

      res.json({
        success: true,
        validationReport: validationReport
      });

    } catch (error) {
      logger.error('Validation failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Validation Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Apply watermarking
  router.post('/watermark', async (req, res) => {
    try {
      const { error, value } = watermarkSchema.validate(req.body);
      if (error) {
        return res.status(400).json({
          error: 'Validation Error',
          message: error.details[0].message,
          timestamp: new Date().toISOString()
        });
      }

      logger.info('Watermarking request received', { 
        method: value.method,
        requestId: req.id 
      });

      const watermarkResult = await guardian.watermark(value.data, {
        method: value.method,
        strength: value.strength,
        message: value.message,
        key: value.key
      });

      res.json({
        success: true,
        watermarkInfo: watermarkResult
      });

    } catch (error) {
      logger.error('Watermarking failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Watermarking Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Verify watermark
  router.post('/watermark/verify', async (req, res) => {
    try {
      const { data, key, method = 'statistical' } = req.body;

      if (!data) {
        return res.status(400).json({
          error: 'Validation Error',
          message: 'Data is required',
          timestamp: new Date().toISOString()
        });
      }

      logger.info('Watermark verification request received', { requestId: req.id });

      const verificationResult = await guardian.verifyWatermark(data, {
        method: method,
        key: key
      });

      res.json({
        success: true,
        verification: verificationResult
      });

    } catch (error) {
      logger.error('Watermark verification failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Verification Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Get generation status
  router.get('/status/:taskId', async (req, res) => {
    try {
      const { taskId } = req.params;

      if (!taskId) {
        return res.status(400).json({
          error: 'Validation Error',
          message: 'Task ID is required',
          timestamp: new Date().toISOString()
        });
      }

      const status = await guardian.getStatus(taskId);

      res.json({
        success: true,
        status: status
      });

    } catch (error) {
      if (error.message.includes('not found')) {
        res.status(404).json({
          error: 'Not Found',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      } else {
        logger.error('Status check failed', { error: error.message, requestId: req.id });
        
        res.status(500).json({
          error: 'Status Check Failed',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  });

  // Get lineage information
  router.get('/lineage/:lineageId', async (req, res) => {
    try {
      const { lineageId } = req.params;

      if (!lineageId) {
        return res.status(400).json({
          error: 'Validation Error',
          message: 'Lineage ID is required',
          timestamp: new Date().toISOString()
        });
      }

      const lineage = await guardian.getLineage(lineageId);

      res.json({
        success: true,
        lineage: lineage
      });

    } catch (error) {
      if (error.message.includes('not found')) {
        res.status(404).json({
          error: 'Not Found',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      } else {
        logger.error('Lineage retrieval failed', { error: error.message, requestId: req.id });
        
        res.status(500).json({
          error: 'Lineage Retrieval Failed',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  });

  // List pipelines
  router.get('/pipelines', async (req, res) => {
    try {
      const pipelines = guardian.listPipelines();

      res.json({
        success: true,
        pipelines: pipelines,
        count: pipelines.length
      });

    } catch (error) {
      logger.error('Pipeline listing failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Pipeline Listing Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Create pipeline
  router.post('/pipelines', async (req, res) => {
    try {
      const pipelineConfig = req.body;

      if (!pipelineConfig.name || !pipelineConfig.generator || !pipelineConfig.dataType) {
        return res.status(400).json({
          error: 'Validation Error',
          message: 'name, generator, and dataType are required',
          timestamp: new Date().toISOString()
        });
      }

      // Create new pipeline
      const pipeline = new GenerationPipeline(null, logger);
      await pipeline.configure(pipelineConfig);

      // Register with guardian
      guardian.pipelines.set(pipeline.id, pipeline);

      res.status(201).json({
        success: true,
        pipeline: {
          id: pipeline.id,
          name: pipelineConfig.name,
          description: pipelineConfig.description,
          generator: pipelineConfig.generator,
          dataType: pipelineConfig.dataType,
          created: pipeline.created
        }
      });

    } catch (error) {
      logger.error('Pipeline creation failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Pipeline Creation Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Delete pipeline
  router.delete('/pipelines/:pipelineId', async (req, res) => {
    try {
      const { pipelineId } = req.params;

      await guardian.removePipeline(pipelineId);

      res.json({
        success: true,
        message: 'Pipeline deleted successfully'
      });

    } catch (error) {
      if (error.message.includes('not found')) {
        res.status(404).json({
          error: 'Not Found',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      } else {
        logger.error('Pipeline deletion failed', { error: error.message, requestId: req.id });
        
        res.status(500).json({
          error: 'Pipeline Deletion Failed',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  });

  // Get system info
  router.get('/info', (req, res) => {
    res.json({
      success: true,
      system: {
        name: 'Synthetic Data Guardian',
        version: process.env.npm_package_version || '1.0.0',
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development',
        nodeVersion: process.version,
        platform: process.platform,
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
      },
      guardian: {
        initialized: guardian.initialized,
        activeTasks: guardian.activeTasks.size,
        registeredPipelines: guardian.pipelines.size
      }
    });
  });

  // Data export endpoint
  router.get('/export/:taskId', async (req, res) => {
    try {
      const { taskId } = req.params;
      const { format = 'json' } = req.query;

      // Get task result
      const task = guardian.activeTasks.get(taskId);
      if (!task || !task.result) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Task not found or no result available',
          timestamp: new Date().toISOString()
        });
      }

      const result = task.result;

      // Set appropriate headers
      switch (format.toLowerCase()) {
        case 'csv':
          res.setHeader('Content-Type', 'text/csv');
          res.setHeader('Content-Disposition', `attachment; filename="synthetic_data_${taskId}.csv"`);
          res.send(result.exportData('csv'));
          break;

        case 'tsv':
          res.setHeader('Content-Type', 'text/tab-separated-values');
          res.setHeader('Content-Disposition', `attachment; filename="synthetic_data_${taskId}.tsv"`);
          res.send(result.exportData('tsv'));
          break;

        case 'json':
        default:
          res.setHeader('Content-Type', 'application/json');
          res.setHeader('Content-Disposition', `attachment; filename="synthetic_data_${taskId}.json"`);
          res.send(result.exportData('json'));
          break;
      }

    } catch (error) {
      logger.error('Data export failed', { error: error.message, requestId: req.id });
      
      res.status(500).json({
        error: 'Export Failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  return router;
}