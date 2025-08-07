/**
 * LineageTracker - Comprehensive lineage and provenance tracking system
 */

import crypto from 'crypto';

export class LineageTracker {
  constructor(logger) {
    this.logger = logger;
    this.lineageStore = new Map(); // In-memory store (would be Neo4j in production)
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    this.logger.info('Initializing Lineage Tracker...');
    this.initialized = true;
    this.logger.info('Lineage Tracker initialized');
  }

  async recordEvent(lineageEvent) {
    if (!this.initialized) {
      throw new Error('LineageTracker not initialized');
    }

    const lineageId = crypto.randomUUID();

    const lineageRecord = {
      id: lineageId,
      ...lineageEvent,
      recordedAt: new Date().toISOString(),
    };

    // Store lineage record
    this.lineageStore.set(lineageId, lineageRecord);

    // Create relationships
    this.createLineageRelationships(lineageRecord);

    this.logger.info('Lineage event recorded', {
      lineageId,
      eventType: lineageEvent.eventType,
      taskId: lineageEvent.taskId,
    });

    return lineageId;
  }

  createLineageRelationships(lineageRecord) {
    const relationships = {
      pipelineId: lineageRecord.pipeline?.id,
      taskId: lineageRecord.taskId,
      parentDatasets: lineageRecord.input?.referenceData ? [lineageRecord.input.referenceData] : [],
      generatedDataset: lineageRecord.output?.dataHash,
    };

    this.lineageStore.set(`${lineageRecord.id}:relationships`, relationships);
  }

  async getLineage(lineageId) {
    if (!this.initialized) {
      throw new Error('LineageTracker not initialized');
    }

    const lineageRecord = this.lineageStore.get(lineageId);
    if (!lineageRecord) {
      throw new Error(`Lineage record not found: ${lineageId}`);
    }

    const relationships = this.lineageStore.get(`${lineageId}:relationships`) || {};

    return {
      lineage: lineageRecord,
      relationships: relationships,
      graph: this.buildLineageGraph(lineageId),
    };
  }

  buildLineageGraph(lineageId) {
    const visited = new Set();
    const nodes = [];
    const edges = [];

    const traverse = currentId => {
      if (visited.has(currentId)) return;
      visited.add(currentId);

      const record = this.lineageStore.get(currentId);
      if (!record) return;

      const relationships = this.lineageStore.get(`${currentId}:relationships`) || {};

      nodes.push({
        id: currentId,
        type: record.eventType || 'unknown',
        timestamp: record.timestamp,
        pipeline: record.pipeline?.name || 'unknown',
        metadata: {
          generator: record.pipeline?.generator,
          recordCount: record.output?.recordCount,
          qualityScore: record.output?.qualityScore,
        },
      });

      if (relationships.parentDatasets) {
        for (const parentId of relationships.parentDatasets) {
          edges.push({
            from: parentId,
            to: currentId,
            type: 'derived_from',
          });
        }
      }
    };

    traverse(lineageId);
    return { nodes, edges };
  }

  async findDerivatives(sourceDataHash) {
    const derivatives = [];

    for (const [id, record] of this.lineageStore.entries()) {
      if (id.includes(':relationships')) continue;

      const relationships = this.lineageStore.get(`${id}:relationships`);
      if (relationships?.parentDatasets?.includes(sourceDataHash)) {
        derivatives.push({
          lineageId: id,
          pipeline: record.pipeline?.name,
          timestamp: record.timestamp,
          recordCount: record.output?.recordCount,
        });
      }
    }

    return derivatives;
  }

  async generateAuditTrail(datasetHash, format = 'json') {
    const auditEntries = [];

    for (const [id, record] of this.lineageStore.entries()) {
      if (id.includes(':relationships')) continue;

      if (record.output?.dataHash === datasetHash) {
        auditEntries.push({
          timestamp: record.timestamp,
          eventType: record.eventType,
          pipeline: record.pipeline,
          input: record.input,
          output: record.output,
          execution: record.execution,
        });
      }
    }

    auditEntries.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    if (format === 'json') {
      return {
        dataset: datasetHash,
        auditTrail: auditEntries,
        generatedAt: new Date().toISOString(),
        totalEvents: auditEntries.length,
      };
    }

    return auditEntries;
  }

  async getLineageStats() {
    const stats = {
      totalRecords: 0,
      eventTypes: {},
      pipelines: {},
      timeRange: { earliest: null, latest: null },
    };

    for (const [id, record] of this.lineageStore.entries()) {
      if (id.includes(':relationships')) continue;

      stats.totalRecords++;

      const eventType = record.eventType || 'unknown';
      stats.eventTypes[eventType] = (stats.eventTypes[eventType] || 0) + 1;

      const pipelineName = record.pipeline?.name || 'unknown';
      stats.pipelines[pipelineName] = (stats.pipelines[pipelineName] || 0) + 1;

      const timestamp = new Date(record.timestamp);
      if (!stats.timeRange.earliest || timestamp < stats.timeRange.earliest) {
        stats.timeRange.earliest = timestamp;
      }
      if (!stats.timeRange.latest || timestamp > stats.timeRange.latest) {
        stats.timeRange.latest = timestamp;
      }
    }

    return stats;
  }

  async exportLineage(format = 'json') {
    const allRecords = [];

    for (const [id, record] of this.lineageStore.entries()) {
      if (!id.includes(':relationships')) {
        const relationships = this.lineageStore.get(`${id}:relationships`) || {};
        allRecords.push({
          ...record,
          relationships,
        });
      }
    }

    if (format === 'json') {
      return {
        exportType: 'lineage',
        exportedAt: new Date().toISOString(),
        recordCount: allRecords.length,
        records: allRecords,
      };
    }

    throw new Error(`Unsupported export format: ${format}`);
  }

  async cleanup() {
    this.logger.info('Cleaning up Lineage Tracker');
    this.lineageStore.clear();
    this.initialized = false;
  }
}
