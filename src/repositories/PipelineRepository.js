/**
 * PipelineRepository - Data access for pipeline configurations
 */

import { BaseRepository } from './BaseRepository.js';

export class PipelineRepository extends BaseRepository {
  constructor(db, logger) {
    super(db, 'pipelines', logger);
  }

  async findByName(name) {
    const query = `SELECT * FROM ${this.tableName} WHERE name = $1`;
    const result = await this.db.query(query, [name]);
    return result.rows[0] || null;
  }

  async findActive(limit = 100) {
    const query = `
      SELECT * FROM ${this.tableName} 
      WHERE is_active = true 
      ORDER BY created_at DESC 
      LIMIT $1
    `;
    const result = await this.db.query(query, [limit]);
    return result.rows;
  }

  async findByGenerator(generator, limit = 100) {
    const query = `
      SELECT * FROM ${this.tableName} 
      WHERE generator = $1 AND is_active = true
      ORDER BY created_at DESC 
      LIMIT $2
    `;
    const result = await this.db.query(query, [generator, limit]);
    return result.rows;
  }

  async findByDataType(dataType, limit = 100) {
    const query = `
      SELECT * FROM ${this.tableName} 
      WHERE data_type = $1 AND is_active = true
      ORDER BY created_at DESC 
      LIMIT $2
    `;
    const result = await this.db.query(query, [dataType, limit]);
    return result.rows;
  }

  async findByTags(tags, limit = 100) {
    const query = `
      SELECT * FROM ${this.tableName} 
      WHERE tags && $1 AND is_active = true
      ORDER BY created_at DESC 
      LIMIT $2
    `;
    const result = await this.db.query(query, [tags, limit]);
    return result.rows;
  }

  async getStatistics(pipelineId) {
    const query = `SELECT * FROM get_pipeline_stats($1)`;
    const result = await this.db.query(query, [pipelineId]);
    return result.rows[0] || null;
  }

  async getRecentActivity(pipelineId, limit = 10) {
    const query = `
      SELECT 
        gt.id,
        gt.status,
        gt.num_records,
        gt.progress,
        gt.started_at,
        gt.completed_at,
        gt.execution_time_ms,
        gr.quality_score,
        gr.privacy_score,
        gr.record_count
      FROM generation_tasks gt
      LEFT JOIN generation_results gr ON gt.id = gr.task_id
      WHERE gt.pipeline_id = $1
      ORDER BY gt.created_at DESC
      LIMIT $2
    `;
    const result = await this.db.query(query, [pipelineId, limit]);
    return result.rows;
  }

  async deactivate(id) {
    return await this.update(id, { is_active: false });
  }

  async activate(id) {
    return await this.update(id, { is_active: true });
  }

  async incrementVersion(id) {
    const query = `
      UPDATE ${this.tableName} 
      SET version = version + 1, updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.db.query(query, [id]);
    return result.rows[0] || null;
  }

  async search(searchTerm, limit = 100) {
    const query = `
      SELECT * FROM ${this.tableName} 
      WHERE (
        name ILIKE $1 OR 
        description ILIKE $1 OR 
        generator ILIKE $1 OR
        $2 = ANY(tags)
      ) AND is_active = true
      ORDER BY 
        CASE WHEN name ILIKE $1 THEN 1 ELSE 2 END,
        created_at DESC
      LIMIT $3
    `;
    const searchPattern = `%${searchTerm}%`;
    const result = await this.db.query(query, [searchPattern, searchTerm, limit]);
    return result.rows;
  }

  async getDashboardData() {
    const query = `
      SELECT 
        COUNT(*) as total_pipelines,
        COUNT(CASE WHEN is_active = true THEN 1 END) as active_pipelines,
        COUNT(DISTINCT generator) as unique_generators,
        COUNT(DISTINCT data_type) as unique_data_types,
        json_object_agg(
          generator, 
          COUNT(CASE WHEN generator = pipelines.generator THEN 1 END)
        ) as generator_distribution,
        json_object_agg(
          data_type, 
          COUNT(CASE WHEN data_type = pipelines.data_type THEN 1 END)
        ) as data_type_distribution
      FROM ${this.tableName}
      WHERE is_active = true
    `;
    const result = await this.db.query(query);
    return result.rows[0] || null;
  }

  async getPerformanceMetrics(timeWindow = '24 hours') {
    const query = `
      SELECT 
        p.id,
        p.name,
        p.generator,
        COUNT(gt.id) as total_executions,
        COUNT(CASE WHEN gt.status = 'completed' THEN 1 END) as successful_executions,
        COUNT(CASE WHEN gt.status = 'failed' THEN 1 END) as failed_executions,
        AVG(gt.execution_time_ms) as avg_execution_time,
        AVG(gr.quality_score) as avg_quality_score,
        AVG(gr.privacy_score) as avg_privacy_score,
        SUM(gr.record_count) as total_records_generated
      FROM ${this.tableName} p
      LEFT JOIN generation_tasks gt ON p.id = gt.pipeline_id
      LEFT JOIN generation_results gr ON gt.id = gr.task_id
      WHERE gt.created_at > NOW() - INTERVAL $1
      GROUP BY p.id, p.name, p.generator
      ORDER BY total_executions DESC
    `;
    const result = await this.db.query(query, [timeWindow]);
    return result.rows;
  }

  async createWithValidation(data) {
    // Validate required fields
    const required = ['name', 'generator', 'data_type'];
    for (const field of required) {
      if (!data[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }

    // Check for duplicate names
    const existing = await this.findByName(data.name);
    if (existing) {
      throw new Error(`Pipeline with name '${data.name}' already exists`);
    }

    // Set defaults
    const pipelineData = {
      ...data,
      config: data.config || {},
      is_active: data.is_active !== undefined ? data.is_active : true,
      version: 1,
      tags: data.tags || []
    };

    return await this.create(pipelineData);
  }

  async updateConfig(id, config) {
    const query = `
      UPDATE ${this.tableName} 
      SET config = $2, version = version + 1, updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.db.query(query, [id, JSON.stringify(config)]);
    return result.rows[0] || null;
  }

  async addTag(id, tag) {
    const query = `
      UPDATE ${this.tableName} 
      SET tags = array_append(tags, $2), updated_at = NOW()
      WHERE id = $1 AND NOT ($2 = ANY(tags))
      RETURNING *
    `;
    const result = await this.db.query(query, [id, tag]);
    return result.rows[0] || null;
  }

  async removeTag(id, tag) {
    const query = `
      UPDATE ${this.tableName} 
      SET tags = array_remove(tags, $2), updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.db.query(query, [id, tag]);
    return result.rows[0] || null;
  }
}