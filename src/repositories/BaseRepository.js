/**
 * BaseRepository - Base class for data access operations
 */

export class BaseRepository {
  constructor(db, tableName, logger) {
    this.db = db;
    this.tableName = tableName;
    this.logger = logger;
  }

  async findById(id) {
    const query = `SELECT * FROM ${this.tableName} WHERE id = $1`;
    const result = await this.db.query(query, [id]);
    return result.rows[0] || null;
  }

  async findAll(limit = 100, offset = 0, orderBy = 'created_at', orderDirection = 'DESC') {
    const query = `
      SELECT * FROM ${this.tableName} 
      ORDER BY ${orderBy} ${orderDirection}
      LIMIT $1 OFFSET $2
    `;
    const result = await this.db.query(query, [limit, offset]);
    return result.rows;
  }

  async findByCondition(conditions = {}, limit = 100) {
    const whereClause = Object.keys(conditions).length > 0 
      ? 'WHERE ' + Object.keys(conditions).map((key, index) => `${key} = $${index + 1}`).join(' AND ')
      : '';

    const query = `
      SELECT * FROM ${this.tableName} 
      ${whereClause}
      ORDER BY created_at DESC
      LIMIT $${Object.keys(conditions).length + 1}
    `;

    const values = [...Object.values(conditions), limit];
    const result = await this.db.query(query, values);
    return result.rows;
  }

  async count(conditions = {}) {
    const whereClause = Object.keys(conditions).length > 0 
      ? 'WHERE ' + Object.keys(conditions).map((key, index) => `${key} = $${index + 1}`).join(' AND ')
      : '';

    const query = `SELECT COUNT(*) as count FROM ${this.tableName} ${whereClause}`;
    const values = Object.values(conditions);
    const result = await this.db.query(query, values);
    return parseInt(result.rows[0].count);
  }

  async create(data) {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const placeholders = keys.map((_, index) => `$${index + 1}`).join(', ');
    const columns = keys.join(', ');

    const query = `
      INSERT INTO ${this.tableName} (${columns})
      VALUES (${placeholders})
      RETURNING *
    `;

    const result = await this.db.query(query, values);
    return result.rows[0];
  }

  async update(id, data) {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const setClause = keys.map((key, index) => `${key} = $${index + 2}`).join(', ');

    const query = `
      UPDATE ${this.tableName} 
      SET ${setClause}, updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `;

    const result = await this.db.query(query, [id, ...values]);
    return result.rows[0] || null;
  }

  async delete(id) {
    const query = `DELETE FROM ${this.tableName} WHERE id = $1 RETURNING *`;
    const result = await this.db.query(query, [id]);
    return result.rows[0] || null;
  }

  async deleteByCondition(conditions) {
    const whereClause = Object.keys(conditions).map((key, index) => `${key} = $${index + 1}`).join(' AND ');
    const query = `DELETE FROM ${this.tableName} WHERE ${whereClause} RETURNING *`;
    const values = Object.values(conditions);
    const result = await this.db.query(query, values);
    return result.rows;
  }

  async exists(id) {
    const query = `SELECT EXISTS(SELECT 1 FROM ${this.tableName} WHERE id = $1)`;
    const result = await this.db.query(query, [id]);
    return result.rows[0].exists;
  }

  async transaction(operations) {
    const client = await this.db.getPostgresClient();
    
    try {
      await client.query('BEGIN');
      
      const results = [];
      for (const operation of operations) {
        const result = await client.query(operation.query, operation.values);
        results.push(result);
      }
      
      await client.query('COMMIT');
      return results;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
}