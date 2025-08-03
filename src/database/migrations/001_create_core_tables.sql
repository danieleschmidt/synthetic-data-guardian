-- Synthetic Data Guardian - Core Database Schema
-- Migration 001: Create core tables for application data

BEGIN;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for additional cryptographic functions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- PIPELINES TABLE - Store generation pipeline configurations
-- =============================================================================
CREATE TABLE pipelines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    generator VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL CHECK (data_type IN ('tabular', 'timeseries', 'text', 'image', 'graph')),
    config JSONB NOT NULL DEFAULT '{}',
    schema_definition JSONB,
    validation_config JSONB,
    watermarking_config JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    version INTEGER DEFAULT 1,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[]
);

-- Create indexes for pipelines
CREATE INDEX idx_pipelines_name ON pipelines(name);
CREATE INDEX idx_pipelines_generator ON pipelines(generator);
CREATE INDEX idx_pipelines_data_type ON pipelines(data_type);
CREATE INDEX idx_pipelines_created_at ON pipelines(created_at);
CREATE INDEX idx_pipelines_is_active ON pipelines(is_active);

-- =============================================================================
-- GENERATION_TASKS TABLE - Track generation task execution
-- =============================================================================
CREATE TABLE generation_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID REFERENCES pipelines(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    num_records INTEGER NOT NULL CHECK (num_records > 0),
    seed INTEGER,
    conditions JSONB DEFAULT '{}',
    options JSONB DEFAULT '{}',
    progress FLOAT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    error_stack TEXT,
    execution_time_ms INTEGER,
    memory_usage_mb FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Create indexes for generation_tasks
CREATE INDEX idx_generation_tasks_pipeline_id ON generation_tasks(pipeline_id);
CREATE INDEX idx_generation_tasks_status ON generation_tasks(status);
CREATE INDEX idx_generation_tasks_created_at ON generation_tasks(created_at);
CREATE INDEX idx_generation_tasks_started_at ON generation_tasks(started_at);

-- =============================================================================
-- GENERATION_RESULTS TABLE - Store generation results metadata
-- =============================================================================
CREATE TABLE generation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES generation_tasks(id) ON DELETE CASCADE,
    record_count INTEGER NOT NULL,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    privacy_score FLOAT CHECK (privacy_score >= 0 AND privacy_score <= 1),
    data_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of generated data
    data_size_bytes BIGINT,
    data_storage_path TEXT,
    metadata JSONB DEFAULT '{}',
    validation_report JSONB,
    watermark_info JSONB,
    lineage_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE -- Data retention policy
);

-- Create indexes for generation_results
CREATE INDEX idx_generation_results_task_id ON generation_results(task_id);
CREATE INDEX idx_generation_results_lineage_id ON generation_results(lineage_id);
CREATE INDEX idx_generation_results_quality_score ON generation_results(quality_score);
CREATE INDEX idx_generation_results_privacy_score ON generation_results(privacy_score);
CREATE INDEX idx_generation_results_created_at ON generation_results(created_at);
CREATE INDEX idx_generation_results_expires_at ON generation_results(expires_at);

-- =============================================================================
-- VALIDATION_REPORTS TABLE - Store detailed validation results
-- =============================================================================
CREATE TABLE validation_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    result_id UUID REFERENCES generation_results(id) ON DELETE CASCADE,
    validator_name VARCHAR(100) NOT NULL,
    passed BOOLEAN NOT NULL,
    score FLOAT CHECK (score >= 0 AND score <= 1),
    message TEXT,
    metrics JSONB DEFAULT '{}',
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for validation_reports
CREATE INDEX idx_validation_reports_result_id ON validation_reports(result_id);
CREATE INDEX idx_validation_reports_validator_name ON validation_reports(validator_name);
CREATE INDEX idx_validation_reports_passed ON validation_reports(passed);
CREATE INDEX idx_validation_reports_score ON validation_reports(score);

-- =============================================================================
-- AUDIT_LOGS TABLE - Comprehensive audit trail
-- =============================================================================
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}',
    before_state JSONB,
    after_state JSONB,
    request_id UUID,
    session_id VARCHAR(255),
    risk_level VARCHAR(20) DEFAULT 'low' CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit_logs
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_entity_type ON audit_logs(entity_type);
CREATE INDEX idx_audit_logs_entity_id ON audit_logs(entity_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_risk_level ON audit_logs(risk_level);

-- =============================================================================
-- LINEAGE_EVENTS TABLE - Data lineage tracking (also stored in Neo4j)
-- =============================================================================
CREATE TABLE lineage_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    parent_event_id UUID,
    pipeline_id UUID REFERENCES pipelines(id),
    task_id UUID REFERENCES generation_tasks(id),
    source_data_hash VARCHAR(64),
    output_data_hash VARCHAR(64),
    transformation_details JSONB DEFAULT '{}',
    quality_metrics JSONB DEFAULT '{}',
    privacy_metrics JSONB DEFAULT '{}',
    execution_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for lineage_events
CREATE INDEX idx_lineage_events_event_id ON lineage_events(event_id);
CREATE INDEX idx_lineage_events_event_type ON lineage_events(event_type);
CREATE INDEX idx_lineage_events_parent_event_id ON lineage_events(parent_event_id);
CREATE INDEX idx_lineage_events_pipeline_id ON lineage_events(pipeline_id);
CREATE INDEX idx_lineage_events_task_id ON lineage_events(task_id);
CREATE INDEX idx_lineage_events_created_at ON lineage_events(created_at);

-- =============================================================================
-- QUALITY_METRICS TABLE - Time-series quality metrics
-- =============================================================================
CREATE TABLE quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    result_id UUID REFERENCES generation_results(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    threshold FLOAT,
    passed BOOLEAN NOT NULL,
    calculation_method VARCHAR(100),
    reference_data_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for quality_metrics
CREATE INDEX idx_quality_metrics_result_id ON quality_metrics(result_id);
CREATE INDEX idx_quality_metrics_metric_name ON quality_metrics(metric_name);
CREATE INDEX idx_quality_metrics_passed ON quality_metrics(passed);
CREATE INDEX idx_quality_metrics_created_at ON quality_metrics(created_at);

-- =============================================================================
-- PRIVACY_METRICS TABLE - Privacy analysis results
-- =============================================================================
CREATE TABLE privacy_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    result_id UUID REFERENCES generation_results(id) ON DELETE CASCADE,
    epsilon FLOAT,
    delta FLOAT,
    reidentification_risk FLOAT CHECK (reidentification_risk >= 0 AND reidentification_risk <= 1),
    membership_inference_risk FLOAT CHECK (membership_inference_risk >= 0 AND membership_inference_risk <= 1),
    attribute_inference_risk FLOAT CHECK (attribute_inference_risk >= 0 AND attribute_inference_risk <= 1),
    sensitive_attributes TEXT[],
    privacy_mechanism VARCHAR(100),
    privacy_budget_consumed FLOAT,
    analysis_details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for privacy_metrics
CREATE INDEX idx_privacy_metrics_result_id ON privacy_metrics(result_id);
CREATE INDEX idx_privacy_metrics_epsilon ON privacy_metrics(epsilon);
CREATE INDEX idx_privacy_metrics_reidentification_risk ON privacy_metrics(reidentification_risk);
CREATE INDEX idx_privacy_metrics_created_at ON privacy_metrics(created_at);

-- =============================================================================
-- WATERMARKS TABLE - Watermark registry and verification
-- =============================================================================
CREATE TABLE watermarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    result_id UUID REFERENCES generation_results(id) ON DELETE CASCADE,
    method VARCHAR(100) NOT NULL,
    strength FLOAT NOT NULL CHECK (strength >= 0 AND strength <= 1),
    message TEXT,
    key_hash VARCHAR(64) NOT NULL, -- Hash of the watermark key
    signature VARCHAR(255),
    verification_data JSONB,
    is_embedded BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_verified_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for watermarks
CREATE INDEX idx_watermarks_result_id ON watermarks(result_id);
CREATE INDEX idx_watermarks_method ON watermarks(method);
CREATE INDEX idx_watermarks_key_hash ON watermarks(key_hash);
CREATE INDEX idx_watermarks_is_verified ON watermarks(is_verified);
CREATE INDEX idx_watermarks_created_at ON watermarks(created_at);

-- =============================================================================
-- SYSTEM_METRICS TABLE - System performance and health metrics
-- =============================================================================
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(50),
    labels JSONB DEFAULT '{}',
    instance_id VARCHAR(255),
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for system_metrics (time-series optimized)
CREATE INDEX idx_system_metrics_type_name_time ON system_metrics(metric_type, metric_name, collected_at DESC);
CREATE INDEX idx_system_metrics_collected_at ON system_metrics(collected_at DESC);

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- =============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply timestamp triggers to relevant tables
CREATE TRIGGER update_pipelines_updated_at 
    BEFORE UPDATE ON pipelines 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_generation_tasks_updated_at 
    BEFORE UPDATE ON generation_tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- =============================================================================

-- Function to get pipeline statistics
CREATE OR REPLACE FUNCTION get_pipeline_stats(pipeline_uuid UUID)
RETURNS TABLE(
    total_tasks INTEGER,
    completed_tasks INTEGER,
    failed_tasks INTEGER,
    avg_execution_time_ms FLOAT,
    avg_quality_score FLOAT,
    avg_privacy_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_tasks,
        COUNT(CASE WHEN gt.status = 'completed' THEN 1 END)::INTEGER as completed_tasks,
        COUNT(CASE WHEN gt.status = 'failed' THEN 1 END)::INTEGER as failed_tasks,
        AVG(gt.execution_time_ms) as avg_execution_time_ms,
        AVG(gr.quality_score) as avg_quality_score,
        AVG(gr.privacy_score) as avg_privacy_score
    FROM generation_tasks gt
    LEFT JOIN generation_results gr ON gt.id = gr.task_id
    WHERE gt.pipeline_id = pipeline_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Delete expired generation results
    DELETE FROM generation_results 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old audit logs (keep 7 years by default)
    DELETE FROM audit_logs 
    WHERE created_at < NOW() - INTERVAL '7 years';
    
    -- Delete old system metrics (keep 1 year by default)
    DELETE FROM system_metrics 
    WHERE collected_at < NOW() - INTERVAL '1 year';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- View for active pipelines with statistics
CREATE VIEW active_pipelines_stats AS
SELECT 
    p.*,
    COALESCE(ps.total_tasks, 0) as total_tasks,
    COALESCE(ps.completed_tasks, 0) as completed_tasks,
    COALESCE(ps.failed_tasks, 0) as failed_tasks,
    ps.avg_execution_time_ms,
    ps.avg_quality_score,
    ps.avg_privacy_score
FROM pipelines p
LEFT JOIN LATERAL get_pipeline_stats(p.id) ps ON true
WHERE p.is_active = true;

-- View for recent generation activity
CREATE VIEW recent_generation_activity AS
SELECT 
    gt.id as task_id,
    p.name as pipeline_name,
    p.generator,
    p.data_type,
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
JOIN pipelines p ON gt.pipeline_id = p.id
LEFT JOIN generation_results gr ON gt.id = gr.task_id
WHERE gt.created_at > NOW() - INTERVAL '24 hours'
ORDER BY gt.created_at DESC;

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default system configuration
INSERT INTO pipelines (name, description, generator, data_type, config) VALUES 
('default-tabular', 'Default tabular data generator', 'basic', 'tabular', '{"initialized": true}'),
('default-timeseries', 'Default time series generator', 'basic', 'timeseries', '{"initialized": true}'),
('default-text', 'Default text generator', 'basic', 'text', '{"initialized": true}');

-- Insert initial audit log
INSERT INTO audit_logs (event_type, action, details) VALUES 
('system', 'database_initialized', '{"migration": "001_create_core_tables", "timestamp": "' || NOW() || '"}');

COMMIT;

-- Create backup function (optional)
CREATE OR REPLACE FUNCTION create_backup_schema()
RETURNS VOID AS $$
BEGIN
    -- This function can be used to create backup schemas
    EXECUTE 'CREATE SCHEMA IF NOT EXISTS backup_' || to_char(NOW(), 'YYYY_MM_DD_HH24_MI_SS');
END;
$$ LANGUAGE plpgsql;