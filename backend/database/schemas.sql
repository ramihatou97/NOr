-- Enhanced KOO Platform Database Schema
-- Comprehensive schema for all intelligence modules and platform features

-- Extensions for PostgreSQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============================================================================
-- CORE USER MANAGEMENT
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    title VARCHAR(100),
    specialty VARCHAR(100),
    institution VARCHAR(255),
    country VARCHAR(100),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(100) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, category, key)
);

CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INTELLIGENT CHAPTERS
-- ============================================================================

CREATE TABLE chapters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    word_count INTEGER,
    reading_time INTEGER, -- in minutes
    specialty VARCHAR(100),
    status VARCHAR(50) DEFAULT 'draft',
    version INTEGER DEFAULT 1,
    parent_chapter_id UUID REFERENCES chapters(id),
    is_template BOOLEAN DEFAULT false,
    content_vector vector(1536), -- OpenAI embedding dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chapter_tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id UUID REFERENCES chapters(id) ON DELETE CASCADE,
    tag VARCHAR(100) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(50) DEFAULT 'manual', -- manual, ai_generated
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chapter_id, tag)
);

CREATE TABLE chapter_collaborators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id UUID REFERENCES chapters(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- owner, editor, viewer, reviewer
    permissions JSONB DEFAULT '{}',
    invited_by UUID REFERENCES users(id),
    invited_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accepted_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(chapter_id, user_id)
);

-- ============================================================================
-- QUALITY ASSESSMENT SYSTEM
-- ============================================================================

CREATE TABLE quality_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID, -- Can reference chapters or other content
    content_type VARCHAR(50) NOT NULL, -- chapter, research_paper, etc.
    overall_score FLOAT NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    dimension_scores JSONB NOT NULL, -- detailed breakdown by dimension
    strengths TEXT[],
    weaknesses TEXT[],
    improvement_suggestions TEXT[],
    evidence_gaps TEXT[],
    bias_indicators TEXT[],
    factual_accuracy FLOAT,
    clinical_relevance FLOAT,
    currency_score FLOAT,
    predicted_longevity FLOAT, -- years
    comparative_ranking FLOAT,
    assessment_metadata JSONB DEFAULT '{}',
    assessor_type VARCHAR(50) DEFAULT 'ai', -- ai, human, hybrid
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE quality_trends (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    dimension VARCHAR(100) NOT NULL,
    score FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB DEFAULT '{}',
    INDEX (content_id, content_type, dimension, timestamp)
);

CREATE TABLE quality_improvements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL,
    assessment_id UUID REFERENCES quality_assessments(id),
    improvement_type VARCHAR(100) NOT NULL,
    before_score FLOAT,
    after_score FLOAT,
    action_taken TEXT,
    result_description TEXT,
    impact_score FLOAT,
    implemented_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- KNOWLEDGE GRAPH SYSTEM
-- ============================================================================

CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(500) NOT NULL,
    node_type VARCHAR(100) NOT NULL, -- concept, disease, treatment, etc.
    category VARCHAR(100),
    description TEXT,
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    evidence_level VARCHAR(50),
    sources TEXT[],
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    relationship_label VARCHAR(200),
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    bidirectional BOOLEAN DEFAULT false,
    evidence TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_node_id, target_node_id, relationship_type)
);

CREATE TABLE knowledge_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    cluster_type VARCHAR(100),
    centroid_embedding vector(1536),
    node_count INTEGER DEFAULT 0,
    coherence_score FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE knowledge_node_clusters (
    node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    cluster_id UUID REFERENCES knowledge_clusters(id) ON DELETE CASCADE,
    membership_strength FLOAT DEFAULT 1.0,
    PRIMARY KEY (node_id, cluster_id)
);

-- ============================================================================
-- RESEARCH ENGINE
-- ============================================================================

CREATE TABLE research_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    domain VARCHAR(100),
    urgency INTEGER DEFAULT 3 CHECK (urgency >= 1 AND urgency <= 5),
    quality_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 20,
    source_preferences TEXT[],
    time_range VARCHAR(50),
    include_gray_literature BOOLEAN DEFAULT false,
    contextual_expansion BOOLEAN DEFAULT true,
    query_embedding vector(1536),
    enhanced_query JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    results_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE research_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES research_queries(id) ON DELETE CASCADE,
    external_id VARCHAR(255), -- DOI, PubMed ID, etc.
    title TEXT NOT NULL,
    authors TEXT[],
    journal VARCHAR(255),
    publication_date DATE,
    abstract TEXT,
    url TEXT,
    quality_score FLOAT,
    relevance_score FLOAT,
    evidence_level VARCHAR(50),
    citation_count INTEGER,
    access_type VARCHAR(50), -- open, subscription, paywall
    key_findings TEXT[],
    methodology TEXT,
    sample_size INTEGER,
    study_type VARCHAR(100),
    conflicts_potential FLOAT DEFAULT 0.0,
    synthesis_ready BOOLEAN DEFAULT false,
    result_metadata JSONB DEFAULT '{}',
    retrieved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE saved_research (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    result_id UUID REFERENCES research_results(id) ON DELETE CASCADE,
    collection_name VARCHAR(200),
    notes TEXT,
    tags TEXT[],
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, result_id)
);

-- ============================================================================
-- WORKFLOW INTELLIGENCE
-- ============================================================================

CREATE TABLE workflow_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    estimated_duration INTEGER, -- minutes
    actual_duration INTEGER,
    priority VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    complexity INTEGER CHECK (complexity >= 1 AND complexity <= 10),
    energy_required VARCHAR(20) DEFAULT 'medium', -- low, medium, high
    prerequisites UUID[], -- array of task IDs
    tags TEXT[],
    due_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending', -- pending, in_progress, completed, blocked
    completion_percentage INTEGER DEFAULT 0,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE work_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    task_id UUID REFERENCES workflow_tasks(id) ON DELETE SET NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_minutes INTEGER,
    productivity_score FLOAT,
    interruptions INTEGER DEFAULT 0,
    focus_level INTEGER CHECK (focus_level >= 1 AND focus_level <= 10),
    energy_level INTEGER CHECK (energy_level >= 1 AND energy_level <= 10),
    environment_factors JSONB DEFAULT '{}',
    session_notes TEXT,
    completed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE productivity_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    daily_productivity FLOAT,
    tasks_completed INTEGER DEFAULT 0,
    total_work_time INTEGER DEFAULT 0, -- minutes
    focus_time INTEGER DEFAULT 0, -- minutes
    break_time INTEGER DEFAULT 0, -- minutes
    interruptions INTEGER DEFAULT 0,
    average_task_duration FLOAT,
    completion_rate FLOAT,
    energy_pattern FLOAT[], -- hourly energy levels
    focus_pattern FLOAT[], -- hourly focus levels
    optimal_time_slots TEXT[],
    procrastination_pattern VARCHAR(100),
    metrics_metadata JSONB DEFAULT '{}',
    UNIQUE(user_id, date)
);

CREATE TABLE workflow_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    recommendation_type VARCHAR(100) NOT NULL,
    title VARCHAR(300) NOT NULL,
    description TEXT,
    impact_score FLOAT, -- expected productivity increase
    effort_level VARCHAR(20), -- low, medium, high
    timeframe VARCHAR(100),
    action_steps TEXT[],
    confidence FLOAT DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'pending', -- pending, applied, dismissed
    applied_at TIMESTAMP WITH TIME ZONE,
    effectiveness_score FLOAT, -- measured after application
    recommendation_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CONFLICT DETECTION SYSTEM
-- ============================================================================

CREATE TABLE content_conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    confidence FLOAT DEFAULT 1.0,
    description TEXT NOT NULL,
    resolution_strategy TEXT,
    sources JSONB NOT NULL, -- array of source references
    evidence JSONB DEFAULT '{}',
    domain VARCHAR(100),
    context JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'unresolved', -- unresolved, investigating, resolved
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES users(id),
    resolution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conflict_resolutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID REFERENCES content_conflicts(id) ON DELETE CASCADE,
    resolution_type VARCHAR(100) NOT NULL,
    preferred_source VARCHAR(255),
    manual_resolution TEXT,
    confidence FLOAT DEFAULT 1.0,
    reasoning TEXT,
    evidence_summary JSONB DEFAULT '{}',
    resolved_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SYNTHESIS ENGINE
-- ============================================================================

CREATE TABLE synthesis_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    synthesis_type VARCHAR(100) NOT NULL,
    topic VARCHAR(500) NOT NULL,
    source_count INTEGER DEFAULT 0,
    synthesized_content TEXT,
    evidence_hierarchy JSONB DEFAULT '{}',
    consensus_points TEXT[],
    conflicting_evidence JSONB DEFAULT '{}',
    research_gaps TEXT[],
    clinical_implications TEXT[],
    quality_assessment JSONB DEFAULT '{}',
    confidence_level FLOAT,
    methodology_notes TEXT,
    limitations TEXT[],
    recommendations TEXT[],
    synthesis_metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE synthesis_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    synthesis_id UUID REFERENCES synthesis_sessions(id) ON DELETE CASCADE,
    source_id VARCHAR(255) NOT NULL,
    source_type VARCHAR(100), -- research_paper, chapter, etc.
    content TEXT,
    evidence_level VARCHAR(50),
    confidence_score FLOAT,
    weight FLOAT DEFAULT 1.0,
    inclusion_reasoning TEXT,
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- PREDICTIVE INTELLIGENCE
-- ============================================================================

CREATE TABLE user_behavior_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    frequency FLOAT, -- how often this pattern occurs
    last_observed TIMESTAMP WITH TIME ZONE,
    prediction_accuracy FLOAT,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictive_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(200) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    training_data_summary JSONB,
    accuracy_metrics JSONB,
    last_trained TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    model_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    model_id UUID REFERENCES predictive_models(id),
    prediction_type VARCHAR(100) NOT NULL,
    predicted_value JSONB NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    context JSONB DEFAULT '{}',
    actual_outcome JSONB, -- filled in after the prediction period
    accuracy_score FLOAT, -- calculated after outcome is known
    feedback_score INTEGER, -- user feedback on prediction quality
    prediction_horizon INTERVAL, -- how far into the future
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    outcome_recorded_at TIMESTAMP WITH TIME ZONE
);

-- ============================================================================
-- CONTEXTUAL INTELLIGENCE
-- ============================================================================

CREATE TABLE user_contexts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    context_type VARCHAR(100) NOT NULL,
    context_data JSONB NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    importance FLOAT DEFAULT 0.5,
    temporal_weight FLOAT DEFAULT 1.0, -- decreases over time
    spatial_context JSONB DEFAULT '{}',
    active_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE context_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    from_context_id UUID REFERENCES user_contexts(id),
    to_context_id UUID REFERENCES user_contexts(id),
    transition_type VARCHAR(100),
    trigger_event VARCHAR(200),
    probability FLOAT DEFAULT 1.0,
    transition_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PERFORMANCE MONITORING
-- ============================================================================

CREATE TABLE system_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    component VARCHAR(100), -- which system component
    environment VARCHAR(50), -- prod, staging, dev
    additional_data JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    response_time_ms INTEGER NOT NULL,
    status_code INTEGER NOT NULL,
    user_id UUID REFERENCES users(id),
    request_size INTEGER,
    response_size INTEGER,
    cache_hit BOOLEAN DEFAULT false,
    error_message TEXT,
    request_metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES predictive_models(id),
    operation_type VARCHAR(100) NOT NULL, -- inference, training, etc.
    execution_time_ms INTEGER,
    input_size INTEGER,
    output_size INTEGER,
    accuracy_score FLOAT,
    confidence_score FLOAT,
    resource_usage JSONB DEFAULT '{}',
    error_occurred BOOLEAN DEFAULT false,
    error_details TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- AUDIT AND LOGGING
-- ============================================================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id UUID,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    additional_context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL, -- DEBUG, INFO, WARN, ERROR, CRITICAL
    logger_name VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    exception_info TEXT,
    additional_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- NOTIFICATIONS SYSTEM
-- ============================================================================

CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    notification_type VARCHAR(100) NOT NULL,
    title VARCHAR(300) NOT NULL,
    message TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium', -- low, medium, high, urgent
    category VARCHAR(100),
    related_resource_type VARCHAR(100),
    related_resource_id VARCHAR(255),
    action_url TEXT,
    is_read BOOLEAN DEFAULT false,
    is_dismissed BOOLEAN DEFAULT false,
    read_at TIMESTAMP WITH TIME ZONE,
    dismissed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    notification_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    notification_type VARCHAR(100) NOT NULL,
    channel VARCHAR(50) NOT NULL, -- email, in_app, push, sms
    enabled BOOLEAN DEFAULT true,
    frequency VARCHAR(50) DEFAULT 'immediate', -- immediate, daily, weekly
    quiet_hours_start TIME,
    quiet_hours_end TIME,
    additional_settings JSONB DEFAULT '{}',
    UNIQUE(user_id, notification_type, channel)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- User indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_specialty ON users(specialty);
CREATE INDEX idx_users_is_active ON users(is_active);

-- Chapter indexes
CREATE INDEX idx_chapters_user_id ON chapters(user_id);
CREATE INDEX idx_chapters_specialty ON chapters(specialty);
CREATE INDEX idx_chapters_status ON chapters(status);
CREATE INDEX idx_chapters_created_at ON chapters(created_at);
CREATE INDEX idx_chapters_content_vector ON chapters USING ivfflat (content_vector vector_cosine_ops);

-- Quality assessment indexes
CREATE INDEX idx_quality_assessments_content ON quality_assessments(content_id, content_type);
CREATE INDEX idx_quality_assessments_score ON quality_assessments(overall_score);
CREATE INDEX idx_quality_trends_content_time ON quality_trends(content_id, timestamp);

-- Knowledge graph indexes
CREATE INDEX idx_knowledge_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_knowledge_edges_source ON knowledge_edges(source_node_id);
CREATE INDEX idx_knowledge_edges_target ON knowledge_edges(target_node_id);
CREATE INDEX idx_knowledge_edges_relationship ON knowledge_edges(relationship_type);

-- Research indexes
CREATE INDEX idx_research_queries_user ON research_queries(user_id);
CREATE INDEX idx_research_queries_domain ON research_queries(domain);
CREATE INDEX idx_research_results_query ON research_results(query_id);
CREATE INDEX idx_research_results_quality ON research_results(quality_score);

-- ============================================================================
-- NUANCE MERGE INTELLIGENCE SYSTEM
-- ============================================================================

-- Nuance merge detection and management with enterprise-grade audit trails
CREATE TABLE nuance_merges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id UUID REFERENCES chapters(id) ON DELETE CASCADE,
    section_id UUID, -- Optional for granular tracking

    -- Content versioning with vector embeddings for semantic analysis
    original_content TEXT NOT NULL,
    updated_content TEXT NOT NULL,
    merged_content TEXT,
    original_content_vector vector(1536),
    updated_content_vector vector(1536),

    -- Advanced similarity metrics with enterprise precision
    semantic_similarity FLOAT CHECK (semantic_similarity BETWEEN 0 AND 1),
    jaccard_similarity FLOAT CHECK (jaccard_similarity BETWEEN 0 AND 1),
    levenshtein_distance INTEGER,
    cosine_similarity FLOAT CHECK (cosine_similarity BETWEEN 0 AND 1),

    -- Nuance classification and intelligence metadata
    nuance_type VARCHAR(100) DEFAULT 'enhancement', -- enhancement, refinement, expansion, clarification
    merge_category VARCHAR(100) DEFAULT 'content_improvement',
    confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score BETWEEN 0 AND 1),
    clinical_relevance_score FLOAT CHECK (clinical_relevance_score BETWEEN 0 AND 1),

    -- AI analysis integration with multi-model support
    ai_analysis JSONB DEFAULT '{}' NOT NULL,
    processing_models JSONB DEFAULT '[]' NOT NULL,
    ai_recommendations JSONB DEFAULT '{}' NOT NULL,

    -- Medical context and specialty-specific analysis
    medical_concepts_added JSONB DEFAULT '[]' NOT NULL,
    anatomical_references JSONB DEFAULT '[]' NOT NULL,
    procedure_references JSONB DEFAULT '[]' NOT NULL,
    specialty_context VARCHAR(100),

    -- Processing metadata and performance tracking
    detection_algorithm VARCHAR(100) DEFAULT 'hybrid_semantic_analysis',
    processing_time_ms INTEGER,
    memory_usage_mb FLOAT,
    algorithm_version VARCHAR(20) DEFAULT '1.0',

    -- Workflow and approval management
    detected_by UUID REFERENCES users(id),
    reviewed_by UUID REFERENCES users(id),
    approved_by UUID REFERENCES users(id),

    -- Status lifecycle with enterprise workflow support
    status VARCHAR(50) DEFAULT 'detected', -- detected, under_review, approved, applied, rejected, superseded
    priority_level INTEGER DEFAULT 5 CHECK (priority_level BETWEEN 1 AND 10),
    auto_apply_eligible BOOLEAN DEFAULT false,
    manual_review_required BOOLEAN DEFAULT true,

    -- Review and approval audit trail
    review_notes TEXT,
    reviewer_confidence FLOAT CHECK (reviewer_confidence BETWEEN 0 AND 1),
    approval_reason TEXT,
    rejection_reason TEXT,
    superseded_by UUID REFERENCES nuance_merges(id),

    -- Enterprise audit timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    approved_at TIMESTAMP WITH TIME ZONE,
    applied_at TIMESTAMP WITH TIME ZONE,
    superseded_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- High-performance similarity computation cache for optimization
CREATE TABLE similarity_computation_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_hash_a VARCHAR(64) NOT NULL,
    content_hash_b VARCHAR(64) NOT NULL,
    similarity_type VARCHAR(50) NOT NULL, -- semantic, jaccard, levenshtein, cosine
    similarity_score FLOAT NOT NULL,
    computation_model VARCHAR(100),
    model_version VARCHAR(20),
    context_metadata JSONB DEFAULT '{}',
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Prevent duplicate computations
    UNIQUE(content_hash_a, content_hash_b, similarity_type, computation_model)
);

-- Sentence-level nuance analysis for granular insights
CREATE TABLE sentence_nuances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    nuance_merge_id UUID REFERENCES nuance_merges(id) ON DELETE CASCADE,

    -- Sentence content and positioning
    original_sentence TEXT NOT NULL,
    enhanced_sentence TEXT NOT NULL,
    sentence_position INTEGER NOT NULL,
    paragraph_position INTEGER,

    -- Granular difference analysis
    added_parts JSONB DEFAULT '[]' NOT NULL,
    modified_parts JSONB DEFAULT '[]' NOT NULL,
    removed_parts JSONB DEFAULT '[]' NOT NULL,
    word_level_changes JSONB DEFAULT '{}' NOT NULL,

    -- Sentence-level intelligence metrics
    sentence_similarity FLOAT CHECK (sentence_similarity BETWEEN 0 AND 1),
    medical_concept_density FLOAT,
    clinical_importance_score FLOAT CHECK (clinical_importance_score BETWEEN 0 AND 1),
    readability_improvement FLOAT,

    -- Medical analysis at sentence level
    medical_terms_added JSONB DEFAULT '[]' NOT NULL,
    anatomical_references JSONB DEFAULT '[]' NOT NULL,
    procedure_references JSONB DEFAULT '[]' NOT NULL,
    drug_references JSONB DEFAULT '[]' NOT NULL,

    -- Sentence categorization
    change_type VARCHAR(100), -- addition, modification, enhancement, clarification
    impact_category VARCHAR(100), -- low, medium, high, critical

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Configurable detection parameters per specialty with enterprise flexibility
CREATE TABLE nuance_detection_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    specialty VARCHAR(100) NOT NULL UNIQUE,

    -- Similarity thresholds with specialty-specific tuning
    exact_duplicate_threshold FLOAT DEFAULT 0.98,
    nuance_threshold_high FLOAT DEFAULT 0.90,
    nuance_threshold_medium FLOAT DEFAULT 0.75,
    nuance_threshold_low FLOAT DEFAULT 0.60,
    significant_change_threshold FLOAT DEFAULT 0.50,

    -- Auto-application settings with safety controls
    auto_apply_threshold FLOAT DEFAULT 0.95,
    auto_apply_enabled BOOLEAN DEFAULT false,
    require_review_threshold FLOAT DEFAULT 0.80,
    manual_approval_required BOOLEAN DEFAULT true,

    -- AI model configuration with version control
    primary_similarity_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    secondary_similarity_model VARCHAR(100),
    ai_analysis_models JSONB DEFAULT '["gpt-4", "claude-3"]' NOT NULL,
    enable_semantic_analysis BOOLEAN DEFAULT true,
    enable_medical_concept_analysis BOOLEAN DEFAULT true,
    enable_clinical_validation BOOLEAN DEFAULT true,

    -- Processing limits and performance controls
    max_content_length INTEGER DEFAULT 50000,
    max_processing_time_ms INTEGER DEFAULT 30000,
    max_concurrent_analyses INTEGER DEFAULT 5,
    batch_processing_size INTEGER DEFAULT 10,

    -- Quality assurance settings
    minimum_confidence_threshold FLOAT DEFAULT 0.70,
    require_human_validation BOOLEAN DEFAULT true,
    enable_cross_validation BOOLEAN DEFAULT true,

    -- Enterprise audit and versioning
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true
);

-- Performance and analytics tracking for continuous optimization
CREATE TABLE nuance_processing_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_type VARCHAR(50) NOT NULL, -- detection, similarity, ai_analysis, merge_application

    -- Performance metrics with detailed tracking
    processing_time_ms INTEGER NOT NULL,
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    gpu_usage_percent FLOAT,

    -- Content characteristics for analysis optimization
    content_length INTEGER,
    sentence_count INTEGER,
    paragraph_count INTEGER,
    medical_term_count INTEGER,
    complexity_score FLOAT,

    -- Model and configuration tracking
    model_used VARCHAR(100),
    algorithm_version VARCHAR(20),
    configuration_id UUID REFERENCES nuance_detection_config(id),
    batch_size INTEGER,

    -- Success/failure tracking with detailed error analysis
    success BOOLEAN NOT NULL,
    error_type VARCHAR(100),
    error_message TEXT,
    error_stack_trace TEXT,

    -- Contextual information for debugging and optimization
    chapter_id UUID REFERENCES chapters(id),
    user_id UUID REFERENCES users(id),
    session_id VARCHAR(255),
    request_metadata JSONB DEFAULT '{}',

    -- Enterprise monitoring timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Nuance merge indexes for high-performance queries
CREATE INDEX idx_nuance_merges_chapter_status ON nuance_merges(chapter_id, status);
CREATE INDEX idx_nuance_merges_created_at ON nuance_merges(created_at DESC);
CREATE INDEX idx_nuance_merges_similarity_scores ON nuance_merges(semantic_similarity DESC, confidence_score DESC);
CREATE INDEX idx_nuance_merges_specialty ON nuance_merges(specialty_context);
CREATE INDEX idx_nuance_merges_workflow ON nuance_merges(status, priority_level DESC, created_at DESC);

CREATE INDEX idx_similarity_cache_lookup ON similarity_computation_cache(content_hash_a, content_hash_b, similarity_type);
CREATE INDEX idx_similarity_cache_expiry ON similarity_computation_cache(expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX idx_sentence_nuances_merge ON sentence_nuances(nuance_merge_id);
CREATE INDEX idx_sentence_nuances_importance ON sentence_nuances(clinical_importance_score DESC);

CREATE INDEX idx_processing_metrics_operation ON nuance_processing_metrics(operation_type, created_at DESC);
CREATE INDEX idx_processing_metrics_performance ON nuance_processing_metrics(processing_time_ms, memory_usage_mb);
CREATE INDEX idx_processing_metrics_success ON nuance_processing_metrics(success, created_at DESC);

CREATE INDEX idx_detection_config_specialty ON nuance_detection_config(specialty, is_active);

-- Vector similarity indexes for high-performance semantic search
CREATE INDEX idx_nuance_merges_original_vector ON nuance_merges USING ivfflat (original_content_vector vector_cosine_ops);
CREATE INDEX idx_nuance_merges_updated_vector ON nuance_merges USING ivfflat (updated_content_vector vector_cosine_ops);

-- Insert default enterprise configurations for medical specialties
INSERT INTO nuance_detection_config (specialty, nuance_threshold_high, nuance_threshold_medium, auto_apply_enabled, require_review_threshold) VALUES
('neurosurgery', 0.92, 0.78, false, 0.85),
('neuroradiology', 0.90, 0.75, false, 0.82),
('neuroanatomy', 0.88, 0.72, false, 0.80),
('neuropathology', 0.91, 0.76, false, 0.83),
('neuropharmacology', 0.89, 0.74, false, 0.81),
('clinical_neurology', 0.87, 0.70, false, 0.78),
('pediatric_neurosurgery', 0.93, 0.80, false, 0.87),
('spinal_surgery', 0.90, 0.75, false, 0.82),
('skull_base_surgery', 0.92, 0.78, false, 0.85),
('general_medicine', 0.85, 0.68, false, 0.75)
ON CONFLICT (specialty) DO NOTHING;

-- Workflow indexes
CREATE INDEX idx_workflow_tasks_user ON workflow_tasks(user_id);
CREATE INDEX idx_workflow_tasks_status ON workflow_tasks(status);
CREATE INDEX idx_workflow_tasks_due_date ON workflow_tasks(due_date);
CREATE INDEX idx_work_sessions_user_time ON work_sessions(user_id, start_time);

-- Performance indexes
CREATE INDEX idx_system_performance_time ON system_performance(recorded_at);
CREATE INDEX idx_api_performance_endpoint_time ON api_performance(endpoint, recorded_at);

-- Full-text search indexes
CREATE INDEX idx_chapters_content_fts ON chapters USING gin(to_tsvector('english', title || ' ' || content));
CREATE INDEX idx_knowledge_nodes_search ON knowledge_nodes USING gin(to_tsvector('english', name || ' ' || coalesce(description, '')));

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_chapters_updated_at BEFORE UPDATE ON chapters FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_nodes_updated_at BEFORE UPDATE ON knowledge_nodes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_edges_updated_at BEFORE UPDATE ON knowledge_edges FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflow_tasks_updated_at BEFORE UPDATE ON workflow_tasks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate chapter word count
CREATE OR REPLACE FUNCTION calculate_word_count()
RETURNS TRIGGER AS $$
BEGIN
    NEW.word_count = array_length(string_to_array(NEW.content, ' '), 1);
    NEW.reading_time = CEIL(NEW.word_count / 200.0); -- Assuming 200 words per minute
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_chapter_stats BEFORE INSERT OR UPDATE ON chapters FOR EACH ROW EXECUTE FUNCTION calculate_word_count();

-- Function to maintain knowledge graph statistics
CREATE OR REPLACE FUNCTION update_knowledge_cluster_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE knowledge_clusters
        SET node_count = node_count + 1
        WHERE id = NEW.cluster_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE knowledge_clusters
        SET node_count = node_count - 1
        WHERE id = OLD.cluster_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_cluster_stats
AFTER INSERT OR DELETE ON knowledge_node_clusters
FOR EACH ROW EXECUTE FUNCTION update_knowledge_cluster_stats();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for user statistics
CREATE VIEW user_stats AS
SELECT
    u.id,
    u.full_name,
    u.specialty,
    COUNT(DISTINCT c.id) as total_chapters,
    COUNT(DISTINCT CASE WHEN c.status = 'published' THEN c.id END) as published_chapters,
    AVG(qa.overall_score) as avg_quality_score,
    COUNT(DISTINCT rq.id) as research_queries,
    COUNT(DISTINCT ws.id) as work_sessions,
    AVG(pm.daily_productivity) as avg_productivity
FROM users u
LEFT JOIN chapters c ON u.id = c.user_id
LEFT JOIN quality_assessments qa ON c.id = qa.content_id AND qa.content_type = 'chapter'
LEFT JOIN research_queries rq ON u.id = rq.user_id
LEFT JOIN work_sessions ws ON u.id = ws.user_id
LEFT JOIN productivity_metrics pm ON u.id = pm.user_id
GROUP BY u.id, u.full_name, u.specialty;

-- View for trending research topics
CREATE VIEW trending_research AS
SELECT
    domain,
    COUNT(*) as query_count,
    AVG(results_count) as avg_results,
    AVG(execution_time_ms) as avg_execution_time,
    MAX(created_at) as last_query
FROM research_queries
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY domain
ORDER BY query_count DESC;

-- View for knowledge graph insights
CREATE VIEW knowledge_insights AS
SELECT
    kn.node_type,
    COUNT(*) as node_count,
    AVG(kn.confidence) as avg_confidence,
    AVG(kn.importance) as avg_importance,
    COUNT(DISTINCT ke.id) as total_connections,
    AVG(ke.strength) as avg_connection_strength
FROM knowledge_nodes kn
LEFT JOIN knowledge_edges ke ON kn.id = ke.source_node_id OR kn.id = ke.target_node_id
GROUP BY kn.node_type
ORDER BY node_count DESC;

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default notification types
INSERT INTO notification_preferences (user_id, notification_type, channel, enabled)
SELECT
    u.id,
    unnest(ARRAY['quality_alert', 'research_found', 'workflow_suggestion', 'conflict_detected']) as notification_type,
    'in_app' as channel,
    true as enabled
FROM users u
ON CONFLICT (user_id, notification_type, channel) DO NOTHING;

-- Insert default predictive models
INSERT INTO predictive_models (model_name, model_type, version, parameters, is_active) VALUES
('user_behavior_predictor', 'neural_network', '1.0', '{"layers": [128, 64, 32], "activation": "relu"}', true),
('quality_predictor', 'gradient_boosting', '1.0', '{"n_estimators": 100, "max_depth": 6}', true),
('workflow_optimizer', 'reinforcement_learning', '1.0', '{"algorithm": "ppo", "learning_rate": 0.001}', true);

-- Create partitioned tables for high-volume data
CREATE TABLE audit_logs_2024 PARTITION OF audit_logs FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE system_logs_2024 PARTITION OF system_logs FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE api_performance_2024 PARTITION OF api_performance FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE users IS 'Core user accounts with profile information and preferences';
COMMENT ON TABLE chapters IS 'Intelligent chapters with AI-enhanced content and metadata';
COMMENT ON TABLE quality_assessments IS 'AI-powered quality assessments for all content types';
COMMENT ON TABLE knowledge_nodes IS 'Nodes in the medical knowledge graph representing concepts, diseases, treatments, etc.';
COMMENT ON TABLE knowledge_edges IS 'Relationships between knowledge graph nodes with confidence scores';
COMMENT ON TABLE research_queries IS 'User research queries with enhanced AI processing';
COMMENT ON TABLE workflow_tasks IS 'User tasks with AI-powered scheduling and optimization';
COMMENT ON TABLE work_sessions IS 'Tracked work sessions for productivity analysis';
COMMENT ON TABLE content_conflicts IS 'Detected conflicts between different sources of information';
COMMENT ON TABLE synthesis_sessions IS 'AI-powered synthesis of multiple sources into coherent content';
COMMENT ON TABLE predictions IS 'AI predictions about user behavior and system optimization';
COMMENT ON TABLE user_contexts IS 'Contextual intelligence tracking user patterns and preferences';