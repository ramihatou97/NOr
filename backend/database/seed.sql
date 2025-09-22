-- KOO Platform Seed Data
-- Initial data for testing and demonstration

-- ============================================================================
-- SYSTEM CONFIGURATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Basic system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('platform_version', '"1.0.0"', 'Current platform version'),
('ai_features_enabled', 'true', 'Whether AI features are enabled globally'),
('max_file_upload_size', '104857600', 'Maximum file upload size in bytes (100MB)'),
('supported_file_types', '["pdf", "docx", "txt", "md"]', 'Supported file types for upload'),
('quality_score_thresholds', '{"minimum": 0.6, "good": 0.8, "excellent": 0.9}', 'Quality assessment thresholds'),
('research_api_limits', '{"daily_requests": 1000, "concurrent_requests": 10}', 'Research API usage limits')
ON CONFLICT (config_key) DO NOTHING;

-- ============================================================================
-- SAMPLE USERS
-- ============================================================================

-- Sample medical specialties and institutions
INSERT INTO users (
    email, password_hash, full_name, title, specialty, institution,
    country, is_active, is_verified
) VALUES
(
    'dr.smith@medicenter.edu',
    '$2b$12$K8gPV.XvTxLVZr8mN9QJ2eH5YrF7VwJjH3Kt9B6Lr4mPq2Ns8Wd6u', -- "password123"
    'Dr. Sarah Smith',
    'Chief of Cardiology',
    'Cardiology',
    'Medical Center University',
    'United States',
    true,
    true
),
(
    'prof.johnson@research.org',
    '$2b$12$K8gPV.XvTxLVZr8mN9QJ2eH5YrF7VwJjH3Kt9B6Lr4mPq2Ns8Wd6u', -- "password123"
    'Prof. Michael Johnson',
    'Professor of Neurology',
    'Neurology',
    'Global Research Institute',
    'United Kingdom',
    true,
    true
),
(
    'dr.garcia@hospital.es',
    '$2b$12$K8gPV.XvTxLVZr8mN9QJ2eH5YrF7VwJjH3Kt9B6Lr4mPq2Ns8Wd6u', -- "password123"
    'Dr. Maria Garcia',
    'Senior Physician',
    'Internal Medicine',
    'General Hospital Madrid',
    'Spain',
    true,
    true
)
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- SAMPLE CHAPTERS
-- ============================================================================

-- Get user IDs for sample data (using email as identifier)
WITH user_data AS (
    SELECT id as user_id FROM users WHERE email = 'dr.smith@medicenter.edu' LIMIT 1
)
INSERT INTO chapters (
    id, user_id, title, content, summary, specialty, status, word_count, reading_time, metadata
)
SELECT
    '550e8400-e29b-41d4-a716-446655440001'::uuid,
    user_data.user_id,
    'Introduction to Cardiovascular Disease',
    'Cardiovascular disease (CVD) represents a class of diseases that involve the heart or blood vessels. CVD includes coronary artery diseases (CAD) such as angina and myocardial infarction (commonly known as a heart attack). Other CVDs include stroke, heart failure, hypertensive heart disease, rheumatic heart disease, cardiomyopathy, abnormal heart rhythms, congenital heart disease, valvular heart disease, carditis, aortic aneurysms, peripheral artery disease, thromboembolic disease, and venous thrombosis.

The underlying mechanisms vary depending on the disease. Coronary artery disease, stroke, and peripheral artery disease involve atherosclerosis. This may be caused by high blood pressure, smoking, diabetes mellitus, lack of exercise, obesity, high blood cholesterol, poor diet, and excessive alcohol consumption, among others. High blood pressure is estimated to account for approximately 13% of CVD deaths, while tobacco accounts for 9%, diabetes 6%, lack of exercise 6%, and obesity 5%.

Risk factors for cardiovascular disease are often interconnected. For example, diabetes increases the risk of heart disease, while obesity contributes to diabetes and high blood pressure. Understanding these relationships is crucial for effective prevention and treatment strategies.',
    'Overview of cardiovascular diseases, their causes, and interconnected risk factors.',
    'Cardiology',
    'published',
    247,
    2,
    '{"tags": ["cardiovascular", "disease", "prevention"], "difficulty": "intermediate"}'
FROM user_data
ON CONFLICT (id) DO NOTHING;

WITH user_data AS (
    SELECT id as user_id FROM users WHERE email = 'prof.johnson@research.org' LIMIT 1
)
INSERT INTO chapters (
    id, user_id, title, content, summary, specialty, status, word_count, reading_time, metadata
)
SELECT
    '550e8400-e29b-41d4-a716-446655440002'::uuid,
    user_data.user_id,
    'Neuroplasticity and Brain Recovery',
    'Neuroplasticity, also known as brain plasticity or neural plasticity, is the ability of neural networks in the brain to change through growth and reorganization. These changes range from individual neuron pathways making new connections, to systematic adjustments like cortical remapping. Examples of neuroplasticity include circuit and network changes that result from learning a new ability, environmental influences, practice, and psychological stress.

The concept of neuroplasticity has profound implications for rehabilitation medicine. Following brain injury, such as stroke or traumatic brain injury, the brain can reorganize itself to compensate for damaged areas. This process involves both structural and functional changes, including the formation of new neural pathways and the strengthening of existing ones.

Clinical applications of neuroplasticity principles include targeted rehabilitation therapies, cognitive training programs, and innovative treatment approaches that harness the brain''s natural ability to adapt and heal. Understanding neuroplasticity mechanisms is essential for developing effective interventions for neurological conditions.',
    'Exploration of brain plasticity mechanisms and their clinical applications in rehabilitation.',
    'Neurology',
    'published',
    178,
    2,
    '{"tags": ["neuroplasticity", "rehabilitation", "brain"], "difficulty": "advanced"}'
FROM user_data
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SAMPLE KNOWLEDGE NODES
-- ============================================================================

INSERT INTO knowledge_nodes (
    id, name, node_type, category, description, confidence, importance, evidence_level
) VALUES
(
    '660e8400-e29b-41d4-a716-446655440001'::uuid,
    'Myocardial Infarction',
    'disease',
    'cardiovascular',
    'Death of heart muscle due to insufficient blood supply, commonly known as heart attack',
    0.95,
    0.9,
    'high'
),
(
    '660e8400-e29b-41d4-a716-446655440002'::uuid,
    'Atherosclerosis',
    'pathology',
    'cardiovascular',
    'Disease in which plaque builds up inside arteries, narrowing them and restricting blood flow',
    0.93,
    0.85,
    'high'
),
(
    '660e8400-e29b-41d4-a716-446655440003'::uuid,
    'Aspirin',
    'treatment',
    'medication',
    'Antiplatelet medication used for cardiovascular disease prevention and treatment',
    0.9,
    0.8,
    'high'
),
(
    '660e8400-e29b-41d4-a716-446655440004'::uuid,
    'Neuroplasticity',
    'concept',
    'neurology',
    'The brain''s ability to reorganize and adapt by forming new neural connections',
    0.88,
    0.9,
    'high'
),
(
    '660e8400-e29b-41d4-a716-446655440005'::uuid,
    'Stroke Rehabilitation',
    'treatment',
    'neurology',
    'Therapeutic interventions designed to help stroke patients recover function',
    0.85,
    0.8,
    'medium'
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SAMPLE KNOWLEDGE EDGES (RELATIONSHIPS)
-- ============================================================================

INSERT INTO knowledge_edges (
    source_node_id, target_node_id, relationship_type, relationship_label,
    strength, confidence, evidence
) VALUES
(
    '660e8400-e29b-41d4-a716-446655440002'::uuid, -- Atherosclerosis
    '660e8400-e29b-41d4-a716-446655440001'::uuid, -- Myocardial Infarction
    'causes',
    'leads to',
    0.85,
    0.9,
    ARRAY['Atherosclerotic plaque rupture can cause coronary artery occlusion leading to myocardial infarction']
),
(
    '660e8400-e29b-41d4-a716-446655440003'::uuid, -- Aspirin
    '660e8400-e29b-41d4-a716-446655440001'::uuid, -- Myocardial Infarction
    'prevents',
    'reduces risk of',
    0.75,
    0.88,
    ARRAY['Clinical trials show aspirin reduces myocardial infarction risk by approximately 20-25%']
),
(
    '660e8400-e29b-41d4-a716-446655440004'::uuid, -- Neuroplasticity
    '660e8400-e29b-41d4-a716-446655440005'::uuid, -- Stroke Rehabilitation
    'enables',
    'is the basis for',
    0.9,
    0.85,
    ARRAY['Neuroplasticity mechanisms allow brain reorganization during stroke rehabilitation']
)
ON CONFLICT (source_node_id, target_node_id, relationship_type) DO NOTHING;

-- ============================================================================
-- SAMPLE QUALITY ASSESSMENTS
-- ============================================================================

INSERT INTO quality_assessments (
    content_id, content_type, overall_score, confidence, dimension_scores,
    strengths, weaknesses, improvement_suggestions, factual_accuracy,
    clinical_relevance, currency_score, predicted_longevity
) VALUES
(
    '550e8400-e29b-41d4-a716-446655440001'::uuid, -- Cardiovascular chapter
    'chapter',
    0.82,
    0.9,
    '{"accuracy": 0.85, "completeness": 0.80, "clarity": 0.85, "evidence": 0.78, "relevance": 0.90}',
    ARRAY['Comprehensive coverage of risk factors', 'Clear explanation of disease mechanisms', 'Good clinical relevance'],
    ARRAY['Limited discussion of recent treatment advances', 'Could benefit from more specific statistics'],
    ARRAY['Add section on current treatment guidelines', 'Include more recent epidemiological data', 'Expand prevention strategies'],
    0.85,
    0.90,
    0.88,
    5.2
),
(
    '550e8400-e29b-41d4-a716-446655440002'::uuid, -- Neurology chapter
    'chapter',
    0.88,
    0.85,
    '{"accuracy": 0.90, "completeness": 0.85, "clarity": 0.90, "evidence": 0.85, "relevance": 0.88}',
    ARRAY['Excellent scientific accuracy', 'Clear explanation of complex concepts', 'Strong clinical applications'],
    ARRAY['Could include more recent research findings', 'Limited discussion of contraindications'],
    ARRAY['Add latest neuroplasticity research', 'Include patient case studies', 'Expand on measurement techniques'],
    0.90,
    0.88,
    0.82,
    4.8
)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- SAMPLE USER PREFERENCES
-- ============================================================================

-- Set preferences for sample users
WITH user_data AS (
    SELECT id as user_id FROM users WHERE email = 'dr.smith@medicenter.edu' LIMIT 1
)
INSERT INTO user_preferences (user_id, category, key, value)
SELECT
    user_data.user_id,
    'ai_settings',
    'quality_threshold',
    '0.8'
FROM user_data
UNION ALL
SELECT
    user_data.user_id,
    'ui_settings',
    'theme',
    '"light"'
FROM user_data
UNION ALL
SELECT
    user_data.user_id,
    'research_settings',
    'preferred_sources',
    '["pubmed", "semantic_scholar"]'
FROM user_data
ON CONFLICT (user_id, category, key) DO NOTHING;

-- ============================================================================
-- SAMPLE CHAPTER TAGS
-- ============================================================================

INSERT INTO chapter_tags (chapter_id, tag, confidence, source) VALUES
('550e8400-e29b-41d4-a716-446655440001'::uuid, 'cardiovascular', 0.95, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440001'::uuid, 'disease', 0.90, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440001'::uuid, 'prevention', 0.85, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440001'::uuid, 'risk-factors', 0.88, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440002'::uuid, 'neuroplasticity', 0.95, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440002'::uuid, 'rehabilitation', 0.90, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440002'::uuid, 'brain', 0.92, 'ai_generated'),
('550e8400-e29b-41d4-a716-446655440002'::uuid, 'neurology', 0.88, 'ai_generated')
ON CONFLICT (chapter_id, tag) DO NOTHING;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- User indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_specialty ON users(specialty);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Chapter indexes
CREATE INDEX IF NOT EXISTS idx_chapters_user_id ON chapters(user_id);
CREATE INDEX IF NOT EXISTS idx_chapters_specialty ON chapters(specialty);
CREATE INDEX IF NOT EXISTS idx_chapters_status ON chapters(status);
CREATE INDEX IF NOT EXISTS idx_chapters_created_at ON chapters(created_at);
CREATE INDEX IF NOT EXISTS idx_chapters_title_gin ON chapters USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_chapters_content_gin ON chapters USING gin(to_tsvector('english', content));

-- Quality assessment indexes
CREATE INDEX IF NOT EXISTS idx_quality_content_id ON quality_assessments(content_id);
CREATE INDEX IF NOT EXISTS idx_quality_content_type ON quality_assessments(content_type);
CREATE INDEX IF NOT EXISTS idx_quality_overall_score ON quality_assessments(overall_score);
CREATE INDEX IF NOT EXISTS idx_quality_created_at ON quality_assessments(created_at);

-- Knowledge graph indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_category ON knowledge_nodes(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_name_gin ON knowledge_nodes USING gin(to_tsvector('english', name));
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source ON knowledge_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target ON knowledge_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_type ON knowledge_edges(relationship_type);

-- Tag indexes
CREATE INDEX IF NOT EXISTS idx_chapter_tags_chapter_id ON chapter_tags(chapter_id);
CREATE INDEX IF NOT EXISTS idx_chapter_tags_tag ON chapter_tags(tag);

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);

-- Preferences indexes
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_category ON user_preferences(category);

-- ============================================================================
-- TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chapters_updated_at BEFORE UPDATE ON chapters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_nodes_updated_at BEFORE UPDATE ON knowledge_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_edges_updated_at BEFORE UPDATE ON knowledge_edges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

-- Insert completion marker
INSERT INTO system_config (config_key, config_value, description) VALUES
('database_seeded', 'true', 'Indicates that the database has been seeded with sample data')
ON CONFLICT (config_key) DO UPDATE SET
    config_value = EXCLUDED.config_value,
    updated_at = CURRENT_TIMESTAMP;