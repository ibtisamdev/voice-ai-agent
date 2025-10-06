-- Voice AI Agent Database Initialization
-- This file runs when the postgres container starts for the first time

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For optimized indexing

-- Create fallback database if application tries to connect to 'voiceai' instead of 'voiceai_db'
CREATE DATABASE voiceai;
GRANT ALL PRIVILEGES ON DATABASE voiceai TO voiceai;

-- Create test database for development and testing
CREATE DATABASE voiceai_test_db;
GRANT ALL PRIVILEGES ON DATABASE voiceai_test_db TO voiceai;

-- Create additional schemas in main database
\c voiceai_db;

-- Conversation management schema
CREATE SCHEMA IF NOT EXISTS conversations;
CREATE SCHEMA IF NOT EXISTS voice_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant permissions to schemas
GRANT USAGE ON SCHEMA conversations TO voiceai;
GRANT USAGE ON SCHEMA voice_data TO voiceai;
GRANT USAGE ON SCHEMA analytics TO voiceai;

GRANT CREATE ON SCHEMA conversations TO voiceai;
GRANT CREATE ON SCHEMA voice_data TO voiceai;
GRANT CREATE ON SCHEMA analytics TO voiceai;

-- Create initial tables for voice services
-- Note: These will be managed by Alembic migrations in production

-- Conversation sessions table
CREATE TABLE IF NOT EXISTS conversations.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    call_id VARCHAR(255),
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    state VARCHAR(50) NOT NULL,
    phone_number VARCHAR(50),
    caller_id VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    
    -- Duration and metrics
    duration_seconds FLOAT,
    total_turns INTEGER DEFAULT 0,
    user_satisfaction_score FLOAT CHECK (user_satisfaction_score >= 0 AND user_satisfaction_score <= 5),
    resolution_status VARCHAR(100),
    
    -- AI model information
    ai_model_used VARCHAR(100),
    voice_id VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    
    -- JSON fields for flexible data
    participants JSONB DEFAULT '[]',
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Conversation turns table
CREATE TABLE IF NOT EXISTS conversations.turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    turn_id VARCHAR(255) UNIQUE NOT NULL,
    session_id UUID REFERENCES conversations.sessions(id) ON DELETE CASCADE,
    
    -- Turn details
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    speaker_id VARCHAR(255) NOT NULL,
    speaker_role VARCHAR(50) NOT NULL,
    
    -- Content
    input_text TEXT,
    input_audio_duration_ms FLOAT,
    response_text TEXT,
    response_audio_duration_ms FLOAT,
    
    -- AI analysis
    intent VARCHAR(100),
    intent_confidence FLOAT CHECK (intent_confidence >= 0 AND intent_confidence <= 1),
    entities JSONB DEFAULT '{}',
    
    -- Performance metrics
    processing_time_ms FLOAT,
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'
);

-- Voice data tables
CREATE TABLE IF NOT EXISTS voice_data.transcriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES conversations.sessions(id) ON DELETE CASCADE,
    
    -- Audio details
    audio_duration_ms FLOAT NOT NULL,
    audio_format VARCHAR(20) DEFAULT 'wav',
    sample_rate INTEGER DEFAULT 16000,
    
    -- Transcription results
    transcribed_text TEXT NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    language_detected VARCHAR(10),
    language_confidence FLOAT,
    
    -- Model information
    model_used VARCHAR(100),
    processing_time_ms FLOAT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Segments and word-level data
    segments JSONB DEFAULT '[]',
    word_timestamps JSONB DEFAULT '[]',
    
    -- Speaker diarization
    speaker_id VARCHAR(50),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS voice_data.syntheses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES conversations.sessions(id) ON DELETE CASCADE,
    
    -- Text input
    input_text TEXT NOT NULL,
    
    -- Audio output details
    audio_duration_ms FLOAT,
    audio_format VARCHAR(20) DEFAULT 'wav',
    sample_rate INTEGER DEFAULT 22050,
    
    -- Synthesis settings
    voice_id VARCHAR(100),
    engine_used VARCHAR(50),
    speed FLOAT DEFAULT 1.0,
    pitch FLOAT DEFAULT 1.0,
    volume FLOAT DEFAULT 1.0,
    
    -- Performance
    processing_time_ms FLOAT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'
);

-- Analytics tables
CREATE TABLE IF NOT EXISTS analytics.daily_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    
    -- Call statistics
    total_sessions INTEGER DEFAULT 0,
    completed_sessions INTEGER DEFAULT 0,
    avg_duration_seconds FLOAT,
    
    -- Voice processing stats
    total_transcriptions INTEGER DEFAULT 0,
    avg_transcription_confidence FLOAT,
    total_syntheses INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_response_time_ms FLOAT,
    total_conversation_turns INTEGER DEFAULT 0,
    
    -- Quality metrics
    avg_user_satisfaction FLOAT,
    
    -- Updated timestamp
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON conversations.sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_phone_number ON conversations.sessions(phone_number);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON conversations.sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_state ON conversations.sessions(state);

CREATE INDEX IF NOT EXISTS idx_turns_session_id ON conversations.turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON conversations.turns(timestamp);
CREATE INDEX IF NOT EXISTS idx_turns_intent ON conversations.turns(intent);

CREATE INDEX IF NOT EXISTS idx_transcriptions_session_id ON voice_data.transcriptions(session_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON voice_data.transcriptions(created_at);

CREATE INDEX IF NOT EXISTS idx_syntheses_session_id ON voice_data.syntheses(session_id);
CREATE INDEX IF NOT EXISTS idx_syntheses_created_at ON voice_data.syntheses(created_at);

-- GIN indexes for JSONB fields
CREATE INDEX IF NOT EXISTS idx_sessions_context_gin ON conversations.sessions USING gin(context);
CREATE INDEX IF NOT EXISTS idx_sessions_metadata_gin ON conversations.sessions USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_turns_entities_gin ON conversations.turns USING gin(entities);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON conversations.sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for session analytics
CREATE OR REPLACE VIEW analytics.session_summary AS
SELECT 
    s.id,
    s.session_id,
    s.direction,
    s.state,
    s.duration_seconds,
    s.total_turns,
    s.user_satisfaction_score,
    s.created_at,
    s.ended_at,
    COUNT(t.id) as actual_turns,
    AVG(t.processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN t.intent IS NOT NULL THEN 1 END) as turns_with_intent,
    AVG(t.intent_confidence) as avg_intent_confidence
FROM conversations.sessions s
LEFT JOIN conversations.turns t ON s.id = t.session_id
GROUP BY s.id, s.session_id, s.direction, s.state, s.duration_seconds, 
         s.total_turns, s.user_satisfaction_score, s.created_at, s.ended_at;

-- Grant permissions on all created objects
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA conversations TO voiceai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA voice_data TO voiceai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO voiceai;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA conversations TO voiceai;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA voice_data TO voiceai;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO voiceai;

-- Log completion
\echo 'Voice AI Agent database initialization completed successfully!';