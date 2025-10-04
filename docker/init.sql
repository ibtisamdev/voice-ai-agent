-- Initialize database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create initial schemas if needed
-- This file runs when the postgres container starts for the first time

-- Example: Create additional databases or users
-- CREATE DATABASE voiceai_test;
-- GRANT ALL PRIVILEGES ON DATABASE voiceai_test TO voiceai;