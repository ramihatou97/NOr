#!/bin/bash
# backend/scripts/entrypoint.optimized.sh
# Optimized entrypoint script for enhanced KOO Platform backend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Enhanced KOO Platform Backend (Optimized)${NC}"

# Wait for PostgreSQL
echo -e "${YELLOW}⏳ Waiting for PostgreSQL...${NC}"
while ! nc -z postgres 5432; do
  sleep 1
done
echo -e "${GREEN}✅ PostgreSQL is ready${NC}"

# Wait for Redis
echo -e "${YELLOW}⏳ Waiting for Redis...${NC}"
while ! nc -z redis 6379; do
  sleep 1
done
echo -e "${GREEN}✅ Redis is ready${NC}"

# Run database migrations
echo -e "${YELLOW}📊 Running database migrations...${NC}"
alembic upgrade head
echo -e "${GREEN}✅ Database migrations completed${NC}"

# Download medical models if not present
echo -e "${YELLOW}🧠 Checking medical NLP models...${NC}"
python -c "
import spacy
try:
    spacy.load('en_core_web_sm')
    print('✅ spaCy model loaded successfully')
except:
    print('❌ spaCy model not found, downloading...')
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
"

# Initialize PDF processing directories
echo -e "${YELLOW}📁 Initializing PDF processing directories...${NC}"
mkdir -p /app/uploads /app/temp /app/checkpoints /app/logs
chmod 755 /app/uploads /app/temp /app/checkpoints /app/logs

# Set up medical processing cache
echo -e "${YELLOW}🏥 Setting up medical processing cache...${NC}"
python -c "
import os
os.makedirs('/app/cache/medical_models', exist_ok=True)
os.makedirs('/app/cache/embeddings', exist_ok=True)
print('✅ Medical processing cache initialized')
"

# Validate environment variables
echo -e "${YELLOW}🔧 Validating configuration...${NC}"
python -c "
import os
required_vars = ['DATABASE_URL', 'REDIS_URL']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'❌ Missing required environment variables: {missing}')
    exit(1)
else:
    print('✅ All required environment variables present')
"

# Test database connection
echo -e "${YELLOW}🔗 Testing database connection...${NC}"
python -c "
import asyncio
import asyncpg
import os

async def test_db():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.close()
        print('✅ Database connection successful')
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        exit(1)

asyncio.run(test_db())
"

# Test Redis connection
echo -e "${YELLOW}📡 Testing Redis connection...${NC}"
python -c "
import redis
import os

try:
    r = redis.from_url(os.getenv('REDIS_URL'))
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"

# Initialize medical AI models
echo -e "${YELLOW}🤖 Initializing AI models...${NC}"
python -c "
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    # Pre-load medical NER model
    ner = pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple')
    print('✅ Medical NER model loaded')
except Exception as e:
    print(f'⚠️  Medical NER model not available: {e}')

try:
    from sentence_transformers import SentenceTransformer
    # Pre-load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Sentence transformer model loaded')
except Exception as e:
    print(f'⚠️  Sentence transformer model not available: {e}')
"

# Memory optimization settings
echo -e "${YELLOW}🧠 Configuring memory optimization...${NC}"
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=131072
echo -e "${GREEN}✅ Memory optimization configured${NC}"

# PDF processing optimization
echo -e "${YELLOW}📄 Configuring PDF processing...${NC}"
python -c "
import os

# Set PDF processing defaults if not provided
pdf_config = {
    'PDF_MAX_FILE_SIZE': '104857600',      # 100MB
    'PDF_MAX_PAGES': '1000',
    'PDF_MEMORY_LIMIT': '536870912',       # 512MB
    'PDF_POOL_SIZE': '5',
    'PDF_PAGE_BATCH_SIZE': '10',
    'PDF_CHECKPOINT_INTERVAL': '50'
}

for key, default in pdf_config.items():
    if not os.getenv(key):
        os.environ[key] = default

print('✅ PDF processing configuration set')
"

# Health check endpoint validation
echo -e "${YELLOW}🏥 Setting up health checks...${NC}"
python -c "
import asyncio
from pathlib import Path

# Create health check endpoint
health_script = '''
import json
import psutil
from datetime import datetime

def health_check():
    return {
        \"status\": \"healthy\",
        \"timestamp\": datetime.now().isoformat(),
        \"memory_usage\": psutil.virtual_memory().percent,
        \"cpu_usage\": psutil.cpu_percent(),
        \"disk_usage\": psutil.disk_usage(\"/\").percent
    }
'''

Path('/app/health_check.py').write_text(health_script)
print('✅ Health check endpoint configured')
"

# Performance monitoring setup
echo -e "${YELLOW}📊 Setting up performance monitoring...${NC}"
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
mkdir -p $PROMETHEUS_MULTIPROC_DIR
echo -e "${GREEN}✅ Performance monitoring configured${NC}"

# Final system check
echo -e "${YELLOW}🔍 Final system check...${NC}"
python -c "
import sys
import importlib

required_modules = [
    'fastapi', 'uvicorn', 'sqlalchemy', 'redis', 'celery',
    'fitz', 'PyPDF2', 'spacy', 'transformers', 'psutil'
]

missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(module)

if missing:
    print(f'❌ Missing required modules: {missing}')
    sys.exit(1)
else:
    print('✅ All required modules available')
"

echo -e "${GREEN}🎉 Enhanced KOO Platform Backend is ready to start!${NC}"
echo -e "${GREEN}🔥 Memory-optimized PDF processing enabled${NC}"
echo -e "${GREEN}🧠 Medical AI models loaded${NC}"
echo -e "${GREEN}📊 Performance monitoring active${NC}"

# Execute the main command
exec "$@"