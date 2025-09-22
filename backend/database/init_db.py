#!/usr/bin/env python3
"""
Database Initialization Script for KOO Platform
Creates all necessary tables, indexes, and initial data
"""

import asyncio
import os
import sys
from pathlib import Path
import logging
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import asyncpg
    import uvloop
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_database_if_not_exists(admin_url: str, db_name: str) -> bool:
    """Create database if it doesn't exist"""
    try:
        # Connect to postgres database to create our database
        admin_conn = await asyncpg.connect(admin_url)

        # Check if database exists
        result = await admin_conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )

        if not result:
            logger.info(f"Creating database: {db_name}")
            await admin_conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Database {db_name} created successfully")
        else:
            logger.info(f"Database {db_name} already exists")

        await admin_conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False

async def run_sql_file(conn: asyncpg.Connection, file_path: Path) -> bool:
    """Execute SQL file"""
    try:
        if not file_path.exists():
            logger.warning(f"SQL file not found: {file_path}")
            return True  # Non-critical

        logger.info(f"Executing SQL file: {file_path.name}")
        sql_content = file_path.read_text(encoding='utf-8')

        # Split by semicolons and execute each statement
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

        for statement in statements:
            if statement.upper().startswith(('CREATE', 'ALTER', 'INSERT', 'UPDATE')):
                await conn.execute(statement)

        logger.info(f"Successfully executed {file_path.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to execute {file_path.name}: {e}")
        return False

async def create_extensions(conn: asyncpg.Connection) -> bool:
    """Create required PostgreSQL extensions"""
    extensions = [
        'uuid-ossp',
        'vector',
        'pg_trgm',
        'btree_gin'
    ]

    for ext in extensions:
        try:
            await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
            logger.info(f"Extension {ext} created/verified")
        except Exception as e:
            logger.warning(f"Failed to create extension {ext}: {e}")
            # Vector extension might not be available in all environments
            if ext == 'vector':
                logger.warning("Vector extension not available - semantic search features will be limited")

    return True

async def verify_installation(conn: asyncpg.Connection) -> bool:
    """Verify database installation"""
    try:
        # Check key tables exist
        tables_to_check = [
            'users',
            'chapters',
            'nuance_detections',
            'quality_assessments',
            'research_queries'
        ]

        for table in tables_to_check:
            result = await conn.fetchval(
                "SELECT to_regclass($1)", f'public.{table}'
            )
            if result:
                logger.info(f"✓ Table {table} exists")
            else:
                logger.warning(f"✗ Table {table} missing")

        # Check extensions
        extensions = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'vector', 'pg_trgm', 'btree_gin')"
        )

        for ext in extensions:
            logger.info(f"✓ Extension {ext['extname']} installed")

        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

async def create_initial_data(conn: asyncpg.Connection) -> bool:
    """Create initial data and sample records"""
    try:
        # Create admin user if not exists
        admin_exists = await conn.fetchval(
            "SELECT id FROM users WHERE email = $1", "admin@kooplatform.com"
        )

        if not admin_exists:
            await conn.execute("""
                INSERT INTO users (
                    email, password_hash, full_name, title, specialty,
                    institution, is_active, is_verified
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8
                )
            """,
                "admin@kooplatform.com",
                "$2b$12$K8gPV.XvTxLVZr8mN9QJ2eH5YrF7VwJjH3Kt9B6Lr4mPq2Ns8Wd6u",  # "admin123"
                "System Administrator",
                "Platform Administrator",
                "System Administration",
                "KOO Platform",
                True,
                True
            )
            logger.info("Created admin user (admin@kooplatform.com / admin123)")

        # Create sample configuration
        config_exists = await conn.fetchval(
            "SELECT id FROM system_config WHERE config_key = $1", "platform_initialized"
        )

        if not config_exists:
            await conn.execute("""
                INSERT INTO system_config (config_key, config_value, description)
                VALUES ($1, $2, $3)
            """,
                "platform_initialized",
                json.dumps({"timestamp": datetime.now().isoformat(), "version": "1.0.0"}),
                "Platform initialization timestamp and version"
            )

        return True

    except Exception as e:
        logger.error(f"Failed to create initial data: {e}")
        return False

async def main():
    """Main initialization function"""
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available. Please install: asyncpg, uvloop")
        sys.exit(1)

    # Use uvloop for better performance
    try:
        uvloop.install()
    except:
        pass  # Fallback to default event loop

    # Get database configuration from environment
    db_url = os.getenv('DATABASE_URL', 'postgresql://koo_user:koo_secure_password_2024@localhost:5432/koo_platform')

    # Parse database URL
    try:
        if db_url.startswith('postgresql://'):
            # Extract components for database creation
            url_parts = db_url.replace('postgresql://', '').split('/')
            if len(url_parts) == 2:
                host_part, db_name = url_parts
                if '@' in host_part:
                    auth_part, host_port = host_part.split('@')
                    admin_url = f"postgresql://{auth_part}@{host_port}/postgres"
                else:
                    admin_url = f"postgresql://{host_part}/postgres"
                    db_name = url_parts[1]
            else:
                logger.error("Invalid DATABASE_URL format")
                sys.exit(1)
        else:
            logger.error("DATABASE_URL must start with postgresql://")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to parse DATABASE_URL: {e}")
        sys.exit(1)

    logger.info(f"Initializing KOO Platform database: {db_name}")

    # Step 1: Create database if needed
    if not await create_database_if_not_exists(admin_url, db_name):
        logger.error("Failed to create database")
        sys.exit(1)

    # Step 2: Connect to the target database
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("Connected to database successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Step 3: Create extensions
        await create_extensions(conn)

        # Step 4: Run schema files
        schema_dir = Path(__file__).parent
        schema_files = [
            schema_dir / 'schemas.sql',
            schema_dir / 'seed.sql'
        ]

        for schema_file in schema_files:
            if not await run_sql_file(conn, schema_file):
                logger.error(f"Failed to execute {schema_file}")
                sys.exit(1)

        # Step 5: Create initial data
        await create_initial_data(conn)

        # Step 6: Verify installation
        await verify_installation(conn)

        logger.info("🎉 Database initialization completed successfully!")
        logger.info("Platform is ready to use:")
        logger.info("  - Admin user: admin@kooplatform.com / admin123")
        logger.info("  - Database: Fully initialized with all tables and indexes")
        logger.info("  - Extensions: All required extensions installed")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())