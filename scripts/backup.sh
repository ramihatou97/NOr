#!/bin/bash

# KOO Platform Database Backup Script
# Creates timestamped backups of the PostgreSQL database

set -e

# Configuration
DB_HOST="postgres"
DB_PORT="5432"
DB_NAME="koo_platform"
DB_USER="koo_user"
BACKUP_DIR="/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/koo_platform_backup_${TIMESTAMP}.sql"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create database backup
echo "Starting database backup at $(date)"
pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_FILE"

# Compress the backup
gzip "$BACKUP_FILE"

echo "Database backup completed: ${BACKUP_FILE}.gz"

# Clean up old backups (keep only last 7 days)
find "$BACKUP_DIR" -name "koo_platform_backup_*.sql.gz" -mtime +7 -delete

echo "Backup cleanup completed"