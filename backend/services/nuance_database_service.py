# backend/services/nuance_database_service.py
"""
Enterprise Database Service Layer for Nuance Merge System
Provides complete database operations with connection pooling, caching, and error handling
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import logging

# Core imports with graceful fallbacks
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.nuance_merge_engine import (
    DetectedNuance,
    NuanceStatus,
    NuanceType,
    MergeCategory,
    SimilarityMetrics,
    MedicalContext,
    SentenceAnalysis,
    NuanceDetectionConfig
)

logger = logging.getLogger(__name__)

class NuanceDatabaseService:
    """Enterprise-grade database service for nuance merge operations"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or "postgresql://localhost/koo_platform"
        self.connection_pool = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 1000

    async def initialize(self) -> bool:
        """Initialize database connection pool with error handling"""

        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, using mock database operations")
            return False

        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                timeout=30,
                command_timeout=30
            )

            # Test connection
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            logger.info("Database connection pool initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            return False

    async def close(self):
        """Clean shutdown of database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Database connection pool closed")

    async def store_detected_nuance(self, nuance: DetectedNuance) -> str:
        """Store a detected nuance with full audit trail"""

        if not self.connection_pool:
            return await self._mock_store_nuance(nuance)

        try:
            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    # Store main nuance record
                    nuance_id = await conn.fetchval(
                        """
                        INSERT INTO nuance_merges (
                            id, chapter_id, section_id, original_content, updated_content,
                            original_content_vector, updated_content_vector,
                            semantic_similarity, jaccard_similarity, levenshtein_distance, cosine_similarity,
                            nuance_type, merge_category, confidence_score, clinical_relevance_score,
                            ai_analysis, processing_models, ai_recommendations,
                            medical_concepts_added, anatomical_references, procedure_references, specialty_context,
                            detection_algorithm, processing_time_ms, memory_usage_mb, algorithm_version,
                            status, priority_level, auto_apply_eligible, manual_review_required,
                            created_at, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                            $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32
                        ) RETURNING id
                        """,
                        nuance.nuance_id, nuance.chapter_id, nuance.section_id,
                        nuance.original_content, nuance.updated_content,
                        None, None,  # Vector embeddings (to be added when sentence-transformers available)
                        nuance.similarity_metrics.semantic_similarity,
                        nuance.similarity_metrics.jaccard_similarity,
                        nuance.similarity_metrics.levenshtein_distance,
                        nuance.similarity_metrics.cosine_similarity,
                        nuance.nuance_type.value, nuance.merge_category.value,
                        nuance.confidence_score, nuance.medical_context.clinical_relevance_score,
                        json.dumps(nuance.ai_analysis), json.dumps(nuance.processing_models),
                        json.dumps(nuance.ai_recommendations),
                        json.dumps(nuance.medical_context.medical_concepts_added),
                        json.dumps(nuance.medical_context.anatomical_references),
                        json.dumps(nuance.medical_context.procedure_references),
                        nuance.medical_context.specialty_context,
                        nuance.detection_algorithm, nuance.processing_time_ms,
                        nuance.memory_usage_mb, nuance.algorithm_version,
                        nuance.status.value, nuance.priority_level,
                        nuance.auto_apply_eligible, nuance.manual_review_required,
                        nuance.detected_at, nuance.detected_at
                    )

                    # Store sentence-level analyses
                    for sentence_analysis in nuance.sentence_analyses:
                        await conn.execute(
                            """
                            INSERT INTO sentence_nuances (
                                nuance_merge_id, original_sentence, enhanced_sentence,
                                sentence_position, paragraph_position,
                                added_parts, modified_parts, removed_parts, word_level_changes,
                                sentence_similarity, medical_concept_density,
                                clinical_importance_score, change_type, impact_category,
                                medical_terms_added, anatomical_references,
                                procedure_references, drug_references,
                                created_at
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                            )
                            """,
                            nuance_id, sentence_analysis.original_sentence,
                            sentence_analysis.enhanced_sentence,
                            sentence_analysis.sentence_position,
                            sentence_analysis.paragraph_position or 0,
                            json.dumps(sentence_analysis.added_parts),
                            json.dumps(sentence_analysis.modified_parts),
                            json.dumps(sentence_analysis.removed_parts),
                            json.dumps(sentence_analysis.word_level_changes),
                            sentence_analysis.sentence_similarity,
                            sentence_analysis.medical_concept_density,
                            sentence_analysis.clinical_importance_score,
                            sentence_analysis.change_type,
                            sentence_analysis.impact_category,
                            json.dumps(getattr(sentence_analysis, 'medical_terms_added', [])),
                            json.dumps(getattr(sentence_analysis, 'anatomical_references', [])),
                            json.dumps(getattr(sentence_analysis, 'procedure_references', [])),
                            json.dumps(getattr(sentence_analysis, 'drug_references', [])),
                            datetime.utcnow()
                        )

                    # Record processing metrics
                    await self._record_processing_metric(
                        conn, "nuance_storage", nuance.processing_time_ms,
                        True, None, nuance.chapter_id
                    )

                    # Invalidate cache
                    self._invalidate_cache(f"chapter_{nuance.chapter_id}")

                    logger.info(f"Stored nuance {nuance_id} for chapter {nuance.chapter_id}")
                    return nuance_id

        except Exception as e:
            logger.error(f"Failed to store nuance: {e}")
            await self._record_processing_metric(
                None, "nuance_storage", 0, False, str(e), nuance.chapter_id
            )
            raise

    async def get_chapter_nuances(self, chapter_id: str, status_filter: str = None,
                                limit: int = 50, offset: int = 0) -> Tuple[List[Dict], int]:
        """Retrieve nuances for a chapter with filtering and pagination"""

        if not self.connection_pool:
            return await self._mock_get_chapter_nuances(chapter_id, status_filter, limit, offset)

        try:
            cache_key = f"chapter_{chapter_id}_{status_filter}_{limit}_{offset}"
            cached_result = self._get_cached(cache_key)
            if cached_result:
                return cached_result

            async with self.connection_pool.acquire() as conn:
                # Build query with optional status filter
                base_query = """
                    SELECT nm.*,
                           array_agg(
                               json_build_object(
                                   'original_sentence', sn.original_sentence,
                                   'enhanced_sentence', sn.enhanced_sentence,
                                   'sentence_position', sn.sentence_position,
                                   'added_parts', sn.added_parts,
                                   'modified_parts', sn.modified_parts,
                                   'similarity', sn.sentence_similarity,
                                   'clinical_importance', sn.clinical_importance_score
                               )
                           ) as sentence_analyses
                    FROM nuance_merges nm
                    LEFT JOIN sentence_nuances sn ON nm.id = sn.nuance_merge_id
                    WHERE nm.chapter_id = $1
                """

                params = [chapter_id]
                if status_filter:
                    base_query += " AND nm.status = $2"
                    params.append(status_filter)

                base_query += """
                    GROUP BY nm.id
                    ORDER BY nm.created_at DESC
                    LIMIT $%d OFFSET $%d
                """ % (len(params) + 1, len(params) + 2)

                params.extend([limit, offset])

                # Get nuances
                nuances = await conn.fetch(base_query, *params)

                # Get total count
                count_query = "SELECT COUNT(*) FROM nuance_merges WHERE chapter_id = $1"
                count_params = [chapter_id]
                if status_filter:
                    count_query += " AND status = $2"
                    count_params.append(status_filter)

                total_count = await conn.fetchval(count_query, *count_params)

                # Convert to dictionary format
                result_nuances = []
                for row in nuances:
                    nuance_dict = dict(row)
                    # Parse JSON fields
                    for field in ['ai_analysis', 'processing_models', 'ai_recommendations',
                                'medical_concepts_added', 'anatomical_references', 'procedure_references']:
                        if nuance_dict.get(field):
                            nuance_dict[field] = json.loads(nuance_dict[field])

                    result_nuances.append(nuance_dict)

                result = (result_nuances, total_count)
                self._cache_result(cache_key, result)

                return result

        except Exception as e:
            logger.error(f"Failed to get chapter nuances: {e}")
            raise

    async def update_nuance_status(self, nuance_id: str, status: str,
                                 reviewer_id: str = None, review_notes: str = None,
                                 reviewer_confidence: float = None) -> bool:
        """Update nuance status with audit trail"""

        if not self.connection_pool:
            return await self._mock_update_status(nuance_id, status)

        try:
            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    # Update main record
                    result = await conn.execute(
                        """
                        UPDATE nuance_merges
                        SET status = $1, reviewed_by = $2, review_notes = $3,
                            reviewer_confidence = $4, reviewed_at = $5, updated_at = $5
                        WHERE id = $6
                        """,
                        status, reviewer_id, review_notes, reviewer_confidence,
                        datetime.utcnow(), nuance_id
                    )

                    if result == "UPDATE 0":
                        return False

                    # Get chapter_id for cache invalidation
                    chapter_id = await conn.fetchval(
                        "SELECT chapter_id FROM nuance_merges WHERE id = $1", nuance_id
                    )

                    # Invalidate cache
                    self._invalidate_cache(f"chapter_{chapter_id}")

                    logger.info(f"Updated nuance {nuance_id} status to {status}")
                    return True

        except Exception as e:
            logger.error(f"Failed to update nuance status: {e}")
            raise

    async def apply_nuance(self, nuance_id: str, apply_method: str,
                         applied_by: str, selected_changes: List[str] = None) -> bool:
        """Mark nuance as applied with application details"""

        if not self.connection_pool:
            return await self._mock_apply_nuance(nuance_id, apply_method)

        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE nuance_merges
                    SET status = 'applied', applied_at = $1, updated_at = $1
                    WHERE id = $2 AND status = 'approved'
                    """,
                    datetime.utcnow(), nuance_id
                )

                if result == "UPDATE 0":
                    return False

                # Get chapter_id for cache invalidation
                chapter_id = await conn.fetchval(
                    "SELECT chapter_id FROM nuance_merges WHERE id = $1", nuance_id
                )

                # Invalidate cache
                self._invalidate_cache(f"chapter_{chapter_id}")

                logger.info(f"Applied nuance {nuance_id} using method {apply_method}")
                return True

        except Exception as e:
            logger.error(f"Failed to apply nuance: {e}")
            raise

    async def get_nuance_analytics(self, chapter_id: str = None,
                                 time_period: str = "30d") -> Dict[str, Any]:
        """Get comprehensive analytics for nuance detection and application"""

        if not self.connection_pool:
            return await self._mock_get_analytics(chapter_id, time_period)

        try:
            # Parse time period
            days = self._parse_time_period(time_period)
            since_date = datetime.utcnow() - timedelta(days=days)

            async with self.connection_pool.acquire() as conn:
                # Base condition
                base_condition = "WHERE created_at >= $1"
                params = [since_date]

                if chapter_id:
                    base_condition += " AND chapter_id = $2"
                    params.append(chapter_id)

                # Detection metrics
                detection_metrics = await conn.fetchrow(f"""
                    SELECT
                        COUNT(*) as total_detected,
                        AVG(confidence_score) as avg_confidence,
                        AVG(processing_time_ms) as avg_processing_time,
                        COUNT(*) FILTER (WHERE status = 'approved') as approved_count,
                        COUNT(*) FILTER (WHERE status = 'applied') as applied_count,
                        COUNT(*) FILTER (WHERE auto_apply_eligible = true) as auto_applicable
                    FROM nuance_merges {base_condition}
                """, *params)

                # Quality metrics
                quality_metrics = await conn.fetchrow(f"""
                    SELECT
                        AVG(clinical_relevance_score) as avg_clinical_relevance,
                        AVG(semantic_similarity) as avg_semantic_similarity,
                        COUNT(*) FILTER (WHERE confidence_score >= 0.8) as high_confidence_count
                    FROM nuance_merges {base_condition}
                """, *params)

                # Performance metrics
                performance_metrics = await conn.fetchrow(f"""
                    SELECT
                        AVG(processing_time_ms) as avg_processing_time,
                        MAX(processing_time_ms) as max_processing_time,
                        MIN(processing_time_ms) as min_processing_time
                    FROM nuance_processing_metrics
                    WHERE created_at >= $1 AND success = true
                """, since_date)

                return {
                    "detection_metrics": dict(detection_metrics) if detection_metrics else {},
                    "quality_metrics": dict(quality_metrics) if quality_metrics else {},
                    "performance_metrics": dict(performance_metrics) if performance_metrics else {},
                    "time_period": time_period,
                    "generated_at": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            raise

    async def get_specialty_config(self, specialty: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a medical specialty"""

        if not self.connection_pool:
            return await self._mock_get_config(specialty)

        try:
            cache_key = f"config_{specialty}"
            cached_config = self._get_cached(cache_key)
            if cached_config:
                return cached_config

            async with self.connection_pool.acquire() as conn:
                config = await conn.fetchrow(
                    "SELECT * FROM nuance_detection_config WHERE specialty = $1 AND is_active = true",
                    specialty
                )

                if config:
                    config_dict = dict(config)
                    # Parse JSON fields
                    if config_dict.get('ai_analysis_models'):
                        config_dict['ai_analysis_models'] = json.loads(config_dict['ai_analysis_models'])

                    self._cache_result(cache_key, config_dict, ttl=7200)  # Cache for 2 hours
                    return config_dict

                return None

        except Exception as e:
            logger.error(f"Failed to get specialty config: {e}")
            raise

    async def store_similarity_cache(self, content_hash_a: str, content_hash_b: str,
                                   similarity_type: str, similarity_score: float,
                                   computation_model: str = None) -> bool:
        """Store similarity computation result for caching"""

        if not self.connection_pool:
            return True  # Mock success

        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO similarity_computation_cache
                    (content_hash_a, content_hash_b, similarity_type, similarity_score,
                     computation_model, computed_at, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (content_hash_a, content_hash_b, similarity_type, computation_model)
                    DO UPDATE SET
                        similarity_score = EXCLUDED.similarity_score,
                        computed_at = EXCLUDED.computed_at,
                        expires_at = EXCLUDED.expires_at
                    """,
                    content_hash_a, content_hash_b, similarity_type, similarity_score,
                    computation_model, datetime.utcnow(),
                    datetime.utcnow() + timedelta(days=7)  # Cache for 1 week
                )
                return True

        except Exception as e:
            logger.error(f"Failed to store similarity cache: {e}")
            return False

    async def get_similarity_cache(self, content_hash_a: str, content_hash_b: str,
                                 similarity_type: str, computation_model: str = None) -> Optional[float]:
        """Retrieve cached similarity computation result"""

        if not self.connection_pool:
            return None

        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval(
                    """
                    SELECT similarity_score FROM similarity_computation_cache
                    WHERE content_hash_a = $1 AND content_hash_b = $2
                    AND similarity_type = $3 AND computation_model = $4
                    AND (expires_at IS NULL OR expires_at > NOW())
                    """,
                    content_hash_a, content_hash_b, similarity_type, computation_model
                )
                return result

        except Exception as e:
            logger.error(f"Failed to get similarity cache: {e}")
            return None

    async def _record_processing_metric(self, conn, operation_type: str,
                                      processing_time_ms: int, success: bool,
                                      error_message: str = None, chapter_id: str = None):
        """Record processing metrics for performance monitoring"""

        if not conn and not self.connection_pool:
            return

        try:
            connection = conn or self.connection_pool

            if conn:
                await conn.execute(
                    """
                    INSERT INTO nuance_processing_metrics
                    (operation_type, processing_time_ms, success, error_message, chapter_id, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    operation_type, processing_time_ms, success, error_message,
                    chapter_id, datetime.utcnow()
                )
            else:
                async with connection.acquire() as new_conn:
                    await new_conn.execute(
                        """
                        INSERT INTO nuance_processing_metrics
                        (operation_type, processing_time_ms, success, error_message, chapter_id, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        operation_type, processing_time_ms, success, error_message,
                        chapter_id, datetime.utcnow()
                    )

        except Exception as e:
            logger.error(f"Failed to record processing metric: {e}")

    # Caching utilities
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.utcnow() < entry['expires']:
                return entry['data']
            else:
                del self.cache[key]
        return None

    def _cache_result(self, key: str, data: Any, ttl: int = None):
        """Cache result with TTL"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'data': data,
            'created': datetime.utcnow(),
            'expires': datetime.utcnow() + timedelta(seconds=ttl or self.cache_ttl)
        }

    def _invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.cache[key]

    def _parse_time_period(self, time_period: str) -> int:
        """Parse time period string to days"""
        if time_period.endswith('d'):
            return int(time_period[:-1])
        elif time_period.endswith('w'):
            return int(time_period[:-1]) * 7
        elif time_period.endswith('m'):
            return int(time_period[:-1]) * 30
        elif time_period.endswith('y'):
            return int(time_period[:-1]) * 365
        else:
            return 30  # Default to 30 days

    # Mock implementations for when database is not available
    async def _mock_store_nuance(self, nuance: DetectedNuance) -> str:
        """Mock implementation for storing nuance"""
        logger.info(f"Mock: Stored nuance {nuance.nuance_id}")
        return nuance.nuance_id

    async def _mock_get_chapter_nuances(self, chapter_id: str, status_filter: str,
                                      limit: int, offset: int) -> Tuple[List[Dict], int]:
        """Mock implementation for getting nuances"""
        mock_nuance = {
            "id": "mock_nuance_123",
            "chapter_id": chapter_id,
            "status": status_filter or "detected",
            "confidence_score": 0.85,
            "nuance_type": "enhancement",
            "created_at": datetime.utcnow()
        }
        return ([mock_nuance], 1)

    async def _mock_update_status(self, nuance_id: str, status: str) -> bool:
        """Mock implementation for updating status"""
        logger.info(f"Mock: Updated nuance {nuance_id} to status {status}")
        return True

    async def _mock_apply_nuance(self, nuance_id: str, apply_method: str) -> bool:
        """Mock implementation for applying nuance"""
        logger.info(f"Mock: Applied nuance {nuance_id} using {apply_method}")
        return True

    async def _mock_get_analytics(self, chapter_id: str, time_period: str) -> Dict[str, Any]:
        """Mock implementation for analytics"""
        return {
            "detection_metrics": {
                "total_detected": 10,
                "avg_confidence": 0.82,
                "approved_count": 8,
                "applied_count": 6
            },
            "quality_metrics": {
                "avg_clinical_relevance": 0.75,
                "avg_semantic_similarity": 0.88
            },
            "performance_metrics": {
                "avg_processing_time": 245
            }
        }

    async def _mock_get_config(self, specialty: str) -> Dict[str, Any]:
        """Mock implementation for specialty config"""
        return {
            "specialty": specialty,
            "nuance_threshold_high": 0.90,
            "nuance_threshold_medium": 0.75,
            "auto_apply_threshold": 0.95,
            "require_review_threshold": 0.80
        }

# Global instance
nuance_db_service = NuanceDatabaseService()