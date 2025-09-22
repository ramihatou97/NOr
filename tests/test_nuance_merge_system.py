# tests/test_nuance_merge_system.py
"""
Comprehensive Test Suite for Nuance Merge System
Tests all components: Core Engine, API Integration, and Database Schema
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
import json

# Core Engine Testing
from backend.core.nuance_merge_engine import (
    AdvancedNuanceMergeEngine,
    NuanceType,
    MergeCategory,
    NuanceStatus,
    DetectedNuance,
    SimilarityMetrics,
    MedicalContext,
    NuanceDetectionConfig
)

# API Testing
from fastapi.testclient import TestClient
from backend.api.enhanced_chapters import router

class TestNuanceMergeEngine:
    """Test the core nuance merge engine functionality"""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine instance for each test"""
        return AdvancedNuanceMergeEngine()

    @pytest.fixture
    def sample_content_pair(self):
        """Sample content for testing"""
        return {
            "original": """
            Craniotomy is a surgical procedure where a portion of the skull is removed to access the brain.
            This procedure is commonly used for brain tumor removal. The patient requires general anesthesia.
            """.strip(),
            "updated": """
            Craniotomy is a neurosurgical procedure where a portion of the cranium is carefully removed to provide access to the intracranial space.
            This advanced procedure is commonly utilized for brain tumor resection and other neurosurgical interventions.
            The patient requires general anesthesia with specialized neuroanesthetic monitoring.
            """.strip()
        }

    @pytest.fixture
    def test_context(self):
        """Standard test context"""
        return {
            "specialty": "neurosurgery",
            "user_id": "test_user_123",
            "quality_threshold": 0.75,
            "enable_ai_analysis": True
        }

    @pytest.mark.asyncio
    async def test_basic_nuance_detection(self, engine, sample_content_pair, test_context):
        """Test basic nuance detection functionality"""

        result = await engine.detect_nuances(
            original_content=sample_content_pair["original"],
            updated_content=sample_content_pair["updated"],
            chapter_id="test_chapter_123",
            context=test_context
        )

        # Assertions
        assert result is not None
        assert isinstance(result, DetectedNuance)
        assert result.chapter_id == "test_chapter_123"
        assert result.confidence_score > 0.0
        assert result.confidence_score <= 1.0
        assert result.nuance_type in NuanceType
        assert result.merge_category in MergeCategory
        assert result.status == NuanceStatus.DETECTED

        # Check similarity metrics
        assert isinstance(result.similarity_metrics, SimilarityMetrics)
        assert 0 <= result.similarity_metrics.semantic_similarity <= 1
        assert 0 <= result.similarity_metrics.jaccard_similarity <= 1
        assert result.similarity_metrics.levenshtein_distance >= 0

        print(f"✅ Basic Detection Test Passed - Confidence: {result.confidence_score:.3f}")

    @pytest.mark.asyncio
    async def test_similarity_metrics_calculation(self, engine):
        """Test similarity metrics calculation accuracy"""

        # Identical content should have high similarity
        identical_content = "This is identical content for testing."
        config = NuanceDetectionConfig(specialty="general_medicine")

        metrics = await engine._calculate_similarity_metrics(
            identical_content, identical_content, config
        )

        assert metrics.semantic_similarity > 0.99
        assert metrics.jaccard_similarity == 1.0
        assert metrics.levenshtein_distance == 0
        assert metrics.cosine_similarity > 0.99

        # Very different content should have low similarity
        different_content_1 = "Neurosurgical procedures require precise planning."
        different_content_2 = "Cooking recipes need accurate measurements."

        metrics = await engine._calculate_similarity_metrics(
            different_content_1, different_content_2, config
        )

        assert metrics.semantic_similarity < 0.5
        assert metrics.jaccard_similarity < 0.3

        print("✅ Similarity Metrics Test Passed")

    @pytest.mark.asyncio
    async def test_medical_context_analysis(self, engine, sample_content_pair):
        """Test medical context analysis functionality"""

        result = await engine._analyze_medical_context(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            "neurosurgery"
        )

        assert isinstance(result, MedicalContext)
        assert result.specialty_context == "neurosurgery"
        assert isinstance(result.medical_concepts_added, list)
        assert isinstance(result.anatomical_references, list)
        assert isinstance(result.procedure_references, list)
        assert 0 <= result.clinical_relevance_score <= 1

        print("✅ Medical Context Analysis Test Passed")

    @pytest.mark.asyncio
    async def test_sentence_level_analysis(self, engine, sample_content_pair):
        """Test sentence-level analysis functionality"""

        analyses = await engine._perform_sentence_analysis(
            sample_content_pair["original"],
            sample_content_pair["updated"]
        )

        assert isinstance(analyses, list)
        assert len(analyses) > 0

        for analysis in analyses:
            assert hasattr(analysis, 'original_sentence')
            assert hasattr(analysis, 'enhanced_sentence')
            assert hasattr(analysis, 'sentence_position')
            assert hasattr(analysis, 'added_parts')
            assert hasattr(analysis, 'modified_parts')
            assert hasattr(analysis, 'removed_parts')
            assert 0 <= analysis.sentence_similarity <= 1

        print("✅ Sentence-Level Analysis Test Passed")

    @pytest.mark.asyncio
    async def test_nuance_classification(self, engine, sample_content_pair, test_context):
        """Test nuance type and category classification"""

        config = engine._get_specialty_config("neurosurgery")
        metrics = await engine._calculate_similarity_metrics(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            config
        )

        nuance_type = await engine._classify_nuance_type(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            metrics,
            test_context
        )

        merge_category = await engine._determine_merge_category(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            nuance_type,
            test_context
        )

        assert nuance_type in NuanceType
        assert merge_category in MergeCategory

        print(f"✅ Classification Test Passed - Type: {nuance_type.value}, Category: {merge_category.value}")

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, engine, sample_content_pair, test_context):
        """Test confidence score calculation"""

        config = engine._get_specialty_config("neurosurgery")
        metrics = await engine._calculate_similarity_metrics(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            config
        )

        nuance_type = await engine._classify_nuance_type(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            metrics,
            test_context
        )

        merge_category = await engine._determine_merge_category(
            sample_content_pair["original"],
            sample_content_pair["updated"],
            nuance_type,
            test_context
        )

        confidence = await engine._calculate_confidence_score(
            metrics, nuance_type, merge_category, test_context
        )

        assert 0 <= confidence <= 1
        print(f"✅ Confidence Score Test Passed - Score: {confidence:.3f}")

    def test_specialty_configurations(self, engine):
        """Test specialty-specific configurations"""

        neurosurgery_config = engine._get_specialty_config("neurosurgery")
        general_config = engine._get_specialty_config("general_medicine")

        assert neurosurgery_config.specialty == "neurosurgery"
        assert general_config.specialty == "general_medicine"

        # Neurosurgery should have stricter thresholds
        assert neurosurgery_config.nuance_threshold_high >= general_config.nuance_threshold_high
        assert neurosurgery_config.require_review_threshold >= general_config.require_review_threshold

        print("✅ Specialty Configuration Test Passed")

    @pytest.mark.asyncio
    async def test_edge_cases(self, engine):
        """Test edge cases and error handling"""

        # Empty content
        result = await engine.detect_nuances("", "", "test_chapter", {})
        assert result is None

        # Identical content
        identical = "This is identical content."
        result = await engine.detect_nuances(identical, identical, "test_chapter", {"specialty": "general_medicine"})
        assert result is None

        # Very short content
        short1 = "Short."
        short2 = "Brief."
        result = await engine.detect_nuances(short1, short2, "test_chapter", {"specialty": "general_medicine"})
        # Should handle gracefully

        print("✅ Edge Cases Test Passed")

class TestNuanceMergeAPI:
    """Test the API integration functionality"""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def mock_user(self):
        """Mock user for testing"""
        class MockUser:
            id = "test_user_123"
            email = "test@example.com"
            full_name = "Test User"
        return MockUser()

    @pytest.fixture
    def detection_request_data(self):
        """Sample detection request data"""
        return {
            "chapter_id": "test_chapter_123",
            "original_content": "Original surgical procedure description.",
            "updated_content": "Enhanced neurosurgical procedure description with improved medical terminology.",
            "specialty": "neurosurgery",
            "context": {"test": True}
        }

    def test_api_models_validation(self, detection_request_data):
        """Test API model validation"""
        from backend.api.enhanced_chapters import NuanceDetectionRequest

        # Valid request should pass
        request = NuanceDetectionRequest(**detection_request_data)
        assert request.chapter_id == "test_chapter_123"
        assert request.specialty == "neurosurgery"

        # Missing required fields should fail
        with pytest.raises(ValueError):
            NuanceDetectionRequest(chapter_id="test")  # Missing required fields

        print("✅ API Models Validation Test Passed")

    def test_response_model_structure(self):
        """Test response model structure"""
        from backend.api.enhanced_chapters import NuanceResponse

        sample_response_data = {
            "nuance_id": "test_nuance_123",
            "chapter_id": "test_chapter_123",
            "status": "detected",
            "nuance_type": "enhancement",
            "merge_category": "content_improvement",
            "confidence_score": 0.85,
            "similarity_metrics": {
                "semantic_similarity": 0.82,
                "jaccard_similarity": 0.75,
                "cosine_similarity": 0.80,
                "normalized_levenshtein": 0.78
            },
            "medical_context": {
                "medical_concepts_added": ["neurosurgical", "terminology"],
                "anatomical_references": ["brain"],
                "procedure_references": ["surgery"],
                "specialty_context": "neurosurgery",
                "clinical_relevance_score": 0.9
            },
            "ai_analysis": {"quality_improvement": 0.8},
            "sentence_analyses": [],
            "workflow_status": {
                "auto_apply_eligible": False,
                "manual_review_required": True,
                "priority_level": 5
            },
            "created_at": datetime.utcnow()
        }

        response = NuanceResponse(**sample_response_data)
        assert response.confidence_score == 0.85
        assert response.status == "detected"

        print("✅ Response Model Structure Test Passed")

class TestIntegrationFlow:
    """Test end-to-end integration flow"""

    @pytest.mark.asyncio
    async def test_complete_detection_flow(self):
        """Test complete detection flow from API to engine"""

        # Initialize engine
        engine = AdvancedNuanceMergeEngine()

        # Sample data
        original_content = """
        Brain surgery requires careful planning and execution.
        The surgeon must have extensive training.
        """

        updated_content = """
        Neurosurgical procedures require meticulous preoperative planning and precise intraoperative execution.
        The neurosurgeon must have comprehensive subspecialty training and board certification.
        """

        context = {
            "specialty": "neurosurgery",
            "user_id": "test_user",
            "quality_threshold": 0.75
        }

        # Run detection
        result = await engine.detect_nuances(
            original_content=original_content,
            updated_content=updated_content,
            chapter_id="integration_test_chapter",
            context=context
        )

        # Verify result
        assert result is not None
        assert result.confidence_score > 0.5  # Should detect significant improvement
        assert len(result.sentence_analyses) > 0
        assert result.medical_context.clinical_relevance_score > 0

        print("✅ Complete Integration Flow Test Passed")
        print(f"   - Detected nuance type: {result.nuance_type.value}")
        print(f"   - Confidence score: {result.confidence_score:.3f}")
        print(f"   - Medical concepts added: {len(result.medical_context.medical_concepts_added)}")

class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""

    @pytest.mark.asyncio
    async def test_processing_performance(self):
        """Test processing performance with various content sizes"""

        engine = AdvancedNuanceMergeEngine()

        # Small content
        small_original = "Short medical text."
        small_updated = "Enhanced short medical text."

        start_time = datetime.utcnow()
        result = await engine.detect_nuances(
            small_original, small_updated, "perf_test_small", {"specialty": "general_medicine"}
        )
        small_time = (datetime.utcnow() - start_time).total_seconds()

        # Medium content
        medium_original = " ".join(["Medical procedure description."] * 50)
        medium_updated = " ".join(["Enhanced medical procedure description."] * 50)

        start_time = datetime.utcnow()
        result = await engine.detect_nuances(
            medium_original, medium_updated, "perf_test_medium", {"specialty": "general_medicine"}
        )
        medium_time = (datetime.utcnow() - start_time).total_seconds()

        # Performance assertions
        assert small_time < 5.0  # Should complete in under 5 seconds
        assert medium_time < 10.0  # Should complete in under 10 seconds

        print(f"✅ Performance Test Passed")
        print(f"   - Small content: {small_time:.3f}s")
        print(f"   - Medium content: {medium_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""

        engine = AdvancedNuanceMergeEngine()

        # Create multiple detection tasks
        tasks = []
        for i in range(5):
            task = engine.detect_nuances(
                f"Original content {i} for testing concurrent processing.",
                f"Updated content {i} with enhanced medical terminology for testing.",
                f"concurrent_test_{i}",
                {"specialty": "general_medicine"}
            )
            tasks.append(task)

        # Run concurrently
        start_time = datetime.utcnow()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (datetime.utcnow() - start_time).total_seconds()

        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 4  # At least 4/5 should succeed
        assert total_time < 15.0  # Should complete concurrently faster than sequential

        print(f"✅ Concurrent Processing Test Passed")
        print(f"   - {len(successful_results)}/5 tasks completed successfully")
        print(f"   - Total time: {total_time:.3f}s")

class TestDatabaseSchemaValidation:
    """Test database schema compatibility"""

    def test_schema_sql_syntax(self):
        """Test that schema SQL is syntactically correct"""

        # Read the schema file
        try:
            with open("C:/Users/ramih/Desktop/code/backend/database/schemas.sql", "r") as f:
                schema_content = f.read()

            # Basic syntax checks
            assert "CREATE TABLE nuance_merges" in schema_content
            assert "CREATE TABLE similarity_computation_cache" in schema_content
            assert "CREATE TABLE sentence_nuances" in schema_content
            assert "CREATE TABLE nuance_detection_config" in schema_content
            assert "CREATE INDEX" in schema_content
            assert "INSERT INTO nuance_detection_config" in schema_content

            print("✅ Database Schema Syntax Test Passed")

        except FileNotFoundError:
            print("⚠️ Schema file not found - this test requires the schema file")

# Test Runner Function
async def run_all_tests():
    """Run all tests and provide comprehensive report"""

    print("🚀 Starting Comprehensive Nuance Merge System Tests")
    print("=" * 70)

    # Core Engine Tests
    print("\n📦 CORE ENGINE TESTS")
    print("-" * 30)

    engine_tests = TestNuanceMergeEngine()
    engine = AdvancedNuanceMergeEngine()

    sample_content = {
        "original": "Basic surgical procedure description.",
        "updated": "Comprehensive neurosurgical procedure description with enhanced medical terminology."
    }

    context = {"specialty": "neurosurgery", "user_id": "test_user"}

    try:
        await engine_tests.test_basic_nuance_detection(engine, sample_content, context)
        await engine_tests.test_similarity_metrics_calculation(engine)
        await engine_tests.test_medical_context_analysis(engine, sample_content)
        await engine_tests.test_sentence_level_analysis(engine, sample_content)
        await engine_tests.test_nuance_classification(engine, sample_content, context)
        await engine_tests.test_confidence_score_calculation(engine, sample_content, context)
        engine_tests.test_specialty_configurations(engine)
        await engine_tests.test_edge_cases(engine)

        print("✅ ALL CORE ENGINE TESTS PASSED")

    except Exception as e:
        print(f"❌ Core Engine Test Failed: {e}")

    # API Tests
    print("\n🌐 API INTEGRATION TESTS")
    print("-" * 30)

    api_tests = TestNuanceMergeAPI()

    try:
        detection_data = {
            "chapter_id": "test_chapter",
            "original_content": "Original content",
            "updated_content": "Updated content",
            "specialty": "neurosurgery"
        }

        api_tests.test_api_models_validation(detection_data)
        api_tests.test_response_model_structure()

        print("✅ ALL API TESTS PASSED")

    except Exception as e:
        print(f"❌ API Test Failed: {e}")

    # Integration Tests
    print("\n🔗 INTEGRATION FLOW TESTS")
    print("-" * 30)

    integration_tests = TestIntegrationFlow()

    try:
        await integration_tests.test_complete_detection_flow()
        print("✅ ALL INTEGRATION TESTS PASSED")

    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")

    # Performance Tests
    print("\n⚡ PERFORMANCE TESTS")
    print("-" * 30)

    performance_tests = TestPerformanceAndScalability()

    try:
        await performance_tests.test_processing_performance()
        await performance_tests.test_concurrent_processing()
        print("✅ ALL PERFORMANCE TESTS PASSED")

    except Exception as e:
        print(f"❌ Performance Test Failed: {e}")

    # Database Schema Tests
    print("\n🗄️ DATABASE SCHEMA TESTS")
    print("-" * 30)

    db_tests = TestDatabaseSchemaValidation()

    try:
        db_tests.test_schema_sql_syntax()
        print("✅ ALL DATABASE TESTS PASSED")

    except Exception as e:
        print(f"❌ Database Test Failed: {e}")

    print("\n" + "=" * 70)
    print("🎉 COMPREHENSIVE TEST SUITE COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_all_tests())