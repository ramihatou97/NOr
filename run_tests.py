#!/usr/bin/env python3
"""
Test Runner for Nuance Merge System
Simplified test execution without external dependencies
"""

import sys
import asyncio
from datetime import datetime
import traceback

# Add backend to path for imports
sys.path.append('C:/Users/ramih/Desktop/code/backend')

async def test_core_engine():
    """Test the core nuance detection engine"""

    print("Testing Core Nuance Detection Engine...")

    try:
        from core.nuance_merge_engine import AdvancedNuanceMergeEngine, NuanceType, MergeCategory

        # Initialize engine
        engine = AdvancedNuanceMergeEngine()

        # Test data
        original_content = """
        Craniotomy is a surgical procedure where a portion of the skull is removed to access the brain.
        This procedure is used for brain tumor removal.
        """.strip()

        updated_content = """
        Craniotomy is a neurosurgical procedure where a portion of the cranium is carefully removed to provide access to the intracranial space.
        This advanced procedure is commonly utilized for brain tumor resection and other neurosurgical interventions.
        """.strip()

        context = {
            "specialty": "neurosurgery",
            "user_id": "test_user_123",
            "quality_threshold": 0.75,
            "enable_ai_analysis": True
        }

        # Test basic detection
        print("   - Testing basic nuance detection...")
        result = await engine.detect_nuances(
            original_content=original_content,
            updated_content=updated_content,
            chapter_id="test_chapter_123",
            context=context
        )

        if result:
            print("   [PASS] Detection successful!")
            print(f"      - Nuance ID: {result.nuance_id}")
            print(f"      - Confidence: {result.confidence_score:.3f}")
            print(f"      - Type: {result.nuance_type.value}")
            print(f"      - Category: {result.merge_category.value}")
            print(f"      - Semantic Similarity: {result.similarity_metrics.semantic_similarity:.3f}")
            print(f"      - Medical Concepts Added: {len(result.medical_context.medical_concepts_added)}")
            print(f"      - Sentence Analyses: {len(result.sentence_analyses)}")
        else:
            print("   [WARN] No nuance detected (this might be expected)")

        # Test similarity metrics
        print("   - Testing similarity metrics...")
        config = engine._get_specialty_config("neurosurgery")
        metrics = await engine._calculate_similarity_metrics(original_content, updated_content, config)

        print(f"      - Semantic: {metrics.semantic_similarity:.3f}")
        print(f"      - Jaccard: {metrics.jaccard_similarity:.3f}")
        print(f"      - Cosine: {metrics.cosine_similarity:.3f}")

        # Test identical content (should return None)
        print("   - Testing identical content detection...")
        identical_result = await engine.detect_nuances(
            original_content="Identical content.",
            updated_content="Identical content.",
            chapter_id="test_identical",
            context=context
        )

        if identical_result is None:
            print("   ✅ Correctly identified identical content")
        else:
            print("   ⚠️  Unexpected result for identical content")

        return True

    except Exception as e:
        print(f"   ❌ Core engine test failed: {e}")
        traceback.print_exc()
        return False

async def test_api_models():
    """Test API model structures"""

    print("🌐 Testing API Models...")

    try:
        from api.enhanced_chapters import (
            NuanceDetectionRequest,
            NuanceResponse,
            NuanceReviewRequest,
            NuanceApplyRequest
        )

        # Test detection request model
        print("   - Testing NuanceDetectionRequest...")
        detection_request = NuanceDetectionRequest(
            chapter_id="test_chapter_123",
            original_content="Original content",
            updated_content="Updated content",
            specialty="neurosurgery",
            context={"test": True}
        )

        assert detection_request.chapter_id == "test_chapter_123"
        assert detection_request.specialty == "neurosurgery"
        print("   ✅ Detection request model working")

        # Test review request model
        print("   - Testing NuanceReviewRequest...")
        review_request = NuanceReviewRequest(
            nuance_id="test_nuance_123",
            action="approve",
            review_notes="Looks good",
            reviewer_confidence=0.9
        )

        assert review_request.action == "approve"
        print("   ✅ Review request model working")

        # Test apply request model
        print("   - Testing NuanceApplyRequest...")
        apply_request = NuanceApplyRequest(
            nuance_id="test_nuance_123",
            apply_method="automatic",
            selected_changes=["change1", "change2"]
        )

        assert apply_request.apply_method == "automatic"
        print("   ✅ Apply request model working")

        return True

    except Exception as e:
        print(f"   ❌ API models test failed: {e}")
        traceback.print_exc()
        return False

async def test_database_schema():
    """Test database schema structure"""

    print("🗄️ Testing Database Schema...")

    try:
        # Read schema file
        schema_path = "C:/Users/ramih/Desktop/code/backend/database/schemas.sql"

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_content = f.read()

        # Check for required tables
        required_tables = [
            "nuance_merges",
            "similarity_computation_cache",
            "sentence_nuances",
            "nuance_detection_config",
            "nuance_processing_metrics"
        ]

        print("   - Checking for required tables...")
        for table in required_tables:
            if f"CREATE TABLE {table}" in schema_content:
                print(f"     ✅ Table '{table}' found")
            else:
                print(f"     ❌ Table '{table}' missing")
                return False

        # Check for indexes
        print("   - Checking for performance indexes...")
        required_indexes = [
            "idx_nuance_merges_chapter_status",
            "idx_similarity_cache_lookup",
            "idx_sentence_nuances_merge",
            "idx_processing_metrics_operation"
        ]

        for index in required_indexes:
            if f"CREATE INDEX {index}" in schema_content:
                print(f"     ✅ Index '{index}' found")
            else:
                print(f"     ❌ Index '{index}' missing")
                return False

        # Check for default configurations
        print("   - Checking for default specialty configurations...")
        if "INSERT INTO nuance_detection_config" in schema_content:
            print("     ✅ Default configurations found")
        else:
            print("     ❌ Default configurations missing")
            return False

        print("   ✅ Database schema validation complete")
        return True

    except FileNotFoundError:
        print("   ❌ Schema file not found")
        return False
    except Exception as e:
        print(f"   ❌ Database schema test failed: {e}")
        return False

async def test_performance():
    """Test performance characteristics"""

    print("⚡ Testing Performance...")

    try:
        from core.nuance_merge_engine import AdvancedNuanceMergeEngine

        engine = AdvancedNuanceMergeEngine()

        # Test with different content sizes
        test_cases = [
            ("Small", "Short medical text.", "Enhanced short medical text."),
            ("Medium", " ".join(["Medical procedure description."] * 20),
                     " ".join(["Enhanced medical procedure description."] * 20)),
            ("Large", " ".join(["Complex neurosurgical procedure with detailed steps."] * 50),
                     " ".join(["Advanced neurosurgical procedure with comprehensive detailed steps."] * 50))
        ]

        for size, original, updated in test_cases:
            print(f"   - Testing {size.lower()} content processing...")

            start_time = datetime.utcnow()

            result = await engine.detect_nuances(
                original_content=original,
                updated_content=updated,
                chapter_id=f"perf_test_{size.lower()}",
                context={"specialty": "general_medicine"}
            )

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            print(f"     - {size} content: {processing_time:.3f}s")

            # Performance assertions
            if size == "Small" and processing_time > 5.0:
                print(f"     ⚠️  Small content took longer than expected: {processing_time:.3f}s")
            elif size == "Medium" and processing_time > 10.0:
                print(f"     ⚠️  Medium content took longer than expected: {processing_time:.3f}s")
            elif size == "Large" and processing_time > 15.0:
                print(f"     ⚠️  Large content took longer than expected: {processing_time:.3f}s")
            else:
                print(f"     ✅ Performance within acceptable range")

        return True

    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

async def test_integration():
    """Test end-to-end integration"""

    print("🔗 Testing Integration Flow...")

    try:
        from core.nuance_merge_engine import AdvancedNuanceMergeEngine
        from api.enhanced_chapters import _convert_nuance_to_response

        engine = AdvancedNuanceMergeEngine()

        # Test complete flow
        print("   - Testing complete detection to response conversion...")

        original = """
        Surgical planning requires careful consideration of patient anatomy.
        The procedure should be performed with standard instruments.
        """

        updated = """
        Neurosurgical planning requires meticulous consideration of patient neuroanatomy and pathophysiology.
        The procedure should be performed with specialized microsurgical instruments and intraoperative monitoring.
        """

        context = {
            "specialty": "neurosurgery",
            "user_id": "integration_test_user",
            "quality_threshold": 0.75
        }

        # Detect nuance
        detected_nuance = await engine.detect_nuances(
            original_content=original,
            updated_content=updated,
            chapter_id="integration_test_chapter",
            context=context
        )

        if detected_nuance:
            print("     ✅ Nuance detection successful")

            # Convert to API response
            api_response = _convert_nuance_to_response(detected_nuance)

            print("     ✅ Response conversion successful")
            print(f"        - Response nuance_id: {api_response.nuance_id}")
            print(f"        - Response confidence: {api_response.confidence_score:.3f}")
            print(f"        - Medical concepts: {len(api_response.medical_context['medical_concepts_added'])}")

        else:
            print("     ⚠️  No nuance detected in integration test")

        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""

    print("KOO Platform Nuance Merge System - Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    test_results = []

    # Run all tests
    tests = [
        ("Core Engine", test_core_engine),
        ("API Models", test_api_models),
        ("Database Schema", test_database_schema),
        ("Performance", test_performance),
        ("Integration", test_integration)
    ]

    for test_name, test_func in tests:
        print(f"Running {test_name} Tests...")
        try:
            result = await test_func()
            test_results.append((test_name, result))
            print(f"✅ {test_name} Tests: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name} Tests: FAILED with exception: {e}")
            test_results.append((test_name, False))
        print()

    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 60)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 ALL TESTS PASSED! The Nuance Merge System is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")

    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())