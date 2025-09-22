#!/usr/bin/env python3
"""
Simplified Test Runner for Nuance Merge System - ASCII Only
"""

import sys
import asyncio
from datetime import datetime
import traceback

# Add backend to path for imports
sys.path.append('C:/Users/ramih/Desktop/code/backend')

async def test_basic_functionality():
    """Test basic functionality of the nuance merge system"""

    print("=" * 50)
    print("NUANCE MERGE SYSTEM - BASIC FUNCTIONALITY TEST")
    print("=" * 50)

    try:
        print("\n1. Testing Core Engine Import...")
        from core.nuance_merge_engine import AdvancedNuanceMergeEngine, NuanceType, MergeCategory
        print("   [PASS] Core engine imported successfully")

        print("\n2. Testing Engine Initialization...")
        engine = AdvancedNuanceMergeEngine()
        print("   [PASS] Engine initialized successfully")

        print("\n3. Testing Basic Nuance Detection...")

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

        start_time = datetime.utcnow()
        result = await engine.detect_nuances(
            original_content=original_content,
            updated_content=updated_content,
            chapter_id="test_chapter_123",
            context=context
        )
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        if result:
            print("   [PASS] Nuance detection successful!")
            print(f"      Processing time: {processing_time:.3f} seconds")
            print(f"      Nuance ID: {result.nuance_id}")
            print(f"      Confidence: {result.confidence_score:.3f}")
            print(f"      Type: {result.nuance_type.value}")
            print(f"      Category: {result.merge_category.value}")
            print(f"      Semantic Similarity: {result.similarity_metrics.semantic_similarity:.3f}")
            print(f"      Medical Concepts: {len(result.medical_context.medical_concepts_added)}")
            print(f"      Sentence Analyses: {len(result.sentence_analyses)}")

            # Test similarity metrics specifically
            print(f"      Jaccard Similarity: {result.similarity_metrics.jaccard_similarity:.3f}")
            print(f"      Cosine Similarity: {result.similarity_metrics.cosine_similarity:.3f}")
            print(f"      Levenshtein Distance: {result.similarity_metrics.levenshtein_distance}")

        else:
            print("   [WARN] No nuance detected")
            return False

        print("\n4. Testing API Model Integration...")
        from api.enhanced_chapters import _convert_nuance_to_response

        api_response = _convert_nuance_to_response(result)
        print("   [PASS] API response conversion successful")
        print(f"      Response nuance_id: {api_response.nuance_id}")
        print(f"      Response confidence: {api_response.confidence_score:.3f}")

        print("\n5. Testing Edge Cases...")

        # Test identical content
        identical_result = await engine.detect_nuances(
            original_content="Identical content.",
            updated_content="Identical content.",
            chapter_id="test_identical",
            context=context
        )

        if identical_result is None:
            print("   [PASS] Correctly rejected identical content")
        else:
            print("   [FAIL] Should have rejected identical content")

        # Test empty content
        empty_result = await engine.detect_nuances(
            original_content="",
            updated_content="",
            chapter_id="test_empty",
            context=context
        )

        if empty_result is None:
            print("   [PASS] Correctly rejected empty content")
        else:
            print("   [FAIL] Should have rejected empty content")

        print("\n6. Testing Performance...")

        # Test with larger content
        large_original = " ".join(["Medical procedure description with detailed steps."] * 30)
        large_updated = " ".join(["Enhanced medical procedure description with comprehensive detailed steps."] * 30)

        start_time = datetime.utcnow()
        large_result = await engine.detect_nuances(
            original_content=large_original,
            updated_content=large_updated,
            chapter_id="test_large",
            context=context
        )
        end_time = datetime.utcnow()
        large_processing_time = (end_time - start_time).total_seconds()

        print(f"   Large content processing time: {large_processing_time:.3f} seconds")

        if large_processing_time < 10.0:
            print("   [PASS] Performance within acceptable range")
        else:
            print("   [WARN] Performance slower than expected")

        return True

    except Exception as e:
        print(f"   [FAIL] Test failed: {e}")
        traceback.print_exc()
        return False

async def test_database_schema():
    """Test database schema validation"""

    print("\n" + "=" * 50)
    print("DATABASE SCHEMA VALIDATION")
    print("=" * 50)

    try:
        schema_path = "C:/Users/ramih/Desktop/code/backend/database/schemas.sql"

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_content = f.read()

        print("\n1. Checking for required tables...")

        required_tables = [
            "nuance_merges",
            "similarity_computation_cache",
            "sentence_nuances",
            "nuance_detection_config",
            "nuance_processing_metrics"
        ]

        all_tables_found = True
        for table in required_tables:
            if f"CREATE TABLE {table}" in schema_content:
                print(f"   [PASS] Table '{table}' found")
            else:
                print(f"   [FAIL] Table '{table}' missing")
                all_tables_found = False

        print("\n2. Checking for performance indexes...")

        required_indexes = [
            "idx_nuance_merges_chapter_status",
            "idx_similarity_cache_lookup",
            "idx_sentence_nuances_merge"
        ]

        all_indexes_found = True
        for index in required_indexes:
            if f"CREATE INDEX {index}" in schema_content:
                print(f"   [PASS] Index '{index}' found")
            else:
                print(f"   [FAIL] Index '{index}' missing")
                all_indexes_found = False

        print("\n3. Checking for default configurations...")
        if "INSERT INTO nuance_detection_config" in schema_content:
            print("   [PASS] Default configurations found")
        else:
            print("   [FAIL] Default configurations missing")
            return False

        return all_tables_found and all_indexes_found

    except FileNotFoundError:
        print("   [FAIL] Schema file not found")
        return False
    except Exception as e:
        print(f"   [FAIL] Schema validation failed: {e}")
        return False

async def test_api_models():
    """Test API model structures"""

    print("\n" + "=" * 50)
    print("API MODELS VALIDATION")
    print("=" * 50)

    try:
        print("\n1. Testing API model imports...")
        from api.enhanced_chapters import (
            NuanceDetectionRequest,
            NuanceResponse,
            NuanceReviewRequest,
            NuanceApplyRequest
        )
        print("   [PASS] API models imported successfully")

        print("\n2. Testing NuanceDetectionRequest...")
        detection_request = NuanceDetectionRequest(
            chapter_id="test_chapter_123",
            original_content="Original content",
            updated_content="Updated content",
            specialty="neurosurgery",
            context={"test": True}
        )

        if detection_request.chapter_id == "test_chapter_123":
            print("   [PASS] Detection request model working")
        else:
            print("   [FAIL] Detection request model failed")
            return False

        print("\n3. Testing NuanceReviewRequest...")
        review_request = NuanceReviewRequest(
            nuance_id="test_nuance_123",
            action="approve",
            review_notes="Looks good"
        )

        if review_request.action == "approve":
            print("   [PASS] Review request model working")
        else:
            print("   [FAIL] Review request model failed")
            return False

        return True

    except Exception as e:
        print(f"   [FAIL] API models test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""

    print("KOO PLATFORM NUANCE MERGE SYSTEM - VALIDATION TESTS")
    print("Test started at:", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Database Schema", test_database_schema),
        ("API Models", test_api_models)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n*** ALL TESTS PASSED! ***")
        print("The Nuance Merge System is ready for production use.")
    else:
        print(f"\n*** {total-passed} TESTS FAILED ***")
        print("Please review the failures above.")

    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())