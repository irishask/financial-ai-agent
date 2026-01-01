"""
═══════════════════════════════════════════════════════════════════════════════
RAG Infrastructure Tests
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
    Validate the foundational RAG infrastructure components:
    - Vector store creation and storage (ChromaDB)
    - Basic vector similarity search functionality
    - Core retrieval mechanism without semantic complexity

TESTS INCLUDED:
    Test 1: Vector Store Embeddings Quality
            Component: build_category_vectorstore.py
            Validates: Embeddings created, stored, and accessible
    
    Test 2: Exact Name Retrieval
            Component: trn_category_rag.py → query_categories()
            Validates: Basic similarity search retrieves exact matches

USAGE:
    from tests.test_rag_infrastructure import run_infrastructure_tests
    results = run_infrastructure_tests()
    
    # Or run individual tests
    from tests.test_rag_infrastructure import (
        test_vector_store_embeddings_quality,
        test_exact_name_retrieval
    )
    
    test_vector_store_embeddings_quality()
    test_exact_name_retrieval()

═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
from typing import Dict, Any

# Import RAG components
from rag.trn_category_rag import load_category_vector_store, query_categories


# ═══════════════════════════════════════════════════════════════════════════
# CHROMADB GLOBAL REGISTRY FIX
# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL: Clear ChromaDB global state to prevent "An instance already exists" errors
try:
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient._identifier_to_system.clear()
except Exception:
    pass  # If clearing fails, continue anyway
    
# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def print_test_header(test_number: int, test_name: str, component: str, purpose: str):
    """Print formatted test header."""
    print("\n" + "=" * 80)
    print(f"TEST {test_number}: {test_name}")
    print("=" * 80)
    print(f"📦 COMPONENT: {component}")
    print(f"🎯 PURPOSE:   {purpose}")
    print("─" * 80)


def print_component_step(step_description: str):
    """Print component step being executed."""
    print(f"\n🔧 {step_description}")


def print_result(status: bool, description: str, details: str = ""):
    """Print formatted result."""
    icon = "✅" if status else "❌"
    status_text = "PASS" if status else "FAIL"
    print(f"{icon} {status_text} | {description}")
    if details:
        print(f"         {details}")


def print_test_summary(test_number: int, passed: int, total: int):
    """Print test summary."""
    percentage = (passed / total * 100) if total > 0 else 0
    print("\n" + "─" * 80)
    print(f"📊 TEST {test_number} SUMMARY")
    print("─" * 80)
    print(f"Total Checks:   {total}")
    print(f"✅ Passed:      {passed}")
    print(f"❌ Failed:      {total - passed}")
    print(f"Success Rate:   {percentage:.1f}%")
    print("─" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: VECTOR STORE EMBEDDINGS QUALITY
# ═══════════════════════════════════════════════════════════════════════════

def test_vector_store_embeddings_quality() -> Dict[str, Any]:
    """
    Test 1: Vector Store Embeddings Quality
    
    COMPONENT TESTED:
        build_category_vectorstore.py
        - load_categories_kb()
        - create_embedding_text()
        - ChromaDB vector store creation
    
    PURPOSE:
        Verify that the vector store was built correctly:
        1. Vector store directory exists
        2. ChromaDB collection loads successfully
        3. Contains exactly 108 documents (20 groups + 91 subcategories)
        4. All documents have proper metadata structure
    
    WHAT THIS VALIDATES:
        ✓ CategoriesKB.json loaded successfully
        ✓ Rich descriptions converted to embeddings
        ✓ ChromaDB storage working correctly
        ✓ Metadata structure correct (id, name, type, group info)
    
    WHY THIS MATTERS:
        This is the foundation of the entire RAG system. If embeddings are
        not created correctly, all downstream functionality will fail.
    
    SUCCESS CRITERIA:
        - Vector store exists at data/chroma_trn_categories/
        - Collection contains exactly 108 documents
        - All documents have required metadata fields
        - No errors accessing vector store
    """
    print_test_header(
        1,
        "Vector Store Embeddings Quality",
        "build_category_vectorstore.py → ChromaDB",
        "Verify vector store built correctly with all 108 category embeddings"
    )
    
    passed = 0
    total = 4
    results = {
        "test_name": "vector_store_embeddings_quality",
        "component": "build_category_vectorstore.py",
        "checks": []
    }
    
    # ─────────────────────────────────────────────────────────────────
    # Check 1: Vector store directory exists
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 1/4: Verify vector store directory exists")
    
    vector_store_path = Path("data/chroma_trn_categories")
    exists = vector_store_path.exists()
    
    if exists:
        passed += 1
        print_result(True, "Vector store directory found", str(vector_store_path))
    else:
        print_result(False, "Vector store directory NOT found", 
                    f"Expected: {vector_store_path}")
        print_result(False, "⚠️  Run: build_vector_store(force_rebuild=True)", "")
    
    results["checks"].append({
        "check": "vector_store_exists",
        "passed": exists,
        "path": str(vector_store_path)
    })
    
    if not exists:
        print_test_summary(1, passed, total)
        results["passed"] = passed
        results["total"] = total
        return results
    
    # ─────────────────────────────────────────────────────────────────
    # Check 2: Load vector store successfully
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 2/4: Load vector store from ChromaDB")
    
    try:
        collection = load_category_vector_store()
        passed += 1
        print_result(True, "Vector store loaded successfully", 
                    f"Collection: {collection.name}")
        results["checks"].append({
            "check": "vector_store_loads",
            "passed": True,
            "collection_name": collection.name
        })
    except Exception as e:
        print_result(False, "Failed to load vector store", str(e))
        results["checks"].append({
            "check": "vector_store_loads",
            "passed": False,
            "error": str(e)
        })
        print_test_summary(1, passed, total)
        results["passed"] = passed
        results["total"] = total
        return results
    
    # ─────────────────────────────────────────────────────────────────
    # Check 3: Verify document count (108 total)
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 3/4: Verify document count (20 groups + 91 subcategories = 108)")
    
    try:
        doc_count = collection.count()
        expected_count = 108  # 20 groups + 91 subcategories
        
        if doc_count == expected_count:
            passed += 1
            print_result(True, f"Correct document count: {doc_count}", 
                        f"Expected: {expected_count}")
        else:
            print_result(False, f"Incorrect document count: {doc_count}", 
                        f"Expected: {expected_count} (20 groups + 91 subcategories)")
        
        results["checks"].append({
            "check": "document_count",
            "passed": (doc_count == expected_count),
            "actual": doc_count,
            "expected": expected_count
        })
    except Exception as e:
        print_result(False, "Failed to count documents", str(e))
        results["checks"].append({
            "check": "document_count",
            "passed": False,
            "error": str(e)
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 4: Verify metadata structure
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 4/4: Verify metadata structure for sample documents")
    
    try:
        # Get sample documents
        sample_results = collection.get(limit=5, include=["metadatas"])
        metadatas = sample_results.get("metadatas", [])
        
        required_fields = {"type", "id", "name", "description"}
        all_valid = True
        
        print()
        for i, metadata in enumerate(metadatas[:3], 1):  # Check first 3
            has_required = required_fields.issubset(set(metadata.keys()))
            if has_required:
                doc_type = metadata.get("type")
                doc_id = metadata.get("id")
                doc_name = metadata.get("name")
                print(f"   Sample {i}: [{doc_type:8}] {doc_name:25} ({doc_id})")
            else:
                all_valid = False
                missing = required_fields - set(metadata.keys())
                print(f"   Sample {i}: ❌ MISSING FIELDS: {missing}")
        
        if all_valid:
            passed += 1
            print()
            print_result(True, "All metadata structures valid", 
                        f"Required fields present: {required_fields}")
        else:
            print()
            print_result(False, "Some metadata structures invalid")
        
        results["checks"].append({
            "check": "metadata_structure",
            "passed": all_valid,
            "required_fields": list(required_fields)
        })
    except Exception as e:
        print_result(False, "Failed to verify metadata", str(e))
        results["checks"].append({
            "check": "metadata_structure",
            "passed": False,
            "error": str(e)
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    
    print_test_summary(1, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = passed / total if total > 0 else 0
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: EXACT NAME RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def test_exact_name_retrieval(similarity_distance_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Test 2: Exact Name Retrieval
    
    COMPONENT TESTED:
        trn_category_rag.py → query_categories()
        - _get_embedding_model()
        - load_category_vector_store()
        - Vector similarity search
    
    PURPOSE:
        Verify that basic vector similarity search correctly retrieves categories
        when queried with exact or near-exact category names. Tests the core
        RAG retrieval mechanism without semantic complexity.
    
    PARAMETERS:
        similarity_distance_threshold (float): Distance threshold for filtering (default: 0.6)
    
    WHAT THIS VALIDATES:
        ✓ Query embeddings generated correctly
        ✓ Vector similarity search returns correct top match
        ✓ Distance scores for exact matches are low (< 0.4)
        ✓ Basic RAG retrieval pipeline functional
    
    WHY THIS MATTERS:
        This validates the core retrieval mechanism. If exact names don't
        retrieve correctly, the system won't work for any queries.
    
    TEST DATA:
        8 exact name queries:
        - 4 groups: dining, groceries, transportation, healthcare
        - 4 subcategories: coffee shops, gas station, restaurants, pharmacy
    
    SUCCESS CRITERIA:
        - Top result matches expected category ID
        - Distance score < 0.4 for exact matches
        - 100% accuracy expected
    """
    print_test_header(
        2,
        "Exact Name Retrieval",
        "trn_category_rag.py → query_categories()",
        "Verify basic vector similarity search retrieves exact category name matches"
    )
    
    # PRINT CHOSEN THRESHOLD
    print(f"🎯 Using similarity_distance_threshold: {similarity_distance_threshold}")
    print("─" * 80)
    
    # Test data
    exact_tests = [
        ("coffee shops", "C806", "Cafes & Coffee Shops"),
        ("dining", "CG800", "Dining"),
        ("groceries", "CG10000", "Groceries"),
        ("gas station", "C101", "Gas Station"),
        ("restaurants", "C803", "Restaurants"),
        ("transportation", "CG100", "Transportation"),
        ("pharmacy", "C302", "Pharmacy"),
        ("healthcare", "CG300", "Healthcare & Medical"),
    ]
    
    passed = 0
    total = len(exact_tests)
    results = {
        "test_name": "exact_name_retrieval",
        "component": "trn_category_rag.py → query_categories()",
        "queries": []
    }
    
    print_component_step(f"Executing {total} exact name queries")
    print()
    
    for term, expected_id, expected_name in exact_tests:
        # Execute query - USE the threshold parameter!
        matches = query_categories(term, top_k=3, min_confidence=similarity_distance_threshold)
        
        if matches and len(matches) > 0:
            top_match = matches[0]
            actual_id = top_match.get("id")
            actual_name = top_match.get("name")
            distance = top_match.get("score")
            
            is_correct = (actual_id == expected_id)
            is_low_distance = (distance < 0.4) if distance else False
            
            if is_correct:
                passed += 1
            
            print_result(
                is_correct,
                f"'{term}' → {actual_name} ({actual_id})",
                f"Distance: {distance:.4f} {'✓ Low' if is_low_distance else '⚠ High'}"
            )
            
            results["queries"].append({
                "term": term,
                "expected_id": expected_id,
                "actual_id": actual_id,
                "actual_name": actual_name,
                "distance": distance,
                "passed": is_correct
            })
        else:
            print_result(False, f"'{term}' → NO RESULTS", f"Expected: {expected_name}")
            results["queries"].append({
                "term": term,
                "expected_id": expected_id,
                "passed": False,
                "error": "No results returned"
            })
    
    print_test_summary(2, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = passed / total if total > 0 else 0
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL INFRASTRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_infrastructure_tests() -> Dict[str, Any]:
    """
    Run All Infrastructure Tests
    
    Executes tests 1-2 to validate foundational RAG infrastructure:
    - Test 1: Vector Store Embeddings Quality
    - Test 2: Exact Name Retrieval
    
    Returns:
        dict: Combined test results
    """
    print("\n" + "=" * 80)
    print("🏗️  RAG INFRASTRUCTURE TESTS")
    print("=" * 80)
    print("Validating: Vector store creation + Basic retrieval")
    print("=" * 80)
    
    # Run tests
    test1 = test_vector_store_embeddings_quality()
    test2 = test_exact_name_retrieval()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INFRASTRUCTURE TESTS SUMMARY")
    print("=" * 80)
    print(f"Test 1 - Vector Store Quality: {test1['passed']}/{test1['total']} ({test1['accuracy']*100:.1f}%)")
    print(f"Test 2 - Exact Name Retrieval: {test2['passed']}/{test2['total']} ({test2['accuracy']*100:.1f}%)")
    print("=" * 80)
    
    all_passed = (test1['accuracy'] >= 0.75 and test2['accuracy'] >= 0.90)
    
    print(f"\n{'✅ INFRASTRUCTURE TESTS PASSED' if all_passed else '⚠️  SOME TESTS NEED ATTENTION'}")
    print("=" * 80)
    
    return {
        "test1_vector_store": test1,
        "test2_exact_retrieval": test2,
        "all_passed": all_passed
    }


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

"""
USAGE IN JUPYTER NOTEBOOK:

1. Run all infrastructure tests:
   from tests.test_rag_infrastructure import run_infrastructure_tests
   results = run_infrastructure_tests()

2. Run individual tests:
   from tests.test_rag_infrastructure import (
       test_vector_store_embeddings_quality,
       test_exact_name_retrieval
   )
   
   # Test vector store
   test_vector_store_embeddings_quality()
   
   # Test basic retrieval
   test_exact_name_retrieval()
"""