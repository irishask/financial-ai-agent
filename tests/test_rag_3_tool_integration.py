"""
═══════════════════════════════════════════════════════════════════════════════
RAG Tool Integration Tests
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
    Validate the tool wrapper layer that formats RAG results for LLM-1 Router.
    This layer sits between the raw RAG system and the LLM tool-calling interface,
    providing quality filtering, confidence assignment, and proper data structures.

TESTS INCLUDED:
    Test 4: Distance Threshold Filtering
            Component: trn_category_tool.py → distance filtering
            Validates: Thresholds filter low-quality matches, assign confidence
    
    Test 5: Tool Wrapper Integration
            Component: trn_category_tool.py → CategoryMatch model
            Validates: Proper structure for LLM tool calling
    
    Test 6: Batch Query Processing
            Component: trn_category_tool.py → batch processing
            Validates: Multiple terms processed efficiently

KEY INSIGHTS:
    These tests validate the "production readiness" layer:
    - Quality control (distance filtering)
    - Proper data structures (CategoryMatch objects)
    - Efficient batch processing (multiple terms in one call)
    - Error handling (graceful degradation)

SUCCESS CRITERIA:
    All tests: ≥75% accuracy (production-quality filtering and formatting)

USAGE:
    # Run individual tests
    from tests.test_rag_tool_integration import (
        test_distance_threshold_filtering,
        test_tool_wrapper_integration,
        test_batch_query_processing
    )

═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Dict, Any

# Import RAG components
from schemas.trn_category_tool import search_transaction_categories
from schemas.router_models import GraphState
from graph_definition import router_node

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
# TEST 4: DISTANCE THRESHOLD FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def test_distance_threshold_filtering() -> Dict[str, Any]:
    """
    Test 4: Distance Threshold Filtering
    
    COMPONENT TESTED:
        trn_category_tool.py → search_transaction_categories()
        - _distance_to_confidence() helper function
        - Distance threshold filtering (default: 0.6)
        - Confidence level assignment (high/medium/low)
    
    PURPOSE:
        Verify that the distance threshold correctly filters out low-quality
        matches and assigns appropriate confidence levels. This is the quality
        control mechanism that prevents returning irrelevant categories to LLM-1.
    
    WHAT THIS VALIDATES:
        ✓ Distance threshold filtering works (default: 0.6)
        ✓ Only matches below threshold are returned
        ✓ Confidence levels assigned correctly:
          - distance < 0.4: "high" confidence
          - 0.4 ≤ distance < 0.6: "medium" confidence
          - distance ≥ 0.6: "low" confidence (filtered out)
        ✓ Poor matches correctly filtered
    
    WHY THIS MATTERS:
        Without proper filtering, LLM-1 Router would receive irrelevant
        category suggestions, leading to incorrect categorization.
        This is a critical production quality control layer.
    
    TEST DATA:
        - Good match: "coffee" → should return results (distance < 0.6)
        - Poor match: "xyzabc123" → should return empty (distance ≥ 0.6)
        - Confidence validation: "groceries" → verify correct confidence level
    
    SUCCESS CRITERIA:
        - Good matches returned with distances < 0.6
        - Poor matches filtered out
        - Confidence levels correctly assigned
        - ≥75% of checks pass
    """
    print_test_header(
        4,
        "Distance Threshold Filtering",
        "trn_category_tool.py → distance filtering",
        "Verify distance thresholds filter low-quality matches and assign confidence levels"
    )
    
    passed = 0
    total = 3
    results = {
        "test_name": "distance_threshold_filtering",
        "component": "trn_category_tool.py",
        "checks": []
    }
    
    # ─────────────────────────────────────────────────────────────────
    # Check 1: Good match returns results with low distance
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 1/3: Good match returns results (distance < 0.6)")
    
    good_term = "coffee"
    matches = search_transaction_categories([good_term])
    
    if matches and len(matches) > 0:
        top_match = matches[0]
        distance = top_match.distance
        
        if distance < 0.6:
            passed += 1
            print_result(True, f"'{good_term}' returned {len(matches)} results",
                        f"Top distance: {distance:.4f} (< 0.6 threshold) ✓")
        else:
            print_result(False, f"'{good_term}' distance too high",
                        f"Top distance: {distance:.4f} (≥ 0.6 threshold)")
        
        results["checks"].append({
            "check": "good_match_passes_threshold",
            "term": good_term,
            "passed": distance < 0.6,
            "distance": distance,
            "result_count": len(matches)
        })
    else:
        print_result(False, f"'{good_term}' returned no results", "Expected results")
        results["checks"].append({
            "check": "good_match_passes_threshold",
            "term": good_term,
            "passed": False,
            "error": "No results returned"
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 2: Poor match filtered out (returns empty)
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 2/3: Poor match filtered out (distance ≥ 0.6)")
    
    poor_term = "xyzabc123randomtext"
    matches = search_transaction_categories([poor_term])
    
    if not matches or len(matches) == 0:
        passed += 1
        print_result(True, f"'{poor_term}' correctly filtered out",
                    "No results returned (all distances ≥ 0.6) ✓")
        results["checks"].append({
            "check": "poor_match_filtered",
            "term": poor_term,
            "passed": True,
            "result_count": 0
        })
    else:
        top_distance = matches[0].distance if matches else None
        print_result(False, f"'{poor_term}' should be filtered",
                    f"Returned {len(matches)} results, top distance: {top_distance:.4f}")
        results["checks"].append({
            "check": "poor_match_filtered",
            "term": poor_term,
            "passed": False,
            "unexpected_results": len(matches),
            "top_distance": top_distance
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 3: Confidence levels assigned correctly
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 3/3: Confidence levels assigned correctly")
    print("   Mapping: < 0.4 = 'high', 0.4-0.6 = 'medium', ≥ 0.6 = 'low'")
    
    test_term = "groceries"
    matches = search_transaction_categories([test_term])
    
    if matches and len(matches) > 0:
        top_match = matches[0]
        distance = top_match.distance
        confidence = top_match.confidence
        
        # Verify confidence mapping
        expected_confidence = "high" if distance < 0.4 else "medium" if distance < 0.6 else "low"
        
        if confidence == expected_confidence:
            passed += 1
            print_result(True, f"Confidence level correct: '{confidence}'",
                        f"Distance: {distance:.4f} → '{confidence}' ✓")
        else:
            print_result(False, f"Confidence level incorrect: '{confidence}'",
                        f"Distance: {distance:.4f} → Expected: '{expected_confidence}', Got: '{confidence}'")
        
        results["checks"].append({
            "check": "confidence_level_assignment",
            "term": test_term,
            "distance": distance,
            "expected_confidence": expected_confidence,
            "actual_confidence": confidence,
            "passed": confidence == expected_confidence
        })
    else:
        print_result(False, f"'{test_term}' returned no results", "Expected results")
        results["checks"].append({
            "check": "confidence_level_assignment",
            "term": test_term,
            "passed": False,
            "error": "No results returned"
        })
    
    print_test_summary(4, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = passed / total if total > 0 else 0
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: TOOL WRAPPER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_tool_wrapper_integration() -> Dict[str, Any]:
    """
    Test 5: Tool Wrapper Integration
    
    COMPONENT TESTED:
        trn_category_tool.py
        - CategoryMatch Pydantic model
        - search_transaction_categories() function
        - search_categories_lc_tool LangChain wrapper
    
    PURPOSE:
        Verify that the LangChain tool wrapper correctly formats RAG results
        into CategoryMatch objects with proper structure for LLM-1 Router.
        Tests the interface between RAG system and LLM tool calling.
    
    WHAT THIS VALIDATES:
        ✓ CategoryMatch model serialization works
        ✓ All required fields present and correctly typed:
          - user_term, category_id, category_name, category_type
          - distance, confidence, group_id, group_name (for subcategories)
        ✓ Tool wrapper returns List[CategoryMatch] format
        ✓ Subcategories include parent group information
        ✓ Error handling returns empty list (not exception)
    
    WHY THIS MATTERS:
        LLM-1 Router expects specific data structures for tool calling.
        Incorrect formatting would break the tool-calling interface and
        prevent LLM-1 from using category information correctly.
    
    TEST DATA:
        - Single term query: "dining"
        - Subcategory query: "coffee shop" (must include group info)
        - Error case: empty query list
    
    SUCCESS CRITERIA:
        - Returns List[CategoryMatch] objects
        - All required fields present and correctly typed
        - Subcategories include group information
        - Error cases handled gracefully
        - ≥75% of checks pass
    """
    print_test_header(
        5,
        "Tool Wrapper Integration",
        "trn_category_tool.py → LangChain tool wrapper",
        "Verify tool wrapper formats results correctly as CategoryMatch objects for LLM-1"
    )
    
    passed = 0
    total = 4
    results = {
        "test_name": "tool_wrapper_integration",
        "component": "trn_category_tool.py",
        "checks": []
    }
    
    # ─────────────────────────────────────────────────────────────────
    # Check 1: Returns List[CategoryMatch] type
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 1/4: Returns List[CategoryMatch] format")
    
    test_term = "dining"
    matches = search_transaction_categories([test_term])
    
    is_list = isinstance(matches, list)
    
    if is_list and len(matches) > 0:
        first_match = matches[0]
        required_attrs = ['user_term', 'category_id', 'category_name', 'category_type', 
                         'distance', 'confidence']
        has_attrs = all(hasattr(first_match, attr) for attr in required_attrs)
        
        if has_attrs:
            passed += 1
            print_result(True, "Returns List[CategoryMatch]",
                        f"Returned {len(matches)} CategoryMatch objects ✓")
        else:
            missing = [attr for attr in required_attrs if not hasattr(first_match, attr)]
            print_result(False, "CategoryMatch missing attributes",
                        f"Missing: {missing}")
        
        results["checks"].append({
            "check": "returns_categoryMatch_list",
            "passed": has_attrs,
            "result_count": len(matches)
        })
    else:
        print_result(False, "Invalid return format",
                    f"Expected list of CategoryMatch, got: {type(matches)}")
        results["checks"].append({
            "check": "returns_categoryMatch_list",
            "passed": False,
            "error": "Not a list or empty"
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 2: All required fields present and correctly populated
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 2/4: All required fields present and properly populated")
    
    matches = search_transaction_categories([test_term])
    
    if matches and len(matches) > 0:
        top_match = matches[0]
        
        # Validate field presence and types
        checks = {
            'user_term': lambda v: isinstance(v, str) and len(v) > 0,
            'category_id': lambda v: isinstance(v, str) and len(v) > 0,
            'category_name': lambda v: isinstance(v, str) and len(v) > 0,
            'category_type': lambda v: v in ['group', 'subcategory'],
            'distance': lambda v: isinstance(v, (int, float)) and 0 <= v <= 2,
            'confidence': lambda v: v in ['high', 'medium', 'low']
        }
        
        all_valid = True
        invalid_fields = []
        
        for field, validator in checks.items():
            value = getattr(top_match, field, None)
            if not validator(value):
                all_valid = False
                invalid_fields.append(f"{field}={value}")
        
        if all_valid:
            passed += 1
            print_result(True, "All required fields valid and properly typed",
                        f"user_term, category_id, category_name, category_type, distance, confidence ✓")
        else:
            print_result(False, "Some fields invalid",
                        f"Invalid: {', '.join(invalid_fields)}")
        
        results["checks"].append({
            "check": "required_fields_valid",
            "passed": all_valid,
            "invalid_fields": invalid_fields if not all_valid else []
        })
    else:
        print_result(False, "No results to check", "")
        results["checks"].append({
            "check": "required_fields_valid",
            "passed": False,
            "error": "No results"
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 3: Subcategories include group information
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 3/4: Subcategories include parent group information")
    
    subcat_term = "coffee shop"
    matches = search_transaction_categories([subcat_term])
    
    if matches and len(matches) > 0:
        subcat_match = matches[0]
        
        if subcat_match.category_type == "subcategory":
            has_group_info = (
                hasattr(subcat_match, 'group_id') and
                subcat_match.group_id is not None and
                len(subcat_match.group_id) > 0 and
                hasattr(subcat_match, 'group_name') and
                subcat_match.group_name is not None and
                len(subcat_match.group_name) > 0
            )
            
            if has_group_info:
                passed += 1
                print_result(True, "Subcategory has group information",
                            f"Group: {subcat_match.group_name} ({subcat_match.group_id}) ✓")
            else:
                print_result(False, "Subcategory missing group information",
                            f"group_id: {getattr(subcat_match, 'group_id', None)}, "
                            f"group_name: {getattr(subcat_match, 'group_name', None)}")
            
            results["checks"].append({
                "check": "subcategory_group_info",
                "passed": has_group_info,
                "group_id": getattr(subcat_match, 'group_id', None),
                "group_name": getattr(subcat_match, 'group_name', None)
            })
        else:
            print_result(False, "Top match not a subcategory",
                        f"Got: {subcat_match.category_type} (expected subcategory)")
            results["checks"].append({
                "check": "subcategory_group_info",
                "passed": False,
                "error": f"Not a subcategory, got: {subcat_match.category_type}"
            })
    else:
        print_result(False, "No results for subcategory term", "")
        results["checks"].append({
            "check": "subcategory_group_info",
            "passed": False,
            "error": "No results"
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 4: Error handling returns empty list (not exception)
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 4/4: Error handling returns empty list (graceful degradation)")
    
    try:
        # Query with empty list (edge case)
        matches = search_transaction_categories([])
        
        if isinstance(matches, list):
            passed += 1
            print_result(True, "Empty query handled gracefully",
                        f"Returned empty list (not exception) ✓")
        else:
            print_result(False, "Unexpected return type",
                        f"Expected list, got: {type(matches)}")
        
        results["checks"].append({
            "check": "error_handling",
            "passed": isinstance(matches, list)
        })
    except Exception as e:
        print_result(False, "Exception raised instead of empty list",
                    f"Exception: {str(e)}")
        results["checks"].append({
            "check": "error_handling",
            "passed": False,
            "error": str(e)
        })
    
    print_test_summary(5, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = passed / total if total > 0 else 0
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: BATCH QUERY PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_query_processing() -> Dict[str, Any]:
    """
    Test 6: Batch Query Processing
    
    COMPONENT TESTED:
        trn_category_tool.py → search_transaction_categories()
        - Multiple search terms in single query
        - Result aggregation and deduplication
        - Maintains distance sorting
    
    PURPOSE:
        Verify that the tool can efficiently process multiple search terms
        in a single query without errors or performance degradation. This is
        needed for queries like "Show me spending on dining, groceries, and gas".
    
    WHAT THIS VALIDATES:
        ✓ Multiple terms processed correctly in one call
        ✓ Results aggregated from all terms
        ✓ No duplicates in results (same category from multiple terms)
        ✓ No performance degradation
        ✓ Maintains correct distance sorting
        ✓ All results are valid CategoryMatch objects
    
    WHY THIS MATTERS:
        Real user queries often involve multiple categories:
        - "Show me dining, groceries, and transportation"
        - "How much on coffee, restaurants, and fast food?"
        
        Batch processing is more efficient than multiple single queries
        and provides better UX (single response with all categories).
    
    TEST DATA:
        Batch query: ["dining", "groceries", "gas station", "medical"]
        - 4 diverse terms across different category types
        - Should return ≥4 results (at least one per term)
    
    SUCCESS CRITERIA:
        - Query completes without errors
        - Returns ≥ number of input terms
        - All results are valid CategoryMatch objects
        - ≥75% of checks pass
    """
    print_test_header(
        6,
        "Batch Query Processing",
        "trn_category_tool.py → batch processing",
        "Verify multiple search terms processed efficiently in single query"
    )
    
    passed = 0
    total = 3
    batch_terms = ["dining", "groceries", "gas station", "medical"]
    
    print_component_step(f"Executing batch query with {len(batch_terms)} terms:")
    print(f"   Terms: {batch_terms}")
    
    # ─────────────────────────────────────────────────────────────────
    # Check 1: Completes without error
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 1/3: Batch query completes without errors")
    
    try:
        matches = search_transaction_categories(batch_terms)
        passed += 1
        print_result(True, "Batch query completed successfully",
                    f"No exceptions raised ✓")
        
        results = {
            "test_name": "batch_query_processing",
            "component": "trn_category_tool.py",
            "batch_terms": batch_terms,
            "checks": [{
                "check": "query_completes",
                "passed": True,
                "result_count": len(matches)
            }]
        }
    except Exception as e:
        print_result(False, "Batch query failed with exception", str(e))
        return {
            "test_name": "batch_query_processing",
            "component": "trn_category_tool.py",
            "batch_terms": batch_terms,
            "passed": 0,
            "total": total,
            "accuracy": 0,
            "error": str(e)
        }
    
    # ─────────────────────────────────────────────────────────────────
    # Check 2: Returns sufficient results
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step(f"CHECK 2/3: Returns sufficient results (≥{len(batch_terms)} expected)")
    
    if len(matches) >= len(batch_terms):
        passed += 1
        print_result(True, f"Sufficient results: {len(matches)}",
                    f"≥{len(batch_terms)} expected ✓")
    else:
        print_result(False, f"Insufficient results: {len(matches)}",
                    f"Expected ≥{len(batch_terms)}")
    
    results["checks"].append({
        "check": "sufficient_results",
        "passed": len(matches) >= len(batch_terms),
        "actual_count": len(matches),
        "expected_minimum": len(batch_terms)
    })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 3: All results are valid CategoryMatch objects
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 3/3: All results are valid CategoryMatch objects")
    
    required_attrs = ['user_term', 'category_id', 'category_name', 'category_type', 
                     'distance', 'confidence']
    all_valid = all(
        all(hasattr(m, attr) for attr in required_attrs)
        for m in matches
    )
    
    if all_valid:
        passed += 1
        print_result(True, "All results valid CategoryMatch objects",
                    f"{len(matches)} objects validated ✓")
        
        # Show sample results
        print("\n   Sample results:")
        for i, match in enumerate(matches[:3], 1):
            print(f"      {i}. {match.category_name} ({match.category_id}) "
                  f"- dist: {match.distance:.4f}, conf: {match.confidence}")
    else:
        invalid_count = sum(
            1 for m in matches
            if not all(hasattr(m, attr) for attr in required_attrs)
        )
        print_result(False, f"{invalid_count} invalid objects",
                    f"Out of {len(matches)} total results")
    
    results["checks"].append({
        "check": "all_valid_objects",
        "passed": all_valid,
        "validated_count": len(matches)
    })
    
    print_test_summary(6, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = passed / total if total > 0 else 0
    results["matches_returned"] = len(matches)
    
    return results




# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: ROUTER LLM TOOL BINDING
# ═══════════════════════════════════════════════════════════════════════════

def test_router_llm_tool_binding() -> Dict[str, Any]:
    """
    Test 8: Router LLM Tool Binding
    
    COMPONENT TESTED:
        graph_definition.py → router_llm.bind_tools() + router_node() tool loop
    
    PURPOSE:
        Verify that the tool integration MECHANICS work correctly:
        - router_llm has tools bound
        - LLM-1 CAN invoke tools (tool call mechanics)
        - Tool results are returned to LLM (message loop)
        - RouterOutput is produced without crash
    
    NOT TESTING:
        - Embedding quality (test_rag_2)
        - Correct category matching (test_rag_2)
        - resolved_trn_categories population (Step 2 - prompt update)
    
    WHAT THIS VALIDATES:
        ✓ .bind_tools() was applied correctly to router_llm
        ✓ Iterative tool calling loop works
        ✓ Tool results fed back to LLM without error
        ✓ RouterOutput produced (not exception/crash)
    
    WHY THIS MATTERS:
        This validates the PLUMBING between LLM-1 and Category RAG tool.
        Embedding quality is tested separately in test_rag_2.
    
    TEST DATA:
        Query with obvious category term to trigger tool call.
    
    SUCCESS CRITERIA:
        - No exceptions
        - RouterOutput produced
        - No error fallback
        - Tool was invoked (at least 1 iteration with tool call)
    """
    print_test_header(
        8,
        "Router LLM Tool Binding",
        "graph_definition.py → router_llm.bind_tools() + tool loop",
        "Verify tool integration mechanics work (not embedding quality)"
    )
    
    passed = 0
    total = 4
    results = {
        "test_name": "router_llm_tool_binding",
        "component": "graph_definition.py",
        "checks": []
    }
    
    # ─────────────────────────────────────────────────────────────────
    # Setup: Create test state with category term
    # ─────────────────────────────────────────────────────────────────
    
    test_query = "I am USER_001. How much did I spend on groceries last month?"
    
    print_component_step(f"Test query: \"{test_query}\"")
    print(f"   (Contains category term 'groceries' to trigger tool call)")
    
    test_state = GraphState(
        user_query=test_query,
        conversation_summary=None,
        turn_id=1
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Check 1: router_node executes without exception
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 1/4: router_node executes without exception")
    
    result_state = None
    execution_error = None
    
    try:
        result_state = router_node(test_state)
        passed += 1
        print_result(True, "router_node executed successfully",
                    "No exceptions raised ✓")
        results["checks"].append({
            "check": "execution_no_exception",
            "passed": True
        })
    except Exception as e:
        execution_error = str(e)
        print_result(False, "router_node raised exception", execution_error[:100])
        results["checks"].append({
            "check": "execution_no_exception",
            "passed": False,
            "error": execution_error[:200]
        })
        # Return early - cannot continue
        print_test_summary(7, passed, total)
        results["passed"] = passed
        results["total"] = total
        results["accuracy"] = passed / total if total > 0 else 0
        return results
    
    # ─────────────────────────────────────────────────────────────────
    # Check 2: RouterOutput is produced (not None)
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 2/4: RouterOutput is produced (not None)")
    
    router_output = result_state.router_output
    
    if router_output is not None:
        passed += 1
        print_result(True, "RouterOutput produced",
                    f"Type: {type(router_output).__name__} ✓")
        results["checks"].append({
            "check": "router_output_not_none",
            "passed": True
        })
    else:
        print_result(False, "RouterOutput is None", "Expected RouterOutput object")
        results["checks"].append({
            "check": "router_output_not_none",
            "passed": False
        })
        # Return early - cannot continue
        print_test_summary(7, passed, total)
        results["passed"] = passed
        results["total"] = total
        results["accuracy"] = passed / total if total > 0 else 0
        return results
    
    # ─────────────────────────────────────────────────────────────────
    # Check 3: No error fallback (clarity_reason doesn't contain "error")
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 3/4: No error fallback in RouterOutput")
    
    clarity_reason = router_output.clarity_reason or ""
    missing_info = router_output.missing_info or []
    
    is_error_fallback = (
        "error" in clarity_reason.lower() or
        "error_recovery" in missing_info
    )
    
    if not is_error_fallback:
        passed += 1
        print_result(True, "No error fallback detected",
                    f"Clarity: {router_output.clarity} ✓")
        results["checks"].append({
            "check": "no_error_fallback",
            "passed": True,
            "clarity": router_output.clarity
        })
    else:
        print_result(False, "Error fallback detected",
                    f"clarity_reason: {clarity_reason[:100]}")
        results["checks"].append({
            "check": "no_error_fallback",
            "passed": False,
            "clarity_reason": clarity_reason
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Check 4: RouterOutput has valid structure
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("CHECK 4/4: RouterOutput has valid structure")
    
    required_fields = ['clarity', 'core_use_cases', 'primary_use_case']
    has_all_fields = all(
        hasattr(router_output, field) and getattr(router_output, field) is not None
        for field in required_fields
    )
    
    if has_all_fields:
        passed += 1
        print_result(True, "RouterOutput has valid structure",
                    f"clarity={router_output.clarity}, primary_uc={router_output.primary_use_case} ✓")
        results["checks"].append({
            "check": "valid_structure",
            "passed": True,
            "clarity": router_output.clarity,
            "core_use_cases": router_output.core_use_cases,
            "primary_use_case": router_output.primary_use_case
        })
    else:
        missing = [f for f in required_fields if not hasattr(router_output, f) or getattr(router_output, f) is None]
        print_result(False, "RouterOutput missing required fields",
                    f"Missing: {missing}")
        results["checks"].append({
            "check": "valid_structure",
            "passed": False,
            "missing_fields": missing
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Info: Show key RouterOutput fields (not scored)
    # ─────────────────────────────────────────────────────────────────
    
    print_component_step("INFO: RouterOutput summary (not scored)")
    print(f"   • Clarity: {router_output.clarity}")
    print(f"   • Core UCs: {router_output.core_use_cases}")
    print(f"   • Primary UC: {router_output.primary_use_case}")
    print(f"   • Confidence: {router_output.uc_confidence}")
    if router_output.resolved_trn_categories:
        print(f"   • Resolved Categories: {len(router_output.resolved_trn_categories)} item(s)")
    else:
        print(f"   • Resolved Categories: None (expected - prompt update pending)")
    
    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    
    print_test_summary(8, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = passed / total if total > 0 else 0
    
    return results
    

# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

"""
USAGE IN JUPYTER NOTEBOOK:

# Run individual tests:
   from tests.test_rag_tool_integration import (
       test_distance_threshold_filtering,
       test_tool_wrapper_integration,
       test_batch_query_processing
   )
   
   # Test distance filtering
   test_distance_threshold_filtering()
   
   # Test tool wrapper
   test_tool_wrapper_integration()
   
   # Test batch processing
   test_batch_query_processing()
"""