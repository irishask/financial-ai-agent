"""
═══════════════════════════════════════════════════════════════════════════════
RAG Semantic Quality Test
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
    Validate the semantic understanding quality of the RAG system.
    This is THE CRITICAL TEST that validates whether rich conceptual descriptions
    enable proper semantic disambiguation between broad (domain-level) and 
    specific (transaction-level) intent.

TESTS INCLUDED:
    Test 3: Semantic Similarity (Broad vs Specific)
            Component: Embedding model + Rich descriptions
            Validates: Broad→Groups, Specific→Subcategories semantic mapping
            
            Part A: 10 BROAD terms → should match category GROUPS
            Part B: 10 SPECIFIC terms → should match SUBCATEGORIES

KEY INSIGHT:
    This test validates the CORE INNOVATION of using rich conceptual descriptions
    to enable semantic understanding rather than keyword matching:
    
    - "medicine" (broad/abstract) → Healthcare & Medical GROUP
    - "dentist" (specific/concrete) → Dental SUBCATEGORY
    
    Without proper descriptions, both would match the same way.

SUCCESS CRITERIA:
    - Part A (Broad → Groups): ≥70% accuracy
    - Part B (Specific → Subcategories): ≥90% accuracy  
    - Overall: ≥80% accuracy

USAGE:
    from tests.test_rag_semantic_quality import run_semantic_quality_test
    results = run_semantic_quality_test()
    
    # Or run directly
    from tests.test_rag_semantic_quality import test_semantic_similarity_broad_vs_specific
    test_semantic_similarity_broad_vs_specific()

═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Dict, Any

# Import RAG components
from rag.trn_category_rag import query_categories


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
# TEST 3: SEMANTIC SIMILARITY (BROAD VS SPECIFIC)
# ═══════════════════════════════════════════════════════════════════════════

def test_semantic_similarity_broad_vs_specific(similarity_distance_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Test 3: Semantic Similarity (Broad vs Specific)
    
    COMPONENT TESTED:
        multilingual-e5-base embedding model + Rich conceptual descriptions
        - Group descriptions (abstract, domain-level language)
        - Subcategory descriptions (concrete, transaction-level language)
    
    PURPOSE:
        Verify that the RAG system understands semantic INTENT and correctly
        distinguishes between:
        - BROAD queries (domain-level) → should match GROUPS
        - SPECIFIC queries (transaction-level) → should match SUBCATEGORIES
        
        This tests semantic understanding, NOT keyword matching.
    
    PARAMETERS:
        similarity_distance_threshold (float): Distance threshold for filtering (default: 0.6)
    
    WHAT THIS VALIDATES:
        ✓ Rich conceptual descriptions enable semantic disambiguation
        ✓ Embedding model captures intent differences (broad vs specific)
        ✓ Groups have stronger semantic signal for abstract terms
        ✓ Subcategories have stronger semantic signal for concrete terms
    
    WHY THIS MATTERS:
        This is THE CRITICAL TEST that proves rich descriptions work.
        
        Example of what we're testing:
        - "medicine" (broad) → Should match Healthcare GROUP (CG300)
        - "dentist" (specific) → Should match Dental SUBCATEGORY (C304)
        
        Without proper descriptions, both might match the same category.
        This test validates that semantic understanding disambiguates them.
    
    TEST DATA:
        Part A: 10 BROAD terms (e.g., "medicine", "food shopping", "eating out")
                → Should match category GROUPS (abstract, domain-level)
        
        Part B: 10 SPECIFIC terms (e.g., "dentist", "coffee shop", "fuel")
                → Should match SUBCATEGORIES (concrete, transaction-level)
    
    SUCCESS CRITERIA:
        - Part A (Broad → Groups): ≥70% accuracy
        - Part B (Specific → Subcategories): ≥90% accuracy
        - Overall: ≥80% accuracy
    """
    print_test_header(
        3,
        "Semantic Similarity (Broad vs Specific)",
        "Embedding Model + Rich Descriptions",
        "Verify semantic understanding distinguishes broad (domain) from specific (transaction) intent"
    )
    
    # PRINT CHOSEN THRESHOLD
    print(f"🎯 Using similarity_distance_threshold: {similarity_distance_threshold}")
    print("─" * 80)
    
    # ═════════════════════════════════════════════════════════════════
    # PART A: BROAD TERMS → CATEGORY GROUPS
    # ═════════════════════════════════════════════════════════════════
    
    broad_tests = [
        ("medicine", "CG300", "Healthcare & Medical"),
        ("food shopping", "CG10000", "Groceries"),
        ("eating out", "CG800", "Dining"),
        ("commute", "CG100", "Transportation"),
        ("home expenses", "CG1400", "Home & Garden"),
        ("kids activities", "CG1600", "Childcare & Kids"),
        ("working out", "CG1700", "Fitness & Sports"),
        ("protection coverage", "CG1000", "Insurance"),
        ("learning", "CG900", "Education"),
        ("charity work", "CG1200", "Charity & Donations"),
    ]
    
    print("\n" + "─" * 80)
    print("PART A: BROAD TERMS → CATEGORY GROUPS")
    print("─" * 80)
    print("Testing: Abstract, domain-level terms should match GROUP categories")
    print_component_step(f"Executing {len(broad_tests)} broad term queries")
    print()
    
    broad_passed = 0
    broad_total = len(broad_tests)
    broad_results = []
    
    for term, expected_id, expected_name in broad_tests:
        matches = query_categories(term, top_k=3, min_confidence=similarity_distance_threshold)
        
        if matches and len(matches) > 0:
            top_match = matches[0]
            actual_id = top_match.get("id")
            actual_name = top_match.get("name")
            actual_type = top_match.get("type")
            distance = top_match.get("score")
            
            # Check: correct ID AND correct type (group)
            is_correct = (actual_id == expected_id and actual_type == "group")
            
            if is_correct:
                broad_passed += 1
            
            type_marker = f"[{actual_type.upper():10}]"
            print_result(
                is_correct,
                f"'{term:20}' → {type_marker} {actual_name:30} ({actual_id})",
                f"Dist: {distance:.4f}"
            )
            
            broad_results.append({
                "term": term,
                "expected_id": expected_id,
                "expected_type": "group",
                "actual_id": actual_id,
                "actual_type": actual_type,
                "distance": distance,
                "passed": is_correct
            })
        else:
            print_result(False, f"'{term:20}' → NO RESULTS", "")
            broad_results.append({
                "term": term,
                "expected_id": expected_id,
                "passed": False
            })
    
    # ═════════════════════════════════════════════════════════════════
    # PART B: SPECIFIC TERMS → SUBCATEGORIES
    # ═════════════════════════════════════════════════════════════════
    
    specific_tests = [
        ("dentist", "C304", "Dental"),
        ("coffee shop", "C806", "Cafes & Coffee Shops"),
        ("fuel", "C101", "Gas Station"),
        ("drugstore", "C302", "Pharmacy"),
        ("barber", "C701", "Hair Salon & Barber"),
        ("streaming", "C403", "Streaming Services"),
        ("veterinarian", "C1301", "Veterinary"),
        ("gym membership", "C1701", "Gym & Fitness Center"),
        ("lawyer fees", "C1801", "Legal Services"),
        ("car repair", "C1902", "Auto Repairs"),
    ]
    
    print("\n" + "─" * 80)
    print("PART B: SPECIFIC TERMS → SUBCATEGORIES")
    print("─" * 80)
    print("Testing: Concrete, transaction-level terms should match SUBCATEGORY items")
    print_component_step(f"Executing {len(specific_tests)} specific term queries")
    print()
    
    specific_passed = 0
    specific_total = len(specific_tests)
    specific_results = []
    
    for term, expected_id, expected_name in specific_tests:
        matches = query_categories(term, top_k=3, min_confidence=similarity_distance_threshold)
        
        if matches and len(matches) > 0:
            top_match = matches[0]
            actual_id = top_match.get("id")
            actual_name = top_match.get("name")
            actual_type = top_match.get("type")
            distance = top_match.get("score")
            
            # Check: correct ID AND correct type (subcategory)
            is_correct = (actual_id == expected_id and actual_type == "subcategory")
            
            if is_correct:
                specific_passed += 1
            
            type_marker = f"[{actual_type.upper():10}]"
            print_result(
                is_correct,
                f"'{term:20}' → {type_marker} {actual_name:30} ({actual_id})",
                f"Dist: {distance:.4f}"
            )
            
            specific_results.append({
                "term": term,
                "expected_id": expected_id,
                "expected_type": "subcategory",
                "actual_id": actual_id,
                "actual_type": actual_type,
                "distance": distance,
                "passed": is_correct
            })
        else:
            print_result(False, f"'{term:20}' → NO RESULTS", "")
            specific_results.append({
                "term": term,
                "expected_id": expected_id,
                "passed": False
            })
    
    # ═════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY AND EVALUATION
    # ═════════════════════════════════════════════════════════════════
    
    total_passed = broad_passed + specific_passed
    total_tests = broad_total + specific_total
    
    broad_accuracy = broad_passed / broad_total if broad_total > 0 else 0
    specific_accuracy = specific_passed / specific_total if specific_total > 0 else 0
    overall_accuracy = total_passed / total_tests if total_tests > 0 else 0
    
    print("\n" + "═" * 80)
    print("📊 COMBINED RESULTS")
    print("═" * 80)
    print(f"Part A - Broad → Groups:           {broad_passed:2}/{broad_total:2} ({broad_accuracy*100:5.1f}%)")
    print(f"Part B - Specific → Subcategories:  {specific_passed:2}/{specific_total:2} ({specific_accuracy*100:5.1f}%)")
    print("─" * 80)
    print(f"Overall Semantic Accuracy:         {total_passed:2}/{total_tests:2} ({overall_accuracy*100:5.1f}%)")
    print("═" * 80)
    
    # Evaluate against success criteria
    broad_pass = broad_accuracy >= 0.70
    specific_pass = specific_accuracy >= 0.90
    overall_pass = overall_accuracy >= 0.80
    
    print(f"\n✅ SUCCESS CRITERIA EVALUATION:")
    print(f"   {'✅' if broad_pass else '❌'} Broad → Groups: ≥70%        (actual: {broad_accuracy*100:.1f}%)")
    print(f"   {'✅' if specific_pass else '❌'} Specific → Subcategories: ≥90% (actual: {specific_accuracy*100:.1f}%)")
    print(f"   {'✅' if overall_pass else '❌'} Overall Accuracy: ≥80%        (actual: {overall_accuracy*100:.1f}%)")
    
    all_criteria_met = broad_pass and specific_pass and overall_pass
    
    print("\n" + "─" * 80)
    if all_criteria_met:
        print("🎉 ALL SUCCESS CRITERIA MET!")
        print("   Semantic understanding is working correctly.")
    else:
        print("⚠️  SOME CRITERIA NOT MET")
        if not broad_pass:
            print("   → Broad terms not mapping well to groups (need better group descriptions)")
        if not specific_pass:
            print("   → Specific terms not mapping well to subcategories (need better subcat descriptions)")
    print("─" * 80)
    
    print_test_summary(3, total_passed, total_tests)
    
    return {
        "test_name": "semantic_similarity_broad_vs_specific",
        "component": "Embedding Model + Rich Descriptions",
        "broad_results": {
            "passed": broad_passed,
            "total": broad_total,
            "accuracy": broad_accuracy,
            "queries": broad_results
        },
        "specific_results": {
            "passed": specific_passed,
            "total": specific_total,
            "accuracy": specific_accuracy,
            "queries": specific_results
        },
        "combined": {
            "total_passed": total_passed,
            "total_tests": total_tests,
            "overall_accuracy": overall_accuracy
        },
        "criteria_met": {
            "broad": broad_pass,
            "specific": specific_pass,
            "overall": overall_pass,
            "all_passed": all_criteria_met
        }
    }



# ═══════════════════════════════════════════════════════════════════════════
# RUN SEMANTIC QUALITY TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_semantic_quality_test() -> Dict[str, Any]:
    """
    Run Semantic Quality Test
    
    Executes Test 3 to validate semantic understanding quality.
    This is the critical test that validates whether rich conceptual
    descriptions enable proper broad vs specific disambiguation.
    
    Returns:
        dict: Test results with accuracy metrics
    """
    print("\n" + "=" * 80)
    print("🧠 RAG SEMANTIC QUALITY TEST")
    print("=" * 80)
    print("Validating: Broad→Groups, Specific→Subcategories semantic mapping")
    print("=" * 80)
    
    # Run test
    results = test_semantic_similarity_broad_vs_specific()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SEMANTIC QUALITY TEST SUMMARY")
    print("=" * 80)
    
    broad_acc = results["broad_results"]["accuracy"]
    specific_acc = results["specific_results"]["accuracy"]
    overall_acc = results["combined"]["overall_accuracy"]
    
    print(f"Broad → Groups:             {broad_acc*100:.1f}% (≥70% required)")
    print(f"Specific → Subcategories:   {specific_acc*100:.1f}% (≥90% required)")
    print(f"Overall Accuracy:           {overall_acc*100:.1f}% (≥80% required)")
    print("=" * 80)
    
    all_passed = results["criteria_met"]["all_passed"]
    
    if all_passed:
        print("\n✅ SEMANTIC QUALITY TEST PASSED")
        print("   Rich descriptions enable proper semantic disambiguation!")
    else:
        print("\n⚠️  SEMANTIC QUALITY TEST NEEDS ATTENTION")
        print("   Some accuracy criteria not met. Check descriptions.")
    
    print("=" * 80)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

"""
USAGE IN JUPYTER NOTEBOOK:

1. Run semantic quality test:
   from tests.test_rag_semantic_quality import run_semantic_quality_test
   results = run_semantic_quality_test()

2. Run test directly:
   from tests.test_rag_semantic_quality import test_semantic_similarity_broad_vs_specific
   test_semantic_similarity_broad_vs_specific()

3. Check specific results:
   results = run_semantic_quality_test()
   print(f"Broad accuracy: {results['broad_results']['accuracy']*100:.1f}%")
   print(f"Specific accuracy: {results['specific_results']['accuracy']*100:.1f}%")
   
4. Analyze failures:
   for query in results['broad_results']['queries']:
       if not query['passed']:
           print(f"Failed: {query['term']} → got {query.get('actual_id')} instead of {query['expected_id']}")
"""