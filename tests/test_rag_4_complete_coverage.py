"""
═══════════════════════════════════════════════════════════════════════════════
RAG Complete Coverage Test
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
    Validate that ALL 60 categories (20 groups + 40 subcategories) are 
    retrievable through the complete RAG pipeline end-to-end. This is the
    final comprehensive test that ensures no categories are missing or broken.

TESTS INCLUDED:
    Test 7: Complete Category Coverage (All 60 Categories)
            Component: Complete RAG pipeline (end-to-end)
            Validates: All categories retrievable with semantic queries
            
            Coverage:
            - 20 category groups (CG100 - CG10000)
            - 40 subcategories (2 per group)

KEY INSIGHTS:
    This test validates the COMPLETENESS of the RAG system:
    - No missing embeddings
    - No broken category mappings
    - Consistent performance across all category types
    - End-to-end pipeline functional for all categories

SUCCESS CRITERIA:
    - Expected category found in top 3 results
    - ≥90% coverage rate (54/60 categories retrievable)
    - Consistent retrieval quality across all categories

USAGE:
    from tests.test_rag_complete_coverage import run_complete_coverage_test
    results = run_complete_coverage_test()
    
    # Or run directly
    from tests.test_rag_complete_coverage import test_complete_category_coverage
    test_complete_category_coverage()

═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Dict, Any, List

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
    print(f"Total Categories:   {total}")
    print(f"✅ Retrievable:     {passed}")
    print(f"❌ Not Found:       {total - passed}")
    print(f"Coverage Rate:      {percentage:.1f}%")
    print("─" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# TEST DATA: ALL 60 CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════

def get_all_category_tests() -> List[tuple]:
    """
    Return all 60 category test cases.
    
    Format: (query_term, expected_id, expected_name, category_type)
    
    Coverage:
    - 20 category groups (CG100 - CG10000)
    - 40 subcategories (2 per group)
    """
    return [
        # CG100: Transportation (+ 2 subcategories)
        ("transportation", "CG100", "Transportation", "group"),
        ("gas station", "C101", "Gas Station", "subcategory"),
        ("parking", "C102", "Parking", "subcategory"),
        
        # CG200: Utilities (+ 2 subcategories)
        ("utilities", "CG200", "Utilities", "group"),
        ("electricity", "C201", "Electricity", "subcategory"),
        ("water bill", "C202", "Water", "subcategory"),
        
        # CG300: Healthcare & Medical (+ 2 subcategories)
        ("healthcare", "CG300", "Healthcare & Medical", "group"),
        ("doctor", "C301", "Doctor & Physician", "subcategory"),
        ("pharmacy", "C302", "Pharmacy", "subcategory"),
        
        # CG400: Entertainment (+ 2 subcategories)
        ("entertainment", "CG400", "Entertainment", "group"),
        ("movies", "C401", "Movies & Cinema", "subcategory"),
        ("concerts", "C402", "Concerts & Events", "subcategory"),
        
        # CG500: Travel (+ 2 subcategories)
        ("travel", "CG500", "Travel", "group"),
        ("airlines", "C501", "Airlines", "subcategory"),
        ("hotels", "C502", "Hotels & Lodging", "subcategory"),
        
        # CG600: Shopping (+ 2 subcategories)
        ("shopping", "CG600", "Shopping", "group"),
        ("department stores", "C601", "Department Stores", "subcategory"),
        ("online shopping", "C602", "Online Shopping", "subcategory"),
        
        # CG700: Personal Care (+ 2 subcategories)
        ("personal care", "CG700", "Personal Care", "group"),
        ("hair salon", "C701", "Hair Salon & Barber", "subcategory"),
        ("spa", "C702", "Spa & Massage", "subcategory"),
        
        # CG800: Dining (+ 2 subcategories)
        ("dining", "CG800", "Dining", "group"),
        ("restaurants", "C803", "Restaurants", "subcategory"),
        ("fast food", "C802", "Fast Food", "subcategory"),
        
        # CG900: Education (+ 2 subcategories)
        ("education", "CG900", "Education", "group"),
        ("tuition", "C901", "Tuition & School Fees", "subcategory"),
        ("books", "C902", "Books & Supplies", "subcategory"),
        
        # CG1000: Insurance (+ 2 subcategories)
        ("insurance", "CG1000", "Insurance", "group"),
        ("health insurance", "C1001", "Health Insurance", "subcategory"),
        ("car insurance", "C1003", "Auto Insurance", "subcategory"),
        
        # CG1100: Taxes (+ 2 subcategories)
        ("taxes", "CG1100", "Taxes", "group"),
        ("federal taxes", "C1101", "Federal Taxes", "subcategory"),
        ("state taxes", "C1102", "State Taxes", "subcategory"),
        
        # CG1200: Charity & Donations (+ 2 subcategories)
        ("charity", "CG1200", "Charity & Donations", "group"),
        ("charitable donations", "C1201", "Charitable Donations", "subcategory"),
        ("religious organizations", "C1202", "Religious Organizations", "subcategory"),
        
        # CG1300: Pets (+ 2 subcategories)
        ("pets", "CG1300", "Pets", "group"),
        ("veterinary", "C1301", "Veterinary", "subcategory"),
        ("pet supplies", "C1302", "Pet Supplies", "subcategory"),
        
        # CG1400: Home & Garden (+ 2 subcategories)
        ("home", "CG1400", "Home & Garden", "group"),
        ("furniture", "C1401", "Furniture & Decor", "subcategory"),
        ("home improvement", "C1402", "Home Improvement", "subcategory"),
        
        # CG1500: Financial Services (+ 2 subcategories)
        ("financial services", "CG1500", "Financial Services", "group"),
        ("bank fees", "C1501", "Bank Fees", "subcategory"),
        ("atm fees", "C1502", "ATM Fees", "subcategory"),
        
        # CG1600: Childcare & Kids (+ 2 subcategories)
        ("childcare", "CG1600", "Childcare & Kids", "group"),
        ("daycare", "C1601", "Daycare & Babysitting", "subcategory"),
        ("toys", "C1602", "Toys & Games", "subcategory"),
        
        # CG1700: Fitness & Sports (+ 2 subcategories)
        ("fitness", "CG1700", "Fitness & Sports", "group"),
        ("gym", "C1701", "Gym & Fitness Center", "subcategory"),
        ("sports equipment", "C1702", "Sports Equipment", "subcategory"),
        
        # CG1800: Professional Services (+ 2 subcategories)
        ("professional services", "CG1800", "Professional Services", "group"),
        ("legal services", "C1801", "Legal Services", "subcategory"),
        ("accounting", "C1802", "Accounting & Tax Prep", "subcategory"),
        
        # CG1900: Auto & Transport Services (+ 2 subcategories)
        ("auto services", "CG1900", "Auto & Transport Services", "group"),
        ("car wash", "C1901", "Car Wash & Detailing", "subcategory"),
        ("auto repairs", "C1902", "Auto Repairs", "subcategory"),
        
        # CG10000: Groceries (+ 2 subcategories)
        ("groceries", "CG10000", "Groceries", "group"),
        ("supermarket", "C10001", "Supermarket", "subcategory"),
        ("organic food", "C10002", "Organic Food Stores", "subcategory"),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: COMPLETE CATEGORY COVERAGE (ALL 60 CATEGORIES)
# ═══════════════════════════════════════════════════════════════════════════


def test_complete_category_coverage(similarity_distance_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Test 7: Complete Category Coverage (All 60 Categories)
    
    COMPONENT TESTED:
        Complete RAG pipeline (end-to-end)
        - build_category_vectorstore.py (embeddings)
        - trn_category_rag.py (retrieval)
        - trn_category_tool.py (formatting)
    
    PURPOSE:
        Verify that ALL 60 categories are retrievable through the complete
        RAG pipeline. This validates:
        1. No missing embeddings in vector store
        2. No broken category mappings
        3. Consistent performance across all category types
        4. End-to-end pipeline functional for entire knowledge base
    
    PARAMETERS:
        similarity_distance_threshold (float): Distance threshold for filtering (default: 0.6)
    
    WHAT THIS VALIDATES:
        ✓ All 20 category groups embedded correctly
        ✓ All 40 subcategories (2 per group) embedded correctly
        ✓ Complete pipeline functions end-to-end for all categories
        ✓ No missing or broken categories in knowledge base
        ✓ Consistent retrieval quality across all categories
    
    WHY THIS MATTERS:
        This is the final validation that the entire system works.
        Missing categories would mean:
        - Users can't get insights for those transaction types
        - LLM-1 Router can't categorize related transactions
        - Incomplete financial understanding for users
    
    TEST DATA:
        60 category queries:
        - 20 groups (CG100 - CG10000)
        - 40 subcategories (2 per group)
        
        Each query tests whether the category is retrievable via
        semantic search (expected in top 3 results).
    
    SUCCESS CRITERIA:
        - Expected category found in top 3 results
        - ≥90% coverage rate (54/60 categories)
        - No systematic failures (all groups represented)
    """
    print_test_header(
        7,
        "Complete Category Coverage (All 60 Categories)",
        "Complete RAG Pipeline (end-to-end)",
        "Verify all 60 categories retrievable: 20 groups + 40 subcategories"
    )
    
    # PRINT CHOSEN THRESHOLD
    print(f"🎯 Using similarity_distance_threshold: {similarity_distance_threshold}")
    print("─" * 80)
    
    # Get all test cases
    all_tests = get_all_category_tests()
    
    passed = 0
    total = len(all_tests)
    
    results = {
        "test_name": "complete_category_coverage",
        "component": "Complete RAG Pipeline",
        "total_categories": total,
        "groups": [],
        "subcategories": [],
        "failed_categories": []
    }
    
    print_component_step(f"Testing {total} categories (20 groups + 40 subcategories)")
    print()
    
    # Track results by type
    group_passed = 0
    group_total = 0
    subcat_passed = 0
    subcat_total = 0
    
    current_group = None
    
    for term, expected_id, expected_name, category_type in all_tests:
        # Print group headers for readability
        if category_type == "group":
            if current_group is not None:
                print()  # Spacing between groups
            current_group = expected_id
            print(f"   {'─' * 76}")
            print(f"   {expected_name} ({expected_id})")
            print(f"   {'─' * 76}")
        
        # Execute query - USE the threshold parameter!
        matches = query_categories(term, top_k=3, min_confidence=similarity_distance_threshold)
        
        # Check if expected category in top 3
        found = False
        if matches:
            for match in matches[:3]:
                if match.get("id") == expected_id:
                    found = True
                    distance = match.get("score")
                    break
        
        # Track results
        if found:
            passed += 1
            if category_type == "group":
                group_passed += 1
            else:
                subcat_passed += 1
        
        if category_type == "group":
            group_total += 1
        else:
            subcat_total += 1
        
        # Print result
        type_marker = f"[{category_type.upper():10}]"
        if found:
            print_result(True, f"{type_marker} '{term:25}' → {expected_name} ({expected_id})",
                        f"Dist: {distance:.4f}")
        else:
            print_result(False, f"{type_marker} '{term:25}' → NOT FOUND",
                        f"Expected: {expected_name} ({expected_id})")
            results["failed_categories"].append({
                "term": term,
                "expected_id": expected_id,
                "expected_name": expected_name,
                "category_type": category_type
            })
        
        # Store result
        result_entry = {
            "term": term,
            "expected_id": expected_id,
            "expected_name": expected_name,
            "found": found,
            "distance": distance if found else None
        }
        
        if category_type == "group":
            results["groups"].append(result_entry)
        else:
            results["subcategories"].append(result_entry)
    
    # ═════════════════════════════════════════════════════════════════
    # DETAILED SUMMARY
    # ═════════════════════════════════════════════════════════════════
    
    group_coverage = (group_passed / group_total * 100) if group_total > 0 else 0
    subcat_coverage = (subcat_passed / subcat_total * 100) if subcat_total > 0 else 0
    overall_coverage = (passed / total * 100) if total > 0 else 0
    
    print("\n" + "═" * 80)
    print("📊 DETAILED COVERAGE ANALYSIS")
    print("═" * 80)
    print(f"Category Groups:     {group_passed:2}/{group_total:2} ({group_coverage:5.1f}%)")
    print(f"Subcategories:       {subcat_passed:2}/{subcat_total:2} ({subcat_coverage:5.1f}%)")
    print("─" * 80)
    print(f"Overall Coverage:    {passed:2}/{total:2} ({overall_coverage:5.1f}%)")
    print("═" * 80)
    
    # Show failed categories if any
    if results["failed_categories"]:
        print(f"\n⚠️  FAILED CATEGORIES ({len(results['failed_categories'])}):")
        for failed in results["failed_categories"]:
            print(f"   • [{failed['category_type'].upper()}] {failed['expected_name']} "
                  f"({failed['expected_id']}) - Query: '{failed['term']}'")
    
    # Evaluate success criteria
    meets_criteria = overall_coverage >= 90.0
    
    print(f"\n✅ SUCCESS CRITERIA:")
    print(f"   {'✅' if meets_criteria else '❌'} Overall Coverage: ≥90% (actual: {overall_coverage:.1f}%)")
    
    print("\n" + "─" * 80)
    if meets_criteria:
        print("🎉 COMPLETE COVERAGE TEST PASSED!")
        print("   All major categories are retrievable through the RAG pipeline.")
    else:
        print("⚠️  COVERAGE BELOW THRESHOLD")
        print(f"   {len(results['failed_categories'])} categories not retrievable.")
        print("   → Check embeddings and descriptions for failed categories")
    print("─" * 80)
    
    print_test_summary(7, passed, total)
    
    results["passed"] = passed
    results["total"] = total
    results["accuracy"] = overall_coverage / 100  # ADD THIS LINE
    results["coverage_rate"] = overall_coverage / 100
    results["group_passed"] = group_passed
    results["group_total"] = group_total
    results["subcat_passed"] = subcat_passed
    results["subcat_total"] = subcat_total
    results["meets_criteria"] = meets_criteria
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# RUN COMPLETE COVERAGE TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_complete_coverage_test() -> Dict[str, Any]:
    """
    Run Complete Coverage Test
    
    Executes Test 7 to validate all 60 categories are retrievable.
    This is the final comprehensive validation of the entire RAG system.
    
    Returns:
        dict: Test results with coverage metrics
    """
    print("\n" + "=" * 80)
    print("🔍 RAG COMPLETE COVERAGE TEST")
    print("=" * 80)
    print("Validating: All 60 categories retrievable (20 groups + 40 subcategories)")
    print("=" * 80)
    
    # Run test
    results = test_complete_category_coverage()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 COMPLETE COVERAGE TEST SUMMARY")
    print("=" * 80)
    
    coverage = results["coverage_rate"] * 100
    print(f"Total Coverage:          {results['passed']}/{results['total']} ({coverage:.1f}%)")
    print(f"Groups:                  {results['group_passed']}/{results['group_total']}")
    print(f"Subcategories:           {results['subcat_passed']}/{results['subcat_total']}")
    print(f"Failed Categories:       {len(results['failed_categories'])}")
    print("=" * 80)
    
    if results["meets_criteria"]:
        print("\n✅ COMPLETE COVERAGE TEST PASSED")
        print("   RAG system has comprehensive category coverage!")
    else:
        print("\n⚠️  COMPLETE COVERAGE TEST NEEDS ATTENTION")
        print(f"   Coverage: {coverage:.1f}% (90% required)")
        print(f"   {len(results['failed_categories'])} categories not retrievable")
    
    print("=" * 80)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

"""
USAGE IN JUPYTER NOTEBOOK:

1. Run complete coverage test:
   from tests.test_rag_complete_coverage import run_complete_coverage_test
   results = run_complete_coverage_test()

2. Run test directly:
   from tests.test_rag_complete_coverage import test_complete_category_coverage
   test_complete_category_coverage()

3. Check coverage details:
   results = run_complete_coverage_test()
   print(f"Coverage rate: {results['coverage_rate']*100:.1f}%")
   print(f"Failed categories: {len(results['failed_categories'])}")
   
4. Identify missing categories:
   results = run_complete_coverage_test()
   for failed in results['failed_categories']:
       print(f"Missing: {failed['expected_name']} ({failed['expected_id']})")
"""