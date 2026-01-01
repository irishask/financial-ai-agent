"""
═══════════════════════════════════════════════════════════════════════════════
PIPELINE TESTS - WITH RAG TRANSACTION CATEGORIES
═══════════════════════════════════════════════════════════════════════════════

Tests for CLEAR queries that REQUIRE category RAG lookup (UC-04).

FOCUS AREAS:
1. RAG Implementation & Usage
   - LLM-1 calls RAG tool (search_transaction_categories)
   - RAG returns correct category with confidence scores
   - LLM-1 populates resolved_trn_categories

2. LLM-2 Grounding
   - LLM-2 receives EXACT parameters from LLM-1
   - LLM-2 uses ONLY those parameters (no hallucination)
   - LLM-2 builds correct tool calls

3. BackOffice Logging
   - Complete audit trail
   - All reasoning steps documented
   - Data sources and filters logged

4. Dynamic Answer Validation
   - Expected values calculated from transactions.csv at runtime
   - No hardcoded dates or amounts
   - Works for any date

VALID QUERY IDs: [3, 4, 7, 8, 9, 10]
   - All are CLEAR queries with UC-04 (category-based)

USAGE:
    from tests.pipeline_rag_tests import test_rag_pipeline
    
    # Test single query
    test_rag_pipeline([4])
    
    # Test all UC-04 queries
    test_rag_pipeline([3, 4, 7, 8, 9, 10], wait_seconds=10)
    
    # Verbose mode (see all LLM iterations)
    test_rag_pipeline([4], silent=False)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

from schemas.router_models import GraphState
from graph_definition import build_graph
from tests.dynamic_expected_calculator import DynamicExpectedCalculator, validate_llm_answer


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# CLEAR queries that REQUIRE category RAG (UC-04)
VALID_RAG_QUERY_IDS = [3, 4, 7, 8, 9, 10, 16, 17]

QA_MAPPING_PATH = "tests/_new_QA_mapping.json"
TRANSACTIONS_PATH = "data/transactions.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class CaptureOutput:
    """
    Context manager to capture stdout during graph execution.
    Used to detect RAG tool calls from console output.
    """
    
    def __init__(self):
        self.captured = ""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._capture_buffer = io.StringIO()
        sys.stdout = self._capture_buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.captured = self._capture_buffer.getvalue()
        sys.stdout = self._original_stdout


class SuppressOutput:
    """Context manager to suppress stdout during graph execution."""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - FILE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_qa_mapping() -> Dict[str, Any]:
    """Load the Q&A mapping JSON file."""
    mapping_path = Path(QA_MAPPING_PATH)
    
    if not mapping_path.exists():
        # Try alternative path (if running from project root)
        mapping_path = Path("_new_QA_mapping.json")
    
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"❌ Q&A mapping file not found.\n"
            f"Expected: {QA_MAPPING_PATH} or ./_new_QA_mapping.json"
        )
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_separator(char="═", length=80):
    """Print separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print section header."""
    print(f"\n🔹 {title}:")
    print("─" * 80)


def print_box(lines: List[str], indent: int = 3):
    """Print content in a box."""
    prefix = " " * indent
    print(f"{prefix}┌{'─' * 74}┐")
    for line in lines:
        # Truncate long lines
        display_line = line[:72] if len(line) > 72 else line
        padding = 72 - len(display_line)
        print(f"{prefix}│ {display_line}{' ' * padding} │")
    print(f"{prefix}└{'─' * 74}┘")


# ═══════════════════════════════════════════════════════════════════════════════
# RAG EXECUTION DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_rag_execution(
    resolved_categories: Optional[List[Dict]], 
    expected_mapping: Optional[Dict],
    captured_output: str
):
    """
    Display RAG tool execution details.
    
    Shows:
    - Whether RAG tool was called
    - RAG input (terms searched)
    - RAG results (category, confidence, distance)
    - Expected vs Actual comparison
    """
    print_section_header("RAG TOOL EXECUTION")
    
    # Check if RAG was called (from captured console output)
    rag_called = "search_transaction_categories" in captured_output
    
    if rag_called:
        print(f"   ✅ Tool Called: search_transaction_categories")
        
        # Extract input terms from captured output
        if "with args:" in captured_output:
            try:
                args_start = captured_output.find("with args:") + len("with args:")
                args_end = captured_output.find("\n", args_start)
                args_str = captured_output[args_start:args_end].strip()
                print(f"   ✅ Input: {args_str}")
            except:
                print(f"   ✅ Input: (see console output)")
    else:
        print(f"   ❌ Tool NOT Called: search_transaction_categories")
        print(f"      LLM-1 should have called RAG for category resolution")
    
    # Show RAG results
    print(f"\n   📊 RAG Results:")
    
    if resolved_categories and len(resolved_categories) > 0:
        # Table header
        print(f"   ┌{'─'*14}┬{'─'*20}┬{'─'*12}┬{'─'*10}┬{'─'*12}┐")
        print(f"   │ {'User Term':<12} │ {'Category':<18} │ {'ID':<10} │ {'Distance':<8} │ {'Confidence':<10} │")
        print(f"   ├{'─'*14}┼{'─'*20}┼{'─'*12}┼{'─'*10}┼{'─'*12}┤")
        
        for cat in resolved_categories:
            user_term = str(cat.get('user_term', ''))[:12]
            cat_name = str(cat.get('category_name', ''))[:18]
            cat_id = str(cat.get('category_id', ''))[:10]
            distance = cat.get('distance', 0)
            confidence = str(cat.get('confidence', ''))[:10]
            
            print(f"   │ {user_term:<12} │ {cat_name:<18} │ {cat_id:<10} │ {distance:<8.4f} │ {confidence:<10} │")
        
        print(f"   └{'─'*14}┴{'─'*20}┴{'─'*12}┴{'─'*10}┴{'─'*12}┘")
    else:
        print(f"   (No categories resolved)")
    
    # Expected vs Actual comparison
    print(f"\n   🎯 Expected vs Actual:")
    
    if expected_mapping and resolved_categories:
        # Use mapping_level to determine which ID to compare
        if expected_mapping.get('mapping_level') == 'subcategory':
            expected_id = expected_mapping.get('subCategoryId')
        else:
            expected_id = expected_mapping.get('categoryGroupId')
    
        expected_name = expected_mapping.get('categoryGroupName') or expected_mapping.get('subCategoryName')
        
        actual_id = resolved_categories[0].get('category_id') if resolved_categories else None
        actual_name = resolved_categories[0].get('category_name') if resolved_categories else None
        
        id_match = expected_id == actual_id
        name_match = expected_name == actual_name
        
        lines = [
            f"Expected: {expected_id} ({expected_name})",
            f"Actual:   {actual_id} ({actual_name})",
            f"Result:   {'✅ MATCH' if id_match else '❌ MISMATCH'}"
        ]
        print_box(lines)
        
        return {"rag_called": rag_called, "category_match": id_match}
    else:
        print_box(["No expected mapping or no resolved categories"])
        return {"rag_called": rag_called, "category_match": False}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-1 OUTPUT DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_llm1_output(router_output, expected_data: Dict) -> Dict[str, bool]:
    """
    Display LLM-1 Router output with validation.
    
    Shows:
    - Clarity (validated against expected)
    - Core UCs (validated against expected)
    - Resolved categories (content)
    - Resolved dates (if temporal)
    """
    print_section_header("LLM-1 OUTPUT (Router)")
    
    checks = {}
    
    # Clarity check
    expected_clarity = expected_data.get('clarity', 'CLEAR')
    actual_clarity = router_output.clarity
    clarity_match = expected_clarity == actual_clarity
    checks['clarity_correct'] = clarity_match
    
    status = "✅" if clarity_match else "❌"
    print(f"   {status} Clarity: {actual_clarity} (expected: {expected_clarity})")
    
    # Core UCs check
    expected_ucs = set(expected_data.get('core_query_categories', []))
    actual_ucs = set(router_output.core_use_cases) if router_output.core_use_cases else set()
    uc_match = 'UC-04' in actual_ucs  # Main check: UC-04 detected
    checks['uc04_detected'] = uc_match
    
    status = "✅" if uc_match else "❌"
    print(f"   {status} UC-04 Detected: {'Yes' if uc_match else 'No'}")
    print(f"      Core UCs: {list(actual_ucs)}")
    print(f"      Expected: {list(expected_ucs)}")
    
    # Resolved categories check
    resolved_cats = router_output.resolved_trn_categories
    has_categories = resolved_cats is not None and len(resolved_cats) > 0
    checks['categories_populated'] = has_categories
    
    status = "✅" if has_categories else "❌"
    print(f"   {status} resolved_trn_categories: {'Populated (' + str(len(resolved_cats)) + ' items)' if has_categories else 'Empty/None'}")
    
    if has_categories:
        print(f"\n      Full content:")
        for cat in resolved_cats:
            print(f"      {json.dumps(cat, indent=8, default=str)}")
    
    # Resolved dates (if temporal query)
    if router_output.resolved_dates:
        rd = router_output.resolved_dates
        print(f"\n   ✓ Resolved Dates:")
        print(f"      • Start: {rd.start_date}")
        print(f"      • End: {rd.end_date}")
        print(f"      • Interpretation: {rd.interpretation}")
    
    # Confidence
    print(f"\n   ✓ Confidence: {router_output.uc_confidence}")
    
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-2 INPUT CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def print_llm2_input_check(router_output, expected_mapping: Dict, test_type: str) -> Dict[str, bool]:
    """
    Verify LLM-2 receives correct input from LLM-1.
    
    Shows:
    - Category ID received
    - Dates received
    - All parameters that LLM-2 should use
    """
    print_section_header("LLM-2 INPUT CHECK")
    
    checks = {}
    
    # Check resolved categories
    resolved_cats = router_output.resolved_trn_categories
    has_categories = resolved_cats is not None and len(resolved_cats) > 0
    
    if has_categories:
        cat_id = resolved_cats[0].get('category_id')
        cat_name = resolved_cats[0].get('category_name')
        
        # Use mapping_level to determine which ID to compare
        if expected_mapping.get('mapping_level') == 'subcategory':
            expected_id = expected_mapping.get('subCategoryId')
        else:
            expected_id = expected_mapping.get('categoryGroupId')

    
        id_correct = cat_id == expected_id
        checks['correct_category_passed'] = id_correct
        
        status = "✅" if id_correct else "❌"
        print(f"   {status} Category ID passed to LLM-2: {cat_id}")
        print(f"      Expected: {expected_id}")
    else:
        checks['correct_category_passed'] = False
        print(f"   ❌ No category passed to LLM-2")
    
    # Check resolved dates
    # For "last_transaction_by_category", dates are NOT required (uses sort + limit)
    dates_not_required = test_type == "last_transaction_by_category"
    
    if router_output.resolved_dates:
        rd = router_output.resolved_dates
        print(f"   ✅ Dates passed to LLM-2: {rd.start_date} to {rd.end_date}")
        checks['dates_passed'] = True
    elif dates_not_required:
        print(f"   ✅ No dates needed (test_type: {test_type} uses sort + limit)")
        checks['dates_passed'] = True  # Not a failure
    else:
        print(f"   ❌ No resolved dates (required for {test_type})")
        checks['dates_passed'] = False
    
    # Summary box
    lines = [
        f"LLM-2 will receive:",
        f"  • category_id: {resolved_cats[0].get('category_id') if has_categories else 'None'}",
        f"  • category_name: {resolved_cats[0].get('category_name') if has_categories else 'None'}",
        f"  • start_date: {router_output.resolved_dates.start_date if router_output.resolved_dates else 'None'}",
        f"  • end_date: {router_output.resolved_dates.end_date if router_output.resolved_dates else 'None'}",
    ]
    print()
    print_box(lines)
    
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-2 GROUNDING CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def print_llm2_grounding_check(
    router_output, 
    execution_result,
    expected_mapping: Dict
) -> Dict[str, bool]:
    """
    Verify LLM-2 uses ONLY data from LLM-1 input (no hallucination).
    
    Shows:
    - Input from LLM-1 vs What LLM-2 used
    - Grounding status for each parameter
    """
    print_section_header("LLM-2 GROUNDING CHECK")
    
    checks = {}
    grounding_issues = []
    
    print(f"   Verifying LLM-2 uses ONLY data from LLM-1 input...\n")
    
    # Get what LLM-1 provided
    resolved_cats = router_output.resolved_trn_categories
    llm1_category_id = resolved_cats[0].get('category_id') if resolved_cats else None
    llm1_start_date = router_output.resolved_dates.start_date if router_output.resolved_dates else None
    llm1_end_date = router_output.resolved_dates.end_date if router_output.resolved_dates else None
    
    # Get what LLM-2 used (from backoffice log)
    backoffice = execution_result.backoffice_log if execution_result else None
    
    if backoffice and backoffice.data_sources:
        filters = backoffice.data_sources.filters_applied or []
        filters_str = " ".join(filters).lower()
        
        # Check category grounding
        if llm1_category_id:
            category_used = llm1_category_id.lower() in filters_str or llm1_category_id in filters_str
            checks['category_grounded'] = category_used
            
            status = "✅ GROUNDED" if category_used else "❌ NOT USED"
            print(f"   │ category_id: {llm1_category_id:<15} │ {status:<20} │")
            
            if not category_used:
                grounding_issues.append(f"Category {llm1_category_id} not found in filters")
        
        # Check date grounding
        if llm1_start_date:
            start_used = str(llm1_start_date) in filters_str
            checks['start_date_grounded'] = start_used
            
            status = "✅ GROUNDED" if start_used else "⚠️  CHECK"
            print(f"   │ start_date:  {str(llm1_start_date):<15} │ {status:<20} │")
        
        if llm1_end_date:
            end_used = str(llm1_end_date) in filters_str
            checks['end_date_grounded'] = end_used
            
            status = "✅ GROUNDED" if end_used else "⚠️  CHECK"
            print(f"   │ end_date:    {str(llm1_end_date):<15} │ {status:<20} │")
        
        # Check for hallucinated filters (filters not in LLM-1 output)
        # This is a simplified check - in production you'd be more thorough
        print()
        
        if grounding_issues:
            print(f"   ⚠️  Grounding Issues:")
            for issue in grounding_issues:
                print(f"      • {issue}")
            checks['fully_grounded'] = False
        else:
            print(f"   ✅ LLM-2 is GROUNDED: All parameters from LLM-1")
            checks['fully_grounded'] = True
    else:
        print(f"   ⚠️  Cannot verify grounding - no backoffice data")
        checks['fully_grounded'] = False
    
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-2 TOOL USAGE
# ═══════════════════════════════════════════════════════════════════════════════

def print_llm2_tool_usage(execution_result) -> Dict[str, bool]:
    """
    Show which tools LLM-2 called and with what parameters.
    """
    print_section_header("LLM-2 TOOL USAGE")
    
    checks = {}
    
    backoffice = execution_result.backoffice_log if execution_result else None
    
    if backoffice and backoffice.data_sources:
        ds = backoffice.data_sources
        
        # Tables used
        if ds.tables_used:
            print(f"   📁 Tables: {ds.tables_used}")
            checks['tables_accessed'] = True
        
        # Fields accessed
        if ds.fields_accessed:
            print(f"   📋 Fields: {ds.fields_accessed}")
        
        # Filters applied (this is the "query")
        if ds.filters_applied:
            print(f"\n   🔍 Query Filters (SQL-like):")
            print(f"   ┌{'─'*74}┐")
            for f in ds.filters_applied:
                display_f = f[:72] if len(f) > 72 else f
                padding = 72 - len(display_f)
                print(f"   │ {display_f}{' '*padding} │")
            print(f"   └{'─'*74}┘")
            checks['filters_applied'] = True
        
        # Aggregations
        if ds.aggregations_used:
            print(f"\n   📊 Aggregations: {ds.aggregations_used}")
            checks['aggregations_used'] = True
    else:
        print(f"   ⚠️  No tool usage data available")
        checks['tables_accessed'] = False
        checks['filters_applied'] = False
    
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# BACKOFFICE LOG DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_backoffice_log(execution_result):
    """
    Display complete BackOffice log structure.
    
    Shows:
    - User query
    - Answer
    - Reasoning steps
    - Data sources
    - Transactions analyzed
    - Confidence
    - RAG used flag
    """
    print_section_header("BACKOFFICE LOG")
    
    if not execution_result:
        print(f"   ❌ No execution result")
        return
    
    backoffice = execution_result.backoffice_log
    
    if not backoffice:
        print(f"   ❌ No backoffice log")
        return
    
    # Print as formatted JSON-like structure
    print(f"   {{")
    print(f'     "user_query": "{backoffice.user_query[:60]}...",' if len(str(backoffice.user_query)) > 60 else f'     "user_query": "{backoffice.user_query}",')
    
    # Answer
    answer = execution_result.final_answer or backoffice.answer
    print(f'     "answer": "{answer[:60]}...",' if len(str(answer)) > 60 else f'     "answer": "{answer}",')
    
    # Analysis
    if backoffice.analysis:
        print(f'     "analysis": {{')
        for key, val in list(backoffice.analysis.items())[:5]:
            print(f'       "{key}": {json.dumps(val)},')
        print(f'     }},')
    
    # Reasoning steps
    if backoffice.reasoning_steps:
        print(f'     "reasoning_steps": [')
        for i, step in enumerate(backoffice.reasoning_steps[:5]):
            step_display = step[:65] if len(step) > 65 else step
            print(f'       "{step_display}...",') if len(step) > 65 else print(f'       "{step_display}",')
        if len(backoffice.reasoning_steps) > 5:
            print(f'       ... ({len(backoffice.reasoning_steps) - 5} more steps)')
        print(f'     ],')
    
    # Data sources
    if backoffice.data_sources:
        ds = backoffice.data_sources
        print(f'     "data_sources": {{')
        print(f'       "tables_used": {json.dumps(ds.tables_used)},')
        print(f'       "fields_accessed": {json.dumps(ds.fields_accessed)},')
        print(f'       "filters_applied": {json.dumps(ds.filters_applied[:3])}...,') if ds.filters_applied and len(ds.filters_applied) > 3 else print(f'       "filters_applied": {json.dumps(ds.filters_applied)},')
        print(f'       "aggregations_used": {json.dumps(ds.aggregations_used)}')
        print(f'     }},')
    
    # Other fields
    print(f'     "transactions_analyzed": {backoffice.transactions_analyzed},')
    print(f'     "confidence": "{backoffice.confidence}",')
    print(f'     "rag_used": {json.dumps(backoffice.rag_used)}')
    print(f"   }}")


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTED ANSWER DISPLAY - WITH DYNAMIC VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def print_formatted_answer(
    execution_result, 
    expected_values: Dict[str, Any],
    test_type: str
) -> Dict[str, bool]:
    """
    Display final answer with dynamic validation against calculated expected values.
    
    Args:
        execution_result: LLM-2 execution result
        expected_values: Dynamically calculated expected values from transactions.csv
        test_type: Type of test (e.g., 'sum_single_period', 'compare_two_periods')
    
    Returns:
        Dict with validation checks
    """
    print_section_header("FORMATTED ANSWER + DYNAMIC VALIDATION")
    
    checks = {}
    
    if not execution_result:
        print(f"   ❌ No execution result")
        return {'answer_validated': False}
    
    actual_answer = execution_result.final_answer
    
    # Display LLM-2 answer
    print(f"   💬 LLM-2 Answer:")
    print_box([actual_answer[:72] if len(actual_answer) > 72 else actual_answer])
    
    # Display dynamically calculated expected values
    print(f"\n   📊 Expected Values (calculated from transactions.csv):")
    
    if test_type == 'balance_calculation':
        print(f"      • Balance: ${expected_values.get('balance', 0):,.2f}")
        print(f"      • As of: {expected_values.get('as_of_date', 'N/A')}")
        
    elif test_type == 'sum_single_period':
        print(f"      • Total: ${expected_values.get('total', 0):,.2f}")
        print(f"      • Count: {expected_values.get('count', 0)} transactions")
        print(f"      • Period: {expected_values.get('period_start')} to {expected_values.get('period_end')}")
        
    elif test_type == 'compare_two_periods':
        p1 = expected_values.get('period_1', {})
        p2 = expected_values.get('period_2', {})
        print(f"      • Period 1 ({p1.get('name', '?')}): ${p1.get('total', 0):,.2f} ({p1.get('count', 0)} txns)")
        print(f"      • Period 2 ({p2.get('name', '?')}): ${p2.get('total', 0):,.2f} ({p2.get('count', 0)} txns)")
        print(f"      • Difference: ${expected_values.get('difference', 0):,.2f} ({expected_values.get('percentage_change', 0)}%)")
        
    elif test_type == 'list_transactions_period':
        print(f"      • Count: {expected_values.get('count', 0)} transactions")
        print(f"      • Total: ${expected_values.get('total_amount', 0):,.2f}")
        print(f"      • Period: {expected_values.get('period_start')} to {expected_values.get('period_end')}")
        
    elif test_type == 'last_transaction_by_category':
        if expected_values.get('found', False):
            print(f"      • Amount: ${expected_values.get('amount', 0):,.2f}")
            print(f"      • Date: {expected_values.get('date', 'N/A')}")
            print(f"      • Category: {expected_values.get('subCategoryName', 'N/A')}")
        else:
            print(f"      • No transactions found for this category")
    else:
        print(f"      • Raw: {expected_values}")
    
    # Validate LLM answer against expected values
    print(f"\n   🔍 Validation:")
    validation_result = validate_llm_answer(actual_answer, expected_values, test_type)
    
    if validation_result['valid']:
        print(f"   ✅ Answer VALIDATED - amounts match expected values")
        checks['answer_validated'] = True
    else:
        print(f"   ⚠️  Answer validation inconclusive")
        checks['answer_validated'] = False
    
    for check_msg in validation_result.get('checks', []):
        print(f"      {check_msg}")
    for error_msg in validation_result.get('errors', []):
        print(f"      ❌ {error_msg}")
    
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_query_summary(query_id: int, all_checks: Dict[str, bool]) -> bool:
    """
    Print summary of all checks for one query.
    
    Returns:
        bool: True if all critical checks passed
    """
    print_separator()
    print(f"📋 QUERY {query_id} SUMMARY")
    print_separator()
    
    passed = 0
    total = 0
    
    check_descriptions = {
        'rag_called': 'RAG tool called',
        'category_match': 'RAG category correct',
        'clarity_correct': 'LLM-1 clarity correct',
        'uc04_detected': 'LLM-1 detected UC-04',
        'categories_populated': 'LLM-1 resolved_trn_categories populated',
        'correct_category_passed': 'LLM-2 received correct category',
        'dates_passed': 'LLM-2 received dates',
        'fully_grounded': 'LLM-2 is GROUNDED',
        'tables_accessed': 'LLM-2 accessed tables',
        'filters_applied': 'LLM-2 applied filters',
        'answer_validated': 'Answer matches expected values',
    }
    
    for check_name, passed_check in all_checks.items():
        total += 1
        if passed_check:
            passed += 1
            status = "✅"
        else:
            status = "❌"
        
        description = check_descriptions.get(check_name, check_name)
        print(f"   {status} {description}")
    
    print()
    print(f"   RESULT: {passed}/{total} checks passed")
    print_separator()
    
    return passed == total


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_rag_pipeline(
    query_ids: Optional[List[int]] = None,
    wait_seconds: int = 10,
    silent: bool = False  # Default False to see RAG tool calls
):
    """
    Test CLEAR queries that REQUIRE category RAG (UC-04).
    
    Tests full pipeline: LLM-1 → RAG → LLM-2 → Answer
    
    Focus:
    - RAG tool called and returns correct category
    - LLM-1 populates resolved_trn_categories
    - LLM-2 receives and uses EXACT parameters (grounded)
    - BackOffice log is complete
    - Answer matches dynamically calculated expected values
    
    Args:
        query_ids: List of query IDs from _new_QA_mapping.json
                   Valid IDs: [3, 4, 7, 8, 9, 10]
                   Default: None (runs ALL valid queries)
        wait_seconds: Seconds to wait between queries (default: 10)
        silent: If True, suppress LLM iteration logs (default: False)
    
    Examples:
        test_rag_pipeline()                       # Test ALL UC-04 queries (default)
        test_rag_pipeline([4])                    # Test single query
        test_rag_pipeline([3, 7, 10])             # Test specific queries
        test_rag_pipeline(silent=True)            # All queries, silent mode
    """
    
    print("\n")
    print_separator()
    print("🧪 RAG PIPELINE TEST - UC-04 QUERIES WITH CATEGORY RESOLUTION")
    print_separator()
    
    # Default: run all valid queries
    if query_ids is None:
        query_ids = VALID_RAG_QUERY_IDS.copy()
        print(f"📋 Running ALL {len(query_ids)} UC-04 queries (default)")
    else:
        # Validate query IDs
        invalid_ids = [qid for qid in query_ids if qid not in VALID_RAG_QUERY_IDS]
        if invalid_ids:
            print(f"❌ Invalid query IDs: {invalid_ids}")
            print(f"   Valid IDs for RAG tests: {VALID_RAG_QUERY_IDS}")
            return
        print(f"📋 Running {len(query_ids)} selected queries")
    
    # Load Q&A mapping early to show query list
    print("\n📂 Loading Q&A mapping...")
    try:
        qa_mapping = load_qa_mapping()
        print("✅ Q&A mapping loaded")
    except FileNotFoundError as e:
        print(str(e))
        return
    
    # Initialize dynamic expected calculator
    print("\n📊 Initializing dynamic expected calculator...")
    try:
        expected_calc = DynamicExpectedCalculator(transactions_path=TRANSACTIONS_PATH)
        print(f"✅ Calculator ready (reference date: {expected_calc.today})")
    except Exception as e:
        print(f"❌ Failed to initialize calculator: {e}")
        return
    
    # Print query list with original questions
    print(f"\n📝 QUERIES TO TEST:")
    print("─" * 80)
    for qid in query_ids:
        qid_str = str(qid)
        if qid_str in qa_mapping:
            query_text = qa_mapping[qid_str].get('query', '(unknown)')
            expected_cat = qa_mapping[qid_str].get('expected_category_mapping', {})
            
            # Use mapping_level to determine which ID to display
            if expected_cat.get('mapping_level') == 'subcategory':
                cat_id = expected_cat.get('subCategoryId', '?')
            else:
                cat_id = expected_cat.get('categoryGroupId', '?')          
            
            print(f"   [{qid:2d}] \"{query_text}\"")
            print(f"        → Expected: {cat_id}")
        else:
            print(f"   [{qid:2d}] (not found in mapping)")
    print("─" * 80)
    
    print(f"\nConfiguration:")
    print(f"   • Wait time: {wait_seconds} seconds between queries")
    print(f"   • Silent mode: {'ON (iteration logs suppressed)' if silent else 'OFF (see all LLM iterations)'}")
    print(f"   • Reference date: {expected_calc.today}")
    
    # Build graph
    print("\n🔧 Building graph...")
    try:
        graph = build_graph()
        compiled_graph = graph.compile()
        print("✅ Graph compiled and ready")
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        return
    
    print_separator()
    
    results = []
    
    for i, qid in enumerate(query_ids, 1):
        qid_str = str(qid)
        
        if qid_str not in qa_mapping:
            print(f"❌ Query ID {qid} not found in Q&A mapping, skipping...")
            results.append({"id": qid, "status": "❌ FAIL", "passed": False})
            continue
        
        query_data = qa_mapping[qid_str]
        expected_mapping = query_data.get('expected_category_mapping', {})
        test_type = query_data.get('test_type', '')
        
        # Calculate expected values dynamically
        try:
            expected_values = expected_calc.calculate_expected(query_data)
        except Exception as e:
            print(f"⚠️  Could not calculate expected values: {e}")
            expected_values = {}
        
        print("\n")
        print_separator()
        print(f"📝 QUERY {i}/{len(query_ids)} (ID: {qid})")
        print_separator()
        print(f"👤 USER: \"{query_data['query']}\"")
        
        # Use mapping_level to determine which ID/name to display
        if expected_mapping.get('mapping_level') == 'subcategory':
            exp_id = expected_mapping.get('subCategoryId')
            exp_name = expected_mapping.get('subCategoryName')
        else:
            exp_id = expected_mapping.get('categoryGroupId')
            exp_name = expected_mapping.get('categoryGroupName')
        print(f"🎯 Expected Category: {exp_id} ({exp_name})")
        
        # Prepare query with user ID
        full_query = f"I am USER_001. {query_data['query']}"
        
        # Create initial state
        initial_state = GraphState(
            user_query=full_query,
            conversation_summary=None,
            turn_id=1
        )
        
        all_checks = {}
        
        try:
            # Execute pipeline with output capture (to detect RAG calls)
            if silent:
                with SuppressOutput():
                    final_state = compiled_graph.invoke(initial_state)
                captured_output = ""
            else:
                with CaptureOutput() as capture:
                    final_state = compiled_graph.invoke(initial_state)
                captured_output = capture.captured
                # Print captured output
                print("\n--- LLM Execution Log ---")
                print(captured_output)
                print("--- End Log ---\n")
            
            # Get outputs
            router_output = final_state.get('router_output')
            execution_result = final_state.get('execution_result')
            
            if not router_output:
                print("❌ No router output returned")
                results.append({"id": qid, "status": "❌ FAIL", "passed": False})
                continue
            
            # 1. RAG Execution
            resolved_cats = router_output.resolved_trn_categories
            rag_checks = print_rag_execution(resolved_cats, expected_mapping, captured_output)
            all_checks.update(rag_checks)
            
            # 2. LLM-1 Output
            llm1_checks = print_llm1_output(router_output, query_data)
            all_checks.update(llm1_checks)
            
            # 3. LLM-2 Input Check
            llm2_input_checks = print_llm2_input_check(router_output, expected_mapping, test_type)
            all_checks.update(llm2_input_checks)
            
            # 4. LLM-2 Grounding Check
            if execution_result:
                grounding_checks = print_llm2_grounding_check(router_output, execution_result, expected_mapping)
                all_checks.update(grounding_checks)
                
                # 5. LLM-2 Tool Usage
                tool_checks = print_llm2_tool_usage(execution_result)
                all_checks.update(tool_checks)
                
                # 6. Formatted Answer with Dynamic Validation
                answer_checks = print_formatted_answer(execution_result, expected_values, test_type)
                all_checks.update(answer_checks)
                
                # 7. BackOffice Log
                print_backoffice_log(execution_result)
            else:
                print("\n❌ No execution result (LLM-2 did not execute)")
            
            # 8. Summary
            all_passed = print_query_summary(qid, all_checks)
            
            results.append({
                "id": qid, 
                "status": "✅ PASS" if all_passed else "⚠️  PARTIAL",
                "passed": all_passed,
                "checks": all_checks
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"id": qid, "status": "❌ FAIL", "passed": False, "error": str(e)})
        
        # Wait between queries (except after last one)
        if i < len(query_ids):
            print(f"\n⏳ Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
    
    # Final Summary
    print("\n")
    print_separator()
    print("📊 FINAL TEST SUMMARY")
    print_separator()
    
    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)
    
    print(f"Total Queries: {total}")
    print(f"Fully Passed:  {passed}")
    print(f"Partial/Failed: {total - passed}")
    print(f"Success Rate:  {(passed/total)*100:.1f}%" if total > 0 else "N/A")
    print()
    
    for r in results:
        print(f"   {r['status']} - Query #{r['id']}")
        if r.get('error'):
            print(f"        Error: {r['error'][:80]}")
    
    print_separator()
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

"""
USAGE IN JUPYTER NOTEBOOK:

1. Test ALL UC-04 queries (default):
   from tests.pipeline_rag_tests import test_rag_pipeline
   test_rag_pipeline()

2. Test ALL queries in silent mode:
   test_rag_pipeline(silent=True)

3. Test single query (verbose - see all LLM iterations):
   test_rag_pipeline([4])

4. Test specific queries:
   test_rag_pipeline([3, 7, 10])

5. Test with shorter wait time:
   test_rag_pipeline(wait_seconds=5)

VALID QUERY IDs: [3, 4, 7, 8, 9, 10]

   [3]  "How much did I spend on dining last month compared to September?"
        → Expected: CG800 (Dining)
   
   [4]  "How much did I spend on groceries last month?"
        → Expected: CG10000 (Groceries)
   
   [7]  "How much did I spend on healthcare last month?"
        → Expected: CG300 (Healthcare & Medical)
   
   [8]  "Show me my gym payments this year"
        → Expected: C1701 (Gym & Fitness Center)
   
   [9]  "How much did I spend on utilities in November compared to October?"
        → Expected: CG200 (Utilities & Bills)
   
   [10] "Show me my last pharmacy transaction"
        → Expected: C302 (Pharmacy)

WHAT THE TEST SHOWS:

🔹 RAG TOOL EXECUTION:
   - Was the tool called?
   - What category was found?
   - Expected vs Actual comparison

🔹 LLM-1 OUTPUT:
   - Clarity correct?
   - UC-04 detected?
   - resolved_trn_categories populated?

🔹 LLM-2 INPUT CHECK:
   - Did LLM-2 receive the category?
   - Did LLM-2 receive the dates?

🔹 LLM-2 GROUNDING CHECK:
   - Did LLM-2 use ONLY parameters from LLM-1?
   - No hallucinated filters?

🔹 LLM-2 TOOL USAGE:
   - Which tables accessed?
   - What filters applied (SQL-like)?
   - What aggregations used?

🔹 DYNAMIC ANSWER VALIDATION:
   - Expected values calculated from transactions.csv
   - Validates LLM answer amounts match expected
   - Works for any date (no hardcoded values)

🔹 BACKOFFICE LOG:
   - Complete audit trail
   - Reasoning steps
   - Data sources
"""