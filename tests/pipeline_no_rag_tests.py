"""
PIPELINE TESTS - NO RAG TRANSACTION CATEGORIES

Tests for queries that DON'T require category RAG lookup:
- CLEAR queries: UC-01, UC-02, UC-03 (without UC-04)
- VAGUE queries: UC-05 (without category ambiguity)

These tests verify the complete pipeline works WITHOUT implementing Category RAG.
"""

import json
import time
import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

from schemas.router_models import GraphState
from graph_definition import build_graph


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Valid query IDs from _new_QA_mapping.json that DON'T need category RAG
VALID_CLEAR_QUERY_IDS = [1, 2, 5, 6]  # UC-01, UC-02, UC-03 only
VALID_VAGUE_QUERY_IDS = [12, 13]      # UC-05 without category ambiguity

QA_MAPPING_PATH = "tests/_new_QA_mapping.json"


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

class SuppressOutput:
    """Context manager to suppress stdout during graph execution."""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


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


def print_separator(char="=", length=100):
    """Print separator line."""
    print("\n" + char * length + "\n")


def print_llm1_detailed(router_output):
    """
    Print detailed LLM-1 router output with all key fields.
    Makes it easy to see what LLM-1 decided and why.
    """
    print("🔹 LLM-1 ROUTER OUTPUT:")
    print(f"   ✓ Clarity: {router_output.clarity}")
    print(f"   ✓ Core UCs: {router_output.core_use_cases}")
    print(f"   ✓ Primary UC: {router_output.primary_use_case}")
    
    # Show uc_operations (detailed subtypes)
    if router_output.uc_operations:
        non_empty_ops = {k: v for k, v in router_output.uc_operations.items() if v}
        if non_empty_ops:
            print(f"   ✓ UC Operations:")
            for uc, ops in non_empty_ops.items():
                print(f"      • {uc}: {ops}")
    
    print(f"   ✓ Complexity Axes: {router_output.complexity_axes}")
    
    # Resolved dates (if temporal query)
    if router_output.resolved_dates:
        rd = router_output.resolved_dates
        print(f"   ✓ Resolved Dates:")
        print(f"      • Start: {rd.start_date}")
        print(f"      • End: {rd.end_date}")
        print(f"      • Interpretation: {rd.interpretation}")
    else:
        print(f"   ✓ Resolved Dates: None (not temporal)")
    
    # Resolved categories (should be None for these tests)
    if router_output.resolved_trn_categories:
        print(f"   ⚠️  Resolved Categories: {router_output.resolved_trn_categories}")
    else:
        print(f"   ✓ Resolved Categories: None (no categories)")
    
    # Resolved amount threshold
    if router_output.resolved_amount_threshold:
        print(f"   ✓ Resolved Amount Threshold: ${router_output.resolved_amount_threshold}")
    
    print(f"   ✓ Needed Tools: {router_output.needed_tools}")
    print(f"   ✓ Confidence: {router_output.uc_confidence}")
    
    if router_output.clarity_reason:
        print(f"   ✓ Clarity Reason: {router_output.clarity_reason}")


def print_llm1_vague(router_output):
    """Print LLM-1 output for VAGUE queries."""
    print("🔹 LLM-1 ROUTER OUTPUT (VAGUE):")
    print(f"   ✓ Clarity: {router_output.clarity}")
    print(f"   ✓ Core UCs: {router_output.core_use_cases}")
    print(f"   ✓ Primary UC: {router_output.primary_use_case}")
    print(f"   ❓ Clarifying Question:")
    print(f"      \"{router_output.clarifying_question}\"")
    print(f"   ❓ Missing Info: {router_output.missing_info}")
    
    if router_output.clarity_reason:
        print(f"   ✓ Clarity Reason: {router_output.clarity_reason}")


def print_llm2_detailed(execution_result):
    """
    Print detailed LLM-2 executor output with all key components.
    Shows the complete back-office log structure.
    """
    print("\n🔹 LLM-2 EXECUTOR OUTPUT:")
    
    # 1. Customer Answer
    print(f"   ✓ Customer Answer:")
    answer_lines = execution_result.final_answer.split('\n')
    for line in answer_lines:
        if line.strip():
            print(f"      \"{line.strip()}\"")
    
    backoffice = execution_result.backoffice_log
    
    # 2. Analysis (Key Metrics)
    if backoffice.analysis:
        print(f"\n   📊 Analysis (Key Metrics):")
        for key, val in backoffice.analysis.items():
            print(f"      • {key}: {val}")
    
    # 3. Reasoning Steps
    if backoffice.reasoning_steps:
        print(f"\n   🔍 Reasoning Steps ({len(backoffice.reasoning_steps)} steps):")
        for i, step in enumerate(backoffice.reasoning_steps, 1):
            # Truncate very long steps
            step_text = step if len(step) <= 150 else step[:147] + "..."
            print(f"      {i}. {step_text}")
    
    # 4. Data Sources
    if backoffice.data_sources:
        ds = backoffice.data_sources
        print(f"\n   🗄️  Data Sources:")
        if ds.tables_used:
            print(f"      • Tables: {ds.tables_used}")
        if ds.fields_accessed:
            print(f"      • Fields: {ds.fields_accessed}")
        if ds.filters_applied:
            print(f"      • Filters:")
            for f in ds.filters_applied[:5]:  # Limit to first 5
                print(f"         - {f}")
            if len(ds.filters_applied) > 5:
                print(f"         - ... and {len(ds.filters_applied) - 5} more")
        if ds.aggregations_used:
            print(f"      • Aggregations: {ds.aggregations_used}")
    
    # 5. Transactions Analyzed
    print(f"\n   📈 Transactions Analyzed: {backoffice.transactions_analyzed}")
    
    # 6. Preferences Used
    if backoffice.preferences_used:
        print(f"\n   ⚙️  Preferences Used:")
        for pref_name, pref_entry in backoffice.preferences_used.items():
            print(f"      • {pref_name}: {pref_entry.value} (source: {pref_entry.source})")
    
    # 7. Confidence
    print(f"\n   🎯 Confidence: {backoffice.confidence}")


def print_error_detailed(error: Exception, stage: str = "unknown"):
    """Print detailed error information."""
    print(f"\n❌ ERROR IN {stage.upper()}:")
    print(f"   Type: {type(error).__name__}")
    print(f"   Message: {str(error)[:300]}")
    
    if hasattr(error, '__cause__') and error.__cause__:
        print(f"   Cause: {str(error.__cause__)[:200]}")


# ═══════════════════════════════════════════════════════════════════
# MAIN TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def test_clear_queries_no_rag_trn_categories(
    query_ids: List[int],
    wait_seconds: int = 10,
    silent: bool = True
):
    """
    Test CLEAR queries WITHOUT transaction category RAG lookup.
    
    Valid for queries with UC-01, UC-02, UC-03 only (no UC-04).
    
    Args:
        query_ids: List of query IDs from _new_QA_mapping.json
                   Valid IDs: [1, 2, 5, 6]
        wait_seconds: Seconds to wait between queries (default: 10)
        silent: Suppress iteration/tool execution logs (default: True)
    
    Examples:
        test_clear_queries_no_rag_trn_categories([1, 2])           # Test 2 queries
        test_clear_queries_no_rag_trn_categories([1, 2, 5, 6])    # Test all 4
        test_clear_queries_no_rag_trn_categories([6], silent=False)  # Verbose
    """
    
    print_separator()
    print("🧪 TEST: CLEAR QUERIES WITHOUT RAG TRANSACTION CATEGORIES")
    print_separator()
    
    # Validate query IDs
    invalid_ids = [qid for qid in query_ids if qid not in VALID_CLEAR_QUERY_IDS]
    if invalid_ids:
        print(f"❌ Invalid query IDs: {invalid_ids}")
        print(f"   Valid IDs for CLEAR non-category queries: {VALID_CLEAR_QUERY_IDS}")
        return
    
    print(f"Testing {len(query_ids)} queries: {query_ids}")
    print(f"Wait time: {wait_seconds} seconds between queries")
    print(f"Silent mode: {'ON' if silent else 'OFF'}")
    
    # Load Q&A mapping
    print("\n📂 Loading Q&A mapping...")
    try:
        qa_mapping = load_qa_mapping()
        print("✅ Q&A mapping loaded")
    except FileNotFoundError as e:
        print(str(e))
        return
    
    # Build graph
    print("🔧 Building graph...")
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
            results.append({"id": qid, "status": "❌ FAIL", "error": "Not in mapping"})
            continue
        
        query_data = qa_mapping[qid_str]
        
        print_separator("-")
        print(f"📝 QUERY {i}/{len(query_ids)} (ID: {qid})")
        print_separator("-")
        print(f"👤 USER: \"{query_data['query']}\"")
        print()
        
        # Prepare query with user ID (format: "I am USER_001. <query>")
        full_query = f"I am USER_001. {query_data['query']}"
        
        # Create initial state
        initial_state = GraphState(
            user_query=full_query,
            conversation_summary=None,
            turn_id=1
        )
        
        try:
            # Execute pipeline
            if silent:
                with SuppressOutput():
                    final_state = compiled_graph.invoke(initial_state)
            else:
                final_state = compiled_graph.invoke(initial_state)
            
            # Check router output
            router_output = final_state.get('router_output')
            
            if not router_output:
                print("❌ No router output returned")
                results.append({"id": qid, "status": "❌ FAIL", "error": "No router output"})
                continue
            
            # Print LLM-1 output
            print_llm1_detailed(router_output)
            
            # Validate clarity
            if router_output.clarity != "CLEAR":
                print(f"\n⚠️  Expected CLEAR but got: {router_output.clarity}")
                results.append({"id": qid, "status": "⚠️  WARN", "error": f"Clarity: {router_output.clarity}"})
                continue
            
            # Check execution result
            execution_result = final_state.get('execution_result')
            
            if not execution_result:
                print("\n❌ No execution result returned (LLM-2 did not execute)")
                results.append({"id": qid, "status": "❌ FAIL", "error": "No execution result"})
                continue
            
            # Print LLM-2 output
            print_llm2_detailed(execution_result)
            
            print(f"\n✅ PASS - Query {qid}")
            results.append({"id": qid, "status": "✅ PASS", "error": None})
            
        except json.JSONDecodeError as e:
            print_error_detailed(e, "JSON Parsing")
            results.append({"id": qid, "status": "❌ FAIL", "error": f"JSON parse: {str(e)[:100]}"})
            
        except Exception as e:
            print_error_detailed(e, "Pipeline Execution")
            results.append({"id": qid, "status": "❌ FAIL", "error": str(e)[:100]})
        
        # Wait between queries (except after last one)
        if i < len(query_ids):
            print(f"\n⏳ Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
    
    # Print summary
    print_separator()
    print("📊 TEST SUMMARY")
    print_separator()
    
    passed = sum(1 for r in results if "PASS" in r["status"])
    total = len(results)
    
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
    print()
    
    for r in results:
        print(f"   {r['status']} - Query #{r['id']}")
        if r['error']:
            print(f"        Error: {r['error']}")
    
    print_separator()
    
    return results


def test_vague_queries_no_rag_trn_categories(
    query_ids: List[int],
    wait_seconds: int = 10,
    silent: bool = True
):
    """
    Test VAGUE queries WITHOUT transaction category RAG lookup.
    
    Valid for UC-05 queries without category ambiguity (only timeframe/threshold ambiguity).
    
    Args:
        query_ids: List of query IDs from _new_QA_mapping.json
                   Valid IDs: [12, 13]
        wait_seconds: Seconds to wait between queries (default: 10)
        silent: Suppress iteration/tool execution logs (default: True)
    
    Examples:
        test_vague_queries_no_rag_trn_categories([12])       # Test query 12 only
        test_vague_queries_no_rag_trn_categories([12, 13])   # Test both
    """
    
    print_separator()
    print("🧪 TEST: VAGUE QUERIES WITHOUT RAG TRANSACTION CATEGORIES")
    print_separator()
    
    # Validate query IDs
    invalid_ids = [qid for qid in query_ids if qid not in VALID_VAGUE_QUERY_IDS]
    if invalid_ids:
        print(f"❌ Invalid query IDs: {invalid_ids}")
        print(f"   Valid IDs for VAGUE non-category queries: {VALID_VAGUE_QUERY_IDS}")
        return
    
    print(f"Testing {len(query_ids)} queries: {query_ids}")
    print(f"Wait time: {wait_seconds} seconds between queries")
    print(f"Silent mode: {'ON' if silent else 'OFF'}")
    
    # Load Q&A mapping
    print("\n📂 Loading Q&A mapping...")
    try:
        qa_mapping = load_qa_mapping()
        print("✅ Q&A mapping loaded")
    except FileNotFoundError as e:
        print(str(e))
        return
    
    # Build graph
    print("🔧 Building graph...")
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
            results.append({"id": qid, "status": "❌ FAIL", "error": "Not in mapping"})
            continue
        
        query_data = qa_mapping[qid_str]
        
        print_separator("-")
        print(f"📝 QUERY {i}/{len(query_ids)} (ID: {qid})")
        print_separator("-")
        print(f"👤 USER: \"{query_data['query']}\"")
        print()
        
        # Prepare query with user ID
        full_query = f"I am USER_001. {query_data['query']}"
        
        # Create initial state
        initial_state = GraphState(
            user_query=full_query,
            conversation_summary=None,
            turn_id=1
        )
        
        try:
            # Execute pipeline
            if silent:
                with SuppressOutput():
                    final_state = compiled_graph.invoke(initial_state)
            else:
                final_state = compiled_graph.invoke(initial_state)
            
            # Check router output
            router_output = final_state.get('router_output')
            
            if not router_output:
                print("❌ No router output returned")
                results.append({"id": qid, "status": "❌ FAIL", "error": "No router output"})
                continue
            
            # Print LLM-1 output for VAGUE
            print_llm1_vague(router_output)
            
            # Validate clarity
            if router_output.clarity != "VAGUE":
                print(f"\n⚠️  Expected VAGUE but got: {router_output.clarity}")
                results.append({"id": qid, "status": "⚠️  WARN", "error": f"Clarity: {router_output.clarity}"})
                continue
            
            # For VAGUE queries, we expect clarifying question but NO execution result
            if router_output.clarifying_question:
                print(f"\n✅ Correctly identified as VAGUE with clarification")
            else:
                print(f"\n⚠️  VAGUE query but no clarifying question generated")
            
            # Verify expected missing info
            expected_missing = query_data.get('missing_info', [])
            actual_missing = router_output.missing_info
            
            print(f"\n   📋 Missing Info Validation:")
            print(f"      Expected: {expected_missing}")
            print(f"      Actual: {actual_missing}")
            
            if set(expected_missing).issubset(set(actual_missing)):
                print(f"      ✅ Missing info correctly identified")
            else:
                print(f"      ⚠️  Missing info mismatch")
            
            print(f"\n✅ PASS - Query {qid} (VAGUE detected)")
            results.append({"id": qid, "status": "✅ PASS", "error": None})
            
        except json.JSONDecodeError as e:
            print_error_detailed(e, "JSON Parsing")
            results.append({"id": qid, "status": "❌ FAIL", "error": f"JSON parse: {str(e)[:100]}"})
            
        except Exception as e:
            print_error_detailed(e, "Pipeline Execution")
            results.append({"id": qid, "status": "❌ FAIL", "error": str(e)[:100]})
        
        # Wait between queries (except after last one)
        if i < len(query_ids):
            print(f"\n⏳ Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
    
    # Print summary
    print_separator()
    print("📊 TEST SUMMARY")
    print_separator()
    
    passed = sum(1 for r in results if "PASS" in r["status"])
    total = len(results)
    
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
    print()
    
    for r in results:
        print(f"   {r['status']} - Query #{r['id']}")
        if r['error']:
            print(f"        Error: {r['error']}")
    
    print_separator()
    
    return results


# ═══════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════

"""
# Test CLEAR queries (no category RAG needed)
from tests.pipeline_tests import test_clear_queries_no_rag_trn_categories

# Test all 4 CLEAR non-category queries
test_clear_queries_no_rag_trn_categories([1, 2, 5, 6])

# Test just 2 queries
test_clear_queries_no_rag_trn_categories([1, 2], wait_seconds=5)

# Verbose mode (see all tool calls)
test_clear_queries_no_rag_trn_categories([6], silent=False)


# Test VAGUE queries (no category RAG needed)
from tests.pipeline_tests import test_vague_queries_no_rag_trn_categories

# Test both VAGUE non-category queries
test_vague_queries_no_rag_trn_categories([12, 13])

# Test just one
test_vague_queries_no_rag_trn_categories([12], wait_seconds=5)
"""