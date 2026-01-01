"""
LLM-1 Multi-Turn Tests
======================

Tests VAGUE query flow with proper multi-turn handling:
1. Turn 1: User sends VAGUE query → LLM-1 returns VAGUE + clarifying_question
2. Turn 2: User answers → LLM-1 receives full conversation history via raw_messages
3. Turn 2: LLM-1 returns CLEAR + summary_update
4. summary_update_node merges preferences into conversation_summary

Does NOT test RAG (that's for end-to-end tests).

Field names (current schema):
- core_use_cases (not core_query_categories)
- primary_use_case (not primary_category)
- uc_operations (not sub_categories)
"""

import json
from typing import Optional, List, Dict, Any

from schemas.router_models import (
    GraphState,
    ConversationSummary,
    RouterOutput,
    PreferenceEntry
)
from graph_definition import (
    router_node,
    summary_update_node
)


# ═══════════════════════════════════════════════════════════════════
# TEST DATA: VAGUE QUERIES
# ═══════════════════════════════════════════════════════════════════

VAGUE_QUERIES = [
    {
        "id": 11,
        "query": "What are my coffee shop expenses?",
        "missing_info": "timeframe",
        "user_answer": "Last month",
        "expected": {
            "turn1_clarity": "VAGUE",
            "turn2_clarity": "CLEAR",
            "turn2_core_use_cases": ["UC-02", "UC-03", "UC-04"],
            "turn2_primary_use_case": "UC-02",
            "summary_key": "time_window",
            "summary_value": "last_month"
        }
    },
    {
        "id": 12,
        "query": "Show me recent transactions",
        "missing_info": "timeframe",
        "user_answer": "Last 7 days",
        "expected": {
            "turn1_clarity": "VAGUE",
            "turn2_clarity": "CLEAR",
            "turn2_core_use_cases": ["UC-01", "UC-03"],
            "turn2_primary_use_case": "UC-03",
            "summary_key": "time_window",
            "summary_value": "last_7_days"
        }
    },
    {
        "id": 13,
        "query": "Show me large purchases",
        "missing_info": "amount threshold",
        "user_answer": "Above $100",
        "expected": {
            "turn1_clarity": "VAGUE",
            "turn2_clarity": "CLEAR",
            "turn2_core_use_cases": ["UC-01"],
            "turn2_primary_use_case": "UC-01",
            "summary_key": "amount_threshold_large",
            "summary_value": 100
        }
    },
    {
        "id": 14,
        "query": "Show me recent large purchases at coffee shops",
        "missing_info": "timeframe + threshold",
        "user_answer": "Last 30 days, above $50",
        "expected": {
            "turn1_clarity": "VAGUE",
            "turn2_clarity": "CLEAR",
            "turn2_core_use_cases": ["UC-01", "UC-03", "UC-04"],
            "turn2_primary_use_case": "UC-01",
            "summary_key": "multiple",  # Both time_window and amount_threshold_large
            "summary_value": "multiple"
        }
    },
    {
        "id": 15,
        "query": "What are my pet expenses?",
        "missing_info": "timeframe",
        "user_answer": "This year",
        "expected": {
            "turn1_clarity": "VAGUE",
            "turn2_clarity": "CLEAR",
            "turn2_core_use_cases": ["UC-02", "UC-03", "UC-04"],
            "turn2_primary_use_case": "UC-02",
            "summary_key": "time_window",
            "summary_value": "this_year"
        }
    }
]


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def print_section(title: str, char: str = "─", width: int = 100):
    """Print a section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_box(title: str, width: int = 98):
    """Print a boxed header."""
    print(f"\n┌{'─' * width}┐")
    print(f"│ {title.ljust(width - 1)}│")
    print(f"└{'─' * width}┘")


def validate_conversation_summary(
    summary: ConversationSummary,
    expected_key: str,
    expected_value: Any
) -> tuple[bool, list[str]]:
    """
    Validate that conversation_summary was updated correctly.
    
    Returns: (passed, list_of_errors)
    """
    errors = []
    
    if expected_key == "multiple":
        # Check both time_window and amount_threshold_large exist
        if summary.time_window is None:
            errors.append("time_window: expected to exist, but is None")
        if summary.amount_threshold_large is None:
            errors.append("amount_threshold_large: expected to exist, but is None")
    
    elif expected_key == "time_window":
        if summary.time_window is None:
            errors.append(f"time_window: expected to exist, but is None")
        elif hasattr(summary.time_window, 'value'):
            # PreferenceEntry - check value
            actual = summary.time_window.value
            if actual != expected_value:
                errors.append(f"time_window.value: expected '{expected_value}', got '{actual}'")
    
    elif expected_key == "amount_threshold_large":
        if summary.amount_threshold_large is None:
            errors.append(f"amount_threshold_large: expected to exist, but is None")
        elif hasattr(summary.amount_threshold_large, 'value'):
            actual = summary.amount_threshold_large.value
            if actual != expected_value:
                errors.append(f"amount_threshold_large.value: expected {expected_value}, got {actual}")
    
    return (len(errors) == 0, errors)


def print_conversation_summary(summary: ConversationSummary):
    """Print conversation summary in a readable format."""
    print("\n   📋 ConversationSummary:")
    
    if summary.time_window:
        val = summary.time_window.value if hasattr(summary.time_window, 'value') else summary.time_window
        print(f"      • time_window: {val}")
    else:
        print(f"      • time_window: None")
    
    if summary.amount_threshold_large:
        val = summary.amount_threshold_large.value if hasattr(summary.amount_threshold_large, 'value') else summary.amount_threshold_large
        print(f"      • amount_threshold_large: {val}")
    else:
        print(f"      • amount_threshold_large: None")
    
    if summary.category_preferences:
        print(f"      • category_preferences: {list(summary.category_preferences.keys())}")


# ═══════════════════════════════════════════════════════════════════
# MAIN TEST: VAGUE QUERIES MULTI-TURN FLOW
# ═══════════════════════════════════════════════════════════════════

def llm1_test_all_vague_queries_multiturn(
    num_examples_to_check: Optional[int] = None,
    query_ids: Optional[List[int]] = None
):
    """
    Test VAGUE queries with proper multi-turn flow.
    
    Args:
        num_examples_to_check: Number of random queries to test.
                               If None, tests all (unless query_ids specified).
        query_ids: List of specific query IDs to test (e.g., [11, 13]).
                   Takes precedence over num_examples_to_check.
    
    Multi-turn flow:
    1. Turn 1: User query (VAGUE) → LLM-1 returns clarifying_question
    2. Turn 2: Build raw_messages with conversation history
    3. Turn 2: LLM-1 processes with context → Returns CLEAR + summary_update
    4. summary_update_node merges into conversation_summary
    
    Usage:
        # 1: All queries (default)
        llm1_test_all_vague_queries_multiturn()
        
        # 2: Random N queries
        llm1_test_all_vague_queries_multiturn(num_examples_to_check=2)
        
        # 3: Specific queries by ID
        llm1_test_all_vague_queries_multiturn(query_ids=[11, 14, 15])
    """
    import random
    
    # Select queries to test (query_ids takes precedence)
    if query_ids:
        queries = [q for q in VAGUE_QUERIES if q['id'] in query_ids]
        print(f"📋 Testing {len(queries)} selected VAGUE queries: {query_ids}")
    elif num_examples_to_check is not None and num_examples_to_check < len(VAGUE_QUERIES):
        queries = random.sample(VAGUE_QUERIES, num_examples_to_check)
        print(f"🎲 Randomly selected {num_examples_to_check} out of {len(VAGUE_QUERIES)} VAGUE queries")
    else:
        queries = VAGUE_QUERIES
        print(f"📋 Testing all {len(VAGUE_QUERIES)} VAGUE queries")
    
    print("=" * 100)
    print("🧪 LLM-1 MULTI-TURN TEST: VAGUE QUERIES")
    print("=" * 100)
    print("\nEach conversation is INDEPENDENT with fresh ConversationSummary")
    print("=" * 100)
    
    results = []
    
    for vq in queries:
        print(f"\n{'=' * 100}")
        print(f"VAGUE Query #{vq['id']}: \"{vq['query']}\"")
        print(f"Missing: {vq['missing_info']}")
        print("=" * 100)
        
        validation_errors = []
        expected = vq['expected']
        
        # ════════════════════════════════════════════════════════════════
        # TURN 1: User sends VAGUE query
        # ════════════════════════════════════════════════════════════════
        print_box("TURN 1: User Query (VAGUE)")
        
        state = GraphState(
            user_query=vq['query'],
            conversation_summary=ConversationSummary(),  # Fresh!
            turn_id=1,
            raw_messages=[]  # Empty initially
        )
        
        print(f"\n👤 User: \"{vq['query']}\"")
        print(f"📝 ConversationSummary: (empty - fresh conversation)")
        
        # Call LLM-1
        state = router_node(state)
        turn1_output = state.router_output
        
        # ════════════════════════════════════════════════════════════════
        # TURN 1 OUTPUT: Clarifying question
        # ════════════════════════════════════════════════════════════════
        print_box("TURN 1: LLM-1 Response (Clarification)")
        
        print(f"\n🤖 LLM-1 Classification:")
        print(f"   clarity: {turn1_output.clarity}")
        print(f"   clarifying_question: {turn1_output.clarifying_question}")
        if turn1_output.missing_info:
            print(f"   missing_info: {turn1_output.missing_info}")
        
        # Validate Turn 1
        print_section("TURN 1 VALIDATION")
        
        if turn1_output.clarity != "VAGUE":
            validation_errors.append(f"Turn 1 clarity: expected VAGUE, got {turn1_output.clarity}")
            print(f"   ❌ clarity: expected VAGUE, got {turn1_output.clarity}")
        else:
            print(f"   ✅ clarity: VAGUE")
        
        if not turn1_output.clarifying_question:
            validation_errors.append("Turn 1: clarifying_question is empty")
            print(f"   ❌ clarifying_question: expected non-empty")
        else:
            print(f"   ✅ clarifying_question: present")
        
        # ════════════════════════════════════════════════════════════════
        # TURN 2: User answers clarification
        # ════════════════════════════════════════════════════════════════
        print_box("TURN 2: User Clarification")
        
        print(f"\n👤 User: \"{vq['user_answer']}\"")
        
        # *** CRITICAL: Build conversation history for LLM-1 ***
        # LLM-1 needs to see the full context to understand that
        # "Last month" is an answer to a clarifying question
        state.raw_messages = [
            {"role": "user", "content": vq['query']},
            {"role": "assistant", "content": turn1_output.clarifying_question},
            {"role": "user", "content": vq['user_answer']}
        ]
        state.user_query = vq['user_answer']
        state.turn_id = 2
        
        # ════════════════════════════════════════════════════════════════
        # TURN 2: LLM-1 processes with context
        # ════════════════════════════════════════════════════════════════
        print_box("TURN 2: LLM-1 Final Response (CLEAR)")
        
        # Call LLM-1 with conversation history
        state = router_node(state)
        turn2_output = state.router_output
        
        print(f"\n🤖 LLM-1 Final Classification:")
        print(f"   clarity: {turn2_output.clarity}")
        print(f"   core_use_cases: {turn2_output.core_use_cases}")
        print(f"   primary_use_case: {turn2_output.primary_use_case}")
        print(f"   complexity_axes: {turn2_output.complexity_axes}")
        print(f"   needed_tools: {turn2_output.needed_tools}")
        
        if turn2_output.summary_update:
            print(f"   summary_update: {turn2_output.summary_update}")
        else:
            print(f"   summary_update: None ⚠️")
        
        # ════════════════════════════════════════════════════════════════
        # Apply summary_update to conversation_summary
        # ════════════════════════════════════════════════════════════════
        state = summary_update_node(state)
        
        print_section("CONVERSATION SUMMARY UPDATE")
        print_conversation_summary(state.conversation_summary)
        
        # ════════════════════════════════════════════════════════════════
        # VALIDATION
        # ════════════════════════════════════════════════════════════════
        print_section("TURN 2 VALIDATION")
        
        # 1. Check clarity is CLEAR
        if turn2_output.clarity != "CLEAR":
            validation_errors.append(f"Turn 2 clarity: expected CLEAR, got {turn2_output.clarity}")
            print(f"   ❌ clarity: expected CLEAR, got {turn2_output.clarity}")
        else:
            print(f"   ✅ clarity: CLEAR")
        
        # 2. Check core_use_cases (order-independent)
        expected_ucs = sorted(expected['turn2_core_use_cases'])
        actual_ucs = sorted(turn2_output.core_use_cases)
        if expected_ucs != actual_ucs:
            validation_errors.append(f"core_use_cases: expected {expected_ucs}, got {actual_ucs}")
            print(f"   ❌ core_use_cases: expected {expected_ucs}, got {actual_ucs}")
        else:
            print(f"   ✅ core_use_cases: {actual_ucs}")
        
        # 3. Check primary_use_case
        if turn2_output.primary_use_case != expected['turn2_primary_use_case']:
            validation_errors.append(f"primary_use_case: expected {expected['turn2_primary_use_case']}, got {turn2_output.primary_use_case}")
            print(f"   ❌ primary_use_case: expected {expected['turn2_primary_use_case']}, got {turn2_output.primary_use_case}")
        else:
            print(f"   ✅ primary_use_case: {turn2_output.primary_use_case}")
        
        # 4. Check summary_update was generated
        if turn2_output.summary_update is None:
            validation_errors.append("summary_update: expected non-null, got None")
            print(f"   ❌ summary_update: expected non-null, got None")
        else:
            print(f"   ✅ summary_update: generated")
        
        # 5. Check conversation_summary was updated
        summary_passed, summary_errors = validate_conversation_summary(
            state.conversation_summary,
            expected['summary_key'],
            expected['summary_value']
        )
        
        if summary_passed:
            print(f"   ✅ conversation_summary: correctly updated")
        else:
            for err in summary_errors:
                validation_errors.append(err)
                print(f"   ❌ {err}")
        
        # ════════════════════════════════════════════════════════════════
        # RESULT
        # ════════════════════════════════════════════════════════════════
        passed = len(validation_errors) == 0
        
        if passed:
            print(f"\n✅ Query #{vq['id']}: PASSED")
        else:
            print(f"\n❌ Query #{vq['id']}: FAILED ({len(validation_errors)} errors)")
            for err in validation_errors:
                print(f"   • {err}")
        
        # ════════════════════════════════════════════════════════════════
        # FULL OUTPUT (for debugging)
        # ════════════════════════════════════════════════════════════════
        print_section("FULL LLM-1 OUTPUT (Turn 2)")
        print(json.dumps(turn2_output.model_dump(mode='json'), indent=2, default=str))
        
        results.append({
            "query_id": vq['id'],
            "query": vq['query'],
            "turn1_clarity": turn1_output.clarity,
            "turn2_clarity": turn2_output.clarity,
            "summary_update_generated": turn2_output.summary_update is not None,
            "conversation_summary_valid": summary_passed,
            "errors": validation_errors,
            "passed": passed
        })
    
    # ════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 100)
    
    passed_count = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"\nTotal: {total} | Passed: {passed_count} | Failed: {total - passed_count}")
    print(f"Success Rate: {(passed_count/total)*100:.1f}%")
    
    print("\nResults by Query:")
    for r in results:
        status = "✅" if r['passed'] else "❌"
        flow = f"{r['turn1_clarity']} → {r['turn2_clarity']}"
        summary = "✓" if r['conversation_summary_valid'] else "✗"
        update = "✓" if r['summary_update_generated'] else "✗"
        print(f"  {status} #{r['query_id']}: {flow} | summary_update:{update} | conv_summary:{summary}")
    
    if total - passed_count > 0:
        print("\nFailed Queries:")
        for r in results:
            if not r['passed']:
                print(f"\n  Query #{r['query_id']}: \"{r['query']}\"")
                for err in r['errors']:
                    print(f"    ❌ {err}")
    
    print("=" * 100)
    
    return results


# ═══════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════
"""
USAGE IN JUPYTER:

import tests.llm1_tests as tst_llm1

# Test VAGUE queries 
# 1: All queries (default)
vague_results = tst_llm1.llm1_test_all_vague_queries_multiturn()

# 2: Random N queries
vague_results = tst_llm1.llm1_test_all_vague_queries_multiturn(num_examples_to_check=2)

# 3: Specific queries by ID
vague_results = tst_llm1.llm1_test_all_vague_queries_multiturn(query_ids=[11, 14, 15])
"""