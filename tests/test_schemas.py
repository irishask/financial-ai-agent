"""
Schema Validation Tests
=======================

Tests that all Pydantic models in router_models.py can be instantiated correctly.
This is a smoke test for schema changes - NOT functional testing.

Usage:
    import tests.test_schemas as ts
    ts.test_all_schemas()
"""

from typing import Dict, Any


def test_all_schemas():
    """Run all schema validation tests."""
    
    print("=" * 80)
    print("🧪 SCHEMA VALIDATION TESTS")
    print("=" * 80)
    
    results = []
    
    # ─────────────────────────────────────────────────────────────────
    # Test 1: PreferenceEntry
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import PreferenceEntry
        
        pref = PreferenceEntry(
            value="last_30_days",
            source="user_defined",
            turn_id=2,
            original_query="Show me recent transactions",
            previous_value=None,
            previous_turn_id=None
        )
        
        assert pref.value == "last_30_days"
        assert pref.source == "user_defined"
        assert pref.turn_id == 2
        assert pref.previous_value is None
        
        print("\n✅ PreferenceEntry")
        print(f"   Fields: value, source, turn_id, original_query, previous_value, previous_turn_id")
        results.append(("PreferenceEntry", True, None))
        
    except Exception as e:
        print(f"\n❌ PreferenceEntry: {e}")
        results.append(("PreferenceEntry", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 2: ResolvedDates
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import ResolvedDates
        from datetime import date
        
        dates = ResolvedDates(
            start_date=date(2025, 11, 1),
            end_date=date(2025, 11, 30),
            interpretation="November 2025"
        )
        
        assert dates.start_date == date(2025, 11, 1)
        assert dates.end_date == date(2025, 11, 30)
        
        print("\n✅ ResolvedDates")
        print(f"   Fields: start_date, end_date, interpretation")
        results.append(("ResolvedDates", True, None))
        
    except Exception as e:
        print(f"\n❌ ResolvedDates: {e}")
        results.append(("ResolvedDates", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 3: ConversationSummary
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import ConversationSummary, PreferenceEntry
        
        summary = ConversationSummary(
            time_window=PreferenceEntry(value="last_month", source="user_defined"),
            amount_threshold_large=PreferenceEntry(value=100, source="user_defined"),
            account_scope=None,
            category_preferences={}
        )
        
        assert summary.time_window.value == "last_month"
        assert summary.amount_threshold_large.value == 100
        
        print("\n✅ ConversationSummary")
        print(f"   Fields: time_window, amount_threshold_large, account_scope, category_preferences")
        results.append(("ConversationSummary", True, None))
        
    except Exception as e:
        print(f"\n❌ ConversationSummary: {e}")
        results.append(("ConversationSummary", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 4: RouterOutput
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import RouterOutput
        
        router_out = RouterOutput(
            clarity="CLEAR",
            core_use_cases=["UC-02", "UC-03", "UC-04"],
            uc_operations={
                "UC-01": [],
                "UC-02": ["sum_spending"],
                "UC-03": ["temporal_filter"],
                "UC-04": ["category_mapping"],
                "UC-05": []
            },
            primary_use_case="UC-02",
            complexity_axes=["temporal", "category"],
            needed_tools=["query_transactions"],
            clarifying_question=None,
            missing_info=[],
            summary_update={"time_window": "last_month"},
            uc_confidence="high",
            clarity_reason="Complete query",
            router_notes="Test"
        )
        
        assert router_out.clarity == "CLEAR"
        assert "UC-02" in router_out.core_use_cases
        assert router_out.primary_use_case == "UC-02"
        
        print("\n✅ RouterOutput")
        print(f"   Fields: clarity, core_use_cases, uc_operations, primary_use_case, ...")
        print(f"   Fields: complexity_axes, needed_tools, clarifying_question, missing_info, ...")
        print(f"   Fields: summary_update, resolved_dates, resolved_trn_categories, resolved_amount_threshold")
        results.append(("RouterOutput", True, None))
        
    except Exception as e:
        print(f"\n❌ RouterOutput: {e}")
        results.append(("RouterOutput", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 5: DataSources
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import DataSources
        
        ds = DataSources(
            tables_used=["transactions"],
            fields_accessed=["amount", "date", "categoryGroupId"],
            filters_applied=["date >= 2025-11-01"],
            aggregations_used=["SUM(amount)"]
        )
        
        assert "transactions" in ds.tables_used
        assert "amount" in ds.fields_accessed
        
        print("\n✅ DataSources")
        print(f"   Fields: tables_used, fields_accessed, filters_applied, aggregations_used")
        results.append(("DataSources", True, None))
        
    except Exception as e:
        print(f"\n❌ DataSources: {e}")
        results.append(("DataSources", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 6: ClarificationStep
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import ClarificationStep
        
        step = ClarificationStep(
            question="What time period?",
            user_answer="Last month",
            turn_id=1
        )
        
        assert step.question == "What time period?"
        assert step.user_answer == "Last month"
        
        print("\n✅ ClarificationStep")
        print(f"   Fields: question, user_answer, turn_id")
        results.append(("ClarificationStep", True, None))
        
    except Exception as e:
        print(f"\n❌ ClarificationStep: {e}")
        results.append(("ClarificationStep", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 7: BackofficeLog
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import BackofficeLog, DataSources
        
        log = BackofficeLog(
            user_query="How much on groceries last month?",
            resolved_query={
                "original": "How much on groceries last month?",
                "resolved_intent": "SUM spending on CG10000 for October 2025",
                "interpretations": {"groceries": "CG10000", "last month": "Oct 2025"}
            },
            answer="You spent $523.45 on groceries last month.",
            analysis={"total": 523.45, "transaction_count": 12},
            reasoning_steps=[
                "Step 1: Mapped 'groceries' to CG10000",
                "Step 2: Resolved 'last month' to October 2025",
                "Step 3: Summed transactions"
            ],
            data_sources=DataSources(
                tables_used=["transactions"],
                fields_accessed=["amount", "categoryGroupId"],
                filters_applied=["categoryGroupId = CG10000"],
                aggregations_used=["SUM(amount)"]
            ),
            transactions_analyzed=12,
            preferences_used={},
            clarification_history=[],
            confidence="high",
            rag_used=False
        )
        
        assert log.user_query == "How much on groceries last month?"
        assert log.resolved_query is not None
        assert log.transactions_analyzed == 12
        
        print("\n✅ BackofficeLog")
        print(f"   Fields: user_query, resolved_query, answer, analysis, reasoning_steps, ...")
        print(f"   Fields: data_sources, transactions_analyzed, preferences_used, ...")
        print(f"   Fields: clarification_history, confidence, rag_used, router_output_snapshot")
        results.append(("BackofficeLog", True, None))
        
    except Exception as e:
        print(f"\n❌ BackofficeLog: {e}")
        results.append(("BackofficeLog", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 8: ExecutionResult
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import ExecutionResult, BackofficeLog, DataSources
        
        result = ExecutionResult(
            final_answer="You spent $523.45 on groceries last month.",
            backoffice_log=BackofficeLog(
                user_query="How much on groceries?",
                answer="$523.45",
                data_sources=DataSources()
            )
        )
        
        assert result.final_answer == "You spent $523.45 on groceries last month."
        assert result.backoffice_log is not None
        
        print("\n✅ ExecutionResult")
        print(f"   Fields: final_answer, backoffice_log")
        results.append(("ExecutionResult", True, None))
        
    except Exception as e:
        print(f"\n❌ ExecutionResult: {e}")
        results.append(("ExecutionResult", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 9: GraphState
    # ─────────────────────────────────────────────────────────────────
    try:
        from schemas.router_models import GraphState, ConversationSummary
        
        state = GraphState(
            user_query="Show me recent transactions",
            conversation_summary=ConversationSummary(),
            router_output=None,
            execution_result=None,
            messages_to_user=[],
            turn_id=1,
            session_id="test-session",
            raw_messages=[]
        )
        
        assert state.user_query == "Show me recent transactions"
        assert state.turn_id == 1
        assert state.raw_messages == []
        
        print("\n✅ GraphState")
        print(f"   Fields: user_query, conversation_summary, router_output, execution_result, ...")
        print(f"   Fields: messages_to_user, turn_id, session_id, raw_messages")
        results.append(("GraphState", True, None))
        
    except Exception as e:
        print(f"\n❌ GraphState: {e}")
        results.append(("GraphState", False, str(e)))
    
    # ─────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 SCHEMA VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {total - passed}")
    
    if passed == total:
        print("\n✅ ALL SCHEMAS VALID!")
    else:
        print("\n❌ Failed schemas:")
        for name, ok, error in results:
            if not ok:
                print(f"   • {name}: {error}")
    
    print("=" * 80)
    
    return results


# ═══════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════
"""
USAGE IN JUPYTER:

import tests.test_schemas as ts
ts.test_all_schemas()
"""