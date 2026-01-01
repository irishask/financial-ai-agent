"""
Unified Back-Office Logging Model

Centralized logging structure that accumulates information from:
- LLM-1 (Router & Clarifier): Routing decisions, UC classification, clarity detection
- LLM-2 (Executor): Query execution, data access, reasoning steps

This creates a complete audit trail for compliance, observability, and debugging.
"""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


###########################################################################################
# UNIFIED BACK-OFFICE LOG MODEL
###########################################################################################

class BackofficeLog(BaseModel):
    """
    Centralized back-office log that accumulates information from both LLM-1 and LLM-2.
    
    This creates a complete audit trail showing:
    - How the query was classified (LLM-1)
    - How the query was executed (LLM-2)
    - What data was accessed
    - What reasoning was used
    - What preferences influenced the result
    
    Shared in GraphState and updated by different nodes:
    - router_node: Adds LLM-1 routing information
    - executor_node: Adds LLM-2 execution information
    - Final log contains complete pipeline trace
    """
    
    # =========================================================================
    # METADATA (Set at start)
    # =========================================================================
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this query was processed"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Session identifier to group related queries"
    )
    
    turn_id: int = Field(
        0,
        description="Turn number within the session"
    )
    
    user_query: str = Field(
        ...,
        description="Original user query that triggered this pipeline"
    )
    
    # =========================================================================
    # LLM-1 ROUTING INFORMATION (Added by router_node)
    # =========================================================================
    
    routing: Optional[Dict[str, Any]] = Field(
        None,
        description="""
        LLM-1 routing decision including:
        - clarity: CLEAR or VAGUE
        - core_use_cases: List of involved UCs
        - primary_use_case: Dominant UC
        - uc_operations: Specific operations per UC
        - complexity_axes: Which complexity challenges involved
        - resolved_dates: Pre-calculated date range (start_date, end_date)
        - resolved_trn_categories: Pre-resolved category IDs from RAG
        - resolved_amount_threshold: Pre-resolved amount threshold
        - uc_confidence: Router's confidence
        - clarity_reason: Why classified as CLEAR/VAGUE
        - router_notes: Internal notes
        - llm1_latency_ms: Time taken by LLM-1
        """
    )
    
    clarification: Optional[Dict[str, Any]] = Field(
        None,
        description="""
        UC-05 clarification information (if query was VAGUE):
        - clarifying_question: Question asked to user
        - missing_info: What was missing
        - user_response: User's clarification
        - preferences_updated: Which preferences were set
        """
    )
    
    # =========================================================================
    # LLM-2 EXECUTION INFORMATION (Added by executor_node)
    # =========================================================================
    
    answer: Optional[str] = Field(
        None,
        description="Final customer-facing answer (simple, conversational)"
    )
    
    analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="""
        Key metrics and computed values:
        - Totals, averages, counts
        - Comparisons and deltas
        - Percentage changes
        - Period-specific breakdowns
        """
    )
    
    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="""
        Step-by-step explanation of execution:
        - What was understood from the query
        - How categories were mapped
        - How time windows were interpreted
        - What filters were applied
        - What calculations were performed
        - How the conclusion was reached
        """
    )
    
    data_sources: Optional[Dict[str, Any]] = Field(
        None,
        description="""
        Complete data lineage:
        - tables_used: Which database tables
        - fields_accessed: Which columns/fields
        - filters_applied: Exact filter conditions with values
        - joins_performed: How tables were connected
        - aggregations_used: Which math operations (SUM, AVG, COUNT, etc.)
        """
    )
    
    transactions_analyzed: int = Field(
        0,
        description="Number of transaction rows processed"
    )
    
    # =========================================================================
    # PREFERENCES & RAG (Can be added by either LLM-1 or LLM-2)
    # =========================================================================
    
    preferences_used: Optional[Dict[str, Any]] = Field(
        None,
        description="""
        Which conversation_summary preferences influenced this answer:
        - time_window: How "recent" was interpreted
        - amount_threshold_large: What counts as "large"
        - category_preferences: Category scope choices
        Each with: value, source, turn_id, original_query
        """
    )
    
    rag_evidence: Optional[Dict[str, Any]] = Field(
        None,
        description="""
        Category RAG search evidence (when UC-04 involved):
        - query: Category phrase searched
        - top_candidates: Similarity scores for top matches
        - selected_category: Chosen category and why
        - decision: How ambiguity was resolved (if any)
        """
    )
    
    # =========================================================================
    # QUALITY & PERFORMANCE METRICS
    # =========================================================================
    
    confidence: Literal["high", "medium", "low"] = Field(
        "medium",
        description="Overall confidence in the answer (considers routing + execution)"
    )
    
    latency: Optional[Dict[str, float]] = Field(
        None,
        description="""
        Performance breakdown in milliseconds:
        - llm1_router_ms: LLM-1 routing time
        - llm2_executor_ms: LLM-2 execution time
        - rag_retrieval_ms: Category RAG time (if used)
        - tool_execution_ms: Tool calls time
        - total_pipeline_ms: End-to-end time
        """
    )
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""
        Any errors encountered during processing:
        - stage: Where error occurred (router, executor, tool)
        - error_type: Type of error
        - message: Error message
        - recovered: Whether error was handled gracefully
        """
    )


###########################################################################################
# HELPER FUNCTIONS FOR UPDATING BACKOFFICE LOG
###########################################################################################

def initialize_backoffice_log(user_query: str, session_id: str, turn_id: int) -> BackofficeLog:
    """
    Initialize a new back-office log at the start of query processing.
    """
    return BackofficeLog(
        user_query=user_query,
        session_id=session_id,
        turn_id=turn_id,
        timestamp=datetime.now()
    )


def add_routing_info(
    log: BackofficeLog,
    router_output: Dict[str, Any],
    llm1_latency_ms: float
) -> BackofficeLog:
    """
    Add LLM-1 routing information to the log.
    Called by router_node after LLM-1 completes.
    """
    log.routing = {
        "clarity": router_output.get("clarity"),
        "core_use_cases": router_output.get("core_use_cases"),
        "primary_use_case": router_output.get("primary_use_case"),
        "uc_operations": router_output.get("uc_operations"),
        "resolved_dates": router_output.get("resolved_dates"),
        "resolved_trn_categories": router_output.get("resolved_trn_categories"),
        "resolved_amount_threshold": router_output.get("resolved_amount_threshold"),
        "complexity_axes": router_output.get("complexity_axes"),
        "uc_confidence": router_output.get("uc_confidence"),
        "clarity_reason": router_output.get("clarity_reason"),
        "router_notes": router_output.get("router_notes"),
        "llm1_latency_ms": llm1_latency_ms
    }
    return log


def add_clarification_info(
    log: BackofficeLog,
    clarifying_question: str,
    missing_info: List[str],
    user_response: Optional[str] = None,
    preferences_updated: Optional[Dict[str, Any]] = None
) -> BackofficeLog:
    """
    Add UC-05 clarification information to the log.
    Called when handling VAGUE queries.
    """
    log.clarification = {
        "clarifying_question": clarifying_question,
        "missing_info": missing_info,
        "user_response": user_response,
        "preferences_updated": preferences_updated
    }
    return log


def add_execution_info(
    log: BackofficeLog,
    executor_result: Dict[str, Any],
    llm2_latency_ms: float
) -> BackofficeLog:
    """
    Add LLM-2 execution information to the log.
    Called by executor_node after LLM-2 completes.
    """
    log.answer = executor_result.get("answer")
    log.analysis = executor_result.get("analysis")
    log.reasoning_steps = executor_result.get("reasoning_steps", [])
    log.data_sources = executor_result.get("data_sources")
    log.transactions_analyzed = executor_result.get("transactions_analyzed", 0)
    log.confidence = executor_result.get("confidence", "medium")
    
    # Update latency info
    if log.latency is None:
        log.latency = {}
    log.latency["llm2_executor_ms"] = llm2_latency_ms
    
    return log


def add_preferences_used(
    log: BackofficeLog,
    conversation_summary: Dict[str, Any]
) -> BackofficeLog:
    """
    Add information about which preferences influenced this answer.
    Can be called by executor_node when conversation_summary was used.
    """
    log.preferences_used = conversation_summary
    return log


def add_rag_evidence(
    log: BackofficeLog,
    rag_query: str,
    top_candidates: List[Dict[str, Any]],
    selected_category: Dict[str, Any],
    decision: str
) -> BackofficeLog:
    """
    Add Category RAG search evidence to the log.
    Called when UC-04 category mapping is performed.
    """
    log.rag_evidence = {
        "query": rag_query,
        "top_candidates": top_candidates,
        "selected_category": selected_category,
        "decision": decision
    }
    return log


def calculate_total_latency(log: BackofficeLog) -> BackofficeLog:
    """
    Calculate total pipeline latency from individual components.
    Called at the end of processing.
    """
    if log.latency:
        total = sum(
            v for k, v in log.latency.items() 
            if k != "total_pipeline_ms" and isinstance(v, (int, float))
        )
        log.latency["total_pipeline_ms"] = total
    return log


###########################################################################################
# USAGE EXAMPLE IN GRAPH NODES
###########################################################################################

"""
EXAMPLE: How to use BackofficeLog in LangGraph nodes

# =========================================================================
# In router_node:
# =========================================================================

def router_node(state: GraphState) -> GraphState:
    import time
    
    # Initialize log if not exists
    if state.backoffice_log is None:
        state.backoffice_log = initialize_backoffice_log(
            user_query=state.user_query,
            session_id=state.session_id,
            turn_id=state.turn_id
        )
    
    # Call LLM-1
    start_time = time.time()
    router_output_dict = call_llm1(state.user_query, state.conversation_summary)
    llm1_latency_ms = (time.time() - start_time) * 1000
    
    # Add routing info to log
    state.backoffice_log = add_routing_info(
        log=state.backoffice_log,
        router_output=router_output_dict,
        llm1_latency_ms=llm1_latency_ms
    )
    
    # If VAGUE, add clarification info
    if router_output_dict["clarity"] == "VAGUE":
        state.backoffice_log = add_clarification_info(
            log=state.backoffice_log,
            clarifying_question=router_output_dict["clarifying_question"],
            missing_info=router_output_dict["missing_info"]
        )
    
    return state


# =========================================================================
# In executor_node:
# =========================================================================

def executor_node(state: GraphState) -> GraphState:
    import time
    
    # Call LLM-2
    start_time = time.time()
    executor_result = call_llm2(
        state.user_query,
        state.router_output,
        state.conversation_summary
    )
    llm2_latency_ms = (time.time() - start_time) * 1000
    
    # Add execution info to log
    state.backoffice_log = add_execution_info(
        log=state.backoffice_log,
        executor_result=executor_result,
        llm2_latency_ms=llm2_latency_ms
    )
    
    # Add preferences used (if any)
    if state.conversation_summary:
        state.backoffice_log = add_preferences_used(
            log=state.backoffice_log,
            conversation_summary=state.conversation_summary.dict()
        )
    
    # Calculate total latency
    state.backoffice_log = calculate_total_latency(state.backoffice_log)
    
    return state


# =========================================================================
# At the end of the graph:
# =========================================================================

# The final state.backoffice_log contains COMPLETE audit trail:
# - LLM-1 routing decisions
# - UC-05 clarifications (if any)
# - LLM-2 execution details
# - Data lineage
# - Performance metrics
# - Everything in ONE place!

# Save to database/logs:
save_backoffice_log(state.backoffice_log)
"""