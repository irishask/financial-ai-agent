"""
Core Pydantic models for routing, preferences, and execution logs in the financial AI agent.
"""
from datetime import date
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class RouterOutput(BaseModel):
    """
    Single preference entry (value + source/turn/query metadata) used inside ConversationSummary
    
    Structured output of LLM-1 (Router & Clarifier).
    This object drives:
    - CLEAR vs VAGUE branching in LangGraph
    - UC type handling in LLM-2
    - UC-05 clarifications
    - Preference updates via `summary_update`
    
    UPDATED: Now supports multi-category queries with both old and new fields for compatibility.
    """
    # --- Core routing fields (UPDATED for multi-category support) ---
    clarity: Literal["CLEAR", "VAGUE"] = Field(
        ...,
        description="Whether the query can be executed as-is (CLEAR) or needs clarification (VAGUE).",
    )
    

    
    # multi-category support
    core_use_cases: List[Literal["UC-01", "UC-02", "UC-03", "UC-04", "UC-05"]] = Field(
        default_factory=list,
        description="List of all UC categories involved in this query (e.g., ['UC-02', 'UC-03', 'UC-04']).",
    )
    uc_operations: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "UC-01": [],
            "UC-02": [], 
            "UC-03": [],
            "UC-04": [],
            "UC-05": []
        },
        description=(
            "Nested structure with all UC categories as keys and their subtypes as arrays. "
            "Empty arrays for unused categories. Example: "
            "{'UC-01': [], 'UC-02': ['sum_spending'], 'UC-03': ['temporal_interpretation'], "
            "'UC-04': ['category_mapping'], 'UC-05': []}"
        ),
    )
    primary_use_case: Optional[Literal["UC-01", "UC-02", "UC-03", "UC-04", "UC-05"]] = Field(
        None,
        description="The dominant/primary UC category that should drive execution logic.",
    )
    
    complexity_axes: List[Literal["temporal", "category", "ambiguity"]] = Field(
        default_factory=list,
        description="Which complexity dimensions are involved in this query.",
    )
    needed_tools: List[str] = Field(
        default_factory=list,
        description="Logical tool names that LLM-2 is expected to use, e.g. => ['query_transactions'].",
    )
    # --- VAGUE-mode specific fields (UC-05 clarifications) ---
    clarifying_question: Optional[str] = Field(
        None,
        description="ONE concise clarification question to ask the user when clarity == 'VAGUE'.",
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description=(
            "List of missing pieces of information, e.g. "
            "['time_window', 'amount_threshold_large', 'account_scope', "
            "'trn_category_scope', 'comparison_baseline']."
        ),
    )
    # --- Preference updates (for conversation_summary, applied in a separate node) ---
    summary_update: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional structured update describing new/overridden user preferences "
            "to be merged into conversation_summary by a dedicated node."
        ),
    )
    # --- Debug / observability metadata ---
    uc_confidence: Literal["high", "medium", "low"] = Field(
        "medium",
        description="Router's confidence in core_query_type and overall interpretation.",
    )
    clarity_reason: Optional[str] = Field(
        None,
        description="Short natural-language explanation for why the query is CLEAR or VAGUE.",
    )
    router_notes: Optional[str] = Field(
        None,
        description="Free-text notes from LLM-1 for internal logs / debugging (not user-facing).",
    )
   
    # Temporal resolution (LLM-1 calculates exact dates) ---
    resolved_dates: Optional["ResolvedDates"] = Field(
        None,
        description=(
            "Pre-calculated date range when the query involves temporal filtering. "
            "LLM-2 should use these dates directly in query_transactions. "
            "Example: 'last 14 days' → {start_date: '2025-11-09', end_date: '2025-11-22', "
            "interpretation: 'last 14 days from Nov 22, 2025'}"
        ),
    )

    # --- NEW: RAG lookup => retrieve Transaction Category resolution
    resolved_trn_categories: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Pre-resolved transaction categories from map_category_from_kb_with_rag_lookup. "
            "LLM-2 should use these category IDs directly in query_transactions. "
            "Example: [{'user_term': 'dining', 'categoryGroupId': 'CG800', "
            "'categoryGroupName': 'Dining', 'confidence': 0.95}]"
        ),
    )

    # --- NEW: Amount threshold resolution
    resolved_amount_threshold: Optional[float] = Field(
        None,
        description=(
            "Pre-resolved amount threshold for queries like 'large purchases'. "
            "LLM-2 should use this directly in query_transactions min_amount filter. "
            "Example: 100.0 (meaning transactions >= $100)"
        ),
    )
#############################################################################################
#############################################################################################

class PreferenceEntry(BaseModel):
    """
    Represents a single user/system preference with metadata (who set it, when, and from which query).
    Used for time windows, thresholds, scopes, and category-specific settings.
    """

    value: Any = Field(
        ...,
        description="The actual preference value, e.g. 'last_30_days', 1000, "
                    "'all_accounts', 'cafes_only', ['FRI', 'SAT'], etc.",
    )
    source: Literal["system_default", "user_defined", "user_override"] = Field(
        ...,
        description="Who/what set this value: a system default, user's first definition, "
                    "or a user override of an existing value.",
    )
    turn_id: Optional[int] = Field(
        None,
        description="Conversation turn in which this value was last set/updated.",
    )
    original_query: Optional[str] = Field(
        None,
        description="The exact user query that defined/overrode this preference.",
    )

    # NEW FIELDS for override tracking:
    previous_value: Optional[Any] = Field(
        None,
        description="The previous value before this preference was overridden. "
                    "None if this is the first time the preference is set.",
    )
    previous_turn_id: Optional[int] = Field(
        None,
        description="The turn_id when the previous value was set. "
                    "None if this is the first time the preference is set.",
    )


    
#############################################################################################

class ResolvedDates(BaseModel):
    """
    Pre-calculated date ranges resolved by LLM-1 for temporal queries.
    LLM-2 uses these directly instead of interpreting temporal phrases.
    """
    
    start_date: Optional[date] = Field(
        None,
        description="Start date (inclusive) for temporal filtering"
    )
    end_date: Optional[date] = Field(
        None,
        description="End date (inclusive) for temporal filtering"
    )
    interpretation: Optional[str] = Field(
        None,
        description="Human-readable explanation of how dates were calculated (for logging)"
    )

#############################################################################################

class ConversationSummary(BaseModel):
    """
    Session-scoped summary of user preferences shared by LLM-1 (router) and LLM-2 (executor).
    Shared between LLM-1 (Router) and LLM-2 (Executor) as a single source of truth.
    """

    time_window: Optional[PreferenceEntry] = Field(
        None,
        description="Default time window for vague queries like 'recent' if not specified explicitly.",
    )
    amount_threshold_large: Optional[PreferenceEntry] = Field(
        None,
        description="What counts as a 'large' transaction by default (e.g. 1000).",
    )
    account_scope: Optional[PreferenceEntry] = Field(
        None,
        description="Which accounts to consider by default, e.g. 'all_accounts', 'primary_account_only'.",
    )
    category_preferences: Dict[str, PreferenceEntry] = Field(
        default_factory=dict,
        description=(
            "Category-specific preferences, keyed by a descriptive name, e.g.: "
            "'coffee_spending_scope' -> cafes_only, "
            "'weekend_definition' -> ['FRI', 'SAT'], "
            "'dining_includes_fast_food' -> True."
        ),
    )

#############################################################################################

class DataSources(BaseModel):
    """Structured description of which data was used to answer the query."""

    tables_used: List[str] = Field(
        default_factory=list,
        description="Logical table names used in this answer, e.g. ['transactions'].",
    )
    fields_accessed: List[str] = Field(
        default_factory=list,
        description="Field/column names accessed, e.g. ['amount', 'date', 'categoryGroupId'].",
    )
    filters_applied: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable descriptions of filters applied, e.g. "
            "'date between 2024-03-01 and 2024-03-31', "
            "'categoryGroupId in [CG800]'."
        ),
    )
    aggregations_used: List[str] = Field(
        default_factory=list,
        description=(
            "Descriptions of aggregations used, e.g. "
            "'SUM(amount) by categoryGroupId', 'COUNT(*)'."
        ),
    )

#############################################################################################

class ClarificationStep(BaseModel):
    """One clarification question/answer pair used as part of UC-05."""

    question: str = Field(
        ...,
        description="Clarifying question shown to the user.",
    )
    user_answer: str = Field(
        ...,
        description="User's reply to the clarifying question.",
    )
    turn_id: Optional[int] = Field(
        None,
        description="Conversation turn in which this clarification was exchanged.",
    )


#############################################################################################

class BackofficeLog(BaseModel):
    """
    Rich internal log for a single CLEAR execution.
    Designed for compliance, observability, and debugging.
    """

    # Core identification
    user_query: str = Field(
        ...,
        description="Original user query that triggered this execution.",
    )

    resolved_query: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Human-readable resolved query showing how ambiguities were interpreted. "
            "Always present (even for CLEAR queries). Structure: "
            "{'original': str, 'resolved_intent': str, 'interpretations': dict}"
        ),
    )

    
    answer: str = Field(
        ...,
        description="Final answer text given to the user (duplicated for convenience).",
    )

    # Analysis & reasoning
    analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key metrics and breakdowns (totals, comparisons, per-category amounts, etc.).",
    )
    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step explanation of how the answer was produced.",
    )

    # Data provenance
    data_sources: DataSources = Field(
        default_factory=DataSources,
        description="Structured description of tables, fields, filters, and aggregations used.",
    )
    transactions_analyzed: int = Field(
        0,
        description="Number of transaction rows analyzed after applying all filters.",
    )

    # Preferences & clarifications
    preferences_used: Dict[str, "PreferenceEntry"] = Field(
        default_factory=dict,
        description=(
            "Snapshot of the preferences from ConversationSummary that were actually used "
            "for this answer, keyed by preference name (e.g. 'time_window')."
        ),
    )
    clarification_history: List[ClarificationStep] = Field(
        default_factory=list,
        description="History of UC-05 clarifications (if any) used before answering.",
    )

    # Confidence & RAG flag
    confidence: Literal["high", "medium", "low"] = Field(
        "medium",
        description="Executor's confidence in the final answer.",
    )
    rag_used: bool = Field(
        False,
        description="Whether RAG/evidence retrieval was used (always False in V1).",
    )

    # Optional trace link back to router decision
    router_output_snapshot: Optional["RouterOutput"] = Field(
        None,
        description="Optional snapshot of the RouterOutput that led to this execution.",
    )


#############################################################################################

class ExecutionResult(BaseModel):
    """
    High-level result of LLM-2 execution for a CLEAR query:
    user-facing answer + rich back-office log.
    """

    final_answer: str = Field(
        ...,
        description="Plain, friendly text returned to the end user.",
    )
    backoffice_log: BackofficeLog = Field(
        ...,
        description="Structured internal log for this answer, for compliance and observability.",
    )

#############################################################################################

# Central state object passed between LangGraph nodes
class GraphState(BaseModel):
    """
    Single state container flowing through the LangGraph:
    user query, shared preferences, router decision, executor result,
    and basic conversation/trace metadata.
    """

    # Core fields
    user_query: str = Field(
        ...,
        description="The latest user query for this turn.",
    )
    conversation_summary: Optional["ConversationSummary"] = Field(
        None,
        description="Session-scoped summary of user preferences shared by all nodes.",
    )
    router_output: Optional["RouterOutput"] = Field(
        None,
        description="Most recent RouterOutput produced by LLM-1 for this query.",
    )
    execution_result: Optional["ExecutionResult"] = Field(
        None,
        description="Executor result for CLEAR queries (final answer + back-office log).",
    )
    messages_to_user: List[str] = Field(
        default_factory=list,
        description="Messages that should be shown to the user in this turn "
                    "(clarifying question or final answer).",
    )

    # Debug / observability helpers
    turn_id: int = Field(
        0,
        description="Monotonic turn counter within the current session, useful for logs and preferences.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier to group turns together.",
    )
    raw_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Low-level conversation history, e.g. [{'role': 'user', 'content': '...'}, "
            "{'role': 'assistant', 'content': '...'}], mainly for debugging or prompt context."
        ),
    )