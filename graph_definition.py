"""
LangGraph for the financial AI agent: nodes, edges, and CLEAR/VAGUE branching over GraphState
"""

from langgraph.graph import StateGraph, START, END
from schemas.router_models import (
    GraphState,
    RouterOutput,
    ConversationSummary,
    PreferenceEntry,      
    ExecutionResult,
    DataSources,       
    BackofficeLog, 
)

import json
from typing import Any, Dict
import os

# LLMs:
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI

from prompts.llm1_prompt import OPTIMIZED_ROUTER_SYSTEM_PROMPT
from prompts.llm2_prompt import llm2_prompt_builder

from schemas.transactions_tool import query_transactions_lc_tool
from schemas.trn_category_tool import search_trans_categories_lc_tool


#############################################################################


def input_node(state: GraphState) -> GraphState:
    """
    Entry point for each turn.
    In the future: normalize user_query, increment turn_id, update raw_messages, etc.
    For now: just return the state as-is.
    """
    return state


#############################################################################

def router_node(state: GraphState) -> GraphState:
    """
    LLM-1 Router & Clarifier with Tool Calling Support.
    
    Supports multi-turn conversations via state.raw_messages.
    
    Architecture:
    - PURE LLM logic: All interpretation (UC type, clarity, tools, missing info, 
      summary_update) is delegated to the model
    - This function builds the prompt, calls the LLM, and parses RouterOutput
    - Tool calling: LLM-1 can call search_transaction_categories for UC-04 queries
    
    Multi-Turn Support:
    When raw_messages is populated, LLM-1 receives conversation history:
    - Previous user query (Turn 1: "Show me recent transactions")
    - Previous assistant clarifying question (Turn 1: "What timeframe?")
    - Current user answer (Turn 2: "Last 30 days")
    
    This enables LLM-1 to:
    - Understand context from previous clarifications
    - Extract preferences from short answers ("Last month", "Just cafes", "$100")
    - Generate summary_update objects to persist preferences
    - Reclassify the original VAGUE query as CLEAR with complete information
    
    Tool Calling:
    - LLM-1 can invoke search_transaction_categories to resolve category terms
    - Results are used to populate resolved_trn_categories in RouterOutput
    """
    payload = build_router_payload(state)

    # Start with system prompt
    messages = [{"role": "system", "content": OPTIMIZED_ROUTER_SYSTEM_PROMPT}]
    
    # Add conversation history if available (for multi-turn clarifications)
    if state.raw_messages:
        messages.extend(state.raw_messages)
    
    # Add current query payload
    messages.append({"role": "user", "content": json.dumps(payload)})

    try:
        # Iterative tool calling loop
        max_iterations = 5
        iteration = 0
        raw_content = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- LLM-1 Router Iteration {iteration} ---")
            
            response = router_llm.invoke(messages)
            
            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"LLM-1 wants to use {len(response.tool_calls)} tool(s)")
                
                tool_result_content = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    print(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    if tool_name == "search_transaction_categories":
                        # Extract terms from args
                        terms = tool_args.get("terms", [])
                        
                        # Call the tool function directly
                        from schemas.trn_category_tool import search_transaction_categories
                        result = search_transaction_categories(terms)
                        
                        # Convert CategoryMatch objects to JSON-serializable format
                        result_json = [match.model_dump() for match in result]
                        
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": json.dumps(result_json, indent=2, default=str)
                        })
                    else:
                        # Unknown tool - return error
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": json.dumps({"error": f"Unknown tool: {tool_name}"})
                        })
                
                # Append assistant message with tool calls and user message with results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_result_content})
                
                continue  # Loop again for LLM to process tool results
            
            else:
                # No tool calls - LLM is done, extract final response
                print("LLM-1 responded without tool calls - parsing RouterOutput")
                raw_content = response.content
                break
        
        if iteration >= max_iterations:
            raise Exception(f"Maximum iterations ({max_iterations}) reached without final response")
        
        # Handle response content format
        if isinstance(raw_content, list):
            # Extract text from content blocks
            text_content = ""
            for item in raw_content:
                if isinstance(item, dict) and 'text' in item:
                    text_content += item['text']
                elif isinstance(item, str):
                    text_content += item
            raw_content = text_content
        
        # Strip markdown code blocks if present
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0]
        elif "```" in raw_content:
            parts = raw_content.split("```")
            if len(parts) >= 3:
                raw_content = parts[1]
        
        raw_content = raw_content.strip()
        
        # Parse and validate the JSON into our typed RouterOutput model
        router_output = RouterOutput.model_validate_json(raw_content)
        
        state.router_output = router_output
        
    except Exception as e:
        print(f"❌ Router error: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal error RouterOutput
        state.router_output = RouterOutput(
            clarity="VAGUE",
            clarifying_question="I encountered an issue understanding your request. Could you please rephrase?",
            missing_info=["error_recovery"],
            core_use_cases=["UC-05"],
            primary_use_case="UC-05",
            uc_confidence="low",
            clarity_reason=f"Error during routing: {str(e)[:100]}"
        )
    
    return state



def vague_handler_node(state: GraphState) -> GraphState:
    """
    UC-05 handler for VAGUE queries.
    In V1 implementation: use router_output.clarifying_question to populate messages_to_user.
    For now: just ensure messages_to_user has something placeholder-like.
    """
    if state.router_output and state.router_output.clarifying_question:
        state.messages_to_user = [state.router_output.clarifying_question]
    else:
        state.messages_to_user = ["I need a bit more information to answer your question."]
    return state



###################################################################################################
""" conversation_summary management

For EVERY turn in the test, show:
1.	📝 summary_update (from LLM-1's RouterOutput.summary_update): 
    o	What LLM-1 proposed to change
    o	Can be null (Turn 1) or a dict with updates (Turn 2)
2.	📦 conversation_summary (the full state after merging - from GraphState.conversation_summary): 
    o	The complete accumulated state AFTER summary_update_node merge
    o	Shows all preferences (time_window, thresholds, etc.)
"""
###################################################################################################
def summary_update_node(state: GraphState) -> GraphState:
    """
    Merge router_output.summary_update into conversation_summary.
    
    Implements 'latest wins' policy for preference overrides.
    This node is the single source of truth for updating conversation_summary.
    
    Architecture:
    - LLM-1 produces summary_update (delta: what changed this turn)
    - This node merges delta into conversation_summary (accumulated state)
    - Supports all preference types: time_window, amount_threshold_large, 
      account_scope, category_preferences
    
    UPDATED: Now handles both simple values and full PreferenceEntry structures
    from LLM-1's summary_update.
    """
    # 1. Check if there's anything to update
    if not state.router_output:
        return state
    
    summary_update = state.router_output.summary_update
    if not summary_update:
        return state  # Nothing to merge
    
    # 2. Initialize conversation_summary if needed
    if state.conversation_summary is None:
        state.conversation_summary = ConversationSummary()
    
    # Helper function to normalize preference data
    def normalize_preference_entry(data: Any, field_name: str) -> PreferenceEntry:
        """
        Convert various input formats to PreferenceEntry.
        
        Handles:
        - Full PreferenceEntry dict: {"value": "last_month", "source": "user_defined", ...}
        - Simple value: "last_month" or 100
        """
        if isinstance(data, dict) and "value" in data:
            # Full PreferenceEntry structure - use as-is
            return PreferenceEntry(**data)
        else:
            # Simple value - wrap in PreferenceEntry with defaults
            return PreferenceEntry(
                value=data,
                source="user_defined",
                turn_id=state.turn_id,
                original_query=state.user_query
            )
    
    # 3. Merge each top-level preference field
    
    # 3a. time_window (PreferenceEntry or None)
    if "time_window" in summary_update:
        time_window_data = summary_update["time_window"]
        if time_window_data is not None:
            state.conversation_summary.time_window = normalize_preference_entry(
                time_window_data, "time_window"
            )
        else:
            state.conversation_summary.time_window = None
    
    # 3b. amount_threshold_large (PreferenceEntry or None)
    if "amount_threshold_large" in summary_update:
        threshold_data = summary_update["amount_threshold_large"]
        if threshold_data is not None:
            state.conversation_summary.amount_threshold_large = normalize_preference_entry(
                threshold_data, "amount_threshold_large"
            )
        else:
            state.conversation_summary.amount_threshold_large = None
    
    # 3c. account_scope (PreferenceEntry or None)
    if "account_scope" in summary_update:
        account_data = summary_update["account_scope"]
        if account_data is not None:
            state.conversation_summary.account_scope = normalize_preference_entry(
                account_data, "account_scope"
            )
        else:
            state.conversation_summary.account_scope = None
    
    # 3d. category_preferences (dict of PreferenceEntry objects)
    if "category_preferences" in summary_update:
        category_updates = summary_update["category_preferences"]
        if category_updates:
            # Merge: add new keys, replace existing keys
            for key, pref_data in category_updates.items():
                if pref_data is not None:
                    state.conversation_summary.category_preferences[key] = normalize_preference_entry(
                        pref_data, f"category_preferences.{key}"
                    )
                else:
                    # Remove preference if explicitly set to None
                    if key in state.conversation_summary.category_preferences:
                        del state.conversation_summary.category_preferences[key]
    
    return state


###################################################################################################

###################################################################################################
# Routing function for conditional branching
###################################################################################################

def route_by_clarity(state: GraphState) -> str:
    """
    Conditional routing function based on router_output.clarity.
    
    Returns:
        - "vague_handler" if clarity == "VAGUE"
        - "executor" if clarity == "CLEAR"
    """
    if state.router_output is None:
        # Safety fallback: if no router_output, go to vague_handler
        return "vague_handler"
    
    clarity = state.router_output.clarity
    
    if clarity == "VAGUE":
        return "vague_handler"
    elif clarity == "CLEAR":
        return "executor"
    else:
        # Safety fallback for unexpected values
        print(f"⚠️  Unexpected clarity value: {clarity}. Routing to vague_handler.")
        return "vague_handler"


###################################################################################################

###################################################################################################

# --- Graph builder ---
def build_graph() -> StateGraph:
    """
    Construct the LangGraph for the financial AI agent with:
    input -> router -> (vague_handler | executor) -> summary_update -> END.
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("input", input_node)
    graph.add_node("router", router_node)
    graph.add_node("vague_handler", vague_handler_node)
    graph.add_node("executor", executor_node)    
    graph.add_node("summary_update", summary_update_node)

    # Edges: START -> input -> router
    graph.add_edge(START, "input")
    graph.add_edge("input", "router")

    # Conditional branching after router based on clarity
    graph.add_conditional_edges("router", route_by_clarity)

    # Both paths join at summary_update
    graph.add_edge("vague_handler", "summary_update")
    graph.add_edge("executor", "summary_update")

    # End after summary_update
    graph.add_edge("summary_update", END)

    return graph

#########################################################################################
# LLM-1 Router & Clarifier
#########################################################################################

def build_router_payload(state: GraphState) -> Dict[str, Any]:
    """
    Prepare the minimal structured input for LLM-1:
    - user_query
    - current conversation_summary (if any) as a dict.
    """
    return {
        "user_query": state.user_query,
        "conversation_summary": (
            state.conversation_summary.model_dump()
            if isinstance(state.conversation_summary, ConversationSummary)
            else None
        ),
    }

#########################################################################################    

router_llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",      
    temperature=0,
).bind_tools(
    [
        search_trans_categories_lc_tool,
        # later we can add more router tools here
    ]
)
# router_llm = ChatOpenAI(model="gpt-4o", temperature=0
#                        ).bind_tools(
#     [
#         search_trans_categories_lc_tool,
#         # later we can add more router tools here
#     ]
# )


def aggregation_calculator(
    operation: str,
    field: str, 
    transactions_json: str,
    filters: str = ""
) -> str:
    """
    Mathematical aggregation tool for UC-02 queries.    
    Args:
        operation: SUM, AVG, COUNT, MIN, MAX
        field: amount, transaction_count
        transactions_json: JSON string of transaction data
        filters: Additional filter criteria    
    Returns:
        JSON string with aggregation result
    """
    import json
    
    try:
        transactions = json.loads(transactions_json)
        
        if operation == "SUM":
            result = sum(float(t.get(field, 0)) for t in transactions)
        elif operation == "AVG":
            amounts = [float(t.get(field, 0)) for t in transactions]
            result = sum(amounts) / len(amounts) if amounts else 0
        elif operation == "COUNT":
            result = len(transactions)
        elif operation == "MIN":
            result = min(float(t.get(field, 0)) for t in transactions) if transactions else 0
        elif operation == "MAX":
            result = max(float(t.get(field, 0)) for t in transactions) if transactions else 0
        else:
            result = 0
            
        return json.dumps({
            "result": result,
            "operation": operation,
            "field": field,
            "transaction_count": len(transactions)
        })
        
    except Exception as e:
        return json.dumps({"error": str(e), "result": 0})

   
########################################################################################################
# LLM-2 Executor: tools, payload builder, and executor node
########################################################################################################

# Bind LLM-2 (Executor) to the tools it can use in Version 1.
# For now we expose:
# - get_date_range_lc_tool: resolve normalized periods to start/end dates (TO DELETE!)
# - query_transactions_lc_tool: query transactions.csv with deterministic filters

executor_llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
).bind_tools(
    [
        query_transactions_lc_tool,
        aggregation_calculator,  
        # later we will also add: map_category_from_kb_lc_tool
    ]
)
# router_llm = ChatOpenAI(model="gpt-4o", temperature=0
#                        ).bind_tools(
#     [
#         search_trans_categories_lc_tool,
#         # later we can add more router tools here
#     ]
# )


def extract_user_id_from_query(user_query: str) -> str:
    """Extract user_id from query like 'I am USER_001. What is my balance?'"""
    import re
    
    # Pattern to match "I am USER_XXX" at start of query
    pattern = r"I am (USER_\d+)\."
    match = re.search(pattern, user_query)
    
    if match:
        return match.group(1)
    
    # Fallback - if no user_id found, use default (but log warning)
    print(f"⚠️  WARNING: No user_id found in query: '{user_query[:50]}...'")
    return "USER_001"  # Fallback only



def build_executor_payload(state: GraphState) -> Dict[str, Any]:
    """
    Data Architecture Hub:
    
    Input Consolidation: Gathers all context from GraphState
    Data Transformation: Converts Pydantic models to JSON for LLM-2
    Context Injection:   Adds execution context (user_id, session info)
    Message Preparation: Creates the exact JSON that LLM-2 receives

    Build JSON payload for LLM-2 with DYNAMIC user context extraction.
    """
    router_obj = state.router_output
    summary_obj = state.conversation_summary
    
    # ✅ EXTRACT USER_ID DYNAMICALLY
    user_id = extract_user_id_from_query(state.user_query)
    
    payload: Dict[str, Any] = {
        "user_query": state.user_query,
        "router_output": router_obj.model_dump(mode='json') if router_obj is not None else None,
        "conversation_summary": summary_obj.model_dump(mode='json') if summary_obj is not None else None,
        "executor_context": {
            "user_id": user_id,  # ✅ DYNAMIC USER CONTEXT!
        },
    }
    return payload





########################################################################################################

def executor_node(state: GraphState) -> GraphState:
    """
    LLM-2 Executor node with iterative tool calling.
    
    Architecture:
    - Calls LLM-2 with tools
    - LLM-2 returns flat JSON: {answer, analysis, reasoning_steps, ...}
    - This function WRAPS it into ExecutionResult structure
    - Adds metadata from GraphState
    """
    if state.router_output is None:
        state.messages_to_user = ["Error: No router output available."]
        return state

    # Build the JSON payload for the user message
    payload = build_executor_payload(state)

    messages = [
        {
            "role": "system",
            "content": llm2_prompt_builder(
                user_query=state.user_query,
                router_output=state.router_output,
                conversation_summary=state.conversation_summary
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    try:
        # Iterative tool calling loop
        max_iterations = 8
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- LLM-2 Iteration {iteration} ---")
            
            response = executor_llm.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"LLM-2 wants to use {len(response.tool_calls)} tools")
                
                tool_result_content = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"] 
                    tool_args = tool_call["args"]
                    
                    print(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    if tool_name == "get_date_range":
                        if 'request' in tool_args:
                            request_data = tool_args['request']
                        else:
                            request_data = tool_args
                        
                        from schemas.executor_models_llm2 import DateRangeRequest
                        request_obj = DateRangeRequest(**request_data)
                        result = get_date_range_lc_tool.func(request_obj)
                        
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": json.dumps(result.model_dump() if hasattr(result, 'model_dump') else result, indent=2, default=str)
                        })
                        
                    elif tool_name == "query_transactions":
                        if 'spec' in tool_args:
                            spec_data = tool_args['spec'] 
                        elif 'request' in tool_args:
                            spec_data = tool_args['request']
                        else:
                            spec_data = tool_args
                            
                        from schemas.executor_models_llm2 import TransactionQuerySpec
                        spec_obj = TransactionQuerySpec(**spec_data)
                        result = query_transactions_lc_tool.func(spec_obj)
                        
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": json.dumps(result.model_dump() if hasattr(result, 'model_dump') else result, indent=2, default=str)
                        })
                    
                    elif tool_name == "aggregation_calculator":
                        result = aggregation_calculator(
                            operation=tool_args.get("operation", ""),
                            field=tool_args.get("field", ""),
                            transactions_json=tool_args.get("transactions_json", ""),
                            filters=tool_args.get("filters", "")
                        )
                        
                        tool_result_content.append({
                            "type": "tool_result", 
                            "tool_use_id": tool_call["id"],
                            "content": result
                        })
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_result_content})
                
                continue
                
            else:
                print("LLM-2 responded without tool calls - expecting final JSON")
                raw_content = response.content
                break
        
        if iteration >= max_iterations:
            raise Exception(f"Maximum iterations ({max_iterations}) reached without final response")

        # =====================================================================
        # CLEAN UP RESPONSE CONTENT FOR JSON PARSING
        # =====================================================================
        if isinstance(raw_content, list):
            text_content = ""
            for item in raw_content:
                if isinstance(item, dict) and 'text' in item:
                    text_content += item['text']
                elif isinstance(item, str):
                    text_content += item
            raw_content = text_content
        
        # =====================================================================
        # ROBUST JSON EXTRACTION (handles multiple LLM output formats)
        # =====================================================================
        
        # Strategy 1: Check for markdown code blocks first
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0]
        elif "```" in raw_content:
            parts = raw_content.split("```")
            if len(parts) >= 3:
                raw_content = parts[1]
        
        # Strategy 2: Find first '{' and last '}' (handles text before/after JSON)
        # This catches cases like: "Now I have the data: {...}"
        first_brace = raw_content.find('{')
        last_brace = raw_content.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            raw_content = raw_content[first_brace:last_brace + 1]
        
        raw_content = raw_content.strip()
        
        print(f"Attempting to parse JSON: {raw_content[:200]}...")
        
        # =====================================================================
        # PARSE LLM-2's FLAT OUTPUT AND WRAP INTO ExecutionResult
        # =====================================================================
        
        # Step 1: Parse LLM-2's flat JSON output
        llm2_output = json.loads(raw_content)
        
        # Step 2: Extract preferences used from conversation_summary
        preferences_used_dict = {}
        if state.conversation_summary:
            if state.conversation_summary.time_window:
                preferences_used_dict["time_window"] = state.conversation_summary.time_window
            if state.conversation_summary.amount_threshold_large:
                preferences_used_dict["amount_threshold_large"] = state.conversation_summary.amount_threshold_large
            if state.conversation_summary.account_scope:
                preferences_used_dict["account_scope"] = state.conversation_summary.account_scope
            if state.conversation_summary.category_preferences:
                for key, pref in state.conversation_summary.category_preferences.items():
                    preferences_used_dict[f"category_{key}"] = pref
        
        # Step 3: Build DataSources from LLM-2 output
        data_sources_data = llm2_output.get("data_sources", {})
        data_sources = DataSources(
            tables_used=data_sources_data.get("tables_used", []),
            fields_accessed=data_sources_data.get("fields_accessed", []),
            filters_applied=data_sources_data.get("filters_applied", []),
            aggregations_used=data_sources_data.get("aggregations_used", [])
        )
        
        # Step 4: Wrap into ExecutionResult structure
        execution_result = ExecutionResult(
            final_answer=llm2_output.get("answer", ""),
            
            backoffice_log=BackofficeLog(
                # Core identification
                user_query=state.user_query,
                
                # NEW: resolved_query from LLM-2
                resolved_query=llm2_output.get("resolved_query"),
                
                # Customer-facing answer (duplicated for audit trail)
                answer=llm2_output.get("answer", ""),
                
                # Analysis & reasoning from LLM-2
                analysis=llm2_output.get("analysis", {}),
                reasoning_steps=llm2_output.get("reasoning_steps", []),
                
                # Data provenance
                data_sources=data_sources,
                transactions_analyzed=llm2_output.get("transactions_analyzed", 0),
                
                # Preferences used (from conversation_summary)
                preferences_used=preferences_used_dict,
                
                # Clarification history (empty for now, can be populated later)
                clarification_history=[],
                
                # Confidence from LLM-2
                confidence=llm2_output.get("confidence", "medium"),
                
                # RAG flag (always False in V1)
                rag_used=False,
                
                # Router snapshot (metadata from LLM-1)
                router_output_snapshot=state.router_output
            )
        )
        
        # =====================================================================
        # STORE RESULT IN STATE
        # =====================================================================
        state.execution_result = execution_result
        state.messages_to_user = [execution_result.final_answer]
        
        print("✅ Successfully wrapped LLM-2 output into ExecutionResult")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw content: {raw_content if 'raw_content' in locals() else 'N/A'}")
        
        # Create error ExecutionResult
        state.execution_result = ExecutionResult(
            final_answer="I encountered an issue processing your request. Please try rephrasing.",
            backoffice_log=BackofficeLog(
                user_query=state.user_query,
                answer="Error: JSON parsing failed",
                analysis={"error": str(e)},
                reasoning_steps=[f"Failed to parse LLM-2 output: {str(e)}"],
                data_sources=DataSources(),
                transactions_analyzed=0,
                preferences_used={},
                clarification_history=[],
                confidence="low",
                rag_used=False,
                router_output_snapshot=state.router_output
            )
        )
        state.messages_to_user = [state.execution_result.final_answer]
        
    except Exception as e:
        print(f"❌ Executor error: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error ExecutionResult
        state.execution_result = ExecutionResult(
            final_answer="I encountered an unexpected error. Please try again.",
            backoffice_log=BackofficeLog(
                user_query=state.user_query,
                answer="Error: Executor exception",
                analysis={"error": str(e)},
                reasoning_steps=[f"Executor exception: {str(e)}"],
                data_sources=DataSources(),
                transactions_analyzed=0,
                preferences_used={},
                clarification_history=[],
                confidence="low",
                rag_used=False,
                router_output_snapshot=state.router_output
            )
        )
        state.messages_to_user = [state.execution_result.final_answer]

    return state
      
# # LLM-2: executor_node function with proper tool calling support
# def executor_node(state: GraphState) -> GraphState:
#     """
#     LLM-2 Executor node with iterative tool calling and UC-01 injection support.
#     """
#     if state.router_output is None:
#         state.messages_to_user = ["Error: No router output available."]
#         return state

#     # Build the JSON payload for the user message
#     payload = build_executor_payload(state)

#     messages = [
#         {
#             "role": "system",
#             "content": llm2_prompt_builder(
#                 user_query=state.user_query,
#                 router_output=state.router_output,
#                 conversation_summary=state.conversation_summary
#             ),
#         },
#         {
#             "role": "user",
#             "content": json.dumps(payload, ensure_ascii=False),
#         },
#     ]

#     try:
#         # Iterative tool calling loop
#         max_iterations = 8 #5
#         iteration = 0
        
#         while iteration < max_iterations:
#             iteration += 1
#             print(f"\n--- LLM-2 Iteration {iteration} ---")
            
#             response = executor_llm.invoke(messages)
            
#             if hasattr(response, 'tool_calls') and response.tool_calls:
#                 print(f"LLM-2 wants to use {len(response.tool_calls)} tools")
                
#                 tool_result_content = []
                
#                 for tool_call in response.tool_calls:
#                     tool_name = tool_call["name"] 
#                     tool_args = tool_call["args"]
                    
#                     print(f"Executing tool: {tool_name} with args: {tool_args}")
                    
#                     if tool_name == "get_date_range":
#                         if 'request' in tool_args:
#                             request_data = tool_args['request']
#                         else:
#                             request_data = tool_args
                        
#                         from schemas.executor_models_llm2 import DateRangeRequest
#                         request_obj = DateRangeRequest(**request_data)
#                         result = get_date_range_lc_tool.func(request_obj)
                        
#                         tool_result_content.append({
#                             "type": "tool_result",
#                             "tool_use_id": tool_call["id"],
#                             "content": json.dumps(result.model_dump() if hasattr(result, 'model_dump') else result, indent=2, default=str)
#                         })
                        
#                     elif tool_name == "query_transactions":
#                         if 'spec' in tool_args:
#                             spec_data = tool_args['spec'] 
#                         elif 'request' in tool_args:
#                             spec_data = tool_args['request']
#                         else:
#                             spec_data = tool_args
                            
#                         from schemas.executor_models_llm2 import TransactionQuerySpec
#                         spec_obj = TransactionQuerySpec(**spec_data)
#                         result = query_transactions_lc_tool.func(spec_obj)
                        
#                         tool_result_content.append({
#                             "type": "tool_result",
#                             "tool_use_id": tool_call["id"],
#                             "content": json.dumps(result.model_dump() if hasattr(result, 'model_dump') else result, indent=2, default=str)
#                         })
                    
#                     elif tool_name == "aggregation_calculator":
#                         result = aggregation_calculator(
#                             operation=tool_args.get("operation", ""),
#                             field=tool_args.get("field", ""),
#                             transactions_json=tool_args.get("transactions_json", ""),
#                             filters=tool_args.get("filters", "")
#                         )
                        
#                         tool_result_content.append({
#                             "type": "tool_result", 
#                             "tool_use_id": tool_call["id"],
#                             "content": result
#                         })
                
#                 messages.append({"role": "assistant", "content": response.content})
#                 messages.append({"role": "user", "content": tool_result_content})
                
#                 continue
                
#             else:
#                 print("LLM-2 responded without tool calls - expecting final JSON")
#                 raw_content = response.content
#                 break
        
#         if iteration >= max_iterations:
#             raise Exception(f"Maximum iterations ({max_iterations}) reached without final response")

#         # Clean up response content for JSON parsing
#         if isinstance(raw_content, list):
#             text_content = ""
#             for item in raw_content:
#                 if isinstance(item, dict) and 'text' in item:
#                     text_content += item['text']
#                 elif isinstance(item, str):
#                     text_content += item
#             raw_content = text_content
        
#         if "```json" in raw_content:
#             raw_content = raw_content.split("```json")[1].split("```")[0]
#         elif "```" in raw_content:
#             parts = raw_content.split("```")
#             if len(parts) >= 3:
#                 raw_content = parts[1]
        
#         raw_content = raw_content.strip()
        
#         print(f"Attempting to parse JSON: {raw_content[:200]}...")
        
#         execution_result = ExecutionResult.model_validate_json(raw_content)

#         state.execution_result = execution_result
#         state.messages_to_user = [execution_result.final_answer]
        
#         print("✅ Successfully parsed ExecutionResult")

#     except json.JSONDecodeError as e:
#         print(f"❌ JSON parsing error: {e}")
#         print(f"Raw content: {raw_content}")
#         state.messages_to_user = [f"I encountered an issue processing your request. Raw response: {raw_content[:100]}..."]
#         state.execution_result = None
        
#     except Exception as e:
#         print(f"❌ Executor error: {e}")
#         state.messages_to_user = [f"Error in executor: {str(e)}"]
#         state.execution_result = None

#     return state

    
