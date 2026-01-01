"""
LangChain StructuredTool for searching transaction categories via RAG.

This module exposes:
- CategoryMatch: Pydantic model for one RAG result
- search_transaction_categories(terms: List[str]) -> List[CategoryMatch]
- search_trans_categories_lc_tool: LangChain StructuredTool for LLM-1 Router

Purpose:
- Wrap the Category RAG system (query_categories) as a LangChain tool
- Transform RAG output into structured CategoryMatch objects
- Enable LLM-1 to resolve natural language category phrases to category IDs

Architecture Position:
- Called by: LLM-1 Router (for UC-04 category-based queries)
- Calls: query_categories() from rag.trn_category_rag
- Output used in: RouterOutput.resolved_trn_categories
"""

from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
from rag.trn_category_rag import query_categories


###########################################################################################
# PYDANTIC MODELS
###########################################################################################

class CategoryMatch(BaseModel):
    """
    Single category match from RAG search.
    
    This model represents one result from the Category RAG system and is designed
    to populate RouterOutput.resolved_trn_categories.
    
    Fields align with RouterOutput expectations and transaction schema:
    - category_id: Maps to categoryGroupId or subCategoryId in transactions
    - category_name: Display name for the category
    - category_type: Whether this is a group (e.g., CG800) or subcategory (e.g., C806)
    - group_id: Parent group ID (for subcategories only)
    - group_name: Parent group name (for subcategories only)
    - distance: Raw similarity score from RAG (lower = better match)
    - confidence: Human-readable confidence level based on distance
    """
    
    user_term: str = Field(
        ...,
        description="Original search term from user query (for traceability)"
    )
    
    category_id: str = Field(
        ...,
        description="categoryGroupId (e.g., 'CG800') or subCategoryId (e.g., 'C806')"
    )
    
    category_name: str = Field(
        ...,
        description="Human-readable category name (e.g., 'Dining', 'Cafes & Coffee Shops')"
    )
    
    category_type: Literal["group", "subcategory"] = Field(
        ...,
        description="Whether this is a category group or subcategory"
    )
    
    group_id: Optional[str] = Field(
        None,
        description="Parent categoryGroupId (only for subcategories)"
    )
    
    group_name: Optional[str] = Field(
        None,
        description="Parent category group name (only for subcategories)"
    )
    
    distance: float = Field(
        ...,
        description="Raw similarity distance from RAG (0.0 = perfect match, lower is better)"
    )
    
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level: high (<0.4), medium (0.4-0.6), low (>0.6)"
    )


###########################################################################################
# HELPER FUNCTIONS
###########################################################################################

def _distance_to_confidence(distance: float) -> Literal["high", "medium", "low"]:
    """
    Map RAG similarity distance to confidence level.
    
    Based on empirical calibration with multilingual-e5-base:
    - Distance < 0.4: Excellent match (high confidence)
    - Distance 0.4-0.6: Good match (medium confidence)
    - Distance > 0.6: Acceptable match (low confidence)
    
    Args:
        distance: Similarity distance from RAG (0.0 = perfect match)
        
    Returns:
        Confidence level: "high", "medium", or "low"
    """
    if distance < 0.4:
        return "high"
    elif distance <= 0.6:
        return "medium"
    else:
        return "low"


###########################################################################################
# CORE TOOL FUNCTION
###########################################################################################

def search_transaction_categories(terms: List[str]) -> List[CategoryMatch]:
    """
    Search for transaction categories matching natural language terms.
    
    This function wraps the Category RAG system (query_categories) and transforms
    the output into structured CategoryMatch objects that LLM-1 can use to populate
    RouterOutput.resolved_trn_categories.
    
    Supports both single-term and batch queries for efficiency:
    - Single term: ["coffee shops"]
    - Multiple terms: ["dining", "groceries", "gas station"]
    
    Args:
        terms: List of natural language category phrases
               (e.g., ["coffee shops"], ["dining", "groceries"])
        
    Returns:
        List of CategoryMatch objects, sorted by relevance (best matches first).
        Deduplicates results if multiple terms match the same category.
        Returns empty list if no matches found or terms list is empty.
        
    Example:
        >>> # Single term
        >>> matches = search_transaction_categories(["coffee shops"])
        >>> for m in matches:
        ...     print(f"{m.category_name} ({m.category_id}) - {m.confidence}")
        Cafes & Coffee Shops (C806) - high
        Restaurants (C803) - medium
        
        >>> # Multiple terms (batch processing)
        >>> matches = search_transaction_categories(["dining", "groceries", "gas"])
        >>> for m in matches:
        ...     print(f"{m.user_term} → {m.category_name} ({m.category_id})")
        dining → Dining (CG800)
        groceries → Groceries (CG10000)
        gas → Gas Station (C101)
    
    Deduplication:
        If multiple terms match the same category, keeps the match with the
        lowest distance (best similarity). Each CategoryMatch.user_term shows
        which search term produced that match.
        
    Error Handling:
        - Empty terms list: Returns empty list (not exception)
        - No matches above threshold: Returns empty list
        - RAG error: Returns empty list (logs error internally)
        
    Performance:
        - ~10ms per term (cached embedding model and vector store)
        - Batch processing more efficient than multiple single calls
        - Uses default threshold 0.6 from RAG configuration
    """
    
    # Validate input - must be a list
    if not terms:
        return []
    
    # Filter out empty or whitespace-only terms
    valid_terms = [t.strip() for t in terms if t and t.strip()]
    
    if not valid_terms:
        return []
    
    try:
        # Collect all matches from all terms
        all_matches: List[CategoryMatch] = []
        
        # Query RAG for each term
        for term in valid_terms:
            # Query RAG system (uses default threshold 0.6, top_k=3)
            rag_results = query_categories(term=term, top_k=3, min_confidence=1.0)
            
            # Transform RAG output to CategoryMatch objects
            for result in rag_results:
                match = CategoryMatch(
                    user_term=term,  # Track which term produced this match
                    category_id=result['id'],
                    category_name=result['name'],
                    category_type=result['type'],  # "group" or "subcategory"
                    group_id=result.get('group_id'),  # None for groups
                    group_name=result.get('group_name'),  # None for groups
                    distance=result['score'],
                    confidence=_distance_to_confidence(result['score'])
                )
                all_matches.append(match)
        
        # Deduplicate: If multiple terms match the same category,
        # keep the match with the best (lowest) distance
        seen: Dict[str, CategoryMatch] = {}
        
        for match in all_matches:
            category_id = match.category_id
            
            if category_id not in seen or match.distance < seen[category_id].distance:
                seen[category_id] = match  # Keep best match for this category
        
        # Convert back to list and sort by distance (best first)
        final_matches = list(seen.values())
        final_matches.sort(key=lambda m: m.distance)
        
        return final_matches
    
    except Exception as e:
        # Log error but don't crash - let LLM-1 handle "no results"
        print(f"⚠️  Category RAG error for terms {valid_terms}: {e}")
        return []


###########################################################################################
# LANGCHAIN TOOL WRAPPER
###########################################################################################

search_trans_categories_lc_tool = StructuredTool.from_function(
    name="search_transaction_categories",
    func=search_transaction_categories,
    description=(
        "Search for transaction categories matching natural language terms. "
        "Accepts a list of search terms (e.g., ['dining'], ['coffee', 'groceries']). "
        "Returns up to 3 best matches per term with category IDs, names, and confidence levels. "
        "Deduplicates results if multiple terms match the same category. "
        "Use this tool when the user query involves category-based filtering (UC-04) "
        "such as 'groceries', 'dining', 'coffee shops', 'transportation', etc. "
        "Returns empty list if no matches found."
    ),
)


###########################################################################################
# USAGE IN JUPYTER NOTEBOOK
###########################################################################################

"""
HOW TO USE IN JUPYTER:

1. Import the tool function:
   from schemas.trn_category_tool import search_transaction_categories

2. Test single category search:
   matches = search_transaction_categories(["coffee shops"])
   for m in matches:
       print(f"{m.category_name} ({m.category_id}) - {m.confidence}")

3. Test batch category search:
   matches = search_transaction_categories(["dining", "groceries", "gas station"])
   for m in matches:
       print(f"{m.user_term} → {m.category_name} ({m.category_id})")

4. Import the LangChain tool (for graph_definition.py):
   from schemas.trn_category_tool import search_trans_categories_lc_tool
   
5. Test with multiple terms (deduplication):
   # These might all match the same category
   test_terms = ["coffee", "cafe", "coffee shop"]
   matches = search_transaction_categories(test_terms)
   print(f"Found {len(matches)} unique categories from {len(test_terms)} terms")
   for m in matches:
       print(f"  {m.user_term} → {m.category_name} (distance: {m.distance:.4f})")

6. Test error handling:
   # Empty list
   matches = search_transaction_categories([])
   print(f"Empty list: {len(matches)} results")
   
   # List with empty strings
   matches = search_transaction_categories(["", "  ", "dining"])
   print(f"Filtered list: {len(matches)} results")
"""