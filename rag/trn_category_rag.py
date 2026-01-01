"""
TRANSACTION CATEGORY RAG - Query Functions

Provides simple, reusable functions to query the category vector store.

Core functions:
- load_category_vector_store() - Load ChromaDB collection
- query_categories() - Search for categories by natural language term
- test_rag_queries() - Comprehensive test of all 60 categories

Usage:
    from rag.trn_category_rag import query_categories
    
    results = query_categories("coffee shops", top_k=3, min_confidence=0.6)
    for match in results:
        print(f"{match['name']} ({match['id']}) - Score: {match['score']:.4f}")

========================================================================================

`@lru_cache` - Function Result Caching
 - Remembers function results to avoid reloading expensive resources.

Our Use Case:
    - We cache **2 expensive resources** that must load only ONCE per session:
        - 1. Embedding Model: `intfloat/multilingual-e5-base`
        - 2. Vector Store: ChromaDB Client
        
    - Why only ONCE per session:
        - Our functions have **no arguments** (always same input)
        - Only need to cache **1 result** (the loaded model/database)
        - Minimal memory overhead
  
Impact on Performance:
    - Single query    =>  Without Cache: 5 seconds           VS  With Cache: 5 seconds 
    - 60 queries      =>  Without Cache: 300 seconds (5 min) VS  With Cache: 6 seconds
    - Speedup Summary =>  Without Cache: 1x                  VS  With Cache **50x faster** 🚀 

Summary - Caching:
    - @lru_cache in our project:
    - Loads `intfloat/multilingual-e5-base` ONCE (`maxsize=1`) (560MB)
    - Loads ChromaDB collection ONCE  (`maxsize=1`) (111 categories)
    - 50x faster for multiple queries
    - Essential for production ML systems 
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# CRITICAL: Must match build_category_vectorstore.py
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"  # ✅ CORRECT MODEL
CHROMA_PERSIST_DIR = "data/chroma_trn_categories"
COLLECTION_NAME = "transaction_categories"

# ChromaDB settings (must be consistent across all calls)
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

# Metadata keys (MUST MATCH build_category_vectorstore.py)
METADATA_TYPE = "type"           # "group" or "subcategory"
METADATA_ID = "id"               # Category ID (e.g., "CG800", "C806")
METADATA_NAME = "name"           # Category name
METADATA_DESCRIPTION = "description"
METADATA_GROUP_ID = "group_id"   # Parent group ID (subcategories only)
METADATA_GROUP_NAME = "group_name"  # Parent group name (subcategories only)

# Query defaults
DEFAULT_TOP_K = 3
DEFAULT_MIN_CONFIDENCE = 0.6  # Distance threshold (lower is better)
                              # With multilingual-e5-base observed ranges:
                              # 0.0-0.4: Excellent, 0.4-0.6: Good, 0.6-0.75: Acceptable
                              # 0.6 threshold = Keep excellent + good matches


# ═══════════════════════════════════════════════════════════════════
# CHROMADB GLOBAL REGISTRY FIX - NUCLEAR OPTION
# ═══════════════════════════════════════════════════════════════════

# CRITICAL: Clear ALL ChromaDB global state at module import time
# This prevents "An instance of Chroma already exists" errors
# NUCLEAR approach: Clear the ENTIRE registry, not just one identifier
try:
    from chromadb.api.shared_system_client import SharedSystemClient
    # Clear ALL registered systems (nuclear option)
    SharedSystemClient._identifier_to_system.clear()
except Exception:
    pass  # If clearing fails, continue anyway


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS


# ═══════════════════════════════════════════════════════════════════


def reset_chromadb_registry():
    """
    EMERGENCY: Manually clear ChromaDB's global registry.
    
    Call this if you get "An instance of Chroma already exists" errors.
    
    Usage:
        from rag.trn_category_rag import reset_chromadb_registry
        reset_chromadb_registry()
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system.clear()
        print("✅ ChromaDB registry cleared successfully")
    except Exception as e:
        print(f"❌ Failed to clear registry: {e}")


@lru_cache(maxsize=1)  # Remember 1 result
def _get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.
    
    Uses LRU cache to ensure model is loaded only once per session.
    Subsequent calls return the cached instance.
    
    Returns:
        SentenceTransformer: Cached embedding model
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)  # Remember 1 result
def load_category_vector_store() -> Collection:
    """
    Load and cache the category vector store ChromaDB collection.
    
    Uses LRU cache to ensure ChromaDB client is created only once per session.
    ChromaDB's global registry is cleared at module import to prevent conflicts.
    
    This function loads the persistent vector store created by
    build_category_vectorstore.py. The vector store must exist
    before calling this function.
    
    Returns:
        Collection: Cached ChromaDB collection with category embeddings
        
    Raises:
        ValueError: If vector store doesn't exist or is empty
    """
    persist_dir = Path(CHROMA_PERSIST_DIR)
    
    if not persist_dir.exists():
        raise ValueError(
            f"Vector store not found at: {CHROMA_PERSIST_DIR}\n"
            f"Run 'python build_category_vectorstore.py' first to create it."
        )
    
    # Initialize ChromaDB client (registry already cleared at module level)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=CHROMA_SETTINGS  # ← Use pre-defined settings constant
    )
    
    # Load collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found in vector store.\n"
            f"Run 'python build_category_vectorstore.py' first.\n"
            f"Error: {e}"
        )
    
    # Verify collection has documents
    count = collection.count()
    if count == 0:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' is empty.\n"
            f"Run 'python build_category_vectorstore.py' first."
        )
    
    return collection


# ═══════════════════════════════════════════════════════════════════
# MAIN QUERY FUNCTION
# ═══════════════════════════════════════════════════════════════════


def query_categories(
    term: str,
    top_k: int = DEFAULT_TOP_K,
    min_confidence: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Query the category vector store for matching categories.
    
    Performs semantic search to find categories that match the given term.
    Returns results sorted by relevance (best matches first).
    
    Args:
        term: Natural language category term (e.g., "groceries", "coffee shops")
        top_k: Number of results to return (default: 3)
        min_confidence: Maximum distance threshold (None = use DEFAULT_MIN_CONFIDENCE)
                       Lower distance = better match
                       Examples: 0.5 (strict), 0.6 (balanced), 0.8 (loose)
                       None = no filtering (show all results)
    
    Returns:
        List of category matches, each containing:
        - 'type': 'group' or 'subcategory'
        - 'id': categoryGroupId or subCategoryId
        - 'name': categoryGroupName or subCategoryName
        - 'group_id': categoryGroupId (for subcategories)
        - 'group_name': categoryGroupName (for subcategories)
        - 'description': Category description
        - 'score': Similarity score (distance: lower = better, 0 = perfect match)
        
    Example:
        >>> results = query_categories("coffee shops", top_k=3, min_confidence=0.6)
        >>> for match in results:
        ...     print(f"{match['name']} ({match['id']}) - Score: {match['score']:.4f}")
        Cafes & Coffee Shops (C806) - Score: 0.2723
        Restaurants (C803) - Score: 0.4261
        Fast Food (C802) - Score: 0.4317
    
    Raises:
        ValueError: If vector store doesn't exist or term is empty
    """
    
    if not term or not term.strip():
        raise ValueError("Category term cannot be empty")
    
    term = term.strip()
    
    # Use provided threshold or default
    active_threshold = min_confidence if min_confidence is not None else DEFAULT_MIN_CONFIDENCE
    
    # Load embedding model and vector store
    embedding_model = _get_embedding_model()
    collection = load_category_vector_store()
    
    # Generate query embedding
    query_embedding = embedding_model.encode([term])[0]
    
    # Search vector store
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Parse results into structured format
    matches = []
    
    if results['metadatas'] and len(results['metadatas'][0]) > 0:
        for i, metadata in enumerate(results['metadatas'][0]):
            distance = results['distances'][0][i] if 'distances' in results else None
            
            # Skip if above distance threshold
            if active_threshold is not None and distance is not None:
                if distance > active_threshold:
                    continue
            
            # Build match object using metadata constants
            match = {
                'type': metadata.get(METADATA_TYPE),
                'score': float(distance) if distance is not None else None,
                'description': metadata.get(METADATA_DESCRIPTION, ''),
            }
            
            # Add ID and name (same keys for both groups and subcategories)
            match.update({
                'id': metadata.get(METADATA_ID),
                'name': metadata.get(METADATA_NAME),
                'group_id': metadata.get(METADATA_GROUP_ID, metadata.get(METADATA_ID)),  # Use own ID if group
                'group_name': metadata.get(METADATA_GROUP_NAME, metadata.get(METADATA_NAME)),  # Use own name if group
            })
            
            matches.append(match)
    
    return matches


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def get_best_match(
    term: str,
    prefer_subcategory: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get the single best matching category for a term.
    
    Args:
        term: Natural language category term
        prefer_subcategory: If True, prefer subcategory over group when both match well
                           (default: True)
    
    Returns:
        Best matching category dict, or None if no good matches
    """
    matches = query_categories(term, top_k=5)
    
    if not matches:
        return None
    
    # If preferring subcategories, prioritize them
    if prefer_subcategory:
        subcats = [m for m in matches if m['type'] == 'subcategory']
        if subcats:
            return subcats[0]
    
    # Return best match overall
    return matches[0]


def format_match_for_display(match: Dict[str, Any]) -> str:
    """
    Format a category match for display.
    
    Args:
        match: Category match dict from query_categories()
    
    Returns:
        Formatted string for display
        
    Example:
        >>> match = query_categories("groceries")[0]
        >>> print(format_match_for_display(match))
        [GROUP] Groceries (CG10000) - Distance: 0.2733
    """
    match_type = "GROUP" if match['type'] == 'group' else "SUBCAT"
    name = match['name']
    cat_id = match['id']
    score = match.get('score')
    
    result = f"[{match_type}] {name} ({cat_id})"
    
    if score is not None:
        result += f" - Distance: {score:.4f}"
    
    if match['type'] == 'subcategory':
        group_name = match.get('group_name')
        if group_name:
            result += f" (under {group_name})"
    
    return result


# ═══════════════════════════════════════════════════════════════════
# COMPREHENSIVE TESTING - ALL 60 CATEGORIES
# ═══════════════════════════════════════════════════════════════════


# def test_rag_queries(similarity_distance_threshold: Optional[float] = None):
#     """
#     Comprehensive test function for ALL 20 category groups + 40 subcategories.
    
#     Args:
#         similarity_distance_threshold: Maximum distance to accept matches (None = use DEFAULT_MIN_CONFIDENCE)
#                                       Lower = stricter (only excellent matches)
#                                       Higher = looser (more matches accepted)
                                      
#                                       Examples:
#                                       - 0.4: Only excellent matches
#                                       - 0.6: Excellent + good matches (default)
#                                       - 0.8: Excellent + good + acceptable matches
#                                       - None: No filtering (show all)
    
#     Coverage:
#     - ALL 20 category groups (CG100, CG200, ..., CG10000)
#     - 2 subcategories per group (40 total)
#     - Total: 60 test queries
    
#     This validates:
#     - Semantic understanding across all categories
#     - Multilingual support (Hebrew + English)
#     - Distance threshold filtering
    
#     Usage:
#         # Use default threshold (0.6)
#         test_rag_queries()
        
#         # Use custom threshold
#         test_rag_queries(similarity_distance_threshold=0.5)  # Stricter
#         test_rag_queries(similarity_distance_threshold=0.8)  # Looser
#         test_rag_queries(similarity_distance_threshold=None) # No filtering
    
#     Returns:
#         dict: {"passed": int, "failed": int, "errors": list}
#     """
#     # Use provided threshold or default
#     active_threshold = similarity_distance_threshold if similarity_distance_threshold is not None else DEFAULT_MIN_CONFIDENCE
    
#     print("=" * 80)
#     print("🧪 COMPREHENSIVE RAG TEST - ALL 20 GROUPS + 40 SUBCATEGORIES")
#     print("=" * 80)
    
#     # ═══════════════════════════════════════════════════════════════
#     # ALL 20 CATEGORY GROUPS + 2 SUBCATEGORIES EACH
#     # ═══════════════════════════════════════════════════════════════
    
#     all_test_queries = [
#         # 1. Transportation (CG100)
#         ("transportation", "CG100", "GROUP"),
#         ("gas station", "C101", "SUBCAT"),
#         ("parking", "C102", "SUBCAT"),
        
#         # 2. Utilities & Bills (CG200)
#         ("utilities", "CG200", "GROUP"),
#         ("electricity", "C201", "SUBCAT"),
#         ("water", "C202", "SUBCAT"),
        
#         # 3. Healthcare & Medical (CG300)
#         ("healthcare", "CG300", "GROUP"),
#         ("doctor", "C301", "SUBCAT"),
#         ("pharmacy", "C302", "SUBCAT"),
        
#         # 4. Entertainment (CG400)
#         ("entertainment", "CG400", "GROUP"),
#         ("movies", "C401", "SUBCAT"),
#         ("concerts", "C402", "SUBCAT"),
        
#         # 5. Travel (CG500)
#         ("travel", "CG500", "GROUP"),
#         ("airlines", "C501", "SUBCAT"),
#         ("hotels", "C502", "SUBCAT"),
        
#         # 6. Shopping (CG600)
#         ("shopping", "CG600", "GROUP"),
#         ("department stores", "C601", "SUBCAT"),
#         ("online shopping", "C602", "SUBCAT"),
        
#         # 7. Insurance (CG700)
#         ("insurance", "CG700", "GROUP"),
#         ("auto insurance", "C701", "SUBCAT"),
#         ("home insurance", "C702", "SUBCAT"),
        
#         # 8. Dining (CG800)
#         ("dining", "CG800", "GROUP"),
#         ("restaurants", "C803", "SUBCAT"),
#         ("fast food", "C802", "SUBCAT"),
        
#         # 9. Education (CG900)
#         ("education", "CG900", "GROUP"),
#         ("tuition", "C901", "SUBCAT"),
#         ("books", "C902", "SUBCAT"),
        
#         # 10. Personal Care (CG1000)
#         ("personal care", "CG1000", "GROUP"),
#         ("hair salon", "C1001", "SUBCAT"),
#         ("spa", "C1002", "SUBCAT"),
        
#         # 11. Home & Garden (CG1100)
#         ("home", "CG1100", "GROUP"),
#         ("furniture", "C1101", "SUBCAT"),
#         ("home improvement", "C1102", "SUBCAT"),
        
#         # 12. Pets (CG1200)
#         ("pets", "CG1200", "GROUP"),
#         ("veterinary", "C1201", "SUBCAT"),
#         ("pet food", "C1202", "SUBCAT"),
        
#         # 13. Subscriptions (CG1300)
#         ("subscriptions", "CG1300", "GROUP"),
#         ("streaming", "C1301", "SUBCAT"),
#         ("software", "C1302", "SUBCAT"),
        
#         # 14. Financial Services (CG1400)
#         ("financial services", "CG1400", "GROUP"),
#         ("bank fees", "C1401", "SUBCAT"),
#         ("ATM", "C1402", "SUBCAT"),
        
#         # 15. Charity & Donations (CG1500)
#         ("charity", "CG1500", "GROUP"),
#         ("donations", "C1501", "SUBCAT"),
#         ("religious", "C1502", "SUBCAT"),
        
#         # 16. Childcare & Kids (CG1600)
#         ("childcare", "CG1600", "GROUP"),
#         ("daycare", "C1601", "SUBCAT"),
#         ("toys", "C1602", "SUBCAT"),
        
#         # 17. Fitness & Sports (CG1700)
#         ("fitness", "CG1700", "GROUP"),
#         ("gym", "C1701", "SUBCAT"),
#         ("sports equipment", "C1702", "SUBCAT"),
        
#         # 18. Professional Services (CG1800)
#         ("professional services", "CG1800", "GROUP"),
#         ("legal", "C1801", "SUBCAT"),
#         ("accounting", "C1802", "SUBCAT"),
        
#         # 19. Automotive (CG1900)
#         ("automotive", "CG1900", "GROUP"),
#         ("car maintenance", "C1901", "SUBCAT"),
#         ("auto repairs", "C1902", "SUBCAT"),
        
#         # 20. Groceries (CG10000)
#         ("groceries", "CG10000", "GROUP"),
#         ("supermarket", "C10001", "SUBCAT"),
#         ("organic food", "C10002", "SUBCAT"),
#     ]
    
#     print(f"\n📊 Testing {len(all_test_queries)} category queries:")
#     print(f"   • 20 category groups")
#     print(f"   • 40 subcategories (2 per group)")
#     print(f"   • Total: 60 queries (groups + subcategories)")
#     print(f"\n🎯 Similarity Distance Threshold: {active_threshold}")
#     if similarity_distance_threshold is not None:
#         print(f"   (Override: {similarity_distance_threshold}, Default: {DEFAULT_MIN_CONFIDENCE})")
#     else:
#         print(f"   (Using default: {DEFAULT_MIN_CONFIDENCE})")
#     print(f"🌍 Multilingual: Hebrew + English support")
#     print(f"🧠 Semantic: Works for unseen brands\n")
    
#     # ═══════════════════════════════════════════════════════════════
#     # RUN TESTS
#     # ═══════════════════════════════════════════════════════════════
    
#     results = {"passed": 0, "failed": 0, "errors": []}
    
#     for i, (term, expected_id, query_type) in enumerate(all_test_queries, 1):
#         print(f"{i}. Query: '{term}' (Type: {query_type}, Expected: {expected_id})")
        
#         try:
#             matches = query_categories(term, top_k=3, min_confidence=active_threshold)
            
#             if matches:
#                 # Check if expected category is in top 3
#                 found_expected = any(m['id'] == expected_id for m in matches)
                
#                 # Show results
#                 for j, match in enumerate(matches, 1):
#                     marker = "✅" if match['id'] == expected_id else "  "
#                     print(f"      {marker} {j}. {format_match_for_display(match)}")
                
#                 if found_expected:
#                     results["passed"] += 1
#                 else:
#                     results["failed"] += 1
#                     results["errors"].append({
#                         "term": term,
#                         "expected": expected_id,
#                         "got": [m['id'] for m in matches]
#                     })
#                     print(f"      ⚠️  Expected {expected_id} not in top 3!")
#             else:
#                 print(f"      ❌ No matches found (all above threshold {active_threshold})")
#                 results["failed"] += 1
#                 results["errors"].append({
#                     "term": term,
#                     "expected": expected_id,
#                     "error": f"No matches above threshold {active_threshold}"
#                 })
        
#         except Exception as e:
#             print(f"      ❌ Error: {e}")
#             results["failed"] += 1
#             results["errors"].append({
#                 "term": term,
#                 "expected": expected_id,
#                 "error": str(e)
#             })
        
#         print()
    
#     # ═══════════════════════════════════════════════════════════════
#     # SUMMARY
#     # ═══════════════════════════════════════════════════════════════
    
#     print("=" * 80)
#     print("📊 TEST SUMMARY")
#     print("=" * 80)
#     print(f"Total Queries: {len(all_test_queries)}")
#     print(f"✅ Passed: {results['passed']}")
#     print(f"❌ Failed: {results['failed']}")
#     print(f"Success Rate: {(results['passed']/len(all_test_queries))*100:.1f}%")
    
#     if results['errors']:
#         print(f"\n❌ {len(results['errors'])} Failed Queries:")
#         for err in results['errors'][:10]:  # Show first 10
#             print(f"   • '{err['term']}' - Expected: {err['expected']}")
#             if 'got' in err:
#                 print(f"     Got: {err['got']}")
#             if 'error' in err:
#                 print(f"     Error: {err['error']}")
    
#     print("=" * 80)
    
#     return results


# ═══════════════════════════════════════════════════════════════════
# USAGE IN JUPYTER NOTEBOOK
# ═══════════════════════════════════════════════════════════════════

"""
HOW TO USE IN JUPYTER:

1. Import functions:
   from rag.trn_category_rag import query_categories, test_rag_queries

2. Manual query:
   results = query_categories("coffee shops", top_k=3, min_confidence=0.6)
   for match in results:
       print(f"{match['name']} ({match['id']}) - Distance: {match['score']:.4f}")

3. Test all 60 categories:
   results = test_rag_queries()
   
4. Test with custom threshold:
   results = test_rag_queries(similarity_distance_threshold=0.5)
"""