"""
═══════════════════════════════════════════════════════════════════════════════
BUILD CATEGORY RAG VECTOR STORE - Production Version
═══════════════════════════════════════════════════════════════════════════════

Purpose:
    Build a semantic search vector store for transaction categories using
    multilingual embeddings and rich conceptual descriptions.

Key Design:
    1. multilingual-e5-base model - Better semantics, Hebrew support
    2. Rich conceptual descriptions - Already in CategoriesKB.json
    3. Simple embedding text generation - Use descriptions as-is
    4. Standard threshold: 0.6 - Industry standard for e5-base

Architecture:
    - ChromaDB: Vector database (persistent, local, free)
    - multilingual-e5-base: Embedding model (560MB, multilingual)
    - 111 categories: 20 groups + 91 subcategories

Usage:
    from build_category_vectorstore import build_vector_store
    
    # First time or after updating CategoriesKB.json
    vectorstore = build_vector_store(force_rebuild=True)
    
    # Subsequent runs (loads existing)
    vectorstore = build_vector_store(force_rebuild=False)

Output:
    - Vector store: data/chroma_trn_categories/ (~20MB)
    - 111 embedded documents with semantic descriptions

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import json
import os
from pathlib import Path
from typing import Dict, List, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ═══════════════════════════════════════════════════════════════════════════════
# CHROMADB GLOBAL REGISTRY FIX
# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Clear ChromaDB global state to prevent "An instance already exists" errors
# This is necessary when using ChromaDB in Jupyter notebooks or when multiple
# modules create ChromaDB clients in the same Python session.
try:
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient._identifier_to_system.clear()
except Exception:
    pass  # If clearing fails, continue anyway


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Data paths
CATEGORIES_JSON_PATH = "data/CategoriesKB.json"
CHROMA_PERSIST_DIR = "data/chroma_trn_categories"
COLLECTION_NAME = "transaction_categories"

# Embedding model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"  # Multilingual, better semantics

# Standard threshold for multilingual-e5-base (used in query script)
# Industry standard based on Hugging Face documentation and production systems
# Distance ranges:
#   0.0 - 0.40 : Excellent matches (exact terms, strong semantic similarity)
#   0.40 - 0.60: Good matches (semantically related, acceptable quality)
#   0.60 - 0.75: Acceptable matches (broader semantic connection)
#   > 0.75     : Weak matches (loosely related or unrelated - filter out)
STANDARD_CONFIDENCE_THRESHOLD = 0.6  # Recommended query threshold


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC EMBEDDING TEXT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_embedding_text(item: Dict[str, Any], item_type: str) -> str:
    """
    Create embedding text from category description.
    
    Design Philosophy:
        CategoriesKB.json now contains rich, conceptual descriptions (200-300 words)
        that explain:
        - GROUPS: The domain, scope, and purpose (abstract language)
        - SUBCATEGORIES: Specific transaction types and mechanisms (concrete language)
        
        These descriptions are production-quality and don't need transformation.
        We use them as-is, only adding the category name at the end for exact
        match emphasis.
    
    Why This Works:
        - Modern embedding models (multilingual-e5-base) understand conceptual text
        - Rich descriptions provide semantic context for similarity matching
        - Category name at end ensures exact matches get high relevance
        - No keyword stuffing - pure semantic understanding
    
    Args:
        item: Category or subcategory dict from CategoriesKB.json
        item_type: 'group' or 'subcategory'
    
    Returns:
        Embedding text: "{description} {category_name}"
    
    Example Output (Healthcare & Medical Group):
        "All expenses related to maintaining physical health, treating medical 
        conditions, and accessing healthcare services. This encompasses professional 
        medical consultations, diagnostic procedures, therapeutic treatments, 
        preventive care, medical products, and health-related services. Represents 
        the complete spectrum of personal healthcare spending across all medical 
        specialties, facilities, and health service providers. Healthcare & Medical"
    
    Example Output (Pharmacy Subcategory):
        "Retail purchases of prescription medications dispensed by licensed 
        pharmacists and over-the-counter drugs at pharmaceutical retail 
        establishments. Transactions involve presenting prescriptions for filling, 
        purchasing medicines, vitamins, and health products from pharmacy stores 
        and drugstores. Distinct from direct medical care as these are product 
        purchases focused on medications and pharmaceutical retail. Pharmacy"
    """
    
    # Get category name based on type
    if item_type == "group":
        name = item.get("categoryGroupName", "")
    elif item_type == "subcategory":
        name = item.get("subCategoryName", "")
    else:
        name = ""
    
    # Get description (already rich and conceptual)
    description = item.get("description", "")
    
    # Return description + name
    # Name at end ensures exact matches get emphasized
    return f"{description} {name}"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_categories_kb(json_path: str) -> Dict[str, Any]:
    """
    Load the categories knowledge base from JSON file.
    
    Expected structure:
        {
            "category_groups": [
                {
                    "categoryGroupId": "CG10000",
                    "categoryGroupName": "Groceries",
                    "description": "All expenses related to purchasing food...",
                    "example_phrases": ["groceries"],
                    "subcategories": [
                        {
                            "subCategoryId": "C10001",
                            "subCategoryName": "Supermarket",
                            "description": "Full-service grocery store purchases...",
                            "example_phrases": ["supermarket"]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    
    Args:
        json_path: Path to CategoriesKB.json
    
    Returns:
        Loaded KB dictionary
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"CategoriesKB.json not found at: {json_path}\n"
            f"Please ensure the file exists in the data/ directory."
        )
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {json_path}: {e.msg}",
            e.doc,
            e.pos
        )
    
    # Validate structure
    if "category_groups" not in kb_data:
        raise ValueError(
            f"Invalid KB structure: 'category_groups' key not found in {json_path}"
        )
    
    return kb_data


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Build or load the category vector store.
    
    Process:
        1. Load CategoriesKB.json
        2. Extract category groups and subcategories
        3. Generate rich embedding text from descriptions
        4. Create/load ChromaDB vector store with multilingual-e5-base embeddings
        5. Return vector store for querying
    
    Args:
        force_rebuild: If True, delete existing store and rebuild from scratch.
                      If False, load existing store if available.
    
    Returns:
        Chroma: Vector store ready for querying
    
    Raises:
        FileNotFoundError: If CategoriesKB.json not found
        ValueError: If JSON structure is invalid
    
    Usage:
        # First time or after updating CategoriesKB.json
        vectorstore = build_vector_store(force_rebuild=True)
        
        # Load existing (fast)
        vectorstore = build_vector_store(force_rebuild=False)
    """
    
    print("=" * 80)
    print("🏗️  BUILDING CATEGORY RAG VECTOR STORE")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Load Categories Knowledge Base
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n📂 Loading categories from: {CATEGORIES_JSON_PATH}")
    
    kb_data = load_categories_kb(CATEGORIES_JSON_PATH)
    category_groups = kb_data.get("category_groups", [])
    
    total_groups = len(category_groups)
    total_subcats = sum(len(g.get("subcategories", [])) for g in category_groups)
    
    print(f"   ✅ Loaded {total_groups} category groups")
    print(f"   ✅ Loaded {total_subcats} subcategories")
    print(f"   ✅ Total categories: {total_groups + total_subcats}")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Load Embedding Model
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    print(f"   ⏳ First time: downloading model (~560MB)...")
    print(f"   ⏳ Subsequent runs: instant loading (cached)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if GPU available)
        encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
    )
    
    print(f"   ✅ Embedding model loaded")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Handle Force Rebuild
    # ─────────────────────────────────────────────────────────────────────
    
    persist_dir = Path(CHROMA_PERSIST_DIR)
    
    if force_rebuild and persist_dir.exists():
        print(f"\n🗑️  Force rebuild: Deleting existing vector store...")
        import shutil
        shutil.rmtree(persist_dir)
        print(f"   ✅ Deleted {persist_dir}")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Check if Vector Store Already Exists
    # ─────────────────────────────────────────────────────────────────────
    
    if persist_dir.exists() and not force_rebuild:
        print(f"\n✅ Vector store already exists at: {persist_dir}")
        print(f"   Use force_rebuild=True to rebuild from scratch")
        
        # Load existing vector store
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_dir)
        )
        
        print("\n" + "=" * 80)
        print("✅ VECTOR STORE LOADED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Location: {persist_dir}")
        print(f"🔍 Collection: {COLLECTION_NAME}")
        print(f"🧠 Model: {EMBEDDING_MODEL_NAME}")
        print(f"💡 Ready for queries via trn_category_rag.query_categories()")
        print("=" * 80)
        
        return vectorstore
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Prepare Documents for Embedding
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n📝 Preparing documents for embedding...")
    
    texts = []      # Semantic embedding text
    metadatas = []  # Category metadata
    ids = []        # Unique IDs
    
    # Process category groups
    group_count = 0
    for group in category_groups:
        group_id = group.get("categoryGroupId")
        group_name = group.get("categoryGroupName")
        
        # Create semantic embedding text from rich description
        embedding_text = create_embedding_text(group, "group")
        
        texts.append(embedding_text)
        metadatas.append({
            "type": "group",
            "id": group_id,
            "name": group_name,
            "description": group.get("description", "")
        })
        ids.append(f"group_{group_id}")
        
        group_count += 1
        
        # Process subcategories
        subcategories = group.get("subcategories", [])
        for subcat in subcategories:
            subcat_id = subcat.get("subCategoryId")
            subcat_name = subcat.get("subCategoryName")
            
            # Create semantic embedding text from rich description
            embedding_text = create_embedding_text(subcat, "subcategory")
            
            texts.append(embedding_text)
            metadatas.append({
                "type": "subcategory",
                "id": subcat_id,
                "name": subcat_name,
                "description": subcat.get("description", ""),
                "group_id": group_id,
                "group_name": group_name
            })
            ids.append(f"subcat_{subcat_id}")
    
    subcat_count = len(texts) - group_count
    
    print(f"   ✅ Prepared {len(texts)} documents:")
    print(f"      • {group_count} category groups")
    print(f"      • {subcat_count} subcategories")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Generate Embeddings and Create Vector Store
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n🔄 Generating embeddings...")
    print(f"   ⏳ This may take 30-60 seconds with multilingual-e5-base...")
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir)
    )
    
    print(f"   ✅ Generated {len(texts)} embeddings")
    print(f"   ✅ Stored in ChromaDB at: {persist_dir}")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: Success Summary
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("✅ VECTOR STORE BUILT SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Total documents: {len(texts)}")
    print(f"   • {group_count} category groups")
    print(f"   • {subcat_count} subcategories")
    print(f"📁 Stored at: {persist_dir}")
    print(f"🔍 Collection: {COLLECTION_NAME}")
    print(f"🧠 Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"🎯 Recommended query threshold: {STANDARD_CONFIDENCE_THRESHOLD}")
    print(f"🌍 Multilingual support: Hebrew + English")
    print(f"💡 Semantic understanding: Conceptual descriptions, not keyword matching")
    print(f"")
    print(f"📖 Next step - Run tests:")
    print(f"   from tests.rag_trn_tests import run_all_rag_tests")
    print(f"   run_all_rag_tests()")
    print("=" * 80)
    
    return vectorstore