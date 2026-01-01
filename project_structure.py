import os
from pathlib import Path

def show_project_structure(root_path=".", max_depth=4, exclude_dirs=None):
    """
    Display project directory structure in a tree format.
    
    Args:
        root_path: Starting directory (default: current directory)
        max_depth: Maximum depth to traverse
        exclude_dirs: List of directory names to exclude (e.g., ['.git', '__pycache__'])
    """
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.ipynb_checkpoints', 'node_modules', '.venv', 'venv'}
    
    root = Path(root_path).resolve()
    
    print(f"📁 PROJECT STRUCTURE: {root.name}")
    print("=" * 80)
    
    def _tree(directory, prefix="", depth=0):
        if depth > max_depth:
            return
        
        try:
            contents = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        # Separate directories and files
        dirs = [item for item in contents if item.is_dir() and item.name not in exclude_dirs]
        files = [item for item in contents if item.is_file()]
        
        # Print directories first
        for i, item in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}📁 {item.name}/")
            
            extension = "    " if is_last_dir else "│   "
            _tree(item, prefix + extension, depth + 1)
        
        # Then print files
        for i, item in enumerate(files):
            is_last = i == len(files) - 1
            connector = "└── " if is_last else "├── "
            
            # Add file size
            size = item.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            
            print(f"{prefix}{connector}📄 {item.name} ({size_str})")
    
    _tree(root)
    print("=" * 80)

# # Run it
# show_project_structure()

#####################################################################
"""
📁 PROJECT STRUCTURE: EXAM_AI_Agents_Personetics
================================================================================
├── 📁 data/
│   ├── 📁 chroma_trn_categories/
│   ├── 📄 CategoriesKB.json (31.2 KB)
│   ├── 📄 CategoriesKB_old.json (4.4 KB)
│   ├── 📄 GenData_notes.txt (2.5 KB)
│   └── 📄 transactions.csv (17.0 KB)
├── 📁 prompts/
│   ├── 📄 __init__.py (325 bytes)
│   ├── 📄 llm1_prompt-Copy1.py (38.7 KB)
│   ├── 📄 llm1_prompt-old.py (34.0 KB)
│   ├── 📄 llm1_prompt.py (38.6 KB)
│   ├── 📄 llm2_prompt-Copy1.py (25.4 KB)
│   ├── 📄 llm2_prompt-old.py (26.9 KB)
│   └── 📄 llm2_prompt.py (25.4 KB)
├── 📁 rag/
├── 📁 schemas/
│   ├── 📄 executor_models_llm2-old.py (8.3 KB)
│   ├── 📄 executor_models_llm2.py (8.5 KB)
│   ├── 📄 router_models-old.py (14.2 KB)
│   ├── 📄 router_models.py (15.0 KB)
│   ├── 📄 transactions_tool-old.py (6.1 KB)
│   ├── 📄 transactions_tool.py (6.1 KB)
│   └── 📄 trn_category_tool.py (0 bytes)
├── 📁 tests/
│   ├── 📄 __init__.py (17 bytes)
│   ├── 📄 _new_QA_mapping.json (15.5 KB)
│   ├── 📄 _orig_QA_mapping_old.json (15.0 KB)
│   ├── 📄 llm1_tests-Copy1.py (36.8 KB)
│   ├── 📄 llm1_tests.py (37.8 KB)
│   ├── 📄 llm2_tests-Copy1.py (43.9 KB)
│   ├── 📄 llm2_tests.py (43.9 KB)
│   ├── 📄 pipeline_tests-Copy1.py (25.2 KB)
│   ├── 📄 pipeline_tests.py (11.8 KB)
│   ├── 📄 sql_data_test-Copy1.py (9.8 KB)
│   └── 📄 sql_data_test.py (9.8 KB)
├── 📄 backoffice_logging-old.py (13.8 KB)
├── 📄 backoffice_logging.py (14.2 KB)
├── 📄 graph_definition-old.py (33.5 KB)
├── 📄 graph_definition.py (33.5 KB)
├── 📄 project_structure.py (2.0 KB)
├── 📄 Ver1-Copy1.ipynb (34.8 KB)
└── 📄 Ver1.ipynb (37.4 KB)
"""