# Financial AI Agent

An AI-powered conversational agent that helps retail banking customers understand their financial behavior through natural language — like asking a financial advisor.

> **Interview Project** | Data Scientist (GenAI) Position | December 2025

---

## 🎯 The Challenge

Build an intelligent system that:
- Understands natural language financial questions
- Provides accurate, simple answers to customers
- Maintains complete reasoning trails for regulatory compliance

**Example:**
```
User: "How much did I spend on dining last month compared to September?"

Agent: "You spent $389.40 on dining in November compared to $668.20 in 
        September. That's a decrease of $278.80 (42% less)."

BackOffice Log: [complete audit trail with data sources, filters, calculations]
```

---

## 🏗️ Architecture

**2-LLM Pipeline orchestrated by LangGraph:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM-1: ROUTER                                                       │
│  • Classify: CLEAR or VAGUE                                          │
│  • Resolve temporal references → exact dates                         │
│  • Map categories via RAG → category IDs                             │
│  • Detect missing info → generate clarifying questions               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        ┌───────────┐                   ┌─────────────┐
        │   CLEAR   │                   │    VAGUE    │
        └─────┬─────┘                   └──────┬──────┘
              │                                │
              ▼                                ▼
┌─────────────────────────┐         ┌──────────────────────┐
│  LLM-2: EXECUTOR        │         │  VAGUE HANDLER       │
│  • Call tools           │         │  • Return question   │
│  • Query transactions   │         │  • Skip LLM-2 (save  │
│  • Generate answer      │         │    cost, no halluc.) │
│  • Log reasoning        │         └──────────────────────┘
└─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DUAL OUTPUT                                                         │
│  • Customer: Simple, conversational answer                           │
│  • BackOffice: Full audit trail (tables, filters, calculations)      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why 2 LLMs?**
- **Cost optimization:** Cheaper model for routing, capable model for execution
- **Better debugging:** Know exactly where issues occur
- **VAGUE queries skip LLM-2:** Saves cost on incomplete queries

---

## ✅ Features

### 5 Core Use Case Categories

| UC | Category | Examples |
|----|----------|----------|
| UC-01 | Direct Retrieval | "What is my current balance?", "Show my last transaction" |
| UC-02 | Aggregation | "How much did I spend last month?", "Total income this year" |
| UC-03 | Temporal | "Spending this week", "Transactions from March" |
| UC-04 | Category-Based | "How much on groceries?", "Show dining transactions" |
| UC-05 | Ambiguity Handling | "Recent transactions" → asks for timeframe |

### 3 Complexity Challenges Solved

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Temporal Logic** | "Last month" = calendar month or rolling 30 days? | LLM-1 resolves to exact dates before LLM-2 |
| **Category Mapping** | "groceries" → which of 100+ categories? | RAG with ChromaDB + semantic search |
| **Intent Disambiguation** | "recent" = 7 days? 30 days? | VAGUE detection → clarifying questions |

### Additional Features

- **Multi-turn conversations:** Collects missing info across turns
- **Conversation Summary:** Remembers user preferences within session
- **Grounding verification:** LLM-2 uses ONLY data from LLM-1 (no hallucination)
- **Back-office logging:** Complete audit trail for compliance

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph (state machine) |
| LLM | Claude 3.5 Sonnet (Anthropic) |
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector Store | ChromaDB (persistent) |
| Framework | LangChain |
| Language | Python 3.12 |

---

## 📁 Project Structure

```
financial-ai-agent/
├── data/
│   ├── transactions.csv          # Demo: 147 transactions (2024-2025)
│   ├── CategoriesKB.json         # 20 groups, 88 subcategories
│   └── chroma_trn_categories/    # Persistent vector store
├── prompts/
│   ├── llm1_prompt.py            # Router prompt + injection functions
│   └── llm2_prompt.py            # Executor prompt builder
├── schemas/
│   ├── router_models.py          # Pydantic models (GraphState, RouterOutput, etc.)
│   ├── transactions_tool.py      # query_transactions tool
│   └── trn_category_tool.py      # RAG search tool
├── tests/
│   ├── _new_QA_mapping.json      # 17 test queries with expectations
│   ├── pipeline_rag_tests.py     # UC-04 tests with validation
│   ├── pipeline_no_rag_tests.py  # UC-01/UC-05 tests
│   ├── llm1_tests.py             # Multi-turn VAGUE→CLEAR tests
│   └── dynamic_expected_calculator.py  # Calculates expected values from CSV
├── graph_definition.py           # LangGraph nodes and edges
├── trn_category_rag.py           # RAG vector store builder
└── FinantialAI_Run_Demo_with_RAG_tests.ipynb  # Main demo notebook
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/financial-ai-agent.git
cd financial-ai-agent

# Create environment
conda create -n financial-agent python=3.12
conda activate financial-agent

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### Requirements

```
langchain>=0.3.0
langchain-anthropic>=0.3.0
langgraph>=0.2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pandas>=2.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

---

## 📅 Demo Date Configuration

The demo uses a **frozen reference date** to align with the synthetic transaction data.

### Two Files Must Be Synchronized

| File | Line | Demo Setting |
|------|------|--------------|
| `prompts/llm1_prompt.py` | ~546 | `current_date = "2025-12-01"` |
| `tests/dynamic_expected_calculator.py` | ~56 | `self.today = datetime(2025, 12, 1).date()` |

### Why This Matters

| Reference Date | "Last month" | "This year" | Data Available |
|----------------|--------------|-------------|----------------|
| ❌ Jan 1, 2026 | Dec 2025 | 2026 | Limited/None |
| ✅ Dec 1, 2025 | **Nov 2025** | **2025** | Rich data |

### Demo Data Coverage (Reference: Dec 1, 2025)

| Category | Nov 2025 | Oct 2025 | Sep 2025 |
|----------|----------|----------|----------|
| Dining (CG800) | $389.40 | $663.60 | $668.20 |
| Groceries (CG10000) | $524.10 | — | — |
| Healthcare (CG300) | $136.75 | $195.80 | — |
| Utilities (CG200) | $222.29 | $205.39 | — |
| Gym (C1701) | $49.99/mo | $49.99/mo | $49.99/mo |

### Switching to Production

```python
# In llm1_prompt.py (line ~546):
current_date = date.today().isoformat()   # PRODUCTION

# In dynamic_expected_calculator.py (line ~56):
self.today = datetime.now().date()         # PRODUCTION
```

---

## 🚀 Running the Demo

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook FinantialAI_Run_Demo_with_RAG_tests.ipynb
```

### Option 2: Run Tests Directly

```bash
# Build/load vector store
python trn_category_rag.py

# Run UC-04 RAG tests
python tests/pipeline_rag_tests.py

# Run UC-01/UC-05 tests
python tests/pipeline_no_rag_tests.py

# Run multi-turn VAGUE tests
python tests/llm1_tests.py
```

---

## 📊 Test Results

**All tests passing: 17/17 = 100%**

| Test Suite | Queries | Result |
|------------|---------|--------|
| CLEAR without RAG (UC-01) | #1, #2 | 2/2 ✅ |
| VAGUE without RAG (UC-05) | #12, #13 | 2/2 ✅ |
| RAG Pipeline (UC-04) | #3, #4, #7, #8, #9, #10, #16, #17 | 8/8 ✅ |
| VAGUE Multi-Turn (UC-05) | #11, #12, #13, #14, #15 | 5/5 ✅ |

### Validation Checks (15 per query)

- ✅ RAG tool called
- ✅ Category mapping correct
- ✅ CLEAR/VAGUE classification correct
- ✅ Dates resolved correctly
- ✅ LLM-2 grounded (uses only LLM-1 data)
- ✅ Correct tables accessed
- ✅ Correct filters applied
- ✅ Answer matches expected values

---

## 🗺️ Roadmap (Not Implemented)

| Component | Purpose | Status |
|-----------|---------|--------|
| Multi-Turn Security | Prompt injection detection | ❌ Designed |
| Resilience | Retry logic, circuit breakers | ❌ Roadmap |
| Observability Dashboards | Grafana metrics | ❌ Roadmap |
| Production Monitoring | Latency, accuracy tracking | ❌ Roadmap |

---

## 📄 Architecture Document

For complete technical details, see: `docs/FINAL_Financial_AI_Agent_Architecture.docx`

---

## 👤 Author

**[Your Name]**  
Data Scientist | GenAI Specialist

---

## 📝 License

This project was created as part of an interview assignment. Not for commercial use.
