# Financial AI Agent

An AI-powered conversational agent that helps retail banking customers understand their financial behavior through natural language.

---

## 🎯 The Challenge

### Business Problem
Bank customers want to ask simple questions like *"How much did I spend on groceries last month?"* — but today this requires navigating complex banking interfaces and manual data analysis.

### Solution Requirements
Build an intelligent system that:
- Understands natural language financial questions
- Provides **simple answers** to customers
- Maintains **complete audit trails** for regulatory compliance

**Example:**
```
User: "Show me expenses related to car ownership and mobility, 
       but exclude public transportation, this year"

Agent: "Your private transportation expenses this year total $2,847.50 
        across 47 transactions (gas, parking, ride-sharing). 
        Public transportation excluded as requested."

BackOffice Log: [full audit trail with RAG category mapping, 
                inclusion/exclusion logic, filters, calculations]
```

### Dual Output
| Output | Audience | Content |
|--------|----------|---------|
| **Customer Answer** | User | Simple, conversational response |
| **Back-Office Log** | Compliance | Full reasoning trail, data sources, calculations |

---

## 🧩 Query Types & Complexity Challenges

### 5 Core Query Types

| UC | Type | Example |
|----|------|---------|
| UC-01 | Direct Retrieval | "What is my current balance?" |
| UC-02 | Aggregation | "How much did I spend last month?" |
| UC-03 | Temporal | "Transactions from March" |
| UC-04 | Category-Based | "Show dining transactions" |
| UC-05 | Ambiguity | "Recent transactions" → needs clarification |

### 3 Complexity Challenges

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Temporal Logic** | "Last month" = calendar month or rolling 30 days? | LLM-1 resolves to exact dates |
| **Category Mapping** | "groceries" → which of 100+ categories? | RAG semantic search |
| **Intent Disambiguation** | "recent" = 7 days? 30 days? | Multi-turn clarification |

---

## 🏗️ Architecture

**2-LLM Pipeline with Multi-Turn Clarification:**

```
                              ┌─────────────────────┐
                              │     USER QUERY      │
                              └──────────┬──────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LLM-1: ROUTER                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │   TEMPORAL   │  │   CATEGORY   │  │    INTENT    │                      │
│  │    LOGIC     │  │  MAPPING     │  │ DISAMBIGUATION│                      │
│  │              │  │   (RAG)      │  │              │                      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                      │
│         │                 │                 │                              │
│         ▼                 ▼                 ▼                              │
│      CLEAR?            CLEAR?            CLEAR?                            │
│         │                 │                 │                              │
│         └────────────┬────┴─────────────────┘                              │
│                      ▼                                                     │
│              ┌──────────────┐                                              │
│              │  ALL CLEAR?  │                                              │
│              └──────┬───────┘                                              │
└─────────────────────┼──────────────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
        YES                      NO
     ┌───────┐              ┌─────────┐
     │ CLEAR │              │  VAGUE  │
     └───┬───┘              └────┬────┘
         │                       │
         │                       ▼
         │              ┌──────────────────────┐
         │              │  CLARIFICATION       │◄───────────────┐
         │              │  ┌────────────────┐  │                │
         │              │  │ Ask user for   │  │                │
         │              │  │ missing info   │  │                │
         │              │  └───────┬────────┘  │                │
         │              │          │           │                │
         │              │          ▼           │                │
         │              │  ┌────────────────┐  │                │
         │              │  │ Update         │  │                │
         │              │  │ Conversation   │  │                │
         │              │  │ Summary        │  │                │
         │              │  └───────┬────────┘  │                │
         │              │          │           │                │
         │              └──────────┼───────────┘                │
         │                         │                            │
         │                         ▼                            │
         │              ┌────────────────────┐                  │
         │              │  USER RESPONDS     │                  │
         │              │  (Multi-Turn)      │                  │
         │              └─────────┬──────────┘                  │
         │                        │                             │
         │                        ▼                             │
         │              ┌────────────────────┐                  │
         │              │  RE-EVALUATE       │                  │
         │              │  with LLM-1        │                  │
         │              └─────────┬──────────┘                  │
         │                        │                             │
         │                   NOW CLEAR?                         │
         │                        │                             │
         │              ┌─────────┴─────────┐                   │
         │              ▼                   ▼                   │
         │             YES                  NO                  │
         │              │                   │                   │
         │              │                   └───────────────────┘
         │              │                   (loop back to clarification)
         ▼              ▼
┌─────────────────────────────┐
│  LLM-2: EXECUTOR            │
│  • Query DB                 │
│  • Calculate                │
│  • Generate answer          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  DUAL OUTPUT                                                    │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  CUSTOMER ANSWER    │    │  BACK-OFFICE LOG                │ │
│  │  Simple response    │    │  • Original query               │ │
│  │  to user            │    │  • Conversation summary         │ │
│  │                     │    │  • Resolved dates & categories  │ │
│  │                     │    │  • SQL filters applied          │ │
│  │                     │    │  • Transactions analyzed        │ │
│  │                     │    │  • Aggregations used            │ │
│  │                     │    │  • Reasoning steps              │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Why 2 LLMs?
- **Cost optimization:** Cheaper model for routing, capable model for execution
- **Better debugging:** Know exactly where issues occur
- **VAGUE queries skip LLM-2:** No hallucinated answers on incomplete queries

---

## 📋 Back-Office Logging

Every query generates a complete audit trail:

```json
{
  "original_query": "Show me expenses related to car ownership and mobility, but exclude public transportation, this year",
  "conversation_summary": {
    "time_window": "this_year",
    "resolved_dates": {"start": "2025-01-01", "end": "2025-12-31"}
  },
  "category_mapping": {
    "user_term": "car ownership and mobility",
    "included": ["C101 (Gas Station)", "C102 (Parking)", "C107 (Taxi & Ride Sharing)"],
    "excluded": ["C103 (Public Transportation)"],
    "method": "RAG semantic search"
  },
  "execution": {
    "tables_accessed": ["transactions"],
    "filters_applied": [
      "categoryId IN ('C101', 'C102', 'C107')",
      "categoryId NOT IN ('C103')",
      "date BETWEEN '2025-01-01' AND '2025-12-31'"
    ],
    "transactions_analyzed": 47,
    "aggregations_used": ["SUM(amount)", "COUNT(*)"]
  },
  "reasoning_steps": [
    "Mapped 'car ownership and mobility' → C101, C102, C107 via RAG",
    "Excluded 'public transportation' → C103",
    "Resolved 'this year' → Jan 1 - Dec 31, 2025",
    "Retrieved 47 transactions totaling $2,847.50"
  ],
  "answer": "Your private transportation expenses this year total $2,847.50 across 47 transactions."
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph (state machine) |
| LLM | Claude 3.5 Sonnet (Anthropic) |
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Language | Python 3.12 |

---

## ☁️ Deployment

**Current:** Local environment

**Cloud-Ready Architecture:**

| Component | Local | Cloud |
|-----------|-------|-------|
| Vector Store | ChromaDB | Pinecone / AWS OpenSearch |
| Transaction DB | CSV | PostgreSQL / DynamoDB |
| LLM API | Anthropic API | Anthropic API / AWS Bedrock |
| Orchestration | Python | AWS Lambda / ECS / Kubernetes |

---

## 📁 Project Structure

```
financial-ai-agent/
├── data/
│   ├── transactions.csv
│   ├── CategoriesKB.json
│   └── chroma_trn_categories/
├── prompts/
│   ├── llm1_prompt.py
│   └── llm2_prompt.py
├── schemas/
│   ├── router_models.py
│   ├── transactions_tool.py
│   └── trn_category_tool.py
├── tests/
│   ├── pipeline_rag_tests.py
│   ├── pipeline_no_rag_tests.py
│   ├── llm1_tests.py
│   └── dynamic_expected_calculator.py
├── graph_definition.py
├── trn_category_rag.py
└── FinantialAI_Run_Demo.ipynb
```

---

## ⚙️ Installation

```bash
git clone https://github.com/irishask/financial-ai-agent.git
cd financial-ai-agent

conda create -n financial-agent python=3.12
conda activate financial-agent

pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

---

## 🚀 Running

> **Note:** Repository uses a fixed reference date for test reproducibility. For production, update date configuration in `prompts/llm1_prompt.py`.

### Jupyter Notebook

```bash
jupyter notebook FinantialAI_Run_Demo.ipynb
```

### Test Suite

```bash
python trn_category_rag.py
python tests/pipeline_rag_tests.py
python tests/pipeline_no_rag_tests.py
python tests/llm1_tests.py
```

---

## 📊 Test Coverage

| Category | Coverage |
|----------|----------|
| Direct Retrieval (UC-01) | Balance queries, last transaction |
| Aggregation (UC-02) | Spending totals, period comparisons |
| Temporal (UC-03) | Date resolution, cross-year queries |
| Category-Based (UC-04) | RAG mapping, hierarchy navigation |
| Ambiguity Handling (UC-05) | VAGUE detection, multi-turn clarification |

---

## 👤 Author

**Irena Shtelman Kravitz**  
Data Scientist | GenAI Specialist

---

## 📝 License

© 2025 Irena Shtelman Kravitz. All rights reserved.
