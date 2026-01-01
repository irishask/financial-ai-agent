# Financial AI Agent

An AI-powered conversational agent that helps retail banking customers understand their financial behavior through natural language.

---

## 🎯 Overview

An intelligent system that:
- Understands natural language financial questions
- Provides accurate, simple answers to customers
- Maintains complete reasoning trails for regulatory compliance

**Example:**
```
User: "How much did I spend on dining last month compared to September?"

Agent: "You spent $389.40 on dining in November compared to $668.20 in 
        September. That's a decrease of $278.80 (42% less)."
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
| Caching | — | Redis / ElastiCache |

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
└── FinantialAI_Run_Demo_with_RAG_tests.ipynb
```

---

## ⚙️ Installation

```bash
git clone https://github.com/irishask/financial-ai-agent.git
cd financial-ai-agent

conda create -n financial-agent python=3.12
conda activate financial-agent

pip install -r requirements.txt

# Set up API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

---

## 🚀 Running

> **Note:** The current repository uses a fixed reference date for test reproducibility. For production, update the date configuration in `prompts/llm1_prompt.py`.

### Jupyter Notebook

```bash
jupyter notebook FinantialAI_Run_Demo_with_RAG_tests.ipynb
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

### Validation

- ✅ RAG category mapping accuracy
- ✅ CLEAR/VAGUE classification
- ✅ Temporal resolution
- ✅ LLM-2 grounding (no hallucination)
- ✅ Back-office logging
- ✅ Answer accuracy

---

## 📄 Documentation

For complete technical details, see: `docs/FINAL_Financial_AI_Agent_Architecture.docx`

---

## 👤 Author

**Irena Shtelman Kravitz**  
Data Scientist | GenAI Specialist

---

## 📝 License

© 2025 Irena Shtelman Kravitz. All rights reserved.