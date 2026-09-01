# TechJam 2026 Track 4 - Conversational Shopping Copilot

A stateful product-retrieval agent that converts a shopper's evolving language
into typed constraints, retrieves catalog evidence, and returns ranked products
plus a useful clarification. 

## Quick Start Requirements

- Python 3.10 or newer.
- For the **Offline Version**: Only the Python standard library is required! No API keys are needed.
- For the **Online Version**: Requires the openai package and an OpenAI API key.

## Installation

`ash
git clone https://github.com/LiuYuan555/jamming.git
cd jamming
mkdir -p data
curl -L https://github.com/TechJam2026/techjam-conversational-search/releases/latest/download/catalog.jsonl.gz -o catalog.jsonl.gz
gzip -dc catalog.jsonl.gz > data/catalog.jsonl
python3 -m venv .venv
# Activate your venv (e.g., .venv\Scripts\activate on Windows or source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
`

## How to Run the Agent

We have configured the starter/agent.py to use an advanced IntentConditionedHybridAgent override block at the very bottom of the file. By default, it is configured to use the **Shannon Entropy** question policy and is completely offline.

### 1. Running the Offline Version (Default)
The offline version is currently active! It uses the Shannon Entropy Gate, Intent Classifier, Intent Router, BM25, and Constraint Reranker. It completely disables the OpenAI Semantic Rewriting tools.
`ash
python3 -m evaluator.local_evaluator
`

### 2. Running the Online Version
To turn on the OpenAI semantic tools:
1. Open starter/agent.py and scroll to the very bottom.
2. In the AgentConfig, change hybrid_config=HybridConfig(enabled=False) to hybrid_config=HybridConfig(enabled=True).
3. Export your OpenAI API key:
`ash
# Windows
set OPENAI_API_KEY=your_key_here
# Mac/Linux
export OPENAI_API_KEY=your_key_here
`
4. Run the evaluator:
` ash
python3 -m evaluator.local_evaluator
`

### Running the Interactive UI
To visually test the agent in your browser:
```bash
pip install -r requirements.txt
python3 -m ui.server
```
Then open http://127.0.0.1:8000 in your web browser.
