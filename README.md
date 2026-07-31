# Knowledge Base Assistant

A LangChain-based knowledge base assistant that answers questions using a local JSON knowledge base with Retrieval-Augmented Generation (RAG).

## Features

- Answer questions using a local knowledge base
- Exact record lookup by record ID (e.g. `KB001`)
- Semantic search using Chroma vector database
- Structured **JSON** responses with Pydantic
- Tool calling with LangChain
- Request logging and token usage tracking
- I added automated evaluation tests with pytest

## Project Structure

```
knowledge_assistant/
├── data/
│   └── knowledge_base.json
├── evaluation/
│   └── test_cases.json
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── main.py
│   ├── models.py
│   ├── retrieval.py
│   └── tools.py
├── tests/
│   └── test_agent.py
├── .env.example
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Hugging Face API token.

```text
HF_TOKEN=your_huggingface_token
```

## Run the Application

From the project root:

```bash
python -m src.main
```

The vector database will be created automatically if it does not already exist. I've also added a question when I was testing system manually.

## Run the Tests

From the project root:

```bash
python -m pytest tests/test_agent.py -v -s
```

## Example Questions

- Tell me about KB001.
- Explain KB015.
- What is the company's annual leave policy?
- What are the password requirements?
- Compare KB001 and KB002.
- Tell me about KB999.
- Who founded Microsoft?
