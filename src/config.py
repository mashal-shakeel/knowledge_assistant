import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-7B-Instruct"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

HF_TOKEN = os.getenv("HF_TOKEN")

DB_PATH = "chroma_db"

DATA_PATH = "data/knowledge_base.json"

LANGCHAIN_TRACING_V2 = os.getenv(
    "LANGCHAIN_TRACING_V2",
    "false"
)

LANGCHAIN_API_KEY = os.getenv(
    "LANGCHAIN_API_KEY"
)