import json

from langchain_core.tools import tool

from src.config import DATA_PATH
from src.retrieval import retrieve_records, format_context


@tool
def lookup_record(record_id: str) -> dict:
    """
    Retrieve an exact knowledge base record by record ID.

    Use this when the user asks about a specific ID
    like KB001 or KB020.
    """

    with open(DATA_PATH, "r") as f:
        records = json.load(f)

    for record in records:
        if record["id"].lower() == record_id.lower():
            return record

    return {
        "error": f"No record found for {record_id}"
    }



@tool
def retrieve_knowledge(question: str) -> str:
    """
    Search the company knowledge base.

    Use this for general questions where the user
    does not provide an exact record ID.
    """

    documents = retrieve_records(question)

    return format_context(documents)