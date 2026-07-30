import json
import logging

from langchain_core.tools import tool

from src.config import DATA_PATH
from src.retrieval import (
    retrieve_records,
    format_context,
    get_retrieved_ids,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def lookup_record(record_id: str) -> dict:
    """
    Retrieve an exact knowledge base record by record ID.

    Use this when the user asks about a specific ID
    like KB001 or KB020.
    """

    logger.info(
        f"Tool called: lookup_record(record_id='{record_id}')"
    )

    with open(DATA_PATH, "r") as f:
        records = json.load(f)

    for record in records:
        if record["id"].lower() == record_id.lower():

            logger.info(
                f"Record found: {record['id']}"
            )

            context = f"""
Record ID: {record['id']}
Title: {record['title']}
Category: {record['category']}
Description: {record['description']}
Keywords: {", ".join(record['keywords'])}
Source: {record['source']}
Last Updated: {record['last_updated']}
"""

            return {
                "context": context,
                "matched_records": [record["id"]],
                "sources": [record["source"]],
            }

    logger.warning(
        f"No record found for: {record_id}"
    )

    return {
        "error": f"No record found for {record_id}",
        "context": "",
        "matched_records": [],
        "sources": [],
    }


@tool
def retrieve_knowledge(question: str) -> dict:
    """
    Search the company knowledge base.

    Use this for general questions where the user
    does not provide an exact record ID.
    """

    logger.info(
        f"Tool called: retrieve_knowledge(question='{question}')"
    )

    documents = retrieve_records(question)

    retrieved_ids = get_retrieved_ids(documents)

    sources = list({
        document.metadata["source"]
        for document in documents
    })

    logger.info(
        f"Retrieved IDs: {retrieved_ids}"
    )

    return {
        "context": format_context(documents),
        "matched_records": retrieved_ids,
        "sources": sources,
    }