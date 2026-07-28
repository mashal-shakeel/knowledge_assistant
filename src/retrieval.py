import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/knowledge_base.json"
DB_PATH = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_documents():

    with open(DATA_PATH, "r") as f:
        records = json.load(f)

    documents = []

    for record in records:
        content = f"""
Record ID: {record['id']}
Title: {record['title']}
Category: {record['category']}
Description: {record['description']}
Keywords: {", ".join(record['keywords'])}
Source: {record['source']}
Last Updated: {record['last_updated']}
"""

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "id": record["id"],
                    "title": record["title"],
                    "source": record["source"],
                    "category": record["category"],
                },
            )
        )

    return documents


def create_vector_store():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    print(f"Indexed {len(chunks)} chunks.")


def load_vector_store():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )


def retrieve_records(question: str, k: int = 3):
    vector_store = load_vector_store()

    return vector_store.similarity_search(
        question,
        k=k,
    )


def format_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)