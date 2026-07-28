from dotenv import load_dotenv

from src.agent import ask_question
from src.retrieval import (
    create_vector_store,
    retrieve_records,
    format_context,
)

load_dotenv()

# only running once 
# create_vector_store()

question = "How many annual leave days do employees receive?"

documents = retrieve_records(question)

context = format_context(documents)

response = ask_question(
    question=question,
    context=context,
)

print(response.model_dump_json(indent=4))