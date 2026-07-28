from src.agent import ask_question

# only run once
# from src.retrieval import create_vector_store
# create_vector_store()

question = "Tell me about KB001"

response = ask_question(
    question
)

print(
    response.model_dump_json(indent=4)
)