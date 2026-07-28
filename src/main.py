from dotenv import load_dotenv
from agent import ask_question
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=256,
    temperature=0,
)

chat = ChatHuggingFace(llm=llm)

#dummy context for testing
context = """
Record ID: KB001
Title: Annual Leave Policy
Category: HR
Source: Employee Handbook

Employees receive 20 days of paid annual leave per calendar year.
Leave requests must be approved by the employee's manager.
"""

# response = ask_question(
#     question="How many annual leave days do employees receive?",
#     context=context,
# )

response = ask_question(
    question="What is the company's maternity leave policy?",
    context=context,
)

print(response.model_dump_json(indent=4))