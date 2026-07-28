from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=256,
    temperature=0,
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("what is LangChain?")

print(response.content)
