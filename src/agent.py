from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from models import KnowledgeResponse

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
    temperature=0,
)

chat = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=KnowledgeResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a company knowledge base assistant.

Use ONLY the provided knowledge records to answer the user's question.

If the answer cannot be determined from the provided records:
- Answer that the information is unavailable.
- Set needs_human_review to true.
- Use a low confidence score.
- Leave matched_records and sources empty if nothing is relevant.

{format_instructions}
"""
        ),
        (
            "human",
            """
Knowledge Records:
{context}

Question:
{question}
"""
        ),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)


def ask_question(question: str, context: str) -> KnowledgeResponse:
    chain = prompt | chat | parser

    return chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )