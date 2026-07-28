from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_core.output_parsers import PydanticOutputParser

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from src.config import MODEL_NAME, HF_TOKEN
from src.models import KnowledgeResponse
from src.tools import (
    lookup_record,
    retrieve_knowledge,
)


llm = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    task="conversational",
    max_new_tokens=512,
    temperature=0,
    huggingfacehub_api_token=HF_TOKEN,
)


chat = ChatHuggingFace(
    llm=llm
)


tools = [
    lookup_record,
    retrieve_knowledge,
]


llm_with_tools = chat.bind_tools(
    tools
)


parser = PydanticOutputParser(
    pydantic_object=KnowledgeResponse
)


SYSTEM_PROMPT = """
You are a company knowledge base assistant.

You have access to tools.

Tool selection rules:

1. If the user provides an exact knowledge base ID
   such as KB001, KB002, use lookup_record.

2. For general questions, use retrieve_knowledge.

3. Never answer from your own knowledge.

4. If tools do not provide enough information:
   - say information is unavailable
   - set needs_human_review=true

Return:

{format_instructions}
"""


def ask_question(question: str):

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                format_instructions=
                parser.get_format_instructions()
            )
        ),
        HumanMessage(
            content=question
        )
    ]


    response = llm_with_tools.invoke(messages)
    print("Tool calls:", response.tool_calls)


    while response.tool_calls:

        messages.append(response)


        for tool_call in response.tool_calls:

            selected_tool = {
                "lookup_record": lookup_record,
                "retrieve_knowledge": retrieve_knowledge,
            }[
                tool_call["name"]
            ]


            tool_result = selected_tool.invoke(
                tool_call["args"]
            )


            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )


        # Ask model again after tool result
        response = llm_with_tools.invoke(
            messages
        )


    return parser.parse(
        response.content
    )