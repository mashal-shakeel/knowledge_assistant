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

import logging
import time
import uuid

logger = logging.getLogger(__name__)

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


def log_token_usage(response):

    usage = response.response_metadata.get("token_usage")

    if usage:
        logger.info(
            f"Token usage - "
            f"Prompt: {usage['prompt_tokens']}, "
            f"Completion: {usage['completion_tokens']}, "
            f"Total: {usage['total_tokens']}"
        )


def ask_question(question: str):

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(f"Request ID: {request_id}")
    logger.info(f"Question: {question}")

    try:

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
        log_token_usage(response)

        logger.info(
            f"Tool calls: {response.tool_calls}"
        )

        while response.tool_calls:

            messages.append(response)

            for tool_call in response.tool_calls:

                logger.info(
                    f"Executing tool: "
                    f"{tool_call['name']} "
                    f"with args: {tool_call['args']}"
                )

                tool_map = {
                    "lookup_record": lookup_record,
                    "retrieve_knowledge": retrieve_knowledge,
                }

                selected_tool = tool_map.get(
                    tool_call["name"]
                )

                if selected_tool is None:
                    raise ValueError(
                        f"Unknown tool: {tool_call['name']}"
                    )

                tool_result = selected_tool.invoke(
                    tool_call["args"]
                )

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

            response = llm_with_tools.invoke(
                messages
            )

            log_token_usage(response)

        elapsed = time.perf_counter() - start_time

        logger.info(
            f"Execution time: {elapsed:.2f} seconds"
        )

        return parser.parse(
            response.content
        )

    except Exception:

        logger.exception(
            f"Request {request_id} failed."
        )

        raise