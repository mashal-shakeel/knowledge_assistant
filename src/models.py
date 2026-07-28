from pydantic import BaseModel, Field


class KnowledgeResponse(BaseModel):
    answer: str = Field(description="Answer to the user's question.")
    matched_records: list[str] = Field(
        description="IDs of the knowledge base records used."
    )
    sources: list[str] = Field(
        description="Sources used to answer the question."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )
    needs_human_review: bool = Field(
        description="True if there is insufficient evidence."
    )