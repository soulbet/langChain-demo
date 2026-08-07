# Pydantic
from typing import Optional, Union

from pydantic import BaseModel, Field

from model_factory.model_factory import model_factory


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]

llm=model_factory().create_model()
structured_llm = llm.with_structured_output(FinalResponse)

# 自动选择哪个模式进行格式化
structured_llm.invoke("Tell me a joke about cats")

structured_llm.invoke("How are you today?")