from typing import TypedDict, Annotated, Optional

from ai_app.model_factory.model_factory import model_factory


# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

llm=model_factory().create_model()
structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")