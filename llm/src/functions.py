import json
from functools import wraps
from typing import Any, Callable
from pydantic import validate_arguments, BaseModel


class openai_function:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.validate_func = validate_arguments(func)
        self.openai_schema = {
            "name": self.func.__name__,
            "description": self.func.__doc__,
            "parameters": self.validate_func.model.schema(),
        }
        self.model = self.validate_func.model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.validate_func(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def from_response(self, completion, throw_error=True):
        """Execute the function from the response of an openai chat completion"""
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == self.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"])
        return self.validate_func(**arguments)


class OpenAISchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        schema = cls.schema()
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": schema,
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"])
        return cls(**arguments)


@openai_function
def get_document_text() -> str:
    """Returns currently selected document's text for the whole document's summarization"""

    return ""


@openai_function
def find_similiar_fragments(prompt: str, top_k: str) -> str:
    """Returns top K similiar document fragments from the database by a given prompt.
    Useful for question answering or searching for similar document fragments from the database
    """

    return ""


@openai_function
def find_documents_that_has_fragment(fragment: str) -> list[str]:
    """Returns a list of K document names in which the specified fragment occurs."""

    return ""


functions_list = [
    get_document_text.openai_schema,
    find_similiar_fragments.openai_schema,
    find_documents_that_has_fragment.openai_schema,
]
