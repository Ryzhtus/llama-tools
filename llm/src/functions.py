from llm.src.functions_converter import convert_python_function_to_openai_function


def get_current_document_text() -> str:
    """Returns currently selected document's text for the whole document's summarization
    or answering questions about the document."""

    return ""


def request_similiar_documents_contents(prompt: str, top_k: str) -> str:
    """Returns top K similiar documents' fragments from the database by a given prompt.
    Useful for question answering or searching for similar document fragments from the database.
    """

    return ""


def request_list_of_documents_names(fragment: str) -> list[str]:
    """Returns a list of document names in which the specified fragment occurs."""

    return ""


functions_list = [
    convert_python_function_to_openai_function(get_current_document_text),
    convert_python_function_to_openai_function(request_similiar_documents_contents),
    convert_python_function_to_openai_function(request_list_of_documents_names),
]
