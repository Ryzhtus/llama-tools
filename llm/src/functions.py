from llm.src.functions_converter import convert_python_function_to_openai_function


def get_document_context_for_summarization() -> str:
    """Use this tool to fetch the entire text of the currently active document only for its summarization."""

    return ""


def search_similar_documents_in_the_database(prompt: str, top_k: str) -> str:
    """Use this tool to find and return fragments from documents in the database that are similar to a provided prompt.
    This is ideal for locating documents that mention specific terms or for compiling information across multiple documents.
    """

    return ""


def request_list_of_documents_names(fragment: str) -> list[str]:
    """Use this tool when asked to provide a simple list of document names that contain a specified fragment."""

    return ""


functions_list = [
    convert_python_function_to_openai_function(get_document_context_for_summarization),
    convert_python_function_to_openai_function(
        search_similar_documents_in_the_database
    ),
    convert_python_function_to_openai_function(request_list_of_documents_names),
]
