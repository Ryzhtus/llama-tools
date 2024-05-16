import requests
import re
import json

retrieval_url = "http://retrieval_app:8002"


def check_function_call(response: str) -> bool:
    if "<functioncall>" in response:
        return True

    return False


def parse_function_call(input_str: str) -> None | dict[str, any] | str:
    """
    Parses a text string to find and extract a function call.
    The function call is expected to be in the format:
    <functioncall> {"name": "<function_name>", "arguments": "<arguments_json_string>"}
    """
    # Regex pattern to extract 'name' and 'arguments'
    pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*("{.*?}"|\'{.*?}\'|\{.*?\})'

    # Search with regex
    match = re.search(pattern, input_str)
    if match:
        try:
            name = match.group(1)
            arguments_str = match.group(2)

            # Parse the arguments JSON
            arguments = json.loads(arguments_str)

            return {"name": name, "arguments": arguments}
        except json.JSONDecodeError:
            # If JSON parsing fails, return an error message
            return {"error": "An error during function call parsing has occurred."}
    return None


def search_similar_documents_in_the_database(prompt: str, top_k: int) -> str:
    response = requests.post(
        retrieval_url + "/search", json={"prompt": prompt, "top_k": 1}
    )

    return response.json()


def request_list_of_documents_names(fragment: str) -> list[str]:
    response = requests.post(
        retrieval_url + "/search_similar", json={"fragment": fragment}
    )

    return response.json()


def get_document_context_for_summarization(name: str) -> str:
    response = requests.post(retrieval_url + "/get_document", json={"name": name})

    return response.json()


functions_map = {
    "search_similar_documents_in_the_database": search_similar_documents_in_the_database,
    "request_list_of_documents_names": request_list_of_documents_names,
    "get_current_document_text": get_document_context_for_summarization,
}
