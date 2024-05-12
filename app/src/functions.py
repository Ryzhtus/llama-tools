import requests
import re
import json

retrieval_url = "http://retrieval_app:8002"


def check_function_call(response: str) -> bool:
    if "<functioncall>" in response:
        return True

    return False


def parse_function_call(input_str: str) -> None | dict[str, any]:
    """
    Parses a text string to find and extract a function call.
    The function call is expected to be in the format:
    <functioncall> {"name": "<function_name>", "arguments": "<arguments_json_string>"}
    """
    # Regex pattern to extract 'name' and 'arguments'
    pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*?)\''

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
            # If JSON parsing fails, return None
            return None
    return None


def find_similiar_fragments(prompt: str, top_k: int) -> str:
    response = requests.post(
        retrieval_url + "/search", json={"prompt": prompt, "top_k": 1}
    )

    return response.json()


def find_documents_that_has_fragment(fragment: str) -> list[str]:
    response = requests.post(
        retrieval_url + "/search_similar", json={"fragment": fragment}
    )

    return response.json()


def get_document_text(name: str) -> str:
    response = requests.post(retrieval_url + "/get_document", json={"name": name})

    return response.json()


functions_map = {
    "find_similiar_fragments": find_similiar_fragments,
    "find_documents_that_has_fragment": find_documents_that_has_fragment,
    "get_document_text": get_document_text,
}


# if __name__ == "__main__":
#     function_calls = [
#         "<functioncall> "
#         + str(
#             {
#                 "name": "find_similiar_fragments",
#                 "arguments": '{"prompt": "Where is The University of Bristol registered?", "top_k": "1"}',
#             }
#         ),
#         "<functioncall> "
#         + str(
#             {
#                 "name": "find_documents_that_has_fragment",
#                 "arguments": '{"fragment": "The University of Bristol"}',
#             }
#         ),
#     ]

#     for func in function_calls:
#         if check_function_call(func):
#             func_name, func_args = parse_function_call(func)
#             print(func_args)
#             response = function_call(func_name, func_args)
#             print(response)
