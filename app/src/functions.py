import requests

retrieval_url = "http://retrieval_app:8002"


def check_function_call(response: str) -> bool:
    if "<functioncall>" in response:
        return True

    return False


def parse_function_call(
    function_call_str: str,
) -> tuple[str, dict[str, str | int | float]]:
    function_dict = eval(function_call_str.replace("<functioncall>", "").strip())
    function_name = function_dict["name"]
    function_args = function_dict["arguments"]

    return function_name, function_args


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
