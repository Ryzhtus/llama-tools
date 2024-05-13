from typing import (
    TypedDict,
    Optional,
    Union,
    Dict,
    Type,
    Tuple,
    Any,
    Sequence,
    Set,
    List,
    Callable,
    cast,
    Literal,
)
from types import FunctionType, MethodType
from copy import deepcopy
from pydantic import BaseModel
import inspect


PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _rm_titles(kv: dict, prev_key: str = "") -> dict:
    new_kv = {}
    for k, v in kv.items():
        if k == "title":
            if isinstance(v, dict) and prev_key == "properties" and "title" in v.keys():
                new_kv[k] = _rm_titles(v, k)
            else:
                continue
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v, k)
        else:
            new_kv[k] = v
    return new_kv


def _retrieve_ref(path: str, schema: dict) -> dict:
    components = path.split("/")
    if components[0] != "#":
        raise ValueError(
            "ref paths are expected to be URI fragments, meaning they should start "
            "with #."
        )
    out = schema
    for component in components[1:]:
        if component.isdigit():
            out = out[int(component)]
        else:
            out = out[component]
    return deepcopy(out)


def _dereference_refs_helper(
    obj: Any,
    full_schema: Dict[str, Any],
    skip_keys: Sequence[str],
    processed_refs: Optional[Set[str]] = None,
) -> Any:
    if processed_refs is None:
        processed_refs = set()

    if isinstance(obj, dict):
        obj_out = {}
        for k, v in obj.items():
            if k in skip_keys:
                obj_out[k] = v
            elif k == "$ref":
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                full_ref = _dereference_refs_helper(
                    ref, full_schema, skip_keys, processed_refs
                )
                processed_refs.remove(v)
                return full_ref
            elif isinstance(v, (list, dict)):
                obj_out[k] = _dereference_refs_helper(
                    v, full_schema, skip_keys, processed_refs
                )
            else:
                obj_out[k] = v
        return obj_out
    elif isinstance(obj, list):
        return [
            _dereference_refs_helper(el, full_schema, skip_keys, processed_refs)
            for el in obj
        ]
    else:
        return obj


def _infer_skip_keys(
    obj: Any, full_schema: dict, processed_refs: Optional[Set[str]] = None
) -> List[str]:
    if processed_refs is None:
        processed_refs = set()

    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref":
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                keys.append(v.split("/")[1])
                keys += _infer_skip_keys(ref, full_schema, processed_refs)
            elif isinstance(v, (list, dict)):
                keys += _infer_skip_keys(v, full_schema, processed_refs)
    elif isinstance(obj, list):
        for el in obj:
            keys += _infer_skip_keys(el, full_schema, processed_refs)
    return keys


class FunctionDescription(TypedDict):
    """Representation of a callable function to send to an LLM."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


def dereference_refs(
    schema_obj: dict,
    *,
    full_schema: Optional[dict] = None,
    skip_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Try to substitute $refs in JSON Schema."""

    full_schema = full_schema or schema_obj
    skip_keys = (
        skip_keys
        if skip_keys is not None
        else _infer_skip_keys(schema_obj, full_schema)
    )
    return _dereference_refs_helper(schema_obj, full_schema, skip_keys)


def _parse_python_function_docstring(function: Callable) -> Tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    docstring = inspect.getdoc(function)
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith("Returns:") or block.startswith("Example:"):
                # Don't break in case Args come after
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + line.strip()
    return description, arg_descriptions


def _get_python_function_name(function: Callable) -> str:
    """Get the name of a Python function."""
    return function.__name__


def _get_python_function_arguments(function: Callable, arg_descriptions: dict) -> dict:
    """Get JsonSchema describing a Python functions arguments.

    Assumes all function arguments are of primitive types (int, float, str, bool) or
    are subclasses of pydantic.BaseModel.
    """
    properties = {}
    annotations = inspect.getfullargspec(function).annotations
    for arg, arg_type in annotations.items():
        if arg == "return":
            continue
        if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            # Mypy error:
            # "type" has no attribute "schema"
            properties[arg] = arg_type.schema()  # type: ignore[attr-defined]
        elif (
            hasattr(arg_type, "__name__")
            and getattr(arg_type, "__name__") in PYTHON_TO_JSON_TYPES
        ):
            properties[arg] = {"type": PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        elif (
            hasattr(arg_type, "__dict__")
            and getattr(arg_type, "__dict__").get("__origin__", None) == Literal
        ):
            properties[arg] = {
                "enum": list(arg_type.__args__),  # type: ignore
                "type": PYTHON_TO_JSON_TYPES[arg_type.__args__[0].__class__.__name__],  # type: ignore
            }
        if arg in arg_descriptions:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]["description"] = arg_descriptions[arg]
    return properties


def _get_python_function_required_args(function: Callable) -> List[str]:
    """Get the required arguments for a Python function."""
    spec = inspect.getfullargspec(function)
    required = spec.args[: -len(spec.defaults)] if spec.defaults else spec.args
    required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]

    is_function_type = isinstance(function, FunctionType)
    is_method_type = isinstance(function, MethodType)
    if required == []:
        return required
    elif is_function_type and required[0] == "self":
        required = required[1:]
    elif is_method_type and required[0] == "cls":
        required = required[1:]
    return required


def convert_python_function_to_openai_function(
    function: Callable,
) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """
    description, arg_descriptions = _parse_python_function_docstring(function)
    return {
        "name": _get_python_function_name(function),
        "description": description,
        "parameters": {
            "type": "object",
            "properties": _get_python_function_arguments(function, arg_descriptions),
            "required": _get_python_function_required_args(function),
        },
    }


def convert_pydantic_to_openai_function(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    rm_titles: bool = True,
) -> FunctionDescription:
    """Converts a Pydantic model to a function description for the OpenAI API."""
    schema = dereference_refs(model.schema())
    schema.pop("definitions", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": _rm_titles(schema) if rm_titles else schema,
    }


def convert_to_openai_function(
    function: Union[Dict[str, Any], Type[BaseModel], Callable],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI function.

    Args:
        function: Either a dictionary, a pydantic.BaseModel class, or a Python function.
            If a dictionary is passed in, it is assumed to already be a valid OpenAI
            function or a JSON schema with top-level 'title' and 'description' keys
            specified.

    Returns:
        A dict version of the passed in function which is compatible with the
            OpenAI function-calling API.
    """

    # already in OpenAI function format
    if isinstance(function, dict) and all(
        k in function for k in ("name", "description", "parameters")
    ):
        return function
    # a JSON schema with title and description
    elif isinstance(function, dict) and all(
        k in function for k in ("title", "description", "properties")
    ):
        function = function.copy()
        return {
            "name": function.pop("title"),
            "description": function.pop("description"),
            "parameters": function,
        }
    elif isinstance(function, type) and issubclass(function, BaseModel):
        return cast(Dict, convert_pydantic_to_openai_function(function))
    elif callable(function):
        return convert_python_function_to_openai_function(function)
    else:
        raise ValueError(
            f"Unsupported function\n\n{function}\n\nFunctions must be passed in"
            " as Dict, pydantic.BaseModel, or Callable. If they're a dict they must"
            " either be in OpenAI function format or valid JSON schema with top-level"
            " 'title' and 'description' keys."
        )


def get_document_text() -> str:
    """Returns currently selected document's text for the whole document's summarization
    or answering questions about the document."""

    return ""


def find_similiar_fragments(prompt: str, top_k: str) -> str:
    """Returns top K similiar document fragments from the database by a given prompt.
    Useful for question answering or searching for similar document fragments from the database.
    """

    return ""


def find_documents_that_has_fragment(fragment: str) -> list[str]:
    """Returns a list of K document names in which the specified fragment occurs."""

    return ""


functions_list = [
    get_document_text,
    find_similiar_fragments,
    find_documents_that_has_fragment,
]
