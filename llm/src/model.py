from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.src.functions_converter import functions_list
import torch

SYSTEM_PROMPT = (
    f"""You are a helpful assistant that helps users to extract and process information from their documents.
Your current capabilities are summarization, document-based question-answering, searching for similar documents' content by provided
fragment and searching the list of document names that are similar to the given text. 
In order to accomplish a user's request that involves any of the listed above capabilities, you have to use one of the following functions: {str(functions_list)}.\n"""
    + """The function must be called only in the following format: <functioncall> {"name": "<function_name>", "arguments": "<arguments_json_string>"}.\n"""
    + """Don't make any assumptions about the required document's content. If after the function call the provided information to you is not enough, you can call another tool
    in order to get more information from the database. 
    In any other scenario that doesn't involve work with the document, you can simply chat with the user."""
)


class GenerativeModel:
    def __init__(self) -> None:
        self.__model_id = "mzbac/llama-3-8B-Instruct-function-calling"

        self.tokenizer = AutoTokenizer.from_pretrained(self.__model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.__model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        self.history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    def generate_response(self, prompt: str) -> str:
        # If the last response was a function call, execute it and return the result
        self.history.append({"role": "user", "content": prompt})

        # Tokenize input and apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            self.history, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # Generate the response from the model
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.1,
        )

        # Decode the response
        llm_response = self.tokenizer.decode(
            outputs[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        # Add interaction to conversation history
        self.history.append({"role": "assistant", "content": llm_response})

        return llm_response

    def reset(self):
        self.history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
