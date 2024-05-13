from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.src.functions import functions_list
import torch

SYSTEM_PROMPT = (
    f"""You are an assistant that helps users extract and process information from their documents. 
Your capabilities include summarization, document-based question-answering, content search by fragment, and finding similar document names.
To fulfill a request, use one of the following functions: {str(functions_list)}."""
    + """Use this format for function calls: <functioncall> {"name": "<function_name>", "arguments": "<arguments_json_string>"}. Name arguments as specified in the function description.
Don't assume document content. If needed, call another tool for more information. For non-document-related queries, chat with the user."""
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
